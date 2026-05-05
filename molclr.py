import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime

import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast # 混合精度支持

# from transformers import AdamW, get_linear_schedule_with_warmup
# from transformers import LAMB  # Hugging Face 提供的 LAMB 实现
# # from transformers import LAMB

from utils import NTXentLoss


# from apex import amp

# apex_support = False
# try:
#     sys.path.append('./apex')
#     from apex import amp
#
#     apex_support = True
# except:
#     print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
#     apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


# 用于实现基于图神经网络（GNN）的分子表示学习任务，结合了监督学习和可能的对比学习（尽管代码中部分对比学习相关内容被注释掉了）。
# 它包括模型训练、验证、预训练权重加载等功能。
class MolCLR(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('ckpt', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数

        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        self.scaler = GradScaler() if config['fp16_precision'] else None  # 混合精度训练
        # 训练稳定性相关配置（config 里没有就用默认值）
        self.grad_clip_norm = float(config.get('grad_clip_norm', 1.0))
        self.nonfinite_max_streak = int(config.get('nonfinite_max_streak', 20))


    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device


    def _step(self, model, data, data_mask, n_iter, epoch_counter):
        with autocast(enabled=self.config['fp16_precision']):  # 混合精度上下文
            # 模型前向传播（返回三个分类任务的预测）
            x, pre_class_label1, pre_class_label2, pre_class_label3 = model(data)
            # loss = self.criterion(x, data.x.flatten())
            # 小于warm_up时，使用mask_loss，否则使用nt_xent_criterion
            # print(epoch_counter, self.config['warm_up'], "-----step-")
            # if epoch_counter < self.config['warm_up']:
            x1, _, _, _ = model(data_mask)
            # 数值稳定：sqrt(0) 的梯度会发散，必须加 eps 避免出现 Inf/NaN 梯度
            diff2 = (x - x1).pow(2).sum(dim=1)
            mask_loss = torch.sqrt(diff2 + 1e-12).mean()

            # if epoch_counter >= self.config['warm_up']:
            #     # print(f"epoch_counter: {epoch_counter}, warm_up threshold: {self.config['warm_up']}")
            #     # get the representations and the projections
            #     zis, _, _, _ = model(xis)  # [N,C]
            #     # get the representations and the projections
            #     zjs, _, _, _ = model(xjs)  # [N,C]
            #
            #     # normalize projection feature vectors
            #     zis = F.normalize(zis, dim=1, eps=1e-8)
            #     zjs = F.normalize(zjs, dim=1, eps=1e-8)
            #     mask_loss = self.nt_xent_criterion(zis, zjs) + mask_loss

            class_loss1 = self.criterion(pre_class_label1, data.y1)
            class_loss2 = self.criterion(pre_class_label2, data.y2)
            class_loss3 = self.criterion(pre_class_label3, data.y3)

            class_loss = class_loss1 + class_loss2 + class_loss3
            loss = mask_loss + class_loss

        return loss

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()
        if self.config['model_type'] == 'gin':
            from models.ginet_molclr import GINet
            model = GINet(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)  # 加载预训练权重
        # elif self.config['model_type'] == 'gcn':                          gin比gcn要好很多
        #     from models.gcn_molclr import GCN
        #     model = GCN(**self.config["model"]).to(self.device)
        #     model = self._load_pre_trained_weights(model)
        else:
            raise ValueError('Undefined GNN model.')
        print(model)

        # model = torch.compile(model)

        print("Model device:", next(model.parameters()).device)

        # 优化器和学习率调度器LAMB
        optimizer = torch.optim.AdamW(
            model.parameters(), self.config['init_lr'],
            weight_decay=eval(self.config['weight_decay'])
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config['epochs'] - self.config['warm_up'],
            eta_min=0, last_epoch=-1
        )

        # if apex_support and self.config['fp16_precision']:
        #     model, optimizer = amp.initialize(
        #         model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
        #     )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            print('Epoch: ', epoch_counter)
            print('Train Loader Batch Count:', len(train_loader))
            print('Train Dataset Size:', len(train_loader.dataset))
            model.train()

            # Adjust batch size dynamically based on epoch
            if epoch_counter < self.config['warm_up']:
                dynamic_batch_size = self.config['batch_size'] // 4  # Smaller batch size in the beginning
            else:
                dynamic_batch_size = self.config['batch_size']  # Larger batch size after warm-up

            # Reinitialize the dataloader with the adjusted batch size
            train_loader, valid_loader = self.dataset.get_data_loaders(batch_size=dynamic_batch_size)

            nonfinite_streak = 0
            for bn, (data, data_mask) in enumerate(train_loader):
                # bn 是 batch number，data 是当前批次的数据。
                optimizer.zero_grad(set_to_none=True)

                data = data.to(self.device)
                data_mask = data_mask.to(self.device)
                # xis = xis.to(self.device)
                # xjs = xjs.to(self.device)

                # print("Batch device:", data.x.device)

                # loss = self._step(model, data, data_mask, xis, xjs, n_iter, epoch_counter)
                loss = self._step(model, data, data_mask, n_iter, epoch_counter)

                # ---- 防止一次 NaN/Inf 把参数污染，后面全程 NaN ----
                if not torch.isfinite(loss):
                    # 注意：这里不要 optimizer.step()
                    nonfinite_streak += 1
                    # 只打印少量，避免日志爆炸
                    if nonfinite_streak <= 5 or (nonfinite_streak % 200 == 0):
                        print(f"[WARN] Non-finite loss. epoch={epoch_counter} bn={bn} n_iter={n_iter} loss={loss.item()}")
                    optimizer.zero_grad(set_to_none=True)
                    n_iter += 1
                    if nonfinite_streak >= self.nonfinite_max_streak:
                        print(f"[FATAL] Non-finite loss streak reached {self.nonfinite_max_streak}. Stop training to avoid wasting compute.")
                        return
                    continue

                if self.config.get('fp16_precision', False):
                    # 混合精度：先 scale -> backward，再 unscale 后做梯度裁剪
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    # 梯度非有限时直接跳过该步，避免 clip 把 inf*0 变成 nan
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    if len(grads) > 0:
                        total_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2)
                        if not torch.isfinite(total_norm):
                            print(f"[WARN] Non-finite grad-norm (fp16). epoch={epoch_counter} bn={bn} n_iter={n_iter} norm={total_norm.item()}")
                            optimizer.zero_grad(set_to_none=True)
                            n_iter += 1
                            continue
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    # 梯度非有限时直接跳过该步，避免 clip 把 inf*0 变成 nan
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    if len(grads) > 0:
                        total_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2)
                        if not torch.isfinite(total_norm):
                            print(f"[WARN] Non-finite grad-norm. epoch={epoch_counter} bn={bn} n_iter={n_iter} norm={total_norm.item()}")
                            optimizer.zero_grad(set_to_none=True)
                            n_iter += 1
                            continue
                    # 梯度裁剪必须在 backward 之后，否则不生效
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip_norm)
                    optimizer.step()

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', float(loss.item()), global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                # if apex_support and self.config['fp16_precision']:
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                #     loss.backward()
                # loss.backward()
                #
                # optimizer.step()
                n_iter += 1

            # warmup for the first few epochs
            if epoch_counter >= self.config['warm_up']:
                scheduler.step()
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                print(epoch_counter, bn, valid_loss, '(validation)')
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    # 保存最佳 覆盖保存
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            if (epoch_counter + 1) % self.config['save_every_n_epochs'] == 0:
                torch.save(model.state_dict(),
                           os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))


    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            # 位置 ./ckpt/pretrained_gin (load_model)  /checkpoints/model.pth
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):  # 验证过程  计算验证集平均损失
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (data, data_mask) in valid_loader:
                data = data.to(self.device)
                data_mask = data_mask.to(self.device)
                # xis = xis.to(self.device)
                # xjs = xjs.to(self.device)

                # loss = self._step(model, data, data_mask, xis, xjs, counter,self.config['warm_up'])

                loss = self._step(model, data, data_mask, counter, epoch_counter=self.config['warm_up'])
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter

        model.train()
        return valid_loss


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)  # 加载配置内容
    print(config)

    if config['aug'] == 'node':
        from dataset.dataset import MoleculeDatasetWrapper
    elif config['aug'] == 'subgraph':
        from dataset.dataset_subgraph import MoleculeDatasetWrapper
    # elif config['aug'] == 'mix':
    #     from dataset.dataset_mix import MoleculeDatasetWrapper    找不到  dataset.dataset_mix
    else:
        raise ValueError('Not defined molecule augmentation!')

    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'])
    molclr = MolCLR(dataset, config)
    molclr.train()


if __name__ == "__main__":
    main()




"""
分子图分割：通过随机掩码原子/键模拟BERT的MLM任务
​MDP任务：预测分子量、LogP、TPSA等物理化学性质
​MFGP任务：识别羟基、胺基等功能团
​联合训练：通过多任务学习融合不同预训练目标
​输出嵌入：使用预训练的GNN层生成固定长度的药物表示
​扩展建议
​数据增强：添加随机旋转/翻转、噪声扰动
​复杂模型：替换为更强大的GNN（如GIN、GraphSAGE）
​跨模态预训练：结合蛋白质序列数据进行联合训练




  # 提取原子特征（原子类型、电荷等）
        atom_features = []
        for atom in mol.GetAtoms():
            atom_feat = [
                atom.GetAtomicNum(),       # 原子类型
                atom.GetFormalCharge(),    # 电荷
                atom.GetTotalDegree(),     # 度数
                atom.GetTotalNumHs(),      # 氢原子数
                atom.IsInRing()            # 是否在环中
            ]
            atom_features.append(atom_feat)
            
        # 提取边特征（键类型）
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_type = bond.GetBondTypeAsDouble()  # 0: SINGLE, 1: DOUBLE, etc.
            edge_attr.append([edge_type])
            edge_attr.append([edge_type])
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        x = torch.tensor(atom_features, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)

"""

