import json
import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
# File-level utilities
import utils  # Ensure local utils.py is imported

import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, precision_recall_curve, auc, \
    cohen_kappa_score, average_precision_score

from dataset.dataset_test import MolTestDatasetWrapper

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef



def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('config_finetune_ulog.yaml', os.path.join(model_checkpoints_folder, 'config_finetune_ulog.yaml'))


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    # Normalize
    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    # De-normalize
    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.classification_metric = self._normalize_cls_metric(
            self.config.get('classification_metric', 'auc')
        )
        self.classification_report_metric = self._normalize_cls_metric(
            self.config.get('classification_report_metric', self.classification_metric)
        )
        self.classification_scheduler_metric = self._normalize_cls_scheduler_metric(
            self.config.get('classification_scheduler_metric', self.classification_metric)
        )
        self.label_smoothing = float(self.config.get('label_smoothing', 0.0))
        # Logit Adjustment strength for imbalanced classification.
        # 0.0 means disabled.
        self.logit_adjust_tau = float(self.config.get('logit_adjust_tau', 0.0))
        self.log_prior = None
        self.roc_auc = None
        self.auprc = None

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['task_name'] + '_' + config['dataset']['target']
        log_dir = os.path.join('scaffold/finetune', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
            device_name = torch.cuda.get_device_name(device)
        else:
            device = 'cpu'
            device_name = 'CPU'
        # print("Running on:", device)
        print(f"Running on: {device} ({device_name})")

        return device

    @staticmethod
    def _normalize_cls_metric(metric_name):
        name = str(metric_name).strip().lower()
        if name in {'auprc', 'aupr', 'ap', 'average_precision', 'pr_auc', 'pr-auc'}:
            return 'auprc'
        return 'auc'

    @staticmethod
    def _normalize_cls_scheduler_metric(metric_name):
        name = str(metric_name).strip().lower()
        if name in {'loss', 'val_loss', 'cross_entropy', 'ce'}:
            return 'loss'
        if name in {'auprc', 'aupr', 'ap', 'average_precision', 'pr_auc', 'pr-auc'}:
            return 'auprc'
        return 'auc'

    def _set_class_weight_from_trainloader(self, train_loader):
        """Set class weights for CE loss using train-set positive/negative ratio."""
        if self.config['dataset']['task'] != 'classification':
            return

        ys = []
        for batch in train_loader:
            y = batch.y

            # Compatible with y shapes: [B], [B,1], [B,T] (use first column for multitask)
            if y.dim() > 1:
                y = y[:, 0]

            y = y.view(-1).detach().cpu()

            # Filter invalid labels (e.g. -1), keep only binary 0/1
            mask = (y == 0) | (y == 1)
            y = y[mask]
            if y.numel() > 0:
                ys.append(y)

        if len(ys) == 0:
            print("[class_weight] No valid labels found, keep default CE.")
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            return

        ys = torch.cat(ys, dim=0)
        pos = (ys == 1).sum().item()
        neg = (ys == 0).sum().item()

        # Guard against degenerate splits
        if pos == 0 or neg == 0:
            print(f"[class_weight] pos={pos}, neg={neg}, keep default CE.")
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            return

        w0 = 1.0
        w1 = neg / (pos + 1e-12)
        w1 = min(w1, float(self.config.get('class_weight_cap', 20.0)))

        # Cache class-prior log vector for optional logit adjustment.
        total = pos + neg
        prior = torch.tensor(
            [neg / (total + 1e-12), pos / (total + 1e-12)],
            device=self.device,
            dtype=torch.float
        )
        self.log_prior = torch.log(torch.clamp(prior, min=1e-12))

        weight = torch.tensor([w0, w1], device=self.device, dtype=torch.float)
        self.criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)

        print(f"[class_weight] pos={pos}, neg={neg}, weight=[{w0:.3f}, {w1:.3f}]")
        if self.logit_adjust_tau > 0:
            print(f"[logit_adjust] enabled, tau={self.logit_adjust_tau}, prior={prior.tolist()}")

    def _step(self, model, data, pred, n_iter):
        # get the prediction
        # __, pred,__ = model(data)  # [N, C] forward pass

        if self.config['dataset']['task'] == 'classification':
            # y = data.y
            # if y.dim() > 1:
            #     y = y.squeeze()  # remove singleton dims
            #     if y.dim() == 0:
            #         y = y.unsqueeze(0)  # keep at least 1D
            # y = y.long()  # ensure integer labels
            logits = pred
            if self.logit_adjust_tau > 0 and self.log_prior is not None:
                logits = logits + self.logit_adjust_tau * self.log_prior.unsqueeze(0)
            loss = self.criterion(logits, data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def train(self):

        print("Using fingerprints and dimensions:")
        for fp in self.config['dataset']['fingerprint_list']:
            if fp == 'ecfp':
                print(f"  ECFP: {self.config['dataset']['ecfp_bits']} bits")
            elif fp == 'maccs':
                print(f"  MACCS: {self.config['dataset']['maccs_bits']} bits")
            elif fp == 'ap':
                print(f"  Atom Pair(ap): {self.config['dataset']['ap_bits']} bits")
            elif fp == 'ext':
                print(f"  Extended Fingerprint(ext): {self.config['dataset']['ext_bits']} bits")
            elif fp == 'torsion':
                print(f"  Topological Torsion(torsion): {self.config['dataset']['torsion_bits']} bits")
            elif fp == 'avalon':
                print(f"  Avalon: {self.config['dataset']['avalon_bits']} bits")


        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()
        if self.config['dataset']['task'] == 'classification' and self.config.get('use_class_weight', False):
            self._set_class_weight_from_trainloader(train_loader)

        print(f"train_size:{len(train_loader.sampler)}")
        print(f"valid_size:{len(valid_loader.sampler)}")
        print(f"test_size:{len(test_loader.sampler)}")

        self.normalizer = None
        # if self.config["task_name"] in ['qm7', 'qm9']:
        #     # qm7/qm9 often benefits from label normalization
        #     labels = []
        #     for batch in train_loader:
        #         labels.append(batch.y)
        #     labels = torch.cat(labels)
        #     self.normalizer = Normalizer(labels)
        #     print(self.normalizer.mean, self.normalizer.std, labels.shape)
        #
        if self.config['model_type'] == 'gin':
            from models.ginet_finetune import GINet
            fps = self.config['dataset']['fingerprint_list']
            model = GINet(self.config['dataset']['task'],fingerprint_list=fps, **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        # elif self.config['model_type'] == 'gcn':
        #     # from models.gcn_finetune import GCN
        #     model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
        #     model = self._load_pre_trained_weights(model)

        # head_keywords = ['pred_head', 'cmf_mlp', 'descgeom_mlp', 'fusion_gate', 'fusion_norm']
        head_keywords = [
            'pred_head', 'fp_mlp', 'descgeom_mlp', 'fusion_gate', 'fusion_ln',
            'mamba', 'mamba_ln',
            'node_mix_gate', 'node_mix_alpha',
            'attn_delta', 'attn_scale',
            'graph_mix_gate', 'graph_mix_alpha',
            # 'att_pool', 'feat_lin', 'node_score_lin'
        ]
        layer_list = []
        for name, param in model.named_parameters():
            if any(k in name for k in head_keywords):
                # print(name, param.requires_grad)
                layer_list.append(name)
        params = [p for n, p in model.named_parameters() if n in layer_list]
        base_params = [p for n, p in model.named_parameters() if n not in layer_list]  # backbone params
        # optimizer = torch.optim.Adam(
        #     [
        #         {'params': base_params, 'lr': self.config['init_base_lr']},
        #         {'params': params}
        #     ],
        #     self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        # )
        optimizer = torch.optim.AdamW(
            [
                {'params': base_params, 'lr': self.config['init_base_lr']},
                {'params': params}
            ],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )
        print(model)

        # ====== Freeze/Unfreeze helpers ======
        def freeze_backbone(base_params, params, optimizer=None, set_base_lr_zero=True):
            """
            Freeze backbone: only train head params.
            - Keep optimizer param groups unchanged.
            - Optionally set backbone lr to 0 for safety.
            """
            for p in base_params:
                p.requires_grad = False
            for p in params:
                p.requires_grad = True
            if optimizer is not None and set_base_lr_zero:
                optimizer.param_groups[0]['lr'] = 0.0  # set backbone lr to 0 during freeze
        def unfreeze_backbone(base_params, optimizer, base_lr):
            """
            Unfreeze backbone and restore backbone learning rate.
            """
            for p in base_params:
                p.requires_grad = True
            optimizer.param_groups[0]['lr'] = float(base_lr)
            # if head_lr is not None:
            #     optimizer.param_groups[1]['lr'] = float(head_lr)

        def print_group_lrs(optimizer, prefix=""):
            lrs = [g.get('lr', None) for g in optimizer.param_groups]
            print(f"{prefix} lrs = {lrs}  (0=base, 1=head)")


        # if apex_support and self.config['fp16_precision']:
        #     model, optimizer = amp.initialize(
        #         model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
        #     )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        model_result_folder = os.path.join(self.writer.log_dir, 'result')
        best_ckpt_path = os.path.join(model_checkpoints_folder, 'model.pth')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = -np.inf
        early_stop_patience = self.config.get('early_stop_patience', None)
        patience_counter = 0

        freeze_epochs = self.config.get('freeze_epochs', 5)
        scheduler = None  # Create scheduler after unfreeze
        # Freeze backbone before training starts
        freeze_backbone(base_params, params, optimizer, set_base_lr_zero=True)
        print_group_lrs(optimizer, "[after freeze]")

        for epoch_counter in range(self.config['epochs']):
            if epoch_counter == freeze_epochs:
                if self.config.get('restore_best_before_unfreeze', True) and os.path.exists(best_ckpt_path):
                    model.load_state_dict(torch.load(best_ckpt_path, map_location=self.device))
                    print(f"[epoch {epoch_counter}] restored best checkpoint before unfreeze.")

                unfreeze_base_lr = float(self.config.get('unfreeze_base_lr', self.config['init_base_lr']))
                unfreeze_head_lr = self.config.get('unfreeze_head_lr', None)
                unfreeze_backbone(
                    base_params, optimizer, base_lr=unfreeze_base_lr,
                )
                if unfreeze_head_lr is not None and len(optimizer.param_groups) > 1:
                    optimizer.param_groups[1]['lr'] = float(unfreeze_head_lr)
                    print(f"[epoch {epoch_counter}] set head lr to {optimizer.param_groups[1]['lr']}")

                if self.config.get('reset_early_stop_on_unfreeze', True):
                    patience_counter = 0
                    print(f"[epoch {epoch_counter}] reset early-stop patience counter.")

                print_group_lrs(optimizer, f"[epoch {epoch_counter} unfreeze]")
                min_lr = float(self.config.get('min_lr', 1e-6))
                factor = float(self.config.get('lr_factor', 0.5))
                patience = int(self.config.get('lr_patience', 3))
                if self.config['dataset']['task'] == 'regression':
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=factor,patience=patience,min_lr = min_lr
                    )
                else:
                    scheduler_mode = 'min' if self.classification_scheduler_metric == 'loss' else 'max'
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode=scheduler_mode, factor=factor, patience=patience, min_lr=min_lr
                    )
                # Optional alternative:
                # scheduler = torch.optim.lr_scheduler.StepLR(
                #     optimizer,
                #     step_size=self.config.get('lr_step_size', 10),
                #     gamma=self.config.get('lr_gamma', 0.5),
                # )

            # Print current lr at the beginning of each epoch
            for i, param_group in enumerate(optimizer.param_groups):
                print(f"Epoch {epoch_counter} lr[{i}] = {param_group['lr']}")

            # Initialize accumulators for this epoch
            epoch_loss = 0.0
            num_samples = 0
            train_preds = []
            train_labels = []

            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                # Forward pass and loss
                __, pred,__,gate_vec = model(data)  # [N, C]
                loss = self._step(model, data, pred, n_iter)

                # Accumulate loss
                batch_size = data.y.size(0)
                epoch_loss += loss.item() * batch_size
                num_samples += batch_size

                if self.config['dataset']['task'] == 'classification':
                    probs = F.softmax(pred, dim=-1)[:, 1].detach().cpu().numpy()
                    labels = data.y.flatten().cpu().numpy()
                    train_preds.extend(probs.tolist())
                    train_labels.extend(labels.tolist())
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                loss.backward()
                # Skip non-finite loss
                if not torch.isfinite(loss):
                    print("Non-finite loss, skip step")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

                optimizer.step()
                n_iter += 1

            # Compute and print training metrics for this epoch
            avg_loss = epoch_loss / num_samples
            if self.config['dataset']['task'] == 'classification' and num_samples > 0:
                train_auc = roc_auc_score(train_labels, train_preds)
                train_preds_bin = [1 if p > 0.5 else 0 for p in train_preds]
                train_acc = accuracy_score(train_labels, train_preds_bin)
                train_f1 = f1_score(train_labels, train_preds_bin)
                train_precision = precision_score(train_labels, train_preds_bin)
                train_recall = recall_score(train_labels, train_preds_bin)
                train_mcc = matthews_corrcoef(train_labels, train_preds_bin)
                train_auprc = average_precision_score(train_labels, train_preds)
                print(
                    f"==> Epoch {epoch_counter} train: avg_loss={avg_loss:.4f}, AUC={train_auc:.4f}, ACC={train_acc:.4f},Auprc={train_auprc:.4f}")

                # Write to TensorBoard
                self.writer.add_scalar('train/loss', avg_loss, epoch_counter)
                self.writer.add_scalar('train/acc', train_acc, epoch_counter)
                self.writer.add_scalar('train/auc', train_auc, epoch_counter)
                self.writer.add_scalar('train/f1', train_f1, epoch_counter)
                self.writer.add_scalar('train/precision', train_precision, epoch_counter)
                self.writer.add_scalar('train/recall', train_recall, epoch_counter)
                self.writer.add_scalar('train/mcc', train_mcc, epoch_counter)

            else:
                print(f"==> Epoch {epoch_counter} train: avg_loss={avg_loss:.4f}")


            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0 or epoch_counter == self.config['epochs']-1 :
                improved = False
                if self.config['dataset']['task'] == 'classification':
                    valid_loss, valid_auc, valid_auprc, valid_acc, valid_f1, valid_precision, valid_recall, valid_mcc \
                        = self._validate(model, valid_loader)
                    valid_main_metric = valid_auprc if self.classification_metric == 'auprc' else valid_auc
                    if self.classification_scheduler_metric == 'loss':
                        valid_scheduler_metric = valid_loss
                    elif self.classification_scheduler_metric == 'auprc':
                        valid_scheduler_metric = valid_auprc
                    else:
                        valid_scheduler_metric = valid_auc
                    if scheduler is not None:
                        scheduler.step(valid_scheduler_metric)
                    if valid_main_metric > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_main_metric
                        torch.save(model.state_dict(), best_ckpt_path)
                        print(
                            f"save vaild epoch:{epoch_counter} , "
                            f"valid_auc:{valid_auc},valid_auprc:{valid_auprc},"
                            f"main_metric({self.classification_metric}):{valid_main_metric},"
                            f"scheduler_metric({self.classification_scheduler_metric}):{valid_scheduler_metric},"
                            f"loss:{valid_loss},acc:{valid_acc},f1:{valid_f1},"
                            f"precision:{valid_precision},recall:{valid_recall},mcc:{valid_mcc}"
                        )
                        improved = True
                    self.writer.add_scalar('valid/loss', valid_loss, epoch_counter)
                    self.writer.add_scalar('valid/auc', valid_auc, epoch_counter)
                    self.writer.add_scalar('valid/auprc', valid_auprc, epoch_counter)
                    self.writer.add_scalar('valid/main_metric', valid_main_metric, epoch_counter)
                    self.writer.add_scalar('valid/scheduler_metric', valid_scheduler_metric, epoch_counter)
                    self.writer.add_scalar('valid/acc', valid_acc, epoch_counter)
                    self.writer.add_scalar('valid/f1', valid_f1, epoch_counter)
                    self.writer.add_scalar('valid/precision', valid_precision, epoch_counter)
                    self.writer.add_scalar('valid/recall', valid_recall, epoch_counter)
                    self.writer.add_scalar('valid/mcc', valid_mcc, epoch_counter)
                elif self.config['dataset']['task'] == 'regression':
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    if scheduler is not None:
                        scheduler.step(valid_loss)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), best_ckpt_path)
                        print(f"save vaild epoch:{epoch_counter} , valid_cls:{valid_rgr}")
                        improved = True

                    self.writer.add_scalar('valid/loss', valid_loss, epoch_counter)
                    auc111 = 'mae' if self.config['task_name'] in ['qm7', 'qm8', 'qm9'] else 'rmse'
                    self.writer.add_scalar(f'valid/{auc111}', valid_rgr, epoch_counter)

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)

                valid_n_iter += 1

                # ---- Early Stopping ----
                if early_stop_patience is not None:
                    patience_counter = 0 if improved else (patience_counter + 1)
                    print(f"Early stop patience: {patience_counter}/{early_stop_patience}")
                    if patience_counter >= early_stop_patience:
                        print(f"Early stopping triggered at epoch {epoch_counter}.")
                        break

            # Optional: call StepLR at end of each epoch
            # if scheduler is not None:
            #     scheduler.step()
            #     print_group_lrs(optimizer, f"[epoch {epoch_counter} after StepLR]")

        self._test(model, test_loader, model_checkpoints_folder,model_result_folder)

    def _load_pre_trained_weights(self, model):
        try:
            # Load manually selected pre-trained checkpoint
            checkpoints_folder = os.path.join('./ckpt/Jan27_23-46-22', 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device, weights_only=True)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            # print("-----------------------------------------------")
            # for name, param in model.named_parameters():
            #     print(name, param.requires_grad, param.shape)
            # print("-----------------------------------------------")
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred,__,gate_vec = model(data)
                loss = self._step(model, data, pred, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data

        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print('Validation loss:', valid_loss, 'MAE:', mae)
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:, 1])
            # tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
            # # acc = (tp + tn) / (tp + fp + tn + fn)
            # acc = accuracy_score(labels, np.argmax(predictions, axis=1))


            valid_probs = predictions[:, 1]  # positive-class probability
            # valid_preds = np.argmax(predictions, axis=1)  # predicted class
            # valid_preds = [1 if p > 0.5 else 0 for p in predictions]
            valid_preds = (predictions[:, 1] > 0.5).astype(int)

            valid_acc = accuracy_score(labels, valid_preds)
            valid_f1 = f1_score(labels, valid_preds)
            valid_precision = precision_score(labels, valid_preds)
            valid_recall = recall_score(labels, valid_preds)
            valid_mcc = matthews_corrcoef(labels, valid_preds)
            valid_auc = roc_auc_score(labels, valid_probs)
            pr_auc = average_precision_score(labels, valid_probs)
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc, 'PR AUC:', pr_auc,'Accuracy:', valid_acc)

            # print('Validation loss:', valid_loss, 'ROC AUC:', valid_auc,'Accuracy:', valid_acc)
            return valid_loss, valid_auc, pr_auc, valid_acc, valid_f1, valid_precision, valid_recall, valid_mcc
        return None

    def _test(self, model, test_loader, model_checkpoints_folder=None,model_result_folder=None):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # Initialize storage for outputs
        all_smiles = []
        all_labels = []
        all_preds = []
        all_attns = []

        all_hfused = [] #← 新增：用于 UMAP + 可解释性（直接复制 h_fused）======================
        all_gate = [] #← 新增：直接复制 gate_vec）======================
        # -------------------------

        # Test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                h_fused, pred, node_attn, gate_vec = model(data)
                loss = self._step(model, data, pred, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)


                # Collect batch-level outputs
                smiles_batch = data.z
                label_batch = data.y.flatten().cpu().tolist()
                node_attn = node_attn.cpu().detach().numpy()
                batch_idx = data.batch.cpu().numpy()
                if self.config['dataset']['task'] == 'classification':
                    pred_vals = pred[:, 1].cpu().tolist()
                else:
                    pred_vals = pred.flatten().cpu().tolist()

                for i, smi in enumerate(smiles_batch):
                    mask = (batch_idx == i)
                    attn_per_graph = node_attn[mask].tolist()
                    all_smiles.append(smi)
                    all_labels.append(label_batch[i])
                    all_preds.append(pred_vals[i])
                    all_attns.append(attn_per_graph)

                    all_hfused.append(h_fused[i].cpu().numpy())  #← 新增：直接复制 h_fused ================
                    all_gate.append(gate_vec[i].cpu().numpy())  #← 新增：s收集 gate weights gate_vec ================
                # -------------------------

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data



        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            df = pd.DataFrame({
                'Label': labels,
                'Prediction': predictions.flatten() if predictions.shape[1] == 1 else predictions[:, 1]
            })


            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                self.mae = mean_absolute_error(labels, predictions)
                print('Test loss:', test_loss, 'Test MAE:', self.mae)
                os.makedirs(model_result_folder, exist_ok=True)
                filename = f"{config['task_name']}_mae_{self.mae:.4f}_finetune.csv"
                filepath = os.path.join(model_result_folder, filename)
                df.to_csv(filepath, mode='w', index=False)
                print(f"Saved MAE results to {filepath}")
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)
                os.makedirs(model_result_folder, exist_ok=True)
                filename = f"{config['task_name']}_rmse_{self.rmse:.4f}_finetune.csv"
                filepath = os.path.join(model_result_folder, filename)
                df.to_csv(filepath, mode='w', index=False)
                print(f"Saved RMSE results to {filepath}")

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)


            df = pd.DataFrame({
                'smiles': all_smiles,
                'Label': all_labels,
                'Prediction': all_preds
            })

            from sklearn.metrics import roc_auc_score, average_precision_score

            self.roc_auc = roc_auc_score(labels, predictions[:, 1])
            print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)
            # df.to_csv(
            #     'scaffold/roc/{}_{}_finetune.csv'.format(config['fine_tune_from'], config['task_name']),
            #     mode='a', index=False,
            # )
            now = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. "20250622_203045"
            # Build output filename using timestamp + metric
            out_dir = model_checkpoints_folder
            os.makedirs(out_dir, exist_ok=True)
            filename = f"auc{self.roc_auc:.4f}_{config['dataset']['fingerprint_list']}_{config['task_name']}_{now}_finetune.csv"
            filepath = os.path.join(out_dir, filename)
            # Save DataFrame (use mode='a' if you need append behavior)
            df.to_csv(filepath, mode='w', index=False)
            print(f"Saved ROC results to {filepath}")

            base_name = f"auc{self.roc_auc:.4f}_{config['task_name']}_{now}_finetune"

            # 2) 整理解释性资产
            gate_arr = np.stack(all_gate, axis=0).astype(np.float32)      # [N, 3]
            hfused_arr = np.stack(all_hfused, axis=0).astype(np.float32)  # [N, feat_dim]
            attn_arr = np.array(
            [np.asarray(x, dtype=np.float32) for x in all_attns],
            dtype=object
            )  # 变长，只能 object

            # 3) 统一保存到一个 npz
            assets_path = os.path.join(out_dir, base_name + "_assets.npz")
            np.savez_compressed(
            assets_path,
            sample_id=np.arange(len(all_smiles), dtype=np.int32),
            smiles=np.array(all_smiles),
            labels=np.array(all_labels),
            preds=np.array(all_preds, dtype=np.float32),
            gates=gate_arr,
            hfused=hfused_arr,
            attns=attn_arr,
            num_nodes=np.array([len(x) for x in all_attns], dtype=np.int32),
            )
            print(f"Saved interpretability assets to {assets_path}")
            print(f"gates shape: {gate_arr.shape}")
            print(f"hfused shape: {hfused_arr.shape}")
            print(f"num attn samples: {len(attn_arr)}")



            


            # Calculate confusion matrix and other metrics
            pred_classes = np.argmax(predictions, axis=1)  # Get predicted class labels
            tn, fp, fn, tp = confusion_matrix(labels, pred_classes).ravel()
            sensitivity = tp / (tp + fn)  # True Positive Rate
            specificity = tn / (tn + fp)  # True Negative Rate
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

            # 1) Basic metrics
            # acc = accuracy_score(labels, pred_classes)
            prec = precision_score(labels, pred_classes)
            recall = recall_score(labels, pred_classes)  # same as sensitivity
            f1 = f1_score(labels, pred_classes)
            kappa = cohen_kappa_score(labels, pred_classes)
            # self.roc_auc = roc_auc_score(labels, predictions[:, 1])
            balanced_accuracy = (sensitivity + specificity) / 2
            # 2) PR-AUC
            auprc = average_precision_score(labels, predictions[:, 1])
            self.auprc = auprc


            # 3) Print summary
            print(f"AUC:       {self.roc_auc:.4f}")
            print(f"Accuracy(Acc):  {accuracy:.4f}")
            print('SEN (recall):', sensitivity)
            print('SPE:', specificity)
            print(f"Precision: {prec:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"Balanced Accuracy (BAC): {balanced_accuracy:.4f}")
            print('MCC:', mcc)
            print(f"AUPRC:     {auprc:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"Kappa:     {kappa:.4f}")






            # Prepare arrays
            y_true = np.array(all_labels)
            y_scores = np.array(all_preds)



            


            # # ==================== 新增：Activity Cliff 正确率（高效采样版） ====================
            # print("\n=== Activity Cliff Analysis (subsample 20,000 for speed) ===")
            # # print("\n=== Activity Cliff Analysis (full dataset) ===")
            # from rdkit import Chem
            # from rdkit.Chem import AllChem
            # from rdkit.DataStructs import FingerprintSimilarity
            # import random

            # idx = list(range(len(y_true)))
            # random.seed(42)
            # sample_idx = random.sample(idx, min(20000, len(y_true)))  # 采样2000个，避免太慢 10000 看看 2000没找到  5000的太少了

            # # sample_idx = list(range(len(y_true)))  # 全量计算（推荐用于最终结果）

            # fps = []
            # for i in sample_idx:
            #     mol = Chem.MolFromSmiles(all_smiles[i])
            #     fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None
            #     fps.append(fp)

            # cliff_correct = 0
            # total_cliffs = 0
            # for a in range(len(sample_idx)):
            #     for b in range(a+1, len(sample_idx)):
            #         if fps[a] is None or fps[b] is None: continue
            #         sim = FingerprintSimilarity(fps[a], fps[b])
            #         if sim > 0.85 and y_true[sample_idx[a]] != y_true[sample_idx[b]]:  # cliff
            #             total_cliffs += 1
            #             # 预测是否正确排序活性
            #             if (y_scores[sample_idx[a]] > y_scores[sample_idx[b]] and y_true[sample_idx[a]] > y_true[sample_idx[b]]) or \
            #                (y_scores[sample_idx[a]] < y_scores[sample_idx[b]] and y_true[sample_idx[a]] < y_true[sample_idx[b]]):
            #                 cliff_correct += 1

            # if total_cliffs > 0:
            #     cliff_acc = cliff_correct / total_cliffs
            #     print(f"Activity Cliff Accuracy: {cliff_acc:.4f} ({total_cliffs} cliffs detected)")
            # else:
            #     print("No activity cliffs found in subsample.")

            # # ==================== 新增：UMAP 可视化（像 GNEprop Fig.1c） ====================
            # print("\n=== Generating UMAP (True label vs Predicted score) ===")
            # import umap
            # import matplotlib.pyplot as plt
            # import seaborn as sns

            # features = np.array(all_hfused)  # 使用融合特征（最能体现可解释性）

            # if len(features) > 5000:  # 大数据集自动采样
            #     idx_sample = random.sample(range(len(features)), 5000)
            #     features = features[idx_sample]
            #     y_true_umap = y_true[idx_sample]
            #     y_scores_umap = y_scores[idx_sample]
            # else:
            #     y_true_umap = y_true
            #     y_scores_umap = y_scores

            # reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
            # embedding = reducer.fit_transform(features)

            # plt.figure(figsize=(14, 6))
            # plt.subplot(1, 2, 1)
            # sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=y_true_umap, palette='coolwarm', alpha=0.8, s=20)
            # plt.title('UMAP - Colored by True Label (1=active)')

            # plt.subplot(1, 2, 2)
            # sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=y_scores_umap, palette='viridis', alpha=0.8, s=20)
            # plt.title('UMAP - Colored by Predicted Score')

            # plt.tight_layout()
            # umap_path = f"umap_{self.config['task_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            # plt.savefig(umap_path, dpi=300)
            # plt.close()
            # print(f"UMAP saved to: {umap_path}")

            # ==================== 新增：简单可解释性总结（Top-10 高活性分子 + attention） ====================
            print("\n=== Explainability Summary ===")
            top10_idx = np.argsort(y_scores)[-10:]
            print("Top 10 predicted active molecules:")
            for i in top10_idx:
                print(f"  SMILES: {all_smiles[i][:80]}... | Pred: {y_scores[i]:.4f} | True: {y_true[i]}")



            # # ==================== 新增：Modality gate weights violin plot ====================
            # print("\n=== Modality Contribution Analysis ===")
            

            # all_gate = np.array(all_gate)  # [N, 3]  shape
            # df_gate = pd.DataFrame(all_gate, columns=['Graph', 'Fingerprint', 'Desc+3D'])
            # df_gate['Task'] = self.config['task_name']  # 可后续扩展多任务

            # plt.figure(figsize=(10, 6))
            # sns.violinplot(data=df_gate.melt(id_vars='Task'), x='variable', y='value', palette='Set2')
            # plt.title(f'Modality Gate Weights Distribution - {self.config["task_name"]}')
            # plt.ylabel('Gate Weight')
            # plt.xlabel('Modality')
            # plt.savefig(f"gate_weights_{self.config['task_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300)
            # plt.close()
            # print("Gate weights violin plot saved.")

            # 保存 attention（已有 all_attns，可后续分析关键原子贡献）

            # # 保存结果 CSV（保持原来）
            # now = datetime.now().strftime("%Y%m%d_%H%M%S")
            # out_dir = model_checkpoints_folder
            # os.makedirs(out_dir, exist_ok=True)
            # filename = f"auc{self.roc_auc:.4f}_{config['dataset']['fingerprint_list']}_{config['task_name']}_{now}_finetune.csv"
            # filepath = os.path.join(out_dir, filename)
            # df.to_csv(filepath, mode='w', index=False)
            # print(f"Saved results to {filepath}")

            # Rank samples by predicted score (descending)
            sorted_indices = np.argsort(y_scores)[::-1]
            sorted_labels = y_true[sorted_indices]

            # ==========================================
            # Bootstrap confidence intervals
            # ==========================================
            from sklearn.metrics import roc_auc_score, average_precision_score

            def compute_confidence_interval(y_true, y_scores, metric_fn, n_bootstraps=1000, confidence_level=0.95):
                rng = np.random.RandomState(42)
                bootstrapped_scores = []

                y_true = np.array(y_true)
                y_scores = np.array(y_scores)

                for i in range(n_bootstraps):
                    # Resample with replacement
                    indices = rng.randint(0, len(y_scores), len(y_scores))

                    if len(np.unique(y_true[indices])) < 2:
                        continue

                    score = metric_fn(y_true[indices], y_scores[indices])
                    bootstrapped_scores.append(score)

                sorted_scores = np.sort(bootstrapped_scores)

                # Confidence interval bounds
                alpha = (1 - confidence_level) / 2
                lower_bound = sorted_scores[int(alpha * len(sorted_scores))]
                upper_bound = sorted_scores[int((1 - alpha) * len(sorted_scores))]
                mean_score = np.mean(bootstrapped_scores)
                std_score = np.std(bootstrapped_scores)

                return mean_score, std_score, lower_bound, upper_bound

            # ------------------------------------------------------
            # 4) Bootstrap AUC/AUPRC
            # ------------------------------------------------------
            print("-" * 30)
            print("Running Bootstrap Analysis (n=1000)...")

            # Input arrays
            y_true_np = np.array(all_labels)
            y_scores_np = np.array(all_preds)

            # (1) AUC CI
            auc_mean, auc_std, auc_low, auc_high = compute_confidence_interval(
                y_true_np, y_scores_np, roc_auc_score
            )
            print(f"ROC-AUC (95% CI): {auc_mean:.4f} +/- {auc_std:.4f} [{auc_low:.4f}, {auc_high:.4f}]")

            # (2) AUPRC CI
            auprc_mean, auprc_std, auprc_low, auprc_high = compute_confidence_interval(
                y_true_np, y_scores_np, average_precision_score
            )
            print(f"AUPRC   (95% CI): {auprc_mean:.4f} +/- {auprc_std:.4f} [{auprc_low:.4f}, {auprc_high:.4f}]")
            print("-" * 30)

            # ------------------------------------------------------
            # 5) Precision@K
            # ------------------------------------------------------
            k_values = [30, 100, 300]
            for k in k_values:
                if k <= len(y_true):
                    top_k_labels = sorted_labels[:k]
                    p_at_k = np.sum(top_k_labels) / k
                    print(f"Precision@{k}: {p_at_k:.4f}")
                else:
                    print(f"Precision@{k}: N/A (Test set size < {k})")

            # ------------------------------------------------------
            # 6) Enrichment Factor (EF@x%)
            # ------------------------------------------------------
            ef_percentages = [0.01, 0.001]  # 1% and 0.1%
            n_total = len(y_true)
            n_actives = np.sum(y_true)
            prevalence = n_actives / n_total

            if n_actives > 0:
                for x in ef_percentages:
                    n_x = int(np.ceil(x * n_total))

                    if n_x > 0:
                        top_x_labels = sorted_labels[:n_x]
                        h_x = np.sum(top_x_labels)
                        precision_at_x = h_x / n_x
                        ef = precision_at_x / prevalence
                        print(f"EF@{x * 100}%:   {ef:.4f}")
                    else:
                        print(f"EF@{x * 100}%:   N/A (Sample too small)")
            else:
                print("EF calculation skipped (No positive samples in test set)")

            # ------------------------------------------------------
            # 7) Recall at fixed precision (Precision >= 0.2)
            # ------------------------------------------------------
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

            target_precision = 0.2
            valid_mask = precisions >= target_precision

            if np.any(valid_mask):
                max_recall_at_prec = np.max(recalls[valid_mask])
                print(f"Recall@Prec>={target_precision}: {max_recall_at_prec:.4f}")
            else:
                print(f"Recall@Prec>={target_precision}: 0.0000 (No threshold met precision requirement)")

            print("-" * 30)
            # ==========================================
            # Plot and save PR / hit-rate figures
            # ==========================================

            # 1) Prepare arrays
            y_true_np = np.array(all_labels)
            y_scores_np = np.array(all_preds)

            # 2) Save folder
            img_save_dir = os.path.join('logs', 'img')

            # 3) Shared time tag for output files
            current_time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Add task name in file tag
            task_name = self.config.get('task_name', 'task')
            final_tag = f"{task_name}_{current_time_tag}"

            print(f"Saving plots to {img_save_dir} with tag: {final_tag}")

            try:
                # Draw PR curve
                utils.plot_pr_with_band_and_save(
                    y_true_np,
                    y_scores_np,
                    B=1000,
                    save_folder=img_save_dir,
                    file_tag=final_tag
                )

                # Draw hit-rate curve
                utils.plot_hit_rate_curve_and_save(
                    y_true_np,
                    y_scores_np,
                    max_k=None,
                    save_folder=img_save_dir,
                    file_tag=final_tag
                )

            except Exception as e:
                print(f"Error during plotting: {e}")
                import traceback
                traceback.print_exc()









        # # ---- Optional: save attention outputs ----
        # now = datetime.now().strftime("%Y%m%d_%H%M%S")
        # torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, f'name_{auc}_{now}.csv'))
        # os.makedirs('scaffold/attn', exist_ok=True)
        # # a) Save prediction CSV
        # df = pd.DataFrame({
        #     'smiles': all_smiles,
        #     'label': all_labels,
        #     'pred': all_preds
        # })
        # df.to_csv(f'scaffold/attn/attn_pred_{now}.csv', index=False)
        # # b) Save attention JSON
        # # attn_data = [
        # #     {'smiles': s, 'node_attention': a}
        # #     for s, a in zip(all_smiles, all_attns)
        # # ]
        # # with open(f'scaffold/attn/attn_values_{now}.json', 'w') as f:
        # #     json.dump(attn_data, f, indent=2)
        # print(f"Saved attention results to scaffold/attn/attn_pred_{now}.csv and .json")
        # # -------------------------
        print(f'========== End of testing for {self.config["task_name"]} ==========')


def main(config):
    results = []
    cls_report_metric = FineTune._normalize_cls_metric(
        config.get('classification_report_metric', config.get('classification_metric', 'auc'))
    )
    for _ in range(4):
        dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
        fine_tune = FineTune(dataset, config)
        fine_tune.train()

        # Store the result based on the task type
        if config['dataset']['task'] == 'classification':
            results.append(fine_tune.auprc if cls_report_metric == 'auprc' else fine_tune.roc_auc)
        if config['dataset']['task'] == 'regression':
            if config['task_name'] in ['qm7', 'qm8', 'qm9']:
                results.append(fine_tune.mae)
            else:
                results.append(fine_tune.rmse)

    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
    fine_tune = FineTune(dataset, config)
    fine_tune.train()

    # Store the result based on the task type
    if config['dataset']['task'] == 'classification':
        results.append(fine_tune.auprc if cls_report_metric == 'auprc' else fine_tune.roc_auc)
    if config['dataset']['task'] == 'regression':
        if config['task_name'] in ['qm7', 'qm8', 'qm9']:
            results.append(fine_tune.mae)
        else:
            results.append(fine_tune.rmse)

    # Convert the list to a numpy array for easy computation of mean and std dev
    results = np.array(results)
    if config['dataset']['task'] == 'classification':
        print(f"classification result metric: {cls_report_metric}")
    print(results)
    print(np.mean(results), np.std(results))
    print(f"{np.mean(results):.6f}+/-{np.std(results):.6f}")
    # Return mean and standard deviation of results
    return np.mean(results), np.std(results)


if __name__ == "__main__":
    config = yaml.load(open("config_finetune_ulog.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)

    if config['task_name'] == 'train':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/train.csv'
        target_list = ["label"]

    elif config['task_name'] == 'S. aureus growth inhibition':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/datasets - S. aureus growth inhibition.csv'
        target_list = ["label"]
    elif config['task_name'] == 'HepG2 cytotoxicity':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/datasets - HepG2 cytotoxicity.csv'
        target_list = ["label"]
    elif config['task_name'] == 'HSkMC cytotoxicity':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/datasets - HSkMC cytotoxicity.csv'
        target_list = ["label"]
    elif config['task_name'] == 'IMR-90 viability':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/datasets - IMR-90 viability.csv'
        target_list = ["label"]
    elif config['task_name'] == 'Proton motive force':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/datasets - Proton motive force.csv'
        target_list = ["label"]
    elif config['task_name'] == 'Antibiotics without quinolones':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/datasets - Antibiotics without quinolones.csv'
        target_list = ["label"]
    elif config['task_name'] == 'Antibiotics without betalactams':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/datasets - Antibiotics without betalactams.csv'
        target_list = ["label"]
    elif config['task_name'] == 'neisseria_gonorrhoeae_binarized':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/neisseria_gonorrhoeae_binarized.csv'
        target_list = ["NG (1321/ 38650)"]
    elif config['task_name'] == 'test':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/dachangganjun-41587_2025_2814_MOESM3_ESM.csv'
        target_list = ["Activity_tolC"]
    else:
        raise ValueError('Undefined downstream task!')

    print(config)
    print()
    print(config["task_name"])

    results_list = []
    for target in target_list:
        config['dataset']['target'] = target
        mean, std = main(config)
        results_list.append([target, mean, std])

    print("======================================================")
    # If there are multiple targets, print aggregate mean/std

    avg_mean = np.mean([result[1] for result in results_list])  # average mean across targets
    avg_std = np.mean([result[2] for result in results_list])  # average std across targets
    if len(target_list) > 1:
        print(f"Multi-task {config['task_name']} average results - Mean: {avg_mean:.6f}, Std: {avg_std:.6f}")
        print(f"{avg_mean:.4f}+/-{avg_std:.4f}")
    else:
        print(f"Single task {config['task_name']} average results - Mean: {avg_mean:.6f}, Std: {avg_std:.6f}")
        print(f"{avg_mean:.4f}+/-{avg_std:.4f}")

    os.makedirs('experiments', exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv(
        'scaffold/experiments/{}_{}_finetune.csv'.format(config['fine_tune_from'], config['task_name']),
        mode='a', index=False, header=False
    )
