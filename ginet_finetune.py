import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, GlobalAttention
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from mamba_ssm import Mamba

num_atom_type = 119  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5  # including aromatic and self-loop edge
num_bond_direction = 3


class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """

    def __init__(self,
                 task='classification', num_layer=5, emb_dim=300, feat_dim=512 ,graph_dim=512,
                 drop_ratio=0, pool='mean', pred_n_layer=2, pred_act='elu',
                 fingerprint_list=None,
                 ecfp_bits=2048,
                 maccs_bits=167,
                 ap_bits=2048,
                 ext_bits=2048,
                 torsion_bits=2048,
                 avalon_bits=1024,
                 fp_hidden_dim=512
                 ):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.graph_dim = graph_dim
        self.drop_ratio = drop_ratio
        self.task = task

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # —— 动态计算总指纹长度 —— #
        total_bits = 0
        if 'ecfp' in fingerprint_list: total_bits += ecfp_bits
        if 'maccs' in fingerprint_list: total_bits += maccs_bits
        if 'ap' in fingerprint_list: total_bits += ap_bits
        if 'ext' in fingerprint_list: total_bits += ext_bits

        if 'torsion' in fingerprint_list: total_bits += torsion_bits
        if 'avalon' in fingerprint_list: total_bits += avalon_bits

        self.fp_total_bits = total_bits
        self.fp_hidden_dim = fp_hidden_dim

        self.desc_dim = 8  # 你 desc_dict 现在是 8 个
        self.geom_dim = 13  # 你 3D 特征是 13 维

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        # 换成全局注意力池化了
        # if pool == 'mean':
        #     self.pool = global_mean_pool
        # elif pool == 'max':
        #     self.pool = global_max_pool
        # elif pool == 'add':
        #     self.pool = global_add_pool
        self.feat_lin = nn.Linear(self.emb_dim, self.graph_dim)

        if self.task == 'classification':
            out_dim = 2
        elif self.task == 'regression':
            out_dim = 1

        # ---- Mamba (optional) ----
        self.use_mamba = True  # 先写死，后续你可改成从 config 读
        if self.use_mamba:
            self.mamba = Mamba(
                d_model=emb_dim,
                d_state=16,
                d_conv=4,
                expand=2,
            )
            self.mamba_ln = nn.LayerNorm(emb_dim)

        # ====== Node-level fusion gate (GNN vs Mamba) ======
        self.node_mix_gate = nn.Linear(2 * emb_dim, 1)
        self.node_mix_alpha = nn.Parameter(torch.tensor(0.0))  # 初始化为0：尽量退化成原模型

        # --- 全局注意力池化 ---=============================================================
        # gate_nn: 输入 emb_dim -> 输出 1 -> sigmoid 得到注意力分数
        self.att_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(emb_dim, 1),
                nn.Sigmoid()
            )
        )

        # ====== Attention reweighting (Mamba-conditioned)  ======
        self.attn_delta = nn.Linear(2 * emb_dim, 1)
        self.attn_scale = nn.Parameter(torch.tensor(0.0))  # 初始化为0：注意力权重保持原样

        # ====== Graph-level Mamba pooling  gated fusion ======
        self.graph_mix_gate = nn.Linear(2 * emb_dim, 1)
        self.graph_mix_alpha = nn.Parameter(torch.tensor(0.0))  # 初始化为0：图级仍用原attention poo



        # ---------------- 建立预测头（Pred Head） ----------------
        self.feat_dim = feat_dim

        self.pred_n_layer = max(1, pred_n_layer)
        if pred_act == 'relu':
            pred_head = [
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.ReLU(inplace=True)
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.feat_dim // 2, self.feat_dim // 2),
                    nn.ReLU(inplace=True),
                ])
            pred_head.append(nn.Linear(self.feat_dim // 2, out_dim))
        elif pred_act == 'softplus':
            pred_head = [
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.Softplus()
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.feat_dim // 2, self.feat_dim // 2),
                    nn.Softplus()
                ])
            pred_head.append(nn.Linear(self.feat_dim // 2, out_dim))
        elif pred_act == 'elu':
            pred_head = [
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.ELU(inplace=True)
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.feat_dim // 2, self.feat_dim // 2),
                    nn.ELU(inplace=True)
                ])
            pred_head.append(nn.Linear(self.feat_dim // 2, out_dim))
        else:
            raise ValueError('Undefined activation function')


        # pred_head.append(nn.Linear(self.feat_dim//2, out_dim))
        self.pred_head = nn.Sequential(*pred_head)

        # ========（2）在这里对 pred_head 里的所有 Linear 调用 reset_parameters() ========
        def init_linear(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        self.pred_head.apply(init_linear)

        # #  门控
        # # gate_h: projects fused features to gating weights for h
        # self.gate_h = nn.Linear(self.graph_dim + self.fp_hidden_dim, self.graph_dim)
        # # gate_fp: projects fused features to gating weights for fp_mapped
        # self.gate_fp = nn.Linear(self.graph_dim + self.fp_hidden_dim, self.fp_hidden_dim)
        # self.gate_h.apply(init_linear)
        # self.gate_fp.apply(init_linear)
        # # —— 修改1：bias 初始为 -1，让 sigmoid(bias)≈0.27，梯度更活跃 ——
        # nn.init.constant_(self.gate_h.bias, -1.0)
        # nn.init.constant_(self.gate_fp.bias, -1.0)

        # ========（2）在这里对 pred_head 里的所有 Linear 调用 reset_parameters() ========

        # elu relu gelu
        self.fp_mlp = nn.Sequential(
            nn.Linear(self.fp_total_bits, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_ratio),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_ratio),  # 手动添加了一下 让dropout为0.3
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_ratio),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_ratio),
            nn.Linear(512, fp_hidden_dim),
            nn.LayerNorm(fp_hidden_dim),
        )
        # self.fp_mlp = nn.Sequential(
        #     nn.Linear(self.fp_total_bits, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(self.drop_ratio),
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(self.drop_ratio),
        #     nn.Linear(512, fp_hidden_dim),
        #     nn.LayerNorm(fp_hidden_dim),
        # )
        self.fp_mlp.apply(init_linear)

        # ===== 描述符 + 3D 分支 =====
        self.descgeom_mlp = nn.Sequential(
            nn.Linear(self.desc_dim + self.geom_dim, 22),
            nn.LayerNorm(22),
            nn.GELU(),
            nn.Dropout(self.drop_ratio),
            nn.Linear(22, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(self.drop_ratio),
            nn.Linear(64, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            # nn.GELU(),
            # nn.Dropout(self.drop_ratio)
        )
        self.descgeom_mlp.apply(init_linear)



        # self.mod_dropout = nn.Dropout(self.drop_ratio)

        # 3-way gating (scalar weights per sample). No Transformer.
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.feat_dim * 3 + 1, self.feat_dim),
            nn.GELU(),
            nn.Dropout(self.drop_ratio),
            nn.Linear(self.feat_dim, 3)
        )
        # ✅ gate 初始尽量别偏某一路，否则容易造成某些路被完全忽略 初始更均匀
        last = self.fusion_gate[-1]  # nn.Linear(feat_dim, 3)
        nn.init.zeros_(last.bias)
        nn.init.normal_(last.weight, mean=0.0, std=1e-3)

        self.fusion_ln = nn.LayerNorm(self.feat_dim)




    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr


        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training) # [batch_size, emb_dim]
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training) # [batch_size, emb_dim]

        # h = self.pool(h, data.batch)  # [batch_size, emb_dim]

        h_gnn = h # 保留一份纯GNN节点表征（并行两路的第一路）

        # ---- Mamba over nodes (per-graph): 第二路 ----
        if getattr(self, "use_mamba", False):
            # data.batch: [num_nodes_total]，每个节点属于哪个图
            num_graphs = int(data.batch.max().item()) + 1
            counts = torch.bincount(data.batch, minlength=num_graphs).tolist()  # 每张图节点数
            # h_list = torch.split(h, counts, dim=0)
            h_list = torch.split(h_gnn, counts, dim=0)

            out_list = []
            for hi in h_list:
                # Mamba expects [B, L, D]
                yi = self.mamba(hi.unsqueeze(0)).squeeze(0)  # [L, D]
                yi = self.mamba_ln(yi)
                out_list.append(yi)
                # out_list.append(hi + yi)  # residual
            h_mamba = torch.cat(out_list, dim=0)
        else:
            h_mamba = h_gnn
            # h = torch.cat(out_list, dim=0)

        # ---- Node-level gated fusion: h_node ----
        g_node = torch.sigmoid(self.node_mix_gate(torch.cat([h_gnn, h_mamba], dim=-1)))  # [N,1]
        # 退化保护：alpha=0 时严格退回到 h_gnn
        h_node = h_gnn + self.node_mix_alpha * g_node * (h_mamba - h_gnn)

        # ---- Attention pooling (explicit weights) ----
        # 基准注意力（保持与你原来 att_pool.gate_nn 一致）
        score_base = self.att_pool.gate_nn(h_gnn).squeeze(1)  # [N]
        # Mamba条件重加权（scale=0 时完全不改变）
        delta = self.attn_delta(torch.cat([h_gnn, h_mamba], dim=-1)).squeeze(1)  # [N]
        score = score_base * (1.0 + self.attn_scale * torch.tanh(delta))

        node_attn = softmax(score, data.batch)  # [N] 每图归一化（可解释：最终权重）
        h_att = global_add_pool(h_node * node_attn.unsqueeze(-1), data.batch)  # [B, emb_dim]

        # ---- Mamba pooling as an auxiliary readout ----
        h_mpool = global_mean_pool(h_mamba, data.batch)  # [B, emb_dim]
        g_graph = torch.sigmoid(self.graph_mix_gate(torch.cat([h_att, h_mpool], dim=-1)))  # [B,1]
        # 退化保护：alpha=0 时严格用 attention pooling
        h = h_att + self.graph_mix_alpha * g_graph * (h_mpool - h_att)






        # # --- 全局注意力池化 -----------------------==================
        # node_attn = self.att_pool.gate_nn(h).squeeze(1)
        # h = self.att_pool(h, data.batch)  # [batch_size, emb_dim]

        h = self.feat_lin(h) # [batch_size, feat_dim]

        # --------------- 指纹映射与拼接 ------------------
        # data.fp 形状应为 [batch_size, fp_total_bits]（从 Dataset 而来）
        fp = data.fp.view(-1, self.fp_total_bits).float()  # 确保是 [batch_size, fp_total_bits]
        # fp = torch.log1p(fp)                  # 如果是count 指纹（Morgan count / AtomPair count） 这种强烈建议做 log1p（最简单有效）
        h_fp = self.fp_mlp(fp)  # 先映射：[batch_size, fp_hidden_dim]

        desc = data.desc.view(-1, self.desc_dim).float()
        geom = data.geom.view(-1, self.geom_dim).float()
        # geom_mask：没有就当全有效
        if hasattr(data, "geom_mask"):
            geom_mask = data.geom_mask.view(-1, 1).float()
        else:
            geom_mask = torch.ones((desc.size(0), 1), device=desc.device, dtype=desc.dtype)
        geom = geom * geom_mask
        desc_geom = torch.cat([desc, geom], dim=-1)
        h_dg = self.descgeom_mlp(desc_geom)

        gate_vec = self.fusion_gate(torch.cat([h, h_fp, h_dg,geom_mask], dim=-1))
        gate_vec = F.softmax(gate_vec, dim=-1)
        h_fused = gate_vec[:, 0:1] * h + gate_vec[:, 1:2] * h_fp + gate_vec[:, 2:3] * h_dg
        h_fused = self.fusion_ln(h_fused)
        out = self.pred_head(h_fused)


        # # 2) 拼接得到 z  11111111111111111111111111111111111
        # z = torch.cat([h, fp_mapped], dim=1)  # [batch_size, feat_dim_without_fp + fp_hidden_dim]
        #
        # # 3) 通过 gate_h, gate_fp 得到两个 “0~1” 之间的门值
        # gate_h = torch.sigmoid(self.gate_h(z))  # [batch_size, feat_dim_without_fp]
        # gate_fp = torch.sigmoid(self.gate_fp(z))  # [batch_size, fp_hidden_dim]
        #
        # # 4) 对原始特征加权
        # h_weighted = gate_h * h  # [batch_size, feat_dim_without_fp]
        # fp_mapped_weighted = gate_fp * fp_mapped  # [batch_size, fp_hidden_dim]
        #
        # # 门控后拼接
        # gated_z = torch.cat([h_weighted, fp_mapped_weighted], dim=1)
        # gate_vec = torch.cat([gate_h, gate_fp], dim=1)

        # # A.标量权重门控（强烈推荐，稳定、好调）
        # # 三路 gate：输入 concat 后输出 3 个logits
        # self.gate = nn.Sequential(
        #     nn.Linear(3 * fusion_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 3)
        # )
        # h_gnn = h_gnn  # [B,512] 你的GNN输出
        # h_fp = self.fp_mlp(fp)  # [B,512]
        # h_dg = self.dg_mlp(dg)  # [B,512]
        #
        # gate_in = torch.cat([h_gnn, h_fp, h_dg], dim=-1)  # [B, 1536]
        # w = torch.softmax(self.gate(gate_in), dim=-1)  # [B,3]
        #
        # h_fused = (
        #         w[:, 0:1] * h_gnn +
        #         w[:, 1:2] * h_fp +
        #         w[:, 2:3] * h_dg
        # )
        # h_fused = self.fusion_norm(h_fused)
        #
        # # B.向量门控（更强表达，但更容易过拟合）
        # self.gate_vec = nn.Linear(3 * fusion_dim, 3 * fusion_dim)
        #
        # # forward
        # g = self.gate_vec(gate_in).view(-1, 3, fusion_dim)  # [B,3,512]
        # g = torch.softmax(g, dim=1)  # 在三路上softmax
        # h_fused = g[:, 0, :] * h_gnn + g[:, 1, :] * h_fp + g[:, 2, :] * h_dg
        # h_fused = self.fusion_norm(h_fused)

        # 残差融合：fused = raw_z + gate_vec * (gated_z - raw_z)
        # fused = z + gate_vec * (gated_z - z)

        # fused = torch.cat([h_weighted, fp_mapped_weighted], dim=1)  # 形状 [batch_size, feat_dim + fp_hidden_dim]

# fused 计算pca或者 tsne可视化

# 生成6种描述符  保存画测试集合的图


        # 将 GIN 特征（h）与指纹特征（fp_mapped）拼接
        # fused = torch.cat([h, fp_mapped], dim=1)  # 形状 [batch_size, feat_dim + fp_hidden_dim]
        # -------------------------------------------------





        # return h, self.pred_head(h)
        # return h_fused, out, node_attn
        return h_fused, out, node_attn,  gate_vec  # 第5个是 gate_vec

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
