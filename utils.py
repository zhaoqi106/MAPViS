from typing import Optional

import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.metrics import precision_recall_curve, average_precision_score


def _get_timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def bootstrap_pr_band(y_true, y_score, B=100, grid_size=200, seed=42):
    """
    (内部辅助函数) 通过 Bootstrap 计算 PR 曲线的 95% 置信区间
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    recall_grid = np.linspace(0, 1, grid_size)

    ap_vals = []
    prec_on_grid = []
    valid_bootstraps = 0

    # 尝试 B*2 次以确保获得 B 个有效样本（防止抽样全为正或全为负）
    for _ in range(B * 2):
        if valid_bootstraps >= B:
            break
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2: continue  # 跳过无效抽样

        valid_bootstraps += 1
        p, r, _ = precision_recall_curve(yt, ys)
        # 插值：r 是降序的，需要翻转为升序
        p_interp = np.interp(recall_grid, r[::-1], p[::-1])
        prec_on_grid.append(p_interp)
        ap_vals.append(average_precision_score(yt, ys))

    prec_on_grid = np.vstack(prec_on_grid)
    ap_vals = np.array(ap_vals)

    return {
        "recall_grid": recall_grid,
        "p_lo": np.quantile(prec_on_grid, 0.025, axis=0),
        "p_hi": np.quantile(prec_on_grid, 0.975, axis=0),
        "ap_mean": np.mean(ap_vals),
        "ap_std": np.std(ap_vals)
    }


def plot_pr_with_band_and_save(y_true, y_score, B=100, save_folder=".", file_tag=None):
    """
    绘制 PR 曲线带置信区间
    参数:
      save_folder: 保存路径 (如 'logs/img')
      file_tag: 文件名后缀 (如时间戳)，若为 None 则自动生成
    """
    if file_tag is None:
        file_tag = _get_timestamp_str()

    # 1. 计算
    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    prevalence = np.mean(y_true)
    band = bootstrap_pr_band(y_true, y_score, B=B)

    # 2. 绘图
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(r, p, color='tab:blue', linewidth=2, label=f"PR Curve (AP={ap:.3f})")
    plt.fill_between(band["recall_grid"], band["p_lo"], band["p_hi"],
                     color='tab:blue', alpha=0.2,
                     label=f"95% CI (Bootstrap B={B})")
    plt.hlines(prevalence, 0, 1, colors="gray", linestyles="--",
               label=f"Baseline (Prev={prevalence:.3f})")

    plt.xlabel("Recall");
    plt.ylabel("Precision")
    plt.title(f"PR Curve")
    plt.legend(loc="best");
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(0, 1.05);
    plt.xlim(0, 1.0)

    # 3. 保存
    os.makedirs(save_folder, exist_ok=True)
    filename = f"PR_Curve_{file_tag}.png"
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"[Utils] PR Curve saved to: {save_path}")


def plot_hit_rate_curve_and_save(y_true, y_score, max_k=None, save_folder=".", file_tag=None):
    """
    绘制 Hit-rate @ Budget 曲线
    """
    if file_tag is None:
        file_tag = _get_timestamp_str()

    y_true = np.array(y_true);
    y_score = np.array(y_score)
    order = np.argsort(-y_score)
    yt_sorted = y_true[order]

    cum_hits = np.cumsum(yt_sorted)
    ks = np.arange(1, len(y_true) + 1)
    if max_k is not None:
        ks = ks[:max_k];
        cum_hits = cum_hits[:max_k]

    cum_precision = cum_hits / ks
    prevalence = np.mean(y_true)

    # 创建子图: 左边累计命中数，右边命中率
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # --- 左图: Cumulative Hits ---
    ax1.plot(ks, cum_hits, linewidth=2, label="Model")
    ax1.plot(ks, ks * prevalence, 'k--', alpha=0.5, label="Random")
    ax1.set_xlabel("Budget K");
    ax1.set_ylabel("Cumulative Hits")
    ax1.set_title("Cumulative Hits @ Budget")
    ax1.legend();
    ax1.grid(True, alpha=0.5)

    # --- 右图: Hit Rate ---
    ax2.plot(ks, cum_precision, color='tab:orange', linewidth=2, label="Hit Rate")
    ax2.hlines(prevalence, ks[0], ks[-1], colors="k", linestyles="--", label=f"Random ({prevalence:.1%})")
    ax2.set_xlabel("Budget K");
    ax2.set_ylabel("Hit Rate (Precision)")
    ax2.set_title("Hit Rate @ Budget")
    ax2.legend();
    ax2.grid(True, alpha=0.5)

    # 保存
    os.makedirs(save_folder, exist_ok=True)
    filename = f"HitRate_Curve_{file_tag}.png"
    save_path = os.path.join(save_folder, filename)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Utils] Hit-rate curves saved to: {save_path}")
