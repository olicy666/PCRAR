"""特性分支编码器: 轻量 PointNet 风格, 编码逐点特征."""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """PointNet 风格特性编码器.

    输入: 逐点特征 [xyz_global, rgb_norm, ...] (B, N0, F_in)
    输出: z_feat ∈ (B, D)

    结构: SharedMLP + MaxPool
    """

    def __init__(self, in_channels: int = 6, hidden_dim: int = 256, out_dim: int = 384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, out_dim, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, N0, F_in) 逐点特征

        Returns:
            z_feat: (B, D)
        """
        x = feat.permute(0, 2, 1)    # (B, F_in, N0)
        x = self.mlp(x)               # (B, D, N0)
        x = x.max(dim=-1)[0]          # (B, D) — max pooling
        return x

