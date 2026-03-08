"""全局统计特征编码器: 小 MLP."""
from __future__ import annotations

import torch
import torch.nn as nn


class GlobalStatEncoder(nn.Module):
    """全局统计特征 MLP 编码器.

    输入: g = [log(N_raw), C_cell, S_cell, bbox_sizes, rgb_stats, ...] (B, G_dim)
    输出: z_global ∈ (B, D)
    """

    def __init__(self, in_dim: int = 14, hidden_dim: int = 128, out_dim: int = 384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: (B, G_dim) 全局统计特征

        Returns:
            z_global: (B, D)
        """
        return self.mlp(g)

