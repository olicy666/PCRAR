"""三分支融合模块: 残差式融合."""
from __future__ import annotations

import torch
import torch.nn as nn


class PointCloudFusion(nn.Module):
    """残差融合三个分支的输出.

    z = LN(z_xyz + W1·z_feat + W2·z_global)
    """

    def __init__(self, dim: int = 384):
        super().__init__()
        self.W1 = nn.Linear(dim, dim, bias=False)
        self.W2 = nn.Linear(dim, dim, bias=False)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(
        self,
        z_xyz: torch.Tensor,
        z_feat: torch.Tensor,
        z_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        All inputs: (B, D)
        Output: (B, D)
        """
        return self.layer_norm(z_xyz + self.W1(z_feat) + self.W2(z_global))

