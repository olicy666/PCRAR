"""VerifyFormer: 候选验证 + 全局自洽 PE 打分."""
from __future__ import annotations

import torch
import torch.nn as nn


class VerifyFormer(nn.Module):
    """VerifyFormer — 候选验证 Transformer.

    对每个候选 j, 把它插入 target 位置后构造"插入后矩阵 token",
    通过小 Transformer 检查全局自洽性.

    输出:
        - z_rec: 每个格子的重建 embedding (用于观测自洽误差)
        - r_verify: 验证规则向量 (用于规则一致性误差)
    """

    def __init__(
        self,
        dim: int = 384,
        n_layers: int = 3,
        n_heads: int = 6,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * ffn_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # 观测自洽重建头
        self.rec_head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

        # 规则一致性投影
        self.rule_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, tokens: torch.Tensor) -> tuple:
        """
        Args:
            tokens: (B, 10, D) — 插入候选后的矩阵 token 序列
                    [rule_token, 9×grid_tokens (target 位置已替换为候选)]

        Returns:
            z_rec: (B, 9, D) 每个格子的重建 embedding
            r_verify: (B, D) 验证规则向量
        """
        v = self.transformer(tokens)  # (B, 10, D)

        v_rule = v[:, 0, :]   # (B, D)
        v_grid = v[:, 1:, :]  # (B, 9, D)

        r_verify = self.rule_proj(v_rule)   # (B, D)
        z_rec = self.rec_head(v_grid)       # (B, 9, D)

        return z_rec, r_verify

