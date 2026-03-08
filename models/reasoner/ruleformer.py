"""RuleFormer: 推断规则向量 r, 并预测目标格表示."""
from __future__ import annotations

import torch
import torch.nn as nn


class RuleFormer(nn.Module):
    """RuleFormer — 规则推断 Transformer.

    输入: 10 个 token ([RULE] + 9 个格子)
    输出:
        - r: 规则向量 (B, D)
        - z_hat_tgt: 预测的 target embedding (B, D)
        - u_rule: [RULE] 输出 token (B, D), 传给 VerifyFormer

    关键设计:
        - 显式 [RULE] token 做规则瓶颈
        - AdaLN/FiLM 调制: r 条件化预测 target
    """

    def __init__(
        self,
        dim: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # Transformer Encoder (Pre-LN)
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

        # 规则向量投影
        self.rule_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        # AdaLN/FiLM 调制: r → γ, β
        self.gamma_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        # 预测头 LayerNorm + MLP
        self.pred_ln = nn.LayerNorm(dim)
        self.pred_head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        target_pos_idx: int = 8,
    ) -> tuple:
        """
        Args:
            tokens: (B, 10, D) — [rule_token, 9×grid_tokens]
            target_pos_idx: target 格在 9 格中的索引 (默认 8, 即 (2,2))

        Returns:
            r: (B, D) 规则向量
            z_hat_tgt: (B, D) 预测的 target embedding
            u_rule: (B, D) [RULE] 输出 token
        """
        # Transformer 编码
        u = self.transformer(tokens)  # (B, 10, D)

        # 提取 [RULE] token 输出 (index 0)
        u_rule = u[:, 0, :]  # (B, D)
        r = self.rule_proj(u_rule)  # (B, D)

        # 提取 target 位置 token (grid token index = target_pos_idx)
        # grid tokens 从 index 1 开始, 所以 target 在 u 中的 index 是 target_pos_idx + 1
        u_tgt = u[:, target_pos_idx + 1, :]  # (B, D)

        # AdaLN/FiLM 调制
        gamma = self.gamma_mlp(r)   # (B, D)
        beta = self.beta_mlp(r)     # (B, D)
        u_tgt_mod = self.pred_ln(u_tgt) * (1.0 + gamma) + beta  # (B, D)

        # 预测 target embedding
        z_hat_tgt = self.pred_head(u_tgt_mod)  # (B, D)

        return r, z_hat_tgt, u_rule

