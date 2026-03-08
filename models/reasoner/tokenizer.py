"""Token 化与位置编码: 将 9 个格子 embedding + [RULE] token 组装为序列."""
from __future__ import annotations

import torch
import torch.nn as nn


class MatrixTokenizer(nn.Module):
    """矩阵 Token 化器.

    将 3×3 格子的 embedding 加上行列位置编码, 连同 [RULE] token
    组装为 Transformer 输入序列 (长度 10):

        [t_rule, t_{0,0}, t_{0,1}, t_{0,2}, t_{1,0}, ..., t_{2,2}]

    - 已观测格子: t = z + E_row[r] + E_col[c]
    - 缺失格子:   t = mask_token_k + E_row[r] + E_col[c]
    - 规则 token: t = w_rule (可学习向量)
    """

    def __init__(self, dim: int = 384, max_missing: int = 3):
        super().__init__()
        self.dim = dim

        # 可学习的行/列位置编码
        self.row_embed = nn.Embedding(3, dim)
        self.col_embed = nn.Embedding(3, dim)

        # 可学习的 [RULE] token
        self.rule_token = nn.Parameter(torch.zeros(1, 1, dim))

        # 多个 mask token (对应不同缺失位置)
        self.mask_tokens = nn.Parameter(torch.zeros(max_missing, dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.rule_token, std=0.02)
        nn.init.trunc_normal_(self.mask_tokens, std=0.02)
        nn.init.trunc_normal_(self.row_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.col_embed.weight, std=0.02)

    def forward(
        self,
        grid_embeddings: torch.Tensor,
        grid_mask: torch.Tensor,
        missing_indices: torch.Tensor,
    ) -> torch.Tensor:
        """组装 token 序列.

        Args:
            grid_embeddings: (B, 9, D) 格子 embedding (缺失格为零)
            grid_mask: (B, 9) bool, True=已观测
            missing_indices: (B, 3) 缺失位置的展平索引 (sorted)

        Returns:
            tokens: (B, 10, D)  [rule, 9×grid]
        """
        B, _, D = grid_embeddings.shape
        device = grid_embeddings.device

        # 位置编码表 (9 个位置)
        rows = torch.arange(3, device=device)
        cols = torch.arange(3, device=device)
        row_enc = self.row_embed(rows)  # (3, D)
        col_enc = self.col_embed(cols)  # (3, D)

        # (3, 3, D) 位置编码网格
        pos_enc = row_enc.unsqueeze(1) + col_enc.unsqueeze(0)  # (3, 3, D)
        pos_enc = pos_enc.reshape(9, D)  # (9, D)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)  # (B, 9, D)

        # 构造 grid tokens
        grid_tokens = grid_embeddings + pos_enc  # (B, 9, D) — 已观测格子

        # 替换缺失位置为 mask token
        for b in range(B):
            for k in range(missing_indices.shape[1]):
                idx = missing_indices[b, k].item()
                if 0 <= idx < 9:
                    grid_tokens[b, idx] = self.mask_tokens[k] + pos_enc[b, idx]

        # [RULE] token
        rule = self.rule_token.expand(B, -1, -1)  # (B, 1, D)

        # 拼接: [rule, 9×grid]
        tokens = torch.cat([rule, grid_tokens], dim=1)  # (B, 10, D)
        return tokens

    def get_pos_enc(self, device: torch.device) -> torch.Tensor:
        """获取 9 个位置的位置编码.

        Returns:
            pos_enc: (9, D)
        """
        rows = torch.arange(3, device=device)
        cols = torch.arange(3, device=device)
        row_enc = self.row_embed(rows)
        col_enc = self.col_embed(cols)
        pos_enc = row_enc.unsqueeze(1) + col_enc.unsqueeze(0)
        return pos_enc.reshape(9, self.dim)

