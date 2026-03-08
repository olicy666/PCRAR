"""PCRAR 训练损失函数."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCRARLoss(nn.Module):
    """PCRAR 综合损失.

    L_total = L_ce + λ · L_margin + μ · L_rule

    组成:
        1. L_ce: 4选1 交叉熵 (主损失)
        2. L_margin: PE margin 辅助损失 (正确候选能量更小)
        3. L_rule: 规则分类辅助损失 (可选)
    """

    def __init__(
        self,
        lambda_margin: float = 0.1,
        mu_rule: float = 0.2,
        tau: float = 0.7,
    ):
        super().__init__()
        self.lambda_margin = lambda_margin
        self.mu_rule = mu_rule
        self.tau = tau
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """计算总损失.

        Args:
            outputs: 模型输出, 包含:
                - scores: (B, 4) 候选得分
                - energies: (B, 4) 候选能量
                - rule_logits: (B, C) 规则分类 logits (可选)
            batch: 数据 batch, 包含:
                - gt_index: (B,) 正确答案索引
                - rule_label: (B,) 规则标签 (可选)

        Returns:
            losses: dict of loss terms
        """
        scores = outputs["scores"]     # (B, 4)
        gt_index = batch["gt_index"]   # (B,)
        B = scores.shape[0]
        device = scores.device

        # === 1. 主损失: 4选1 交叉熵 ===
        L_ce = self.ce_loss(scores, gt_index)

        # === 2. PE Margin 辅助损失 ===
        L_margin = torch.tensor(0.0, device=device)
        if "energies" in outputs:
            energies = outputs["energies"]  # (B, 4)

            # 正确候选能量
            E_pos = energies[torch.arange(B, device=device), gt_index]  # (B,)

            # 构造负例掩码
            neg_mask = torch.ones(B, 4, dtype=torch.bool, device=device)
            neg_mask[torch.arange(B, device=device), gt_index] = False
            E_neg = energies[neg_mask].reshape(B, 3)  # (B, 3)

            # margin loss: E_pos 应该小, E_neg 应该大于 τ
            L_margin = (
                E_pos.pow(2).mean()
                + torch.clamp(self.tau - E_neg, min=0).pow(2).mean()
            )

        # === 3. 规则分类辅助损失 (可选) ===
        L_rule = torch.tensor(0.0, device=device)
        if "rule_logits" in outputs and "rule_label" in batch:
            rule_logits = outputs["rule_logits"]  # (B, C)
            rule_label = batch["rule_label"]      # (B,)
            L_rule = self.ce_loss(rule_logits, rule_label)

        # === 总损失 ===
        L_total = L_ce + self.lambda_margin * L_margin + self.mu_rule * L_rule

        return {
            "total": L_total,
            "ce": L_ce,
            "margin": L_margin,
            "rule_cls": L_rule,
        }

