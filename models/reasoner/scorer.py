"""候选评分器: 基于预测误差 (PE) 计算候选得分."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class CandidateScorer(nn.Module):
    """候选评分器.

    综合三种误差 (Prediction Error) 计算候选能量:
        E[j] = α · E_tgt[j] + β · E_obs[j] + γ · E_rule[j]
        score[j] = -E[j]

    误差项:
        1. E_tgt: 目标匹配误差 ||ReLU(z_cand) - ReLU(z_hat_tgt)||
        2. E_obs: 观测自洽误差 Σ||ReLU(z_obs) - ReLU(z_rec)||
        3. E_rule: 规则一致性误差 ||r_verify - r||
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_energy(
        self,
        z_cand: torch.Tensor,
        z_hat_tgt: torch.Tensor,
        z_obs: torch.Tensor,
        z_rec: torch.Tensor,
        r: torch.Tensor,
        r_verify: torch.Tensor,
        grid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算单个候选的能量.

        Args:
            z_cand: (B, D) 候选 embedding
            z_hat_tgt: (B, D) 预测的 target embedding
            z_obs: (B, 9, D) 观测格子的 embedding
            z_rec: (B, 9, D) VerifyFormer 重建的格子 embedding
            r: (B, D) RuleFormer 输出的规则向量
            r_verify: (B, D) VerifyFormer 输出的验证规则向量
            grid_mask: (B, 9) bool 观测掩码

        Returns:
            E: (B,) 总能量
            details: 各项能量的详细信息
        """
        # 1. 目标匹配误差
        e_tgt = torch.relu(z_cand) - torch.relu(z_hat_tgt)
        E_tgt = torch.norm(e_tgt, dim=-1)  # (B,)

        # 2. 观测自洽误差
        e_obs = torch.relu(z_obs) - torch.relu(z_rec)  # (B, 9, D)
        E_obs_per_cell = torch.norm(e_obs, dim=-1)      # (B, 9)
        # 只计算已观测格子
        E_obs = (E_obs_per_cell * grid_mask.float()).sum(dim=-1)  # (B,)
        n_obs = grid_mask.float().sum(dim=-1).clamp(min=1)
        E_obs = E_obs / n_obs  # 归一化

        # 3. 规则一致性误差
        E_rule = torch.norm(r_verify - r, dim=-1)  # (B,)

        # 总能量
        E = self.alpha * E_tgt + self.beta * E_obs + self.gamma * E_rule

        details = {
            "E_tgt": E_tgt,
            "E_obs": E_obs,
            "E_rule": E_rule,
        }

        return E, details

    def forward(
        self,
        z_cands: torch.Tensor,
        z_hat_tgt: torch.Tensor,
        z_obs: torch.Tensor,
        z_recs: torch.Tensor,
        r: torch.Tensor,
        r_verifys: torch.Tensor,
        grid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算 4 个候选的得分.

        Args:
            z_cands: (B, 4, D) 候选 embeddings
            z_hat_tgt: (B, D) 预测的 target embedding
            z_obs: (B, 9, D) 观测格子的 embeddings
            z_recs: (B, 4, 9, D) 每个候选的重建 embeddings
            r: (B, D) 规则向量
            r_verifys: (B, 4, D) 验证规则向量
            grid_mask: (B, 9) 观测掩码

        Returns:
            scores: (B, 4) 候选得分 (越大越好)
            all_details: 详细信息
        """
        B, n_cand, D = z_cands.shape
        energies = []
        all_E_tgt = []
        all_E_obs = []
        all_E_rule = []

        for j in range(n_cand):
            E, details = self.compute_energy(
                z_cands[:, j],           # (B, D)
                z_hat_tgt,               # (B, D)
                z_obs,                   # (B, 9, D)
                z_recs[:, j],            # (B, 9, D)
                r,                       # (B, D)
                r_verifys[:, j],         # (B, D)
                grid_mask,               # (B, 9)
            )
            energies.append(E)
            all_E_tgt.append(details["E_tgt"])
            all_E_obs.append(details["E_obs"])
            all_E_rule.append(details["E_rule"])

        energies = torch.stack(energies, dim=1)       # (B, 4)
        scores = -energies                             # 得分 = 负能量

        all_details = {
            "energies": energies,
            "E_tgt": torch.stack(all_E_tgt, dim=1),   # (B, 4)
            "E_obs": torch.stack(all_E_obs, dim=1),
            "E_rule": torch.stack(all_E_rule, dim=1),
        }

        return scores, all_details

