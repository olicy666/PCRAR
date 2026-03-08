"""PCRAR 完整模型: Encoder + RuleFormer + VerifyFormer + Scorer."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .encoder import PointCloudEncoder
from .reasoner.tokenizer import MatrixTokenizer
from .reasoner.ruleformer import RuleFormer
from .reasoner.verifyformer import VerifyFormer
from .reasoner.scorer import CandidateScorer


class PCRARModel(nn.Module):
    """PCRAR 点云矩阵推理模型.

    三段式架构:
        1. PointCloudEncoder (共享): 每个点云 → embedding z ∈ R^D
        2. RuleFormer: 显式 [RULE] token → 规则向量 r, 预测 target ẑ_tgt
        3. VerifyFormer: 候选验证 + 全局自洽 PE → 候选得分

    前向传播流程:
        Step 1: Encoder 编码 10 个点云 (6 观测 + 4 候选)
        Step 2: Tokenizer 组装 token 序列 (含 [RULE])
        Step 3: RuleFormer 推断规则 r, 预测 target ẑ_tgt
        Step 4: VerifyFormer × 4, 每个候选插入后验证
        Step 5: Scorer 计算 4 个候选得分
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        D = config["model"]["D"]
        enc_cfg = config["model"].get("encoder", {})
        rf_cfg = config["model"].get("ruleformer", {})
        vf_cfg = config["model"].get("verifyformer", {})
        sc_cfg = config["model"].get("scorer", {})

        # === 1. Point Cloud Encoder (共享) ===
        self.encoder = PointCloudEncoder(
            dim=D,
            n_centers=enc_cfg.get("n_centers", 64),
            k_neighbors=enc_cfg.get("k_neighbors", 32),
            encoder_layers=enc_cfg.get("encoder_layers", 6),
            encoder_heads=enc_cfg.get("encoder_heads", 6),
            feat_in_channels=config["model"].get("feat_in_channels", 6),
            global_feat_dim=config["data"].get("global_feat_dim", 14),
            pretrained_path=config["model"].get("pretrained_path", ""),
            use_native_ops=enc_cfg.get("use_native_ops", False),
            strict_native_ops=enc_cfg.get("strict_native_ops", False),
        )

        # === 2. Tokenizer ===
        self.tokenizer = MatrixTokenizer(dim=D)

        # === 3. RuleFormer ===
        self.ruleformer = RuleFormer(
            dim=D,
            n_layers=rf_cfg.get("n_layers", 6),
            n_heads=rf_cfg.get("n_heads", 6),
            ffn_ratio=rf_cfg.get("ffn_ratio", 4),
            dropout=rf_cfg.get("dropout", 0.1),
        )

        # === 4. VerifyFormer ===
        self.verifyformer = VerifyFormer(
            dim=D,
            n_layers=vf_cfg.get("n_layers", 3),
            n_heads=vf_cfg.get("n_heads", 6),
            ffn_ratio=vf_cfg.get("ffn_ratio", 4),
            dropout=vf_cfg.get("dropout", 0.1),
        )

        # === 5. Scorer ===
        self.scorer = CandidateScorer(
            alpha=sc_cfg.get("alpha", 1.0),
            beta=sc_cfg.get("beta", 0.5),
            gamma=sc_cfg.get("gamma", 0.2),
        )

        # === 6. 规则分类辅助头 ===
        num_rule_types = config["model"].get("num_rule_types", 7)
        self.rule_classifier = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(D // 2, num_rule_types),
        )

        self.D = D

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """完整前向传播.

        Args:
            batch: 数据字典, 包含:
                - grid_xyz_local: (B, 9, N0, 3)
                - grid_feat: (B, 9, N0, F)
                - grid_global_stats: (B, 9, G)
                - grid_mask: (B, 9) bool
                - cand_xyz_local: (B, 4, N0, 3)
                - cand_feat: (B, 4, N0, F)
                - cand_global_stats: (B, 4, G)
                - missing_indices: (B, 3)

        Returns:
            outputs: 包含 scores, energies, rule_logits 等
        """
        device = batch["grid_xyz_local"].device
        B = batch["grid_xyz_local"].shape[0]

        # ============================================
        # Step 1: 编码所有点云
        # ============================================
        # 编码 9 个格子
        grid_z = self.encoder.encode_batch(
            batch["grid_xyz_local"],      # (B, 9, N0, 3)
            batch["grid_feat"],           # (B, 9, N0, F)
            batch["grid_global_stats"],   # (B, 9, G)
            mask=batch["grid_mask"],      # (B, 9)
        )  # → (B, 9, D)

        # 编码 4 个候选
        cand_z = self.encoder.encode_batch(
            batch["cand_xyz_local"],      # (B, 4, N0, 3)
            batch["cand_feat"],           # (B, 4, N0, F)
            batch["cand_global_stats"],   # (B, 4, G)
        )  # → (B, 4, D)

        # ============================================
        # Step 2: 构造 token 序列
        # ============================================
        tokens = self.tokenizer(
            grid_z,
            batch["grid_mask"],
            batch["missing_indices"],
        )  # → (B, 10, D)

        # ============================================
        # Step 3: RuleFormer — 推断规则 + 预测 target
        # ============================================
        # target 位置: (2,2) → 展平索引 8
        r, z_hat_tgt, u_rule = self.ruleformer(tokens, target_pos_idx=8)
        # r: (B, D), z_hat_tgt: (B, D), u_rule: (B, D)

        # ============================================
        # Step 4: VerifyFormer × 4 候选
        # ============================================
        # 获取 target 位置的位置编码
        pos_enc = self.tokenizer.get_pos_enc(device)  # (9, D)
        target_pos_enc = pos_enc[8]  # (D,)

        # 为每个候选构造插入后的 token 序列, 并行处理
        # 先在 (B, 4, 10, D) 维度写入, 再 reshape 成 (B*4, 10, D), 避免 batch/candidate 维错位
        tokens_expanded = tokens.unsqueeze(1).expand(-1, 4, -1, -1).clone()  # (B, 4, 10, D)

        # 替换 target 位置 token (index 9 = 1+8)
        cand_tokens = cand_z + target_pos_enc.view(1, 1, self.D)  # (B, 4, D)
        tokens_expanded[:, :, 9, :] = cand_tokens

        # 使用已推断的 rule token
        tokens_expanded[:, :, 0, :] = u_rule.unsqueeze(1)  # (B, 4, D)

        tokens_all = tokens_expanded.reshape(B * 4, 10, self.D)

        # 并行通过 VerifyFormer
        z_rec_all, r_verify_all = self.verifyformer(tokens_all)
        # z_rec_all: (B*4, 9, D), r_verify_all: (B*4, D)

        # Reshape 回 (B, 4, ...)
        z_recs = z_rec_all.reshape(B, 4, 9, self.D)       # (B, 4, 9, D)
        r_verifys = r_verify_all.reshape(B, 4, self.D)     # (B, 4, D)

        # ============================================
        # Step 5: Scorer — 计算得分
        # ============================================
        scores, score_details = self.scorer(
            z_cands=cand_z,         # (B, 4, D)
            z_hat_tgt=z_hat_tgt,    # (B, D)
            z_obs=grid_z,           # (B, 9, D)
            z_recs=z_recs,          # (B, 4, 9, D)
            r=r,                    # (B, D)
            r_verifys=r_verifys,    # (B, 4, D)
            grid_mask=batch["grid_mask"],  # (B, 9)
        )
        # scores: (B, 4)

        # ============================================
        # 规则分类 (辅助)
        # ============================================
        rule_logits = self.rule_classifier(r)  # (B, num_rule_types)

        return {
            "scores": scores,                    # (B, 4) 候选得分
            "energies": score_details["energies"],  # (B, 4) 候选能量
            "r": r,                              # (B, D) 规则向量
            "z_hat_tgt": z_hat_tgt,              # (B, D) 预测 target
            "rule_logits": rule_logits,           # (B, C) 规则分类
            "E_tgt": score_details["E_tgt"],      # (B, 4)
            "E_obs": score_details["E_obs"],      # (B, 4)
            "E_rule": score_details["E_rule"],    # (B, 4)
        }

    def freeze_encoder(self):
        """冻结预训练点云主干 (阶段 A)."""
        for param in self.encoder.geo_encoder.parameters():
            param.requires_grad = False
        print("[INFO] Froze geo_encoder (pretrained backbone)")

    def unfreeze_encoder(self, last_n: int = 2):
        """解冻预训练主干的最后 N 个 block (阶段 B)."""
        # 先全部冻结
        for param in self.encoder.geo_encoder.parameters():
            param.requires_grad = False

        # 解冻最后 N 个 block
        blocks_obj = self.encoder.geo_encoder.blocks
        blocks = blocks_obj.blocks if hasattr(blocks_obj, "blocks") else blocks_obj
        n_blocks = len(blocks)
        for i in range(max(0, n_blocks - last_n), n_blocks):
            for param in blocks[i].parameters():
                param.requires_grad = True

        # 解冻 final norm
        for param in self.encoder.geo_encoder.norm.parameters():
            param.requires_grad = True

        print(f"[INFO] Unfroze last {last_n} blocks of geo_encoder")

    def get_param_groups(self, lr_backbone: float = 2e-5, lr_heads: float = 1e-4):
        """获取参数组 (不同学习率).

        Returns:
            list of param groups for optimizer
        """
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "geo_encoder" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_heads},
        ]
