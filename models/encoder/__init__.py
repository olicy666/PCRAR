"""点云编码器模块: 预训练几何主干 + 特性分支 + 全局统计分支 + 融合."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .pretrained_encoder import PointMAEEncoder
from .feat_encoder import FeatureEncoder
from .global_mlp import GlobalStatEncoder
from .fusion import PointCloudFusion


class PointCloudEncoder(nn.Module):
    """完整的点云编码器 (共享权重, 对所有格子和候选使用同一个).

    三分支架构:
        A) 预训练几何分支 (Point-MAE): xyz_local → z_xyz
        B) 特性分支 (PointNet-style): [xyz_global, rgb] → z_feat
        C) 全局统计分支 (MLP): global_stats → z_global
    融合: z = LN(z_xyz + W1·z_feat + W2·z_global)
    """

    def __init__(
        self,
        dim: int = 384,
        n_centers: int = 64,
        k_neighbors: int = 32,
        encoder_layers: int = 6,
        encoder_heads: int = 6,
        feat_in_channels: int = 6,
        global_feat_dim: int = 14,
        pretrained_path: Optional[str] = None,
        use_native_ops: bool = False,
        strict_native_ops: bool = False,
    ):
        super().__init__()
        self.dim = dim

        # 分支 A: 预训练几何主干
        self.geo_encoder = PointMAEEncoder(
            embed_dim=dim,
            n_centers=n_centers,
            k_neighbors=k_neighbors,
            depth=encoder_layers,
            num_heads=encoder_heads,
            use_native_ops=use_native_ops,
            strict_native_ops=strict_native_ops,
        )

        # 加载预训练权重
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        # 分支 B: 特性分支
        self.feat_encoder = FeatureEncoder(
            in_channels=feat_in_channels,
            hidden_dim=min(256, dim),
            out_dim=dim,
        )

        # 分支 C: 全局统计分支
        self.global_encoder = GlobalStatEncoder(
            in_dim=global_feat_dim,
            hidden_dim=128,
            out_dim=dim,
        )

        # 融合
        self.fusion = PointCloudFusion(dim=dim)

    def _load_pretrained(self, path: str):
        """加载预训练权重 (兼容 Point-MAE checkpoint 格式)."""
        import os
        if not os.path.exists(path):
            print(f"[WARNING] Pretrained weights not found: {path}, training from scratch.")
            return

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # Point-MAE checkpoints 通常有 'model' 或 'base_model' key
        state = ckpt.get("model", ckpt.get("base_model", ckpt))

        # Normalize common wrapper prefixes from Point-MAE / DDP checkpoints.
        if hasattr(state, "items"):
            normalized = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("module.MAE_encoder."):
                    nk = nk[len("module.MAE_encoder."):]
                elif nk.startswith("MAE_encoder."):
                    nk = nk[len("MAE_encoder."):]
                elif nk.startswith("module."):
                    nk = nk[len("module."):]
                normalized[nk] = v
            state = normalized

        # 尝试加载, 忽略不匹配的 key
        missing, unexpected = self.geo_encoder.load_state_dict(state, strict=False)
        if missing:
            print(f"[INFO] Missing keys in pretrained encoder: {len(missing)}")
        if unexpected:
            print(f"[INFO] Unexpected keys in pretrained encoder: {len(unexpected)}")

    def encode_single(
        self,
        xyz_local: torch.Tensor,
        feat: torch.Tensor,
        global_stats: torch.Tensor,
    ) -> torch.Tensor:
        """编码单个点云.

        Args:
            xyz_local: (B, N0, 3) 局部归一化坐标
            feat: (B, N0, F) 逐点特征
            global_stats: (B, G) 全局统计量

        Returns:
            z: (B, D) 融合后的 embedding
        """
        z_xyz = self.geo_encoder(xyz_local)          # (B, D)
        z_feat = self.feat_encoder(feat)             # (B, D)
        z_global = self.global_encoder(global_stats) # (B, D)
        return self.fusion(z_xyz, z_feat, z_global)  # (B, D)

    def encode_batch(
        self,
        xyz_local: torch.Tensor,
        feat: torch.Tensor,
        global_stats: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """编码一批点云 (多个格子或候选).

        Args:
            xyz_local: (B, K, N0, 3) K 个点云的局部坐标
            feat: (B, K, N0, F) 逐点特征
            global_stats: (B, K, G) 全局统计量
            mask: (B, K) bool, 可选, True=有效

        Returns:
            z: (B, K, D) 每个点云的 embedding
        """
        B, K, N0, _ = xyz_local.shape
        D = self.dim

        # 展平: (B*K, N0, 3)
        xyz_flat = xyz_local.reshape(B * K, N0, -1)
        feat_flat = feat.reshape(B * K, N0, -1)
        gs_flat = global_stats.reshape(B * K, -1)

        # 对无效位置 (mask=False) 也需要编码 (输出会被后续忽略)
        z_flat = self.encode_single(xyz_flat, feat_flat, gs_flat)  # (B*K, D)

        z = z_flat.reshape(B, K, D)

        # 将无效位置的 embedding 置零
        if mask is not None:
            z = z * mask.unsqueeze(-1).float()

        return z
