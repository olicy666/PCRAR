"""Point-MAE compatible encoder (state-dict compatible with MAE_encoder.* keys).

This implementation keeps the original module/key structure used by Point-MAE:
    cls_token / cls_pos / encoder / pos_embed / blocks.blocks / norm
while using pure PyTorch FPS+KNN fallback for grouping.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

try:
    from pointnet2_ops import pointnet2_utils  # type: ignore
    _HAS_POINTNET2 = True
except ImportError:
    pointnet2_utils = None  # type: ignore
    _HAS_POINTNET2 = False

try:
    from knn_cuda import KNN  # type: ignore
    _HAS_KNN_CUDA = True
except ImportError:
    KNN = None  # type: ignore
    _HAS_KNN_CUDA = False


def fps_subsample(xyz: torch.Tensor, n_centers: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Farthest Point Sampling with O(B*N*G) pure PyTorch fallback."""
    bsz, n_pts, _ = xyz.shape
    device = xyz.device

    center_idx = torch.zeros(bsz, n_centers, dtype=torch.long, device=device)
    distances = torch.full((bsz, n_pts), 1e10, device=device)
    farthest = torch.randint(0, n_pts, (bsz,), device=device)
    batch_index = torch.arange(bsz, device=device)

    for i in range(n_centers):
        center_idx[:, i] = farthest
        centroid = xyz[batch_index, farthest].unsqueeze(1)  # (B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        distances = torch.minimum(distances, dist)
        farthest = distances.argmax(dim=-1)

    centers = torch.gather(xyz, 1, center_idx.unsqueeze(-1).expand(-1, -1, 3))
    return center_idx, centers


def knn_group(xyz: torch.Tensor, centers: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """KNN grouping by torch.cdist."""
    bsz, n_pts, _ = xyz.shape
    n_centers = centers.shape[1]
    k = min(k, n_pts)

    dist = torch.cdist(centers, xyz, p=2)  # (B, G, N)
    _, idx = dist.topk(k, dim=-1, largest=False)  # (B, G, K)

    grouped = torch.gather(
        xyz.unsqueeze(1).expand(-1, n_centers, -1, -1),
        2,
        idx.unsqueeze(-1).expand(-1, -1, -1, 3),
    )  # (B, G, K, 3)
    grouped = grouped - centers.unsqueeze(2)
    return idx, grouped


class Group(nn.Module):
    """Point-MAE style group divider: FPS + KNN."""

    def __init__(self, num_group: int, group_size: int, use_native_ops: bool = False, strict_native_ops: bool = False):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.use_native_ops = use_native_ops
        self.strict_native_ops = strict_native_ops
        self.native_ready = _HAS_POINTNET2 and _HAS_KNN_CUDA
        self.knn = KNN(k=group_size, transpose_mode=True) if self.native_ready else None

        if self.use_native_ops and not self.native_ready and self.strict_native_ops:
            missing = []
            if not _HAS_POINTNET2:
                missing.append("pointnet2_ops")
            if not _HAS_KNN_CUDA:
                missing.append("knn_cuda")
            raise ImportError(
                "Native Point-MAE ops required but missing: "
                + ", ".join(missing)
                + ". Install pointnet2_ops and KNN_CUDA first."
            )

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_native_ops and self.native_ready:
            # Original Point-MAE style operators:
            # FPS from pointnet2_ops + KNN from KNN_CUDA.
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.num_group)  # (B, G)
            centers = pointnet2_utils.gather_operation(
                xyz.transpose(1, 2).contiguous(),
                fps_idx,
            ).transpose(1, 2).contiguous()  # (B, G, 3)

            bsz, n_pts, _ = xyz.shape
            _, idx = self.knn(xyz, centers)  # (B, G, K)
            idx = idx.long().contiguous()

            idx_base = torch.arange(0, bsz, device=xyz.device).view(-1, 1, 1) * n_pts
            idx = idx + idx_base
            idx = idx.view(-1)

            neighborhood = xyz.view(bsz * n_pts, -1)[idx, :]
            neighborhood = neighborhood.view(bsz, self.num_group, self.group_size, 3).contiguous()
            neighborhood = neighborhood - centers.unsqueeze(2)
            return neighborhood, centers

        _, centers = fps_subsample(xyz, self.num_group)
        _, neighborhood = knn_group(xyz, centers, self.group_size)
        return neighborhood, centers


class Encoder(nn.Module):
    """Point-MAE token embedding encoder (keeps original parameter names)."""

    def __init__(self, encoder_channel: int):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        # point_groups: (B, G, K, 3)
        bsz, n_groups, n_neighbors, _ = point_groups.shape
        point_groups = point_groups.reshape(bsz * n_groups, n_neighbors, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))  # (B*G, 256, K)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B*G, 256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, n_neighbors), feature], dim=1)  # (B*G, 512, K)
        feature = self.second_conv(feature)  # (B*G, C, K)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (B*G, C)
        return feature_global.reshape(bsz, n_groups, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Point-MAE style self-attention (qkv/proj keys)."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, n_tokens, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(bsz, n_tokens, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    """Point-MAE compatible block: norm1/attn + norm2/mlp."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=False, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden, out_features=dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Container to keep key path blocks.blocks.{i}.*."""

    def __init__(self, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0, drop_rate: float = 0.0, attn_drop_rate: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x + pos)
        return x


class PointMAEEncoder(nn.Module):
    """Point-MAE-compatible geometry encoder."""

    def __init__(
        self,
        embed_dim: int = 384,
        n_centers: int = 64,
        k_neighbors: int = 32,
        depth: int = 6,
        num_heads: int = 6,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
        use_native_ops: bool = False,
        strict_native_ops: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.group_divider = Group(
            num_group=n_centers,
            group_size=k_neighbors,
            use_native_ops=use_native_ops,
            strict_native_ops=strict_native_ops,
        )
        self.encoder = Encoder(encoder_channel=embed_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Keep this wrapper so checkpoints map to blocks.blocks.*
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=float(ffn_ratio),
            drop_rate=dropout,
            attn_drop_rate=dropout,
        )
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        bsz = xyz.shape[0]
        neighborhood, centers = self.group_divider(xyz)  # (B, G, K, 3), (B, G, 3)
        tokens = self.encoder(neighborhood)              # (B, G, D)
        pos = self.pos_embed(centers)                    # (B, G, D)

        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        cls_pos = self.cls_pos.expand(bsz, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)       # (B, 1+G, D)
        pos = torch.cat([cls_pos, pos], dim=1)           # (B, 1+G, D)

        x = self.blocks(x, pos)
        x = self.norm(x)
        cls_out = x[:, 0]
        patch_mean = x[:, 1:].mean(dim=1)
        return cls_out + patch_mean
