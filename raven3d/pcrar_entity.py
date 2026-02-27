"""PCRAR Entity 模块.

定义 Attr(E) 数据结构和布尔几何体点云采样。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .csg import (
    CSGNode, Leaf, OpNode, OpType, PrimType, SizeLevel, DeltaLevel,
    node_from_dict, csg_to_expr, get_all_leaves, copy_csg,
    DISCRETE_ANGLES, SIZE_LEVEL_MAP, DELTA_LEVEL_MAP,
)
from .geometry import rotation_matrix


# 默认参数
DEFAULT_N_POINTS = 8192
DEFAULT_OVERSAMPLE_FACTOR = 5
DEFAULT_BOUNDARY_EPS = 0.005
DEFAULT_BOUNDARY_SAMPLES = 8
DEFAULT_MAX_ITERATIONS = 20

# 密度离散档（总点数档位，leaf 均分）
# 上调一档：中档从 7168 提升到 8192，并保持档位间距一致。
DENSITY_POINT_PRESETS = [11264, 9728, 8192, 6656, 5120]

# 颜色离散档（三档）
COLOR_PRESETS = ["red", "green", "blue"]
COLOR_RGB_PRESETS = [
    (220, 60, 60),   # red
    (60, 170, 80),   # green
    (65, 105, 225),  # blue
]

# 兼容历史接口：密度权重按均分处理
DENSITY_PRESETS_1 = [[1.0], [1.0], [1.0]]
DENSITY_PRESETS_2 = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
DENSITY_PRESETS_3 = [[1 / 3, 1 / 3, 1 / 3]] * 3


def density_point_count(idx: int) -> int:
    """根据密度档位返回总点数"""
    if not DENSITY_POINT_PRESETS:
        return DEFAULT_N_POINTS
    idx = int(idx)
    if idx < 0:
        idx = 0
    elif idx >= len(DENSITY_POINT_PRESETS):
        idx = len(DENSITY_POINT_PRESETS) - 1
    return int(DENSITY_POINT_PRESETS[idx])


def color_label(idx: int) -> str:
    """根据颜色档位返回颜色标签"""
    if not COLOR_PRESETS:
        return "red"
    idx = int(idx)
    if idx < 0:
        idx = 0
    elif idx >= len(COLOR_PRESETS):
        idx = len(COLOR_PRESETS) - 1
    return str(COLOR_PRESETS[idx])


def color_rgb(idx: int) -> Tuple[int, int, int]:
    """根据颜色档位返回 RGB 颜色"""
    if not COLOR_RGB_PRESETS:
        return (220, 60, 60)
    idx = int(idx)
    if idx < 0:
        idx = 0
    elif idx >= len(COLOR_RGB_PRESETS):
        idx = len(COLOR_RGB_PRESETS) - 1
    rgb = COLOR_RGB_PRESETS[idx]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


@dataclass
class ObservationConfig:
    """观测配置 (O)"""
    global_pose_deg: Tuple[int, int, int] = (0, 0, 0)  # 全局离散旋转
    global_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    sampling_mode: str = "surface"
    n_points: int = DEFAULT_N_POINTS
    part_sampling_weights: Optional[List[float]] = None
    density_preset_idx: int = 0  # 密度档位索引
    color_preset_idx: int = 0  # 颜色档位索引（red/green/blue）

    def copy(self) -> "ObservationConfig":
        return ObservationConfig(
            global_pose_deg=self.global_pose_deg,
            global_translation=self.global_translation,
            sampling_mode=self.sampling_mode,
            n_points=self.n_points,
            part_sampling_weights=list(self.part_sampling_weights) if self.part_sampling_weights else None,
            density_preset_idx=self.density_preset_idx,
            color_preset_idx=self.color_preset_idx,
        )

    def get_global_rotation_matrix(self) -> np.ndarray:
        euler_rad = np.deg2rad(self.global_pose_deg)
        return rotation_matrix(euler_rad)

    def get_global_translation(self) -> np.ndarray:
        return np.array(self.global_translation)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_pose_deg": list(self.global_pose_deg),
            "global_translation": list(self.global_translation),
            "sampling_mode": self.sampling_mode,
            "n_points": self.n_points,
            "part_sampling_weights": list(self.part_sampling_weights) if self.part_sampling_weights else None,
            "density_preset_idx": self.density_preset_idx,
            "color_preset_idx": self.color_preset_idx,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ObservationConfig":
        return ObservationConfig(
            global_pose_deg=tuple(d.get("global_pose_deg", [0, 0, 0])),
            global_translation=tuple(d.get("global_translation", [0.0, 0.0, 0.0])),
            sampling_mode=d.get("sampling_mode", "surface"),
            n_points=d.get("n_points", DEFAULT_N_POINTS),
            part_sampling_weights=d.get("part_sampling_weights"),
            density_preset_idx=d.get("density_preset_idx", 0),
            color_preset_idx=d.get("color_preset_idx", 0),
        )


@dataclass
class PCRAREntity:
    """PCRAR 实体 Attr(E) = (CSG, O)
    
    表示一个复合布尔几何体实体。
    """
    csg: CSGNode
    obs: ObservationConfig = field(default_factory=ObservationConfig)

    def copy(self) -> "PCRAREntity":
        return PCRAREntity(
            csg=copy_csg(self.csg),
            obs=self.obs.copy(),
        )

    def get_leaves(self) -> List[Leaf]:
        """获取所有叶节点"""
        return get_all_leaves(self.csg)

    def leaf_count(self) -> int:
        """叶节点数量"""
        return len(self.get_leaves())

    def get_expr(self) -> str:
        """获取可读表达式"""
        return csg_to_expr(self.csg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "csg": self.csg.to_dict(),
            "obs": self.obs.to_dict(),
            "expr": self.get_expr(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PCRAREntity":
        return PCRAREntity(
            csg=node_from_dict(d["csg"]),
            obs=ObservationConfig.from_dict(d["obs"]),
        )

    def sample_point_cloud(
        self,
        rng: np.random.Generator,
        n_points: Optional[int] = None,
        oversample_factor: int = DEFAULT_OVERSAMPLE_FACTOR,
        boundary_eps: float = DEFAULT_BOUNDARY_EPS,
        boundary_samples: int = DEFAULT_BOUNDARY_SAMPLES,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> np.ndarray:
        """采样布尔几何体表面点云
        
        使用边界扰动法近似采样 CSG 结果的表面。
        
        Args:
            rng: 随机数生成器
            n_points: 目标点数，默认使用 obs.n_points
            oversample_factor: 过采样因子
            boundary_eps: 边界判定扰动距离
            boundary_samples: 每个点的扰动采样数
            max_iterations: 最大迭代次数
            
        Returns:
            (n_points, 3) 点云数组
        """
        if n_points is None:
            n_points = density_point_count(self.obs.density_preset_idx)
            self.obs.n_points = n_points

        leaves = self.get_leaves()
        n_leaves = len(leaves)
        
        # 获取采样权重
        weights = self._get_sampling_weights(n_leaves)
        
        all_surface_points = []
        
        for iteration in range(max_iterations):
            # 计算每个 leaf 需要采样的点数
            remaining = n_points - len(all_surface_points)
            if remaining <= 0:
                break
            
            proposal_count = remaining * oversample_factor
            leaf_counts = np.round(weights * proposal_count).astype(int)
            # 确保总数足够
            while leaf_counts.sum() < proposal_count:
                leaf_counts[rng.integers(n_leaves)] += 1
            
            # 从每个 leaf 表面采样
            proposals = []
            for leaf, count in zip(leaves, leaf_counts):
                if count > 0:
                    pts = leaf.sample_surface(int(count), rng)
                    proposals.append(pts)
            
            if not proposals:
                continue
            
            all_proposals = np.vstack(proposals)
            
            # 边界判定：检查每个点是否在 CSG 表面附近
            boundary_mask = self._check_boundary(all_proposals, rng, boundary_eps, boundary_samples)
            
            boundary_points = all_proposals[boundary_mask]
            all_surface_points.append(boundary_points)
            
            if sum(len(p) for p in all_surface_points) >= n_points:
                break
        
        # 合并所有点
        if all_surface_points:
            final_points = np.vstack(all_surface_points)
        else:
            # 如果采样失败，回退到简单采样
            final_points = self._fallback_sample(rng, n_points, leaves, weights)
        
        # 调整点数
        if len(final_points) > n_points:
            indices = rng.choice(len(final_points), n_points, replace=False)
            final_points = final_points[indices]
        elif len(final_points) < n_points:
            # 补充采样
            shortage = n_points - len(final_points)
            extra = self._fallback_sample(rng, shortage, leaves, weights)
            final_points = np.vstack([final_points, extra])
        
        # 应用全局变换
        final_points = self._apply_global_transform(final_points)
        
        # 打乱顺序
        rng.shuffle(final_points)
        
        return final_points[:n_points]

    def _get_sampling_weights(self, n_leaves: int) -> np.ndarray:
        """获取采样权重"""
        if n_leaves <= 0:
            return np.array([], dtype=float)
        if self.obs.part_sampling_weights and len(self.obs.part_sampling_weights) == n_leaves:
            weights = np.array([float(w) for w in self.obs.part_sampling_weights], dtype=float)
            weights = np.clip(weights, 0.0, None)
            if weights.sum() > 0:
                return weights / weights.sum()
        weights = np.ones(n_leaves, dtype=float) / float(n_leaves)
        return weights

    def _check_boundary(
        self,
        points: np.ndarray,
        rng: np.random.Generator,
        eps: float,
        n_samples: int,
    ) -> np.ndarray:
        """检查点是否在 CSG 边界附近
        
        通过随机扰动检测：如果扰动后的内外判定与原点不同，则认为在边界。
        """
        n_points = len(points)
        is_boundary = np.zeros(n_points, dtype=bool)
        
        # 计算原点的内外判定
        base_inside = self.csg.inside(points)
        
        # 对每个点进行多次扰动检测
        for _ in range(n_samples):
            # 随机方向单位向量
            directions = rng.normal(size=(n_points, 3))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
            
            # 扰动点
            perturbed = points + eps * directions
            perturbed_inside = self.csg.inside(perturbed)
            
            # 如果扰动后判定不同，则在边界
            is_boundary |= (base_inside != perturbed_inside)
        
        return is_boundary

    def _fallback_sample(
        self,
        rng: np.random.Generator,
        n_points: int,
        leaves: List[Leaf],
        weights: np.ndarray,
    ) -> np.ndarray:
        """回退采样：简单地从各 leaf 表面采样并过滤"""
        n_leaves = len(leaves)
        leaf_counts = np.round(weights * n_points).astype(int)
        while leaf_counts.sum() < n_points:
            leaf_counts[rng.integers(n_leaves)] += 1
        
        all_pts = []
        for leaf, count in zip(leaves, leaf_counts):
            if count > 0:
                pts = leaf.sample_surface(int(count), rng)
                # 只保留在 CSG 内部或边界的点
                inside = self.csg.inside(pts)
                # 对于 Union 操作，保留所有在边界的点
                # 这里简化处理：保留所有点
                all_pts.append(pts)
        
        if all_pts:
            return np.vstack(all_pts)
        else:
            return np.zeros((n_points, 3))

    def _apply_global_transform(self, points: np.ndarray) -> np.ndarray:
        """应用全局变换"""
        rot = self.obs.get_global_rotation_matrix()
        trans = self.obs.get_global_translation()
        return points @ rot.T + trans


def sample_random_entity(
    rng: np.random.Generator,
    leaf_count: int = 2,
    allowed_ops: Optional[List[OpType]] = None,
) -> PCRAREntity:
    """随机生成一个 PCRAR 实体
    
    Args:
        rng: 随机数生成器
        leaf_count: 叶节点数量（1/2/3）
        allowed_ops: 允许的操作类型
        
    Returns:
        随机 PCRAR 实体
    """
    from .csg import sample_random_csg, DISCRETE_ANGLES
    
    csg = sample_random_csg(rng, leaf_count=leaf_count, allowed_ops=allowed_ops)
    
    # 随机观测配置
    global_pose_deg = tuple(int(rng.choice(DISCRETE_ANGLES)) for _ in range(3))
    
    # 随机密度档位
    density_preset_idx = int(rng.integers(len(DENSITY_POINT_PRESETS)))
    n_points = density_point_count(density_preset_idx)
    color_preset_idx = int(rng.integers(len(COLOR_PRESETS)))

    obs = ObservationConfig(
        global_pose_deg=global_pose_deg,
        n_points=n_points,
        density_preset_idx=density_preset_idx,
        color_preset_idx=color_preset_idx,
    )
    
    return PCRAREntity(csg=csg, obs=obs)


def entities_equal(e1: PCRAREntity, e2: PCRAREntity, check_obs: bool = True) -> bool:
    """检查两个实体是否相等（用于唯一性校验）"""
    # 比较 CSG 结构
    d1 = e1.csg.to_dict()
    d2 = e2.csg.to_dict()
    if d1 != d2:
        return False
    
    if check_obs:
        # 比较观测配置
        o1 = e1.obs.to_dict()
        o2 = e2.obs.to_dict()
        if o1 != o2:
            return False
    
    return True
