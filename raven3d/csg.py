"""CSG (Constructive Solid Geometry) 数据结构模块.

实现布尔几何体的二叉树表示，支持 Union/Intersect/Diff 操作。
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .geometry import rotation_matrix


class PrimType(str, Enum):
    """Primitive 类型枚举，只保留 4 种"""
    SPHERE = "sphere"
    BOX = "box"
    CYLINDER = "cylinder"
    CONE = "cone"


class OpType(str, Enum):
    """布尔操作类型"""
    UNION = "union"
    INTERSECT = "intersect"
    DIFF = "diff"


class SizeLevel(str, Enum):
    """尺寸离散档位"""
    S = "S"
    M = "M"
    L = "L"


class DeltaLevel(str, Enum):
    """位置 delta 离散档位"""
    NEAR = "Near"
    MID = "Mid"
    FAR = "Far"


# 尺寸档位映射到 scale 值
SIZE_LEVEL_MAP: Dict[SizeLevel, float] = {
    SizeLevel.S: 0.8,
    SizeLevel.M: 1.0,
    SizeLevel.L: 1.2,
}

# Delta 档位映射到实际位移量
DELTA_LEVEL_MAP: Dict[DeltaLevel, float] = {
    DeltaLevel.NEAR: 0.35,
    DeltaLevel.MID: 0.50,
    DeltaLevel.FAR: 0.65,
}

# 离散角度列表（度）
DISCRETE_ANGLES = [0, 60, 120, 180, 240, 300]

# Primitive 类型列表（用于循环）
PRIM_TYPE_CYCLE = [PrimType.SPHERE, PrimType.BOX, PrimType.CYLINDER, PrimType.CONE]


@dataclass
class Leaf:
    """CSG 树的叶节点，代表一个基本几何体"""
    id: int
    prim_type: PrimType
    size_level: SizeLevel = SizeLevel.M
    local_pose_deg: Tuple[int, int, int] = (0, 0, 0)  # 离散欧拉角（度）
    slot: int = 0  # 位置槽位 -1, 0, +1
    delta_level: DeltaLevel = DeltaLevel.MID

    def copy(self) -> "Leaf":
        return Leaf(
            id=self.id,
            prim_type=self.prim_type,
            size_level=self.size_level,
            local_pose_deg=self.local_pose_deg,
            slot=self.slot,
            delta_level=self.delta_level,
        )

    def get_scale(self) -> float:
        """获取实际 scale 值"""
        return SIZE_LEVEL_MAP[self.size_level]

    def get_position(self) -> np.ndarray:
        """获取实际位置（基于 slot 和 delta_level）"""
        delta = DELTA_LEVEL_MAP[self.delta_level]
        return np.array([self.slot * delta, 0.0, 0.0])

    def get_rotation_matrix(self) -> np.ndarray:
        """获取旋转矩阵"""
        euler_rad = np.deg2rad(self.local_pose_deg)
        return rotation_matrix(euler_rad)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "leaf",
            "id": self.id,
            "prim": self.prim_type.value,
            "size": self.size_level.value,
            "local_pose_deg": list(self.local_pose_deg),
            "slot": self.slot,
            "delta_level": self.delta_level.value,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Leaf":
        return Leaf(
            id=d["id"],
            prim_type=PrimType(d["prim"]),
            size_level=SizeLevel(d["size"]),
            local_pose_deg=tuple(d["local_pose_deg"]),
            slot=d["slot"],
            delta_level=DeltaLevel(d["delta_level"]),
        )

    def inside(self, points: np.ndarray) -> np.ndarray:
        """判断点是否在该 primitive 内部（向量化）
        
        Args:
            points: (N, 3) 世界坐标点
            
        Returns:
            (N,) bool 数组
        """
        # 逆变换到局部坐标
        center = self.get_position()
        scale = self.get_scale()
        rot = self.get_rotation_matrix()
        
        # x_local = R^T * ((x - center) / scale)
        pts_centered = points - center
        pts_local = (pts_centered @ rot) / scale
        
        if self.prim_type == PrimType.SPHERE:
            # ||x_local|| <= 0.5 (半径 0.5)
            return np.linalg.norm(pts_local, axis=1) <= 0.5
        elif self.prim_type == PrimType.BOX:
            # max(|x|,|y|,|z|) <= 0.5
            return np.max(np.abs(pts_local), axis=1) <= 0.5
        elif self.prim_type == PrimType.CYLINDER:
            # x^2+y^2 <= 0.5^2 且 |z| <= 0.5
            r_sq = pts_local[:, 0]**2 + pts_local[:, 1]**2
            return (r_sq <= 0.25) & (np.abs(pts_local[:, 2]) <= 0.5)
        elif self.prim_type == PrimType.CONE:
            # 圆锥：底部半径 0.5，高度 1.0，顶点在 z=0.5
            # 线性半径随 z 变化: r_at_z = 0.5 * (0.5 - z)
            z = pts_local[:, 2]
            r_sq = pts_local[:, 0]**2 + pts_local[:, 1]**2
            # z 范围: [-0.5, 0.5]
            valid_z = (z >= -0.5) & (z <= 0.5)
            # 在 z 处的最大半径
            max_r = 0.5 * (0.5 - z)
            return valid_z & (r_sq <= max_r**2) & (max_r >= 0)
        else:
            raise ValueError(f"Unknown prim_type: {self.prim_type}")

    def sample_surface(self, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """从 primitive 表面采样点"""
        scale = self.get_scale()
        center = self.get_position()
        rot = self.get_rotation_matrix()
        
        if self.prim_type == PrimType.SPHERE:
            # 球面均匀采样
            phi = rng.uniform(0, 2 * np.pi, n_points)
            cos_theta = rng.uniform(-1, 1, n_points)
            sin_theta = np.sqrt(1 - cos_theta**2)
            pts = 0.5 * np.stack([
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                cos_theta
            ], axis=1)
        elif self.prim_type == PrimType.BOX:
            # 立方体表面采样
            pts = self._sample_box_surface(n_points, rng)
        elif self.prim_type == PrimType.CYLINDER:
            # 圆柱表面采样
            pts = self._sample_cylinder_surface(n_points, rng)
        elif self.prim_type == PrimType.CONE:
            # 圆锥表面采样
            pts = self._sample_cone_surface(n_points, rng)
        else:
            raise ValueError(f"Unknown prim_type: {self.prim_type}")
        
        # 应用变换
        pts_scaled = pts * scale
        pts_rotated = pts_scaled @ rot.T
        return pts_rotated + center

    def _sample_box_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """立方体表面采样"""
        # 6 个面，每个面面积相同
        face_idx = rng.integers(0, 6, n)
        u = rng.uniform(-0.5, 0.5, n)
        v = rng.uniform(-0.5, 0.5, n)
        pts = np.zeros((n, 3))
        for i in range(n):
            f = face_idx[i]
            if f == 0:  # +x
                pts[i] = [0.5, u[i], v[i]]
            elif f == 1:  # -x
                pts[i] = [-0.5, u[i], v[i]]
            elif f == 2:  # +y
                pts[i] = [u[i], 0.5, v[i]]
            elif f == 3:  # -y
                pts[i] = [u[i], -0.5, v[i]]
            elif f == 4:  # +z
                pts[i] = [u[i], v[i], 0.5]
            else:  # -z
                pts[i] = [u[i], v[i], -0.5]
        return pts

    def _sample_cylinder_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """圆柱表面采样（侧面 + 顶底）"""
        r = 0.5
        h = 1.0
        side_area = 2 * np.pi * r * h
        cap_area = 2 * np.pi * r**2
        total = side_area + cap_area
        side_prob = side_area / total
        
        is_side = rng.random(n) < side_prob
        n_side = is_side.sum()
        n_cap = n - n_side
        
        pts = np.zeros((n, 3))
        
        # 侧面
        if n_side > 0:
            theta = rng.uniform(0, 2 * np.pi, n_side)
            z = rng.uniform(-0.5, 0.5, n_side)
            pts[is_side] = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
        
        # 顶底
        if n_cap > 0:
            is_top = rng.random(n_cap) < 0.5
            r_sample = r * np.sqrt(rng.random(n_cap))
            theta = rng.uniform(0, 2 * np.pi, n_cap)
            z = np.where(is_top, 0.5, -0.5)
            cap_pts = np.stack([r_sample * np.cos(theta), r_sample * np.sin(theta), z], axis=1)
            pts[~is_side] = cap_pts
        
        return pts

    def _sample_cone_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """圆锥表面采样（侧面 + 底面）"""
        r = 0.5
        h = 1.0
        slant = np.sqrt(r**2 + h**2)
        side_area = np.pi * r * slant
        base_area = np.pi * r**2
        total = side_area + base_area
        side_prob = side_area / total
        
        is_side = rng.random(n) < side_prob
        n_side = is_side.sum()
        n_base = n - n_side
        
        pts = np.zeros((n, 3))
        
        # 侧面：顶点在 z=0.5，底部在 z=-0.5
        if n_side > 0:
            u = rng.random(n_side)  # 0 到 1，0=顶点，1=底部
            theta = rng.uniform(0, 2 * np.pi, n_side)
            r_at_u = r * u
            z = 0.5 - u * h
            pts[is_side] = np.stack([r_at_u * np.cos(theta), r_at_u * np.sin(theta), z], axis=1)
        
        # 底面
        if n_base > 0:
            r_sample = r * np.sqrt(rng.random(n_base))
            theta = rng.uniform(0, 2 * np.pi, n_base)
            pts[~is_side] = np.stack([r_sample * np.cos(theta), r_sample * np.sin(theta), 
                                      np.full(n_base, -0.5)], axis=1)
        
        return pts


@dataclass
class OpNode:
    """CSG 树的操作节点"""
    op: OpType
    left: Union["OpNode", Leaf]
    right: Union["OpNode", Leaf]

    def copy(self) -> "OpNode":
        return OpNode(
            op=self.op,
            left=self.left.copy(),
            right=self.right.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "op",
            "op": self.op.value,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OpNode":
        left = node_from_dict(d["left"])
        right = node_from_dict(d["right"])
        return OpNode(
            op=OpType(d["op"]),
            left=left,
            right=right,
        )

    def inside(self, points: np.ndarray) -> np.ndarray:
        """递归判断点是否在 CSG 结果内部"""
        left_inside = self.left.inside(points)
        right_inside = self.right.inside(points)
        
        if self.op == OpType.UNION:
            return left_inside | right_inside
        elif self.op == OpType.INTERSECT:
            return left_inside & right_inside
        elif self.op == OpType.DIFF:
            return left_inside & ~right_inside
        else:
            raise ValueError(f"Unknown op: {self.op}")

    def get_leaves(self) -> List[Leaf]:
        """获取所有叶节点"""
        leaves = []
        self._collect_leaves(leaves)
        return leaves

    def _collect_leaves(self, leaves: List[Leaf]) -> None:
        if isinstance(self.left, Leaf):
            leaves.append(self.left)
        else:
            self.left._collect_leaves(leaves)
        if isinstance(self.right, Leaf):
            leaves.append(self.right)
        else:
            self.right._collect_leaves(leaves)

    def get_ops(self) -> List["OpNode"]:
        """获取所有操作节点"""
        ops = [self]
        if isinstance(self.left, OpNode):
            ops.extend(self.left.get_ops())
        if isinstance(self.right, OpNode):
            ops.extend(self.right.get_ops())
        return ops

    def to_expr(self) -> str:
        """生成可读的表达式字符串"""
        left_str = self.left.to_expr() if isinstance(self.left, OpNode) else self.left.prim_type.value.capitalize()
        right_str = self.right.to_expr() if isinstance(self.right, OpNode) else self.right.prim_type.value.capitalize()
        op_name = self.op.value.capitalize()
        return f"{op_name}({left_str},{right_str})"


# CSGNode 类型别名
CSGNode = Union[OpNode, Leaf]


def node_from_dict(d: Dict[str, Any]) -> CSGNode:
    """从字典反序列化 CSG 节点"""
    if d["type"] == "leaf":
        return Leaf.from_dict(d)
    else:
        return OpNode.from_dict(d)


def csg_to_expr(node: CSGNode) -> str:
    """生成可读的表达式字符串"""
    if isinstance(node, Leaf):
        return node.prim_type.value.capitalize()
    else:
        return node.to_expr()


def count_leaves(node: CSGNode) -> int:
    """计算叶节点数量"""
    if isinstance(node, Leaf):
        return 1
    else:
        return len(node.get_leaves())


def get_all_leaves(node: CSGNode) -> List[Leaf]:
    """获取所有叶节点"""
    if isinstance(node, Leaf):
        return [node]
    else:
        return node.get_leaves()


def get_all_ops(node: CSGNode) -> List[OpNode]:
    """获取所有操作节点"""
    if isinstance(node, Leaf):
        return []
    else:
        return node.get_ops()


def _choice_enum(rng: np.random.Generator, enum_list: List) -> Any:
    """从枚举列表中随机选择，确保返回正确的类型"""
    idx = int(rng.integers(len(enum_list)))
    return enum_list[idx]


def has_containment_risk(leaves: List["Leaf"], margin: float = 0.05) -> bool:
    """粗略判断 leaf 是否可能完全包含（仅基于中心距离与尺度）"""
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            pos_i = leaves[i].get_position()[0]
            pos_j = leaves[j].get_position()[0]
            dist = abs(pos_i - pos_j)
            r_i = SIZE_LEVEL_MAP[leaves[i].size_level]
            r_j = SIZE_LEVEL_MAP[leaves[j].size_level]
            if dist <= abs(r_i - r_j) + margin:
                return True
    return False


def enforce_leaf_separation(leaves: List["Leaf"]) -> None:
    """调整 leaf 位置/位移档位，降低完全包含导致不可见的概率"""
    if len(leaves) < 2:
        return

    # 先避免过近的位移档位
    for leaf in leaves:
        if leaf.delta_level == DeltaLevel.NEAR:
            leaf.delta_level = DeltaLevel.MID

    if not has_containment_risk(leaves):
        return

    # 提升位移档位，拉开中心距
    for leaf in leaves:
        leaf.delta_level = DeltaLevel.FAR

    if not has_containment_risk(leaves):
        return

    # 仍有风险时，强制拉开 slot
    target_slots = [-1, 1] if len(leaves) == 2 else [-1, 0, 1]
    for leaf, slot in zip(leaves, target_slots):
        leaf.slot = slot


def sample_random_csg(
    rng: np.random.Generator,
    leaf_count: int = 2,
    allowed_ops: Optional[List[OpType]] = None,
) -> CSGNode:
    """随机生成一个 CSG 树
    
    Args:
        rng: 随机数生成器
        leaf_count: 叶节点数量（1/2/3）
        allowed_ops: 允许的操作类型
        
    Returns:
        CSG 树根节点
    """
    if allowed_ops is None:
        allowed_ops = [OpType.UNION, OpType.DIFF]
    
    leaf_count = max(1, min(3, leaf_count))
    
    # 随机选择 primitive 类型
    prim_types = [_choice_enum(rng, PRIM_TYPE_CYCLE) for _ in range(leaf_count)]
    
    # 随机选择属性
    size_levels = list(SizeLevel)
    delta_levels = list(DeltaLevel)
    slots = [-1, 0, 1]
    
    leaves = []
    for i, prim_type in enumerate(prim_types):
        # 为不同的 leaf 分配不同的 slot
        if leaf_count == 1:
            slot = 0
        elif leaf_count == 2:
            slot = slots[i] if i < 2 else 0
        else:
            slot = slots[i] if i < 3 else 0
        
        leaf = Leaf(
            id=i,
            prim_type=prim_type,
            size_level=_choice_enum(rng, size_levels),
            local_pose_deg=tuple(int(rng.choice(DISCRETE_ANGLES)) for _ in range(3)),
            slot=slot,
            delta_level=_choice_enum(rng, delta_levels),
        )
        leaves.append(leaf)

    enforce_leaf_separation(leaves)

    # 构建二叉树
    if leaf_count == 1:
        return leaves[0]
    if leaf_count == 2:
        op = _choice_enum(rng, allowed_ops)
        return OpNode(op=op, left=leaves[0], right=leaves[1])
    else:  # leaf_count == 3
        # 先组合前两个，再与第三个组合
        op1 = _choice_enum(rng, allowed_ops)
        op2 = _choice_enum(rng, allowed_ops)
        inner = OpNode(op=op1, left=leaves[0], right=leaves[1])
        return OpNode(op=op2, left=inner, right=leaves[2])


def copy_csg(node: CSGNode) -> CSGNode:
    """深拷贝 CSG 树"""
    return node.copy()
