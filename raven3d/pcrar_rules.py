"""PCRAR 规则模块.

实现 7 条规则：Progression, Cycle, Toggle, Count, Conservation, Permutation, Symmetry
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from .csg import (
    CSGNode, Leaf, OpNode, OpType, PrimType, SizeLevel, DeltaLevel,
    PRIM_TYPE_CYCLE, DISCRETE_ANGLES, SIZE_LEVEL_MAP, DELTA_LEVEL_MAP,
    get_all_leaves, get_all_ops, copy_csg, sample_random_csg,
)
from .pcrar_entity import PCRAREntity, ObservationConfig, sample_random_entity


# 尺寸档位列表（用于 Progression）
SIZE_LEVELS = [SizeLevel.S, SizeLevel.M, SizeLevel.L]
# Delta 档位列表
DELTA_LEVELS = [DeltaLevel.NEAR, DeltaLevel.MID, DeltaLevel.FAR]
# 槽位列表
SLOTS = [-1, 0, 1]


class RuleTemplate(str, Enum):
    """规则模板枚举"""
    PROGRESSION = "Progression"
    CYCLE = "Cycle"
    TOGGLE = "Toggle"
    COUNT = "Count"
    CONSERVATION = "Conservation"
    PERMUTATION = "Permutation"
    SYMMETRY = "Symmetry"


# 规则来源对齐
RULE_SOURCE_ALIGN: Dict[RuleTemplate, List[str]] = {
    RuleTemplate.PROGRESSION: ["R1-1", "R1-2", "R1-3", "R1-4", "R1-5"],
    RuleTemplate.CYCLE: ["R1-6", "R3-10"],
    RuleTemplate.TOGGLE: ["R3"],
    RuleTemplate.COUNT: ["R1-11", "R3-4", "R3-5", "R4-3"],
    RuleTemplate.CONSERVATION: ["R2-2"],
    RuleTemplate.PERMUTATION: ["R3-2", "R3-7"],
    RuleTemplate.SYMMETRY: ["R4-7"],
}


@dataclass
class RuleParams:
    """规则参数基类"""
    template: RuleTemplate
    axis: Optional[str] = None  # 作用轴: r(size), R(pose), p(position), d(density)
    leaf_idx: Optional[int] = None  # 目标叶节点索引
    direction: int = 1  # 方向: +1/-1
    rot_axis: Optional[str] = None  # 旋转轴: x/y/z（仅 Progression + R 使用）
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "template": self.template.value,
            "axis": self.axis,
            "leaf_idx": self.leaf_idx,
            "direction": self.direction,
            "rot_axis": self.rot_axis,
        }


class PCRARRule(ABC):
    """PCRAR 规则基类"""
    template: RuleTemplate
    source_align: List[str]
    
    @abstractmethod
    def sample_params(self, rng: np.random.Generator, entity: PCRAREntity) -> RuleParams:
        """采样规则参数"""
        pass
    
    @abstractmethod
    def apply(self, entity: PCRAREntity, params: RuleParams) -> PCRAREntity:
        """应用规则变换"""
        pass
    
    @abstractmethod
    def check(self, entity_a: PCRAREntity, entity_b: PCRAREntity, params: RuleParams) -> bool:
        """检查 B = T(A) 是否成立"""
        pass
    
    def can_apply(self, entity: PCRAREntity, params: RuleParams) -> bool:
        """检查规则是否可以应用（前置条件）"""
        return True


def _choice_from_list(rng: np.random.Generator, lst: list):
    """从列表中随机选择元素"""
    idx = int(rng.integers(len(lst)))
    return lst[idx]


class ProgressionRule(PCRARRule):
    """Rule1 Progression（递进）
    
    同一属性沿固定步长变化：A→B 走一步，C→D 也走同一步
    轴（四选一）：r(size_level +/-1)、R(pose +90deg 等)、p(slot shift +/-1 或 delta_level +/-1)、d(weights 档位 +/-1)
    """
    template = RuleTemplate.PROGRESSION
    source_align = RULE_SOURCE_ALIGN[RuleTemplate.PROGRESSION]
    
    def sample_params(self, rng: np.random.Generator, entity: PCRAREntity) -> RuleParams:
        # 随机选择轴
        axis = _choice_from_list(rng, ["r", "R", "p", "d"])
        # 随机方向
        direction = _choice_from_list(rng, [-1, 1])
        rot_axis = _choice_from_list(rng, ["x", "y", "z"]) if axis == "R" else None
        
        return RuleParams(
            template=self.template,
            axis=axis,
            leaf_idx=None,
            direction=direction,
            rot_axis=rot_axis,
        )
    
    def can_apply(self, entity: PCRAREntity, params: RuleParams) -> bool:
        """检查是否可以应用（避免越界）"""
        leaves = entity.get_leaves()
        direction = params.direction
        
        if params.axis == "r":
            for leaf in leaves:
                idx = SIZE_LEVELS.index(leaf.size_level)
                new_idx = idx + direction
                if not (0 <= new_idx < len(SIZE_LEVELS)):
                    return False
            return True
        elif params.axis == "R":
            # 避免旋转落在高对称结构（球体/相交体）上，导致外观变化不可见
            if any(leaf.prim_type == PrimType.SPHERE for leaf in leaves):
                return False
            if any(op.op == OpType.INTERSECT for op in get_all_ops(entity.csg)):
                return False
            # 旋转总是可以（模 360）
            return True
        elif params.axis == "p":
            # 检查 slot 或 delta_level
            for leaf in leaves:
                slot_idx = SLOTS.index(leaf.slot)
                new_slot_idx = slot_idx + direction
                if not (0 <= new_slot_idx < len(SLOTS)):
                    return False
            return True
        elif params.axis == "d":
            # 密度档位
            leaf_count = len(leaves)
            from .pcrar_entity import DENSITY_PRESETS_2, DENSITY_PRESETS_3
            presets = DENSITY_PRESETS_2 if leaf_count == 2 else DENSITY_PRESETS_3
            new_idx = entity.obs.density_preset_idx + direction
            return 0 <= new_idx < len(presets)
        
        return True
    
    def apply(self, entity: PCRAREntity, params: RuleParams) -> PCRAREntity:
        new_entity = entity.copy()
        leaves = new_entity.get_leaves()
        direction = params.direction
        
        if params.axis == "r":
            # 尺寸递进（整体：所有 leaf 同步）
            for leaf in leaves:
                idx = SIZE_LEVELS.index(leaf.size_level)
                new_idx = max(0, min(len(SIZE_LEVELS) - 1, idx + direction))
                leaf.size_level = SIZE_LEVELS[new_idx]
        elif params.axis == "R":
            # 全局姿态递进（按指定旋转轴步进 60 度）
            pose = list(new_entity.obs.global_pose_deg)
            axis = (params.rot_axis or "x").lower()
            axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
            pose[axis_idx] = (pose[axis_idx] + direction * 60) % 360
            new_entity.obs.global_pose_deg = tuple(pose)
        elif params.axis == "p":
            # 位置递进（整体：所有 leaf 同步 slot shift）
            for leaf in leaves:
                slot_idx = SLOTS.index(leaf.slot)
                new_slot_idx = max(0, min(len(SLOTS) - 1, slot_idx + direction))
                leaf.slot = SLOTS[new_slot_idx]
        elif params.axis == "d":
            # 密度档位递进
            leaf_count = len(leaves)
            from .pcrar_entity import DENSITY_PRESETS_2, DENSITY_PRESETS_3
            presets = DENSITY_PRESETS_2 if leaf_count == 2 else DENSITY_PRESETS_3
            new_idx = max(0, min(len(presets) - 1, new_entity.obs.density_preset_idx + direction))
            new_entity.obs.density_preset_idx = new_idx
        
        return new_entity
    
    def check(self, entity_a: PCRAREntity, entity_b: PCRAREntity, params: RuleParams) -> bool:
        expected = self.apply(entity_a, params)
        return _compare_entities(expected, entity_b)


class CycleRule(PCRARRule):
    """Rule2 Cycle（循环）
    
    某个 leaf 的 prim_type 做离散循环 Sphere→Box→Cylinder→Cone→Sphere
    """
    template = RuleTemplate.CYCLE
    source_align = RULE_SOURCE_ALIGN[RuleTemplate.CYCLE]
    
    def sample_params(self, rng: np.random.Generator, entity: PCRAREntity) -> RuleParams:
        leaves = entity.get_leaves()
        leaf_idx = int(rng.integers(len(leaves)))
        direction = _choice_from_list(rng, [-1, 1])
        
        return RuleParams(
            template=self.template,
            axis="shape",
            leaf_idx=leaf_idx,
            direction=direction,
        )
    
    def apply(self, entity: PCRAREntity, params: RuleParams) -> PCRAREntity:
        new_entity = entity.copy()
        leaves = new_entity.get_leaves()
        leaf = leaves[params.leaf_idx]
        
        # 形状循环
        idx = PRIM_TYPE_CYCLE.index(leaf.prim_type)
        new_idx = (idx + params.direction) % len(PRIM_TYPE_CYCLE)
        leaf.prim_type = PRIM_TYPE_CYCLE[new_idx]
        
        return new_entity
    
    def check(self, entity_a: PCRAREntity, entity_b: PCRAREntity, params: RuleParams) -> bool:
        expected = self.apply(entity_a, params)
        return _compare_entities(expected, entity_b)


class ToggleRule(PCRARRule):
    """Rule3 Toggle（切换）
    
    保持 leaf 不变，只切换某个内部节点的 op：Union ↔ Diff
    """
    template = RuleTemplate.TOGGLE
    source_align = RULE_SOURCE_ALIGN[RuleTemplate.TOGGLE]
    
    def sample_params(self, rng: np.random.Generator, entity: PCRAREntity) -> RuleParams:
        # 找到所有可切换的操作节点（Union / Diff）
        ops = [op for op in get_all_ops(entity.csg) if op.op in (OpType.UNION, OpType.DIFF)]
        op_idx = int(rng.integers(len(ops))) if ops else 0
        return RuleParams(
            template=self.template,
            axis="op",
            leaf_idx=op_idx,
            direction=1,
        )
    
    def can_apply(self, entity: PCRAREntity, params: RuleParams) -> bool:
        # 需要有可切换的操作节点（Union / Diff）
        ops = [op for op in get_all_ops(entity.csg) if op.op in (OpType.UNION, OpType.DIFF)]
        return len(ops) > 0
    
    def apply(self, entity: PCRAREntity, params: RuleParams) -> PCRAREntity:
        new_entity = entity.copy()
        
        # 找到并切换可切换操作节点
        ops = [op for op in get_all_ops(new_entity.csg) if op.op in (OpType.UNION, OpType.DIFF)]
        if not ops:
            return new_entity
        
        op_idx = params.leaf_idx % len(ops)
        op_node = ops[op_idx]
        
        # Union ↔ Diff 切换
        if op_node.op == OpType.UNION:
            op_node.op = OpType.DIFF
        elif op_node.op == OpType.DIFF:
            op_node.op = OpType.UNION
        # Intersect 保持不变
        
        return new_entity
    
    def check(self, entity_a: PCRAREntity, entity_b: PCRAREntity, params: RuleParams) -> bool:
        expected = self.apply(entity_a, params)
        return _compare_entities(expected, entity_b)


class CountRule(PCRARRule):
    """Rule4 Count（增减）
    
    leaf_count: 2→3 或 3→2
    """
    template = RuleTemplate.COUNT
    source_align = RULE_SOURCE_ALIGN[RuleTemplate.COUNT]
    
    def sample_params(self, rng: np.random.Generator, entity: PCRAREntity) -> RuleParams:
        leaf_count = entity.leaf_count()
        # 2→3 增加，3→2 减少
        direction = 1 if leaf_count == 2 else -1
        
        return RuleParams(
            template=self.template,
            axis="count",
            leaf_idx=None,
            direction=direction,
        )
    
    def can_apply(self, entity: PCRAREntity, params: RuleParams) -> bool:
        leaf_count = entity.leaf_count()
        if params.direction == 1 and leaf_count >= 3:
            return False
        if params.direction == -1 and leaf_count <= 2:
            return False
        return True
    
    def apply(self, entity: PCRAREntity, params: RuleParams) -> PCRAREntity:
        new_entity = entity.copy()
        
        if params.direction == 1:
            # 增加一个叶节点（2→3）
            new_entity = self._add_leaf(new_entity)
        else:
            # 减少一个叶节点（3→2）
            new_entity = self._remove_leaf(new_entity)
        
        return new_entity
    
    def _add_leaf(self, entity: PCRAREntity) -> PCRAREntity:
        """添加一个叶节点"""
        leaves = entity.get_leaves()
        new_id = max(leaf.id for leaf in leaves) + 1
        
        # 创建新叶节点
        new_leaf = Leaf(
            id=new_id,
            prim_type=leaves[0].prim_type,  # 继承第一个 leaf 的类型
            size_level=SizeLevel.M,
            local_pose_deg=(0, 0, 0),
            slot=0,  # 放在中间位置
            delta_level=DeltaLevel.MID,
        )
        
        # 重建 CSG 树：将新叶节点添加到树中
        old_csg = entity.csg
        new_csg = OpNode(
            op=OpType.UNION,
            left=copy_csg(old_csg),
            right=new_leaf,
        )
        
        # 更新密度权重
        from .pcrar_entity import DENSITY_PRESETS_3
        new_obs = entity.obs.copy()
        new_obs.density_preset_idx = 0  # 使用均匀分布
        
        return PCRAREntity(csg=new_csg, obs=new_obs)
    
    def _remove_leaf(self, entity: PCRAREntity) -> PCRAREntity:
        """移除一个叶节点"""
        # 简化处理：移除最后一个叶节点
        # 这需要重建树结构
        leaves = entity.get_leaves()
        if len(leaves) <= 2:
            return entity
        
        # 使用前两个叶节点重建简单树
        new_csg = OpNode(
            op=OpType.UNION,
            left=leaves[0].copy(),
            right=leaves[1].copy(),
        )
        
        # 更新密度权重
        from .pcrar_entity import DENSITY_PRESETS_2
        new_obs = entity.obs.copy()
        new_obs.density_preset_idx = 0
        
        return PCRAREntity(csg=new_csg, obs=new_obs)
    
    def check(self, entity_a: PCRAREntity, entity_b: PCRAREntity, params: RuleParams) -> bool:
        # 检查叶节点数量变化
        count_a = entity_a.leaf_count()
        count_b = entity_b.leaf_count()
        expected_count = count_a + params.direction
        return count_b == expected_count


class ConservationRule(PCRARRule):
    """Rule5 Conservation（守恒）
    
    任选两 leaf：size(i)+size(j)=C（离散守恒，一增一减）
    """
    template = RuleTemplate.CONSERVATION
    source_align = RULE_SOURCE_ALIGN[RuleTemplate.CONSERVATION]
    
    def sample_params(self, rng: np.random.Generator, entity: PCRAREntity) -> RuleParams:
        leaves = entity.get_leaves()
        if len(leaves) < 2:
            return RuleParams(template=self.template)
        
        # 选择两个叶节点
        indices = rng.choice(len(leaves), size=2, replace=False)
        
        return RuleParams(
            template=self.template,
            axis="size_conservation",
            leaf_idx=int(indices[0]),  # 第一个 leaf 索引
            direction=int(indices[1]),  # 第二个 leaf 索引（复用 direction 字段）
        )
    
    def can_apply(self, entity: PCRAREntity, params: RuleParams) -> bool:
        leaves = entity.get_leaves()
        if len(leaves) < 2:
            return False
        
        idx1 = params.leaf_idx
        idx2 = params.direction  # 复用 direction 存储第二个索引
        
        if idx1 >= len(leaves) or idx2 >= len(leaves):
            return False
        
        # 检查是否可以一增一减
        leaf1 = leaves[idx1]
        leaf2 = leaves[idx2]
        
        idx1_size = SIZE_LEVELS.index(leaf1.size_level)
        idx2_size = SIZE_LEVELS.index(leaf2.size_level)
        
        # leaf1 增加需要 idx1_size < max，leaf2 减少需要 idx2_size > 0
        return idx1_size < len(SIZE_LEVELS) - 1 and idx2_size > 0
    
    def apply(self, entity: PCRAREntity, params: RuleParams) -> PCRAREntity:
        new_entity = entity.copy()
        leaves = new_entity.get_leaves()
        
        idx1 = params.leaf_idx
        idx2 = params.direction
        
        if idx1 >= len(leaves) or idx2 >= len(leaves):
            return new_entity
        
        leaf1 = leaves[idx1]
        leaf2 = leaves[idx2]
        
        # 守恒变换：一增一减
        idx1_size = SIZE_LEVELS.index(leaf1.size_level)
        idx2_size = SIZE_LEVELS.index(leaf2.size_level)
        
        new_idx1 = min(len(SIZE_LEVELS) - 1, idx1_size + 1)
        new_idx2 = max(0, idx2_size - 1)
        
        leaf1.size_level = SIZE_LEVELS[new_idx1]
        leaf2.size_level = SIZE_LEVELS[new_idx2]
        
        return new_entity
    
    def check(self, entity_a: PCRAREntity, entity_b: PCRAREntity, params: RuleParams) -> bool:
        expected = self.apply(entity_a, params)
        return _compare_entities(expected, entity_b)


class PermutationRule(PCRARRule):
    """Rule6 Permutation（置换）
    
    把 leaf 当槽位对象做循环置换：
    - ShiftSlots：leaf 的 slot assignment 循环左移/右移
    """
    template = RuleTemplate.PERMUTATION
    source_align = RULE_SOURCE_ALIGN[RuleTemplate.PERMUTATION]
    
    def sample_params(self, rng: np.random.Generator, entity: PCRAREntity) -> RuleParams:
        direction = _choice_from_list(rng, [-1, 1])
        
        return RuleParams(
            template=self.template,
            axis="slot_permutation",
            leaf_idx=None,
            direction=direction,
        )
    
    def can_apply(self, entity: PCRAREntity, params: RuleParams) -> bool:
        return entity.leaf_count() >= 2
    
    def apply(self, entity: PCRAREntity, params: RuleParams) -> PCRAREntity:
        new_entity = entity.copy()
        leaves = new_entity.get_leaves()
        
        # 收集所有 slot
        slots = [leaf.slot for leaf in leaves]
        
        # 循环置换
        if params.direction == 1:
            # 右移：[0,1,2] -> [2,0,1]
            slots = [slots[-1]] + slots[:-1]
        else:
            # 左移：[0,1,2] -> [1,2,0]
            slots = slots[1:] + [slots[0]]
        
        # 应用新的 slot
        for leaf, new_slot in zip(leaves, slots):
            leaf.slot = new_slot
        
        return new_entity
    
    def check(self, entity_a: PCRAREntity, entity_b: PCRAREntity, params: RuleParams) -> bool:
        expected = self.apply(entity_a, params)
        return _compare_entities(expected, entity_b)


class SymmetryRule(PCRARRule):
    """Rule7 Symmetry（对称）
    
    选一对 leaf（左/右）作为对称对：
    - A→B 左做 +Δ，则右做 -Δ（作用于 p 或 R）
    - 第三个 leaf（若存在）可当锚点不动
    """
    template = RuleTemplate.SYMMETRY
    source_align = RULE_SOURCE_ALIGN[RuleTemplate.SYMMETRY]
    
    def sample_params(self, rng: np.random.Generator, entity: PCRAREntity) -> RuleParams:
        axis = _choice_from_list(rng, ["p", "R"])  # 位置或姿态
        
        return RuleParams(
            template=self.template,
            axis=axis,
            leaf_idx=0,  # 左侧 leaf 索引
            direction=1,  # 右侧 leaf 索引
        )
    
    def can_apply(self, entity: PCRAREntity, params: RuleParams) -> bool:
        leaves = entity.get_leaves()
        if len(leaves) < 2:
            return False
        
        if params.axis == "p":
            # 检查位置是否可以对称变化
            left = leaves[params.leaf_idx]
            right = leaves[params.direction]
            left_slot_idx = SLOTS.index(left.slot)
            right_slot_idx = SLOTS.index(right.slot)
            # 左侧可以增，右侧可以减（或相反）
            return (left_slot_idx < len(SLOTS) - 1 and right_slot_idx > 0)
        
        return True
    
    def apply(self, entity: PCRAREntity, params: RuleParams) -> PCRAREntity:
        new_entity = entity.copy()
        leaves = new_entity.get_leaves()
        
        if len(leaves) < 2:
            return new_entity
        
        left = leaves[params.leaf_idx]
        right = leaves[params.direction]
        
        if params.axis == "p":
            # 位置对称变化
            left_slot_idx = SLOTS.index(left.slot)
            right_slot_idx = SLOTS.index(right.slot)
            
            new_left_idx = min(len(SLOTS) - 1, left_slot_idx + 1)
            new_right_idx = max(0, right_slot_idx - 1)
            
            left.slot = SLOTS[new_left_idx]
            right.slot = SLOTS[new_right_idx]
        elif params.axis == "R":
            # 姿态对称变化
            left_pose = list(left.local_pose_deg)
            right_pose = list(right.local_pose_deg)
            
            left_pose[0] = (left_pose[0] + 90) % 360
            right_pose[0] = (right_pose[0] - 90) % 360
            
            left.local_pose_deg = tuple(left_pose)
            right.local_pose_deg = tuple(right_pose)
        
        return new_entity
    
    def check(self, entity_a: PCRAREntity, entity_b: PCRAREntity, params: RuleParams) -> bool:
        expected = self.apply(entity_a, params)
        return _compare_entities(expected, entity_b)


def _compare_entities(e1: PCRAREntity, e2: PCRAREntity) -> bool:
    """比较两个实体是否相等"""
    d1 = e1.to_dict()
    d2 = e2.to_dict()
    # 忽略 expr 字段
    d1.pop("expr", None)
    d2.pop("expr", None)
    return d1 == d2


# 规则注册表
PCRAR_RULES: Dict[RuleTemplate, Type[PCRARRule]] = {
    RuleTemplate.PROGRESSION: ProgressionRule,
    RuleTemplate.CYCLE: CycleRule,
    RuleTemplate.TOGGLE: ToggleRule,
    RuleTemplate.COUNT: CountRule,
    RuleTemplate.CONSERVATION: ConservationRule,
    RuleTemplate.PERMUTATION: PermutationRule,
    RuleTemplate.SYMMETRY: SymmetryRule,
}


def get_rule(template: RuleTemplate) -> PCRARRule:
    """获取规则实例"""
    return PCRAR_RULES[template]()


def sample_rule(rng: np.random.Generator) -> PCRARRule:
    """随机采样一条规则"""
    template = _choice_from_list(rng, list(RuleTemplate))
    return get_rule(template)


def sample_applicable_rule(
    rng: np.random.Generator,
    entity: PCRAREntity,
    max_attempts: int = 20,
) -> Tuple[PCRARRule, RuleParams]:
    """采样一条可应用的规则及其参数"""
    templates = list(RuleTemplate)
    rng.shuffle(templates)
    
    for template in templates:
        rule = get_rule(template)
        for _ in range(max_attempts):
            params = rule.sample_params(rng, entity)
            if rule.can_apply(entity, params):
                return rule, params
    
    # 回退到 Cycle（总是可以应用）
    rule = CycleRule()
    params = rule.sample_params(rng, entity)
    return rule, params


def generate_distractor(
    entity: PCRAREntity,
    rule: PCRARRule,
    params: RuleParams,
    rng: np.random.Generator,
) -> Tuple[PCRAREntity, str]:
    """生成一个干扰项
    
    返回 (干扰实体, 错误原因)
    """
    # 策略：应用不同的变换或相同变换的不同参数
    distractor = entity.copy()
    reason = ""
    
    # 随机选择干扰方式
    method = _choice_from_list(rng, ["different_axis", "different_direction", "different_leaf", "different_rule"])
    
    if method == "different_axis" and params.axis:
        # 使用不同的属性轴
        axes = ["r", "R", "p", "d"]
        other_axes = [a for a in axes if a != params.axis]
        if other_axes:
            new_axis = _choice_from_list(rng, other_axes)
            new_params = RuleParams(
                template=params.template,
                axis=new_axis,
                leaf_idx=params.leaf_idx,
                direction=params.direction,
            )
            if rule.can_apply(distractor, new_params):
                distractor = rule.apply(distractor, new_params)
                reason = f"应用了错误的属性轴 {new_axis}，正确应为 {params.axis}"
                return distractor, reason
    
    if method == "different_direction":
        # 使用相反的方向
        new_params = RuleParams(
            template=params.template,
            axis=params.axis,
            leaf_idx=params.leaf_idx,
            direction=-params.direction,
        )
        if rule.can_apply(distractor, new_params):
            distractor = rule.apply(distractor, new_params)
            reason = "应用了相反的变换方向"
            return distractor, reason
    
    if method == "different_leaf":
        # 应用到不同的叶节点
        leaves = entity.get_leaves()
        if len(leaves) > 1 and params.leaf_idx is not None:
            other_idx = (params.leaf_idx + 1) % len(leaves)
            new_params = RuleParams(
                template=params.template,
                axis=params.axis,
                leaf_idx=other_idx,
                direction=params.direction,
            )
            if rule.can_apply(distractor, new_params):
                distractor = rule.apply(distractor, new_params)
                reason = f"应用到了错误的对象（leaf {other_idx} 而非 {params.leaf_idx}）"
                return distractor, reason
    
    # 回退：应用不同的规则
    other_templates = [t for t in RuleTemplate if t != params.template]
    other_template = _choice_from_list(rng, other_templates)
    other_rule = get_rule(other_template)
    other_params = other_rule.sample_params(rng, distractor)
    if other_rule.can_apply(distractor, other_params):
        distractor = other_rule.apply(distractor, other_params)
        reason = f"应用了错误的规则 {other_template.value}，正确应为 {params.template.value}"
    else:
        # 最后回退：简单修改一个属性
        leaves = distractor.get_leaves()
        if leaves:
            leaf = leaves[0]
            idx = SIZE_LEVELS.index(leaf.size_level)
            new_idx = (idx + 1) % len(SIZE_LEVELS)
            leaf.size_level = SIZE_LEVELS[new_idx]
            reason = "尺寸变化不符合规则模式"
    
    return distractor, reason
