"""PCRAR 数据集生成模块.

支持 Relational（2→1）和 Analogical（3→1）两种题型。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .io import write_meta, write_ply, ensure_dir
from .csg import OpType
from .pcrar_entity import PCRAREntity, sample_random_entity, DEFAULT_N_POINTS, entities_equal
from .pcrar_rules import (
    RuleTemplate, RuleParams, PCRARRule,
    sample_applicable_rule, generate_distractor, get_rule,
    RULE_SOURCE_ALIGN,
)


# 颜色映射（与原项目保持一致）
COLOR_MAP = {
    "in_0.ply": (31, 119, 180),    # 深海蓝
    "in_1.ply": (255, 127, 14),    # 鲜亮橙
    "in_2.ply": (44, 160, 44),     # 森林绿
    "cand_0.ply": (214, 39, 40),   # 砖红
    "cand_1.ply": (148, 103, 189), # 柔和紫
    "cand_2.ply": (140, 86, 75),   # 可可棕
    "cand_3.ply": (227, 119, 194), # 粉色
}


@dataclass
class PCRARConfig:
    """PCRAR 数据集生成配置"""
    n_points: int = DEFAULT_N_POINTS
    n_candidates: int = 4  # 候选数量
    task_mix: float = 0.5  # Relational 比例（0.5 表示一半 Relational，一半 Analogical）
    leaf_count_min: int = 2
    leaf_count_max: int = 3
    allowed_ops: Optional[List[OpType]] = field(
        default_factory=lambda: [OpType.UNION]
    )
    rule_filter: Optional[Set[RuleTemplate]] = None


class PCRARDatasetGenerator:
    """PCRAR 数据集生成器"""
    
    def __init__(
        self,
        config: Optional[PCRARConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or PCRARConfig()
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            np.random.seed(seed)
    
    def generate_sample(
        self,
        output_root: Path,
        sample_index: int,
        task_type: Optional[str] = None,
        correct_idx: Optional[int] = None,
        preferred_axis: Optional[str] = None,
    ) -> Dict[str, Any]:
        """生成单个样本
        
        Args:
            output_root: 输出根目录
            sample_index: 样本索引
            task_type: 任务类型 "relational" 或 "analogical"，None 表示随机
            correct_idx: 正确答案索引，None 表示随机
            
        Returns:
            样本元数据字典
        """
        output_root = Path(output_root)
        
        # 确定任务类型
        if task_type is None:
            task_type = "relational" if self.rng.random() < self.config.task_mix else "analogical"
        
        # 确定正确答案位置
        if correct_idx is None:
            correct_idx = int(self.rng.integers(self.config.n_candidates))
        
        if task_type == "relational":
            return self._generate_relational_sample(
                output_root,
                sample_index,
                correct_idx,
                preferred_axis=preferred_axis,
            )
        else:
            return self._generate_analogical_sample(
                output_root,
                sample_index,
                correct_idx,
                preferred_axis=preferred_axis,
            )
    
    def _generate_relational_sample(
        self,
        output_root: Path,
        sample_index: int,
        correct_idx: int,
        preferred_axis: Optional[str] = None,
    ) -> Dict[str, Any]:
        """生成 Relational（2→1）样本
        
        目标：让模型"认知属性/关系规律"
        - 采样 Attr(A)
        - 采样规则 T（含参数）
        - 得到 B = T(A)
        - 正确答案 D* = T(B)
        - 候选 {D1..Dk}，仅 D* 满足同一个 T
        """
        # 生成初始实体 A，并确保规则可连续应用两次
        max_entity_attempts = 50
        max_rule_attempts = 50
        for _ in range(max_entity_attempts):
            leaf_count = int(self.rng.integers(self.config.leaf_count_min, self.config.leaf_count_max + 1))
            entity_a = sample_random_entity(self.rng, leaf_count=leaf_count, allowed_ops=self.config.allowed_ops)
            for _ in range(max_rule_attempts):
                try:
                    rule, params = self._sample_rule(entity_a, preferred_axis=preferred_axis)
                except RuntimeError:
                    continue
                if params.axis in ("r", "p"):
                    entity_a = self._adjust_entity_for_rule(entity_a, rule, params)
                entity_b = rule.apply(entity_a, params)
                if rule.can_apply(entity_b, params):
                    entity_correct = rule.apply(entity_b, params)
                    break
            else:
                continue
            break
        else:
            raise RuntimeError("Failed to sample a relational rule that can be applied twice.")
        
        # 生成干扰项（去重）
        distractors = []
        distractor_reasons = []
        max_attempts = 50
        while len(distractors) < self.config.n_candidates - 1:
            distractor, reason = generate_distractor(entity_b, rule, params, self.rng)
            if entities_equal(distractor, entity_correct, check_obs=True):
                continue
            if any(entities_equal(distractor, d, check_obs=True) for d in distractors):
                continue
            distractors.append(distractor)
            distractor_reasons.append(reason)
            if len(distractors) < self.config.n_candidates - 1 and max_attempts > 0:
                max_attempts -= 1
        if len(distractors) < self.config.n_candidates - 1:
            raise RuntimeError("Failed to generate unique distractors.")
        
        # 组装候选
        candidates = [None] * self.config.n_candidates
        candidate_reasons = [""] * self.config.n_candidates
        candidates[correct_idx] = entity_correct
        candidate_reasons[correct_idx] = "符合规则的正确延续"
        
        d_idx = 0
        for i in range(self.config.n_candidates):
            if candidates[i] is None:
                candidates[i] = distractors[d_idx]
                candidate_reasons[i] = distractor_reasons[d_idx]
                d_idx += 1
        
        # 生成点云并保存
        return self._save_sample(
            output_root=output_root,
            sample_index=sample_index,
            task_type="relational",
            inputs=[entity_a, entity_b],
            candidates=candidates,
            correct_idx=correct_idx,
            candidate_reasons=candidate_reasons,
            rule=rule,
            params=params,
        )
    
    def _generate_analogical_sample(
        self,
        output_root: Path,
        sample_index: int,
        correct_idx: int,
        preferred_axis: Optional[str] = None,
    ) -> Dict[str, Any]:
        """生成 Analogical（3→1）样本
        
        目标：把在 A→B 学到的关系规律迁移到新的几何体 C 上
        - 采样 Attr(A)
        - 采样规则 T
        - B = T(A)
        - 采样 Attr(C)（要求与 A 同分布、但不是 A 的 copy；且与规则前置条件兼容）
        - D* = T(C)
        - 生成候选，仅 D* 满足 T
        """
        # 生成初始实体 A，并确保规则可应用
        max_entity_attempts = 50
        for _ in range(max_entity_attempts):
            leaf_count = int(self.rng.integers(self.config.leaf_count_min, self.config.leaf_count_max + 1))
            entity_a = sample_random_entity(self.rng, leaf_count=leaf_count, allowed_ops=self.config.allowed_ops)
            try:
                rule, params = self._sample_rule(entity_a, preferred_axis=preferred_axis)
            except RuntimeError:
                continue
            entity_b = rule.apply(entity_a, params)
            break
        else:
            raise RuntimeError("Failed to sample an analogical rule for the given config.")
        
        # 生成 C（与 A 结构相似但不同）
        entity_c = self._generate_compatible_entity(entity_a, rule, params)
        
        # 确保规则可以应用到 C
        if not rule.can_apply(entity_c, params):
            # 调整参数或重新生成 C
            entity_c = self._adjust_entity_for_rule(entity_c, rule, params)
        
        # 生成正确答案 D* = T(C)
        entity_correct = rule.apply(entity_c, params)
        
        # 生成干扰项（去重）
        distractors = []
        distractor_reasons = []
        max_attempts = 50
        while len(distractors) < self.config.n_candidates - 1:
            distractor, reason = generate_distractor(entity_c, rule, params, self.rng)
            if entities_equal(distractor, entity_correct, check_obs=True):
                continue
            if any(entities_equal(distractor, d, check_obs=True) for d in distractors):
                continue
            distractors.append(distractor)
            distractor_reasons.append(reason)
            if len(distractors) < self.config.n_candidates - 1 and max_attempts > 0:
                max_attempts -= 1
        if len(distractors) < self.config.n_candidates - 1:
            raise RuntimeError("Failed to generate unique distractors.")
        
        # 组装候选
        candidates = [None] * self.config.n_candidates
        candidate_reasons = [""] * self.config.n_candidates
        candidates[correct_idx] = entity_correct
        candidate_reasons[correct_idx] = "符合规则的正确延续"
        
        d_idx = 0
        for i in range(self.config.n_candidates):
            if candidates[i] is None:
                candidates[i] = distractors[d_idx]
                candidate_reasons[i] = distractor_reasons[d_idx]
                d_idx += 1
        
        # 生成点云并保存
        return self._save_sample(
            output_root=output_root,
            sample_index=sample_index,
            task_type="analogical",
            inputs=[entity_a, entity_b, entity_c],
            candidates=candidates,
            correct_idx=correct_idx,
            candidate_reasons=candidate_reasons,
            rule=rule,
            params=params,
        )
    
    def _sample_rule(
        self,
        entity: PCRAREntity,
        preferred_axis: Optional[str] = None,
    ) -> Tuple[PCRARRule, RuleParams]:
        """采样一条可应用的规则"""
        if self.config.rule_filter:
            # 从过滤后的规则中选择
            templates = list(self.config.rule_filter)
            self.rng.shuffle(templates)
            for template in templates:
                rule = get_rule(template)
                for _ in range(20):
                    if preferred_axis and template == RuleTemplate.PROGRESSION:
                        direction = int(self.rng.choice([-1, 1]))
                        rot_axis = (
                            str(self.rng.choice(["x", "y", "z"]))
                            if preferred_axis == "R"
                            else None
                        )
                        params = RuleParams(
                            template=RuleTemplate.PROGRESSION,
                            axis=preferred_axis,
                            leaf_idx=None,
                            direction=direction,
                            rot_axis=rot_axis,
                        )
                    else:
                        params = rule.sample_params(self.rng, entity)
                    if rule.can_apply(entity, params):
                        return rule, params
            raise RuntimeError("No applicable rules for current rule_filter.")
        
        return sample_applicable_rule(self.rng, entity)
    
    def _generate_compatible_entity(
        self,
        entity_a: PCRAREntity,
        rule: PCRARRule,
        params: RuleParams,
    ) -> PCRAREntity:
        """生成与 A 兼容但不同的实体 C"""
        # 尝试生成一个新实体，保持相同的叶节点数量
        leaf_count = entity_a.leaf_count()
        
        for _ in range(10):
            entity_c = sample_random_entity(self.rng, leaf_count=leaf_count, allowed_ops=self.config.allowed_ops)
            
            # 确保不是 A 的副本
            if entity_c.to_dict() != entity_a.to_dict():
                return entity_c
        
        # 回退：修改 A 的副本
        entity_c = entity_a.copy()
        leaves = entity_c.get_leaves()
        if leaves:
            # 修改第一个 leaf 的形状
            from .csg import PRIM_TYPE_CYCLE
            leaf = leaves[0]
            idx = PRIM_TYPE_CYCLE.index(leaf.prim_type)
            new_idx = (idx + 1) % len(PRIM_TYPE_CYCLE)
            leaf.prim_type = PRIM_TYPE_CYCLE[new_idx]
        
        return entity_c
    
    def _adjust_entity_for_rule(
        self,
        entity: PCRAREntity,
        rule: PCRARRule,
        params: RuleParams,
    ) -> PCRAREntity:
        """调整实体以满足规则前置条件"""
        from .csg import SizeLevel
        from .pcrar_rules import SIZE_LEVELS, SLOTS
        
        new_entity = entity.copy()
        leaves = new_entity.get_leaves()
        
        if params.axis == "r":
            # 确保尺寸可以变化（整体：所有 leaf）
            for leaf in leaves:
                if params.direction == 1:
                    leaf.size_level = SIZE_LEVELS[0]  # 设为最小，可以增加
                else:
                    leaf.size_level = SIZE_LEVELS[-1]  # 设为最大，可以减少
        elif params.axis == "p":
            # 确保位置可以变化（整体：所有 leaf）
            for leaf in leaves:
                if params.direction == 1:
                    leaf.slot = SLOTS[0]
                else:
                    leaf.slot = SLOTS[-1]
        
        return new_entity
    
    def _save_sample(
        self,
        output_root: Path,
        sample_index: int,
        task_type: str,
        inputs: List[PCRAREntity],
        candidates: List[PCRAREntity],
        correct_idx: int,
        candidate_reasons: List[str],
        rule: PCRARRule,
        params: RuleParams,
    ) -> Dict[str, Any]:
        """保存样本并生成元数据"""
        focus = self._build_focus(task_type, rule, params, inputs)
        sample_id = f"sample_{sample_index:06d}"
        sample_dir = output_root / sample_id
        ensure_dir(sample_dir)
        
        # 保存输入点云
        input_paths = []
        input_entities = []
        for i, entity in enumerate(inputs):
            filename = f"in_{i}.ply"
            filepath = sample_dir / filename
            points = entity.sample_point_cloud(self.rng, n_points=self.config.n_points)
            write_ply(filepath, points, color=COLOR_MAP.get(filename))
            input_paths.append(f"{sample_id}/{filename}")
            input_entities.append(entity.to_dict())
        
        # 保存候选点云
        candidate_paths = []
        candidate_entities = []
        for i, entity in enumerate(candidates):
            filename = f"cand_{i}.ply"
            filepath = sample_dir / filename
            points = entity.sample_point_cloud(self.rng, n_points=self.config.n_points)
            write_ply(filepath, points, color=COLOR_MAP.get(filename))
            candidate_paths.append(f"{sample_id}/{filename}")
            candidate_entities.append(entity.to_dict())
        
        # 生成标签
        gt_labels = ["A", "B", "C", "D"]
        gt_label = gt_labels[correct_idx] if correct_idx < len(gt_labels) else str(correct_idx)
        
        # 构建元数据
        meta = {
            "id": sample_id,
            "task_type": task_type,
            "focus": focus,
            "input_paths": input_paths,
            "candidate_paths": candidate_paths,
            "gt_index": correct_idx,
            "gt_label": gt_label,
            "n_points": self.config.n_points,
            "rule": {
                "template": rule.template.value,
                "source_align": RULE_SOURCE_ALIGN.get(rule.template, []),
                "params": params.to_dict(),
            },
            "entities": {
                "inputs": input_entities,
                "candidates": candidate_entities,
            },
            "notes": {
                "distractors": [r for i, r in enumerate(candidate_reasons) if i != correct_idx],
            },
        }
        
        write_meta(sample_dir / "meta.json", meta)
        return meta

    def _build_focus(
        self,
        task_type: str,
        rule: PCRARRule,
        params: RuleParams,
        inputs: List[PCRAREntity],
    ) -> str:
        """生成考点描述"""
        axis = params.axis or ""
        leaf_idx = params.leaf_idx
        leaf_indices = params.leaf_indices or ([leaf_idx] if leaf_idx is not None else [])
        direction = params.direction
        directions = params.directions or ([direction] * len(leaf_indices) if leaf_indices else [])
        dir_sign = "+" if direction >= 0 else "-"
        dir_word = "增加" if direction >= 0 else "减少"
        shift_word = "右移" if direction >= 0 else "左移"
        if task_type == "analogical":
            prefix = "类比推理：先从 A→B 归纳规则，再将同一规则应用到 C 得到答案。"
        else:
            prefix = "关系推理：从 A→B 归纳规则，答案是对 B 的同规则延续。"

        if rule.template == RuleTemplate.PROGRESSION:
            if axis == "r":
                core = f"递进规则：整体尺寸档位{dir_word}1（所有 leaf 同步，S/M/L）。"
            elif axis == "R":
                rot_axis = (params.rot_axis or "X").upper()
                core = f"递进规则：整体绕 {rot_axis} 轴旋转 {dir_sign}60°。"
            elif axis == "p":
                core = f"递进规则：整体槽位 slot {shift_word}1 格（所有 leaf 同步）。"
            elif axis == "d":
                core = f"递进规则：采样密度档位{dir_word}1（权重变化）。"
            else:
                core = "递进规则：沿同一属性做离散步进。"
        elif rule.template == RuleTemplate.CYCLE:
            if leaf_indices:
                parts = []
                for i, idx in enumerate(leaf_indices):
                    d = directions[i] if i < len(directions) else direction
                    sign = "+" if d >= 0 else "-"
                    parts.append(f"leaf{idx}({sign}1)")
                leaf_part = "，".join(parts)
                core = f"循环规则：{leaf_part} 形状按序循环（Sphere→Box→Cylinder→Cone）。"
            else:
                core = f"循环规则：leaf{leaf_idx} 形状按序循环（Sphere→Box→Cylinder→Cone），方向 {dir_sign}1。"
        elif rule.template == RuleTemplate.TOGGLE:
            core = f"切换规则：CSG 操作 Union ↔ Diff 切换（op 索引 {leaf_idx}）。"
        elif rule.template == RuleTemplate.COUNT:
            if direction >= 0:
                core = "增减规则：leaf 数量按 1→2→3→1 正向循环。"
            else:
                core = "增减规则：leaf 数量按 1→3→2→1 反向循环。"
        elif rule.template == RuleTemplate.CONSERVATION:
            core = f"守恒规则：leaf{leaf_idx} 尺寸 +1，leaf{direction} 尺寸 -1（一增一减）。"
        elif rule.template == RuleTemplate.PERMUTATION:
            core = f"置换规则：所有 leaf 的 slot 位置循环{shift_word}。"
        elif rule.template == RuleTemplate.SYMMETRY:
            if axis == "p":
                core = f"对称规则：左右 leaf 位置对称变化（左 +1，右 -1）。"
            else:
                core = f"对称规则：左右 leaf 姿态对称变化（左 +90°，右 -90°）。"
        else:
            core = "规则变换：按指定模板对实体属性做变换。"

        return f"{prefix}{core}"
    
    def generate_dataset(
        self,
        output_root: str | Path,
        num_samples: int,
    ) -> None:
        """生成数据集
        
        Args:
            output_root: 输出根目录
            num_samples: 样本数量
        """
        output_root = Path(output_root)
        ensure_dir(output_root)
        
        # 平衡正确答案位置
        n_candidates = self.config.n_candidates
        base = num_samples // n_candidates
        remainder = num_samples % n_candidates
        correct_indices = []
        for i in range(n_candidates):
            correct_indices.extend([i] * (base + (1 if i < remainder else 0)))
        self.rng.shuffle(correct_indices)
        
        # 平衡任务类型
        n_relational = int(num_samples * self.config.task_mix)
        n_analogical = num_samples - n_relational
        task_types = ["relational"] * n_relational + ["analogical"] * n_analogical
        self.rng.shuffle(task_types)
        
        axis_plan: Optional[List[str]] = None
        if (
            self.config.rule_filter
            and len(self.config.rule_filter) == 1
            and RuleTemplate.PROGRESSION in self.config.rule_filter
        ):
            axes = ["r", "R", "p", "d"]
            base = num_samples // len(axes)
            remainder = num_samples % len(axes)
            axis_plan = []
            for i, axis in enumerate(axes):
                count = base + (1 if i < remainder else 0)
                axis_plan.extend([axis] * count)
            self.rng.shuffle(axis_plan)

        all_entries = []
        for idx in range(num_samples):
            correct_idx = int(correct_indices[idx])
            task_type = task_types[idx]
            preferred_axis = axis_plan[idx] if axis_plan else None
            entry = self.generate_sample(
                output_root,
                idx,
                task_type=task_type,
                correct_idx=correct_idx,
                preferred_axis=preferred_axis,
            )
            all_entries.append(entry)
        
        # 保存汇总元数据
        write_meta(output_root / "meta.json", all_entries)
        
        print(f"Generated {num_samples} PCRAR samples in {output_root}")
        print(f"  - Relational: {n_relational}")
        print(f"  - Analogical: {n_analogical}")
