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
from .pcrar_entity import PCRAREntity, sample_random_entity, DEFAULT_N_POINTS
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
    allowed_ops: Optional[List[OpType]] = None
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
            return self._generate_relational_sample(output_root, sample_index, correct_idx)
        else:
            return self._generate_analogical_sample(output_root, sample_index, correct_idx)
    
    def _generate_relational_sample(
        self,
        output_root: Path,
        sample_index: int,
        correct_idx: int,
    ) -> Dict[str, Any]:
        """生成 Relational（2→1）样本
        
        目标：让模型"认知属性/关系规律"
        - 采样 Attr(A)
        - 采样规则 T（含参数）
        - 得到 B = T(A)
        - 正确答案 D* = T(B)
        - 候选 {D1..Dk}，仅 D* 满足同一个 T
        """
        # 生成初始实体 A
        leaf_count = int(self.rng.integers(self.config.leaf_count_min, self.config.leaf_count_max + 1))
        entity_a = sample_random_entity(self.rng, leaf_count=leaf_count, allowed_ops=self.config.allowed_ops)
        
        # 采样规则和参数
        rule, params = self._sample_rule(entity_a)
        
        # 生成 B = T(A)
        entity_b = rule.apply(entity_a, params)
        
        # 生成正确答案 D* = T(B)
        # 需要确保规则可以继续应用
        if rule.can_apply(entity_b, params):
            entity_correct = rule.apply(entity_b, params)
        else:
            # 如果不能继续应用，使用相反方向
            reverse_params = RuleParams(
                template=params.template,
                axis=params.axis,
                leaf_idx=params.leaf_idx,
                direction=-params.direction,
            )
            if rule.can_apply(entity_b, reverse_params):
                entity_correct = rule.apply(entity_b, reverse_params)
                params = reverse_params
            else:
                # 回退：使用不同的规则
                rule, params = self._sample_rule(entity_b)
                entity_correct = rule.apply(entity_b, params)
        
        # 生成干扰项
        distractors = []
        distractor_reasons = []
        for _ in range(self.config.n_candidates - 1):
            distractor, reason = generate_distractor(entity_b, rule, params, self.rng)
            distractors.append(distractor)
            distractor_reasons.append(reason)
        
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
        # 生成初始实体 A
        leaf_count = int(self.rng.integers(self.config.leaf_count_min, self.config.leaf_count_max + 1))
        entity_a = sample_random_entity(self.rng, leaf_count=leaf_count, allowed_ops=self.config.allowed_ops)
        
        # 采样规则和参数
        rule, params = self._sample_rule(entity_a)
        
        # 生成 B = T(A)
        entity_b = rule.apply(entity_a, params)
        
        # 生成 C（与 A 结构相似但不同）
        entity_c = self._generate_compatible_entity(entity_a, rule, params)
        
        # 确保规则可以应用到 C
        if not rule.can_apply(entity_c, params):
            # 调整参数或重新生成 C
            entity_c = self._adjust_entity_for_rule(entity_c, rule, params)
        
        # 生成正确答案 D* = T(C)
        entity_correct = rule.apply(entity_c, params)
        
        # 生成干扰项
        distractors = []
        distractor_reasons = []
        for _ in range(self.config.n_candidates - 1):
            distractor, reason = generate_distractor(entity_c, rule, params, self.rng)
            distractors.append(distractor)
            distractor_reasons.append(reason)
        
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
    
    def _sample_rule(self, entity: PCRAREntity) -> Tuple[PCRARRule, RuleParams]:
        """采样一条可应用的规则"""
        if self.config.rule_filter:
            # 从过滤后的规则中选择
            templates = list(self.config.rule_filter)
            self.rng.shuffle(templates)
            for template in templates:
                rule = get_rule(template)
                params = rule.sample_params(self.rng, entity)
                if rule.can_apply(entity, params):
                    return rule, params
        
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
        
        if params.axis == "r" and params.leaf_idx is not None and params.leaf_idx < len(leaves):
            # 确保尺寸可以变化
            leaf = leaves[params.leaf_idx]
            if params.direction == 1:
                leaf.size_level = SIZE_LEVELS[0]  # 设为最小，可以增加
            else:
                leaf.size_level = SIZE_LEVELS[-1]  # 设为最大，可以减少
        elif params.axis == "p" and params.leaf_idx is not None and params.leaf_idx < len(leaves):
            # 确保位置可以变化
            leaf = leaves[params.leaf_idx]
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
        
        all_entries = []
        for idx in range(num_samples):
            correct_idx = int(correct_indices[idx])
            task_type = task_types[idx]
            entry = self.generate_sample(output_root, idx, task_type=task_type, correct_idx=correct_idx)
            all_entries.append(entry)
        
        # 保存汇总元数据
        write_meta(output_root / "meta.json", all_entries)
        
        print(f"Generated {num_samples} PCRAR samples in {output_root}")
        print(f"  - Relational: {n_relational}")
        print(f"  - Analogical: {n_analogical}")
