"""PCRAR dataset generator.

Default path is matrix-only 3x3 RAVEN-like generation.
Legacy relational/analogical generation is preserved behind an explicit switch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .candidate_generation import CandidateMixConfig, generate_candidates
from .csg import OpType, PRIM_TYPE_CYCLE, enforce_leaf_separation, has_containment_risk, has_excessive_overlap
from .io import ensure_dir, write_meta, write_ply
from .matrix_grid import (
    MatrixLevelConfig,
    build_matrix_level_config,
    can_apply_k,
    generate_grid,
    grid_quality_checks,
    normalize_entity_levels,
    prepare_entity_for_rule_path,
)
from .pcrar_entity import DEFAULT_N_POINTS, PCRAREntity, color_rgb, density_point_count, sample_random_entity
from .pcrar_rules import (
    CYCLE_AXES,
    CYCLE_AXIS_COLOR,
    CYCLE_AXIS_DENSITY,
    CYCLE_AXIS_SHAPE,
    CYCLE_AXIS_SIZE,
    CYCLE_DENSITY_INDICES,
    CYCLE_SHAPE_LEVELS,
    CYCLE_SIZE_LEVELS,
    PCRARRule,
    RuleParams,
    RuleTemplate,
    RULE_SOURCE_ALIGN,
    SYMMETRY_DENSITY_WEIGHT_STEP,
    get_rule,
)


GRID_COLOR_MAP = {
    (0, 0): (31, 119, 180),
    (0, 1): (255, 127, 14),
    (0, 2): (44, 160, 44),
    (1, 0): (214, 39, 40),
    (1, 1): (148, 103, 189),
    (1, 2): (140, 86, 75),
    (2, 0): (227, 119, 194),
    (2, 1): (127, 127, 127),
}

CAND_COLOR_MAP = {
    0: (188, 189, 34),
    1: (23, 190, 207),
    2: (174, 199, 232),
    3: (255, 187, 120),
    4: (152, 223, 138),
    5: (255, 152, 150),
}

ROW_PERTURB_TYPES = ("density", "jitter", "quantize", "outlier")


@dataclass
class PCRARConfig:
    # Shared
    n_points: int = DEFAULT_N_POINTS
    allowed_ops: Optional[List[OpType]] = field(default_factory=lambda: [OpType.UNION])
    leaf_count_min: int = 2
    leaf_count_max: int = 3

    # Matrix (default)
    num_options: int = 4
    matrix_k_h_choices: Tuple[int, ...] = (1, 2)
    matrix_k_v_choices: Tuple[int, ...] = (1, 2)
    matrix_size_levels: int = 3
    matrix_density_levels: int = 3
    matrix_delta_levels: int = 5
    matrix_slot_levels: Tuple[int, ...] = (-1, 0, 1)
    matrix_missing_one_per_row: bool = True
    generate_confusing_view: bool = True
    view_image_size: Tuple[int, int] = (512, 512)
    rule_filter: Optional[Set[RuleTemplate]] = None
    enable_row_perturbation: bool = True
    apply_density_perturb_on_density_rules: bool = True
    row2_density_keep_ratio: float = 0.75
    row3_density_keep_ratio: float = 0.5
    row2_jitter_std_ratio: float = 0.003
    row3_jitter_std_ratio: float = 0.009
    row2_quantize_step_ratio: float = 0.005
    row3_quantize_step_ratio: float = 0.015
    row2_outlier_replace_ratio: float = 0.01
    row3_outlier_replace_ratio: float = 0.04

    # Legacy path compatibility
    legacy_enabled: bool = False
    legacy_task_mix: float = 0.5
    legacy_n_candidates: int = 4


class PCRARDatasetGenerator:
    """PCRAR dataset generator (matrix-first)."""

    def __init__(
        self,
        config: Optional[PCRARConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or PCRARConfig()
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            np.random.seed(seed)

        self.level_cfg = build_matrix_level_config(
            matrix_size_levels=self.config.matrix_size_levels,
            matrix_density_levels=self.config.matrix_density_levels,
            matrix_delta_levels=self.config.matrix_delta_levels,
            matrix_slot_levels=self.config.matrix_slot_levels,
        )

        self._legacy_generator = None
        if self.config.legacy_enabled:
            from .pcrar_legacy_dataset import PCRARConfig as LegacyConfig
            from .pcrar_legacy_dataset import PCRARDatasetGenerator as LegacyGenerator

            self._legacy_generator = LegacyGenerator(
                config=LegacyConfig(
                    n_points=self.config.n_points,
                    n_candidates=self.config.legacy_n_candidates,
                    task_mix=self.config.legacy_task_mix,
                    leaf_count_min=self.config.leaf_count_min,
                    leaf_count_max=self.config.leaf_count_max,
                    allowed_ops=self.config.allowed_ops,
                    rule_filter=self.config.rule_filter,
                ),
                seed=seed,
            )

    def _sample_matrix_rule(
        self,
        entity: PCRAREntity,
        preferred_axis: Optional[str] = None,
    ) -> Tuple[PCRARRule, RuleParams]:
        templates = self._effective_rule_templates()
        self.rng.shuffle(templates)
        for template in templates:
            rule = get_rule(template)
            if template == RuleTemplate.CYCLE:
                self._enforce_cycle_preconditions(entity, preferred_axis=preferred_axis)
            elif template == RuleTemplate.SYMMETRY:
                self._enforce_symmetry_preconditions(entity, preferred_axis=preferred_axis)
            for _ in range(40):
                params = rule.sample_params(self.rng, entity)
                if preferred_axis:
                    if template == RuleTemplate.CYCLE:
                        expected_cycle_axis = self._normalize_cycle_axis_name(preferred_axis)
                        if expected_cycle_axis is not None and params.axis != expected_cycle_axis:
                            continue
                    elif template in {RuleTemplate.PROGRESSION, RuleTemplate.SYMMETRY} and params.axis != preferred_axis:
                        continue
                if rule.can_apply(entity, params):
                    return rule, params
        raise RuntimeError("No applicable matrix rule found for current entity")

    def _effective_rule_templates(self) -> List[RuleTemplate]:
        if self.config.rule_filter:
            normalized = {self._normalize_template_name(t) for t in self.config.rule_filter}
            return list(normalized)
        # 合并 Cycle + Copy 后，矩阵主路径默认不再单独采样 Copy。
        return [t for t in RuleTemplate if t != RuleTemplate.COPY]

    @staticmethod
    def _normalize_template_name(template: RuleTemplate) -> RuleTemplate:
        return RuleTemplate.CYCLE if template == RuleTemplate.COPY else template

    @staticmethod
    def _normalize_cycle_axis_name(axis: Optional[str]) -> Optional[str]:
        if axis is None:
            return None
        mapping = {
            "shape": CYCLE_AXIS_SHAPE,
            "density": CYCLE_AXIS_DENSITY,
            "size": CYCLE_AXIS_SIZE,
            "color": CYCLE_AXIS_COLOR,
            "copy_shape_cycle": CYCLE_AXIS_SHAPE,
            "copy_size_cycle": CYCLE_AXIS_SIZE,
            "copy_density_cycle": CYCLE_AXIS_DENSITY,
        }
        if axis in mapping:
            return mapping[axis]
        if axis in CYCLE_AXES:
            return axis
        return None

    def _enforce_cycle_preconditions(self, entity: PCRAREntity, preferred_axis: Optional[str] = None) -> None:
        leaves = entity.get_leaves()
        if len(leaves) not in (2, 3):
            return
        axis = self._normalize_cycle_axis_name(preferred_axis)
        if axis is None:
            axis = str(self.rng.choice(np.array(CYCLE_AXES, dtype=object)))

        if axis == CYCLE_AXIS_SHAPE:
            shape_idx = int(self.rng.integers(len(CYCLE_SHAPE_LEVELS)))
            shape = CYCLE_SHAPE_LEVELS[shape_idx]
            for leaf in leaves:
                leaf.prim_type = shape
        elif axis == CYCLE_AXIS_SIZE:
            size_idx = int(self.rng.integers(len(CYCLE_SIZE_LEVELS)))
            size = CYCLE_SIZE_LEVELS[size_idx]
            for leaf in leaves:
                leaf.size_level = size
        elif axis == CYCLE_AXIS_DENSITY:
            density_idx = int(self.rng.choice(np.array(CYCLE_DENSITY_INDICES, dtype=int)))
            entity.obs.density_preset_idx = density_idx
            entity.obs.n_points = density_point_count(density_idx)
        elif axis == CYCLE_AXIS_COLOR:
            entity.obs.color_preset_idx = int(self.rng.integers(3))

        enforce_leaf_separation(leaves)

    def _enforce_copy_preconditions(self, entity: PCRAREntity, preferred_axis: Optional[str] = None) -> None:
        """兼容接口：Copy 轴映射为合并后的 Cycle 轴。"""
        self._enforce_cycle_preconditions(entity, preferred_axis=preferred_axis)

    def _enforce_symmetry_preconditions(self, entity: PCRAREntity, preferred_axis: Optional[str] = None) -> None:
        leaves = entity.get_leaves()
        if len(leaves) != 2:
            return
        if preferred_axis == "R":
            # Symmetry pose axis rejects spheres; force non-spherical primitives.
            non_sphere = [PRIM_TYPE_CYCLE[1], PRIM_TYPE_CYCLE[2]]
            for i, leaf in enumerate(leaves):
                leaf.prim_type = non_sphere[i % len(non_sphere)]
        elif preferred_axis == "r":
            # Keep both leaves away from size boundaries to preserve +/- room.
            mid = len(self.level_cfg.size_levels) // 2
            center = self.level_cfg.size_levels[mid]
            for leaf in leaves:
                leaf.size_level = center

    @staticmethod
    def _min_unique_threshold(rule: PCRARRule) -> int:
        if rule.template in {RuleTemplate.CYCLE, RuleTemplate.COPY, RuleTemplate.PERMUTATION}:
            return 3
        return 5

    @staticmethod
    def _entity_visibility_ok(entity: PCRAREntity) -> bool:
        leaves = entity.get_leaves()
        return not has_containment_risk(leaves) and not has_excessive_overlap(leaves)

    @classmethod
    def _grid_visibility_ok(cls, grid: List[List[PCRAREntity]]) -> bool:
        for row in grid:
            for entity in row:
                if not cls._entity_visibility_ok(entity):
                    return False
        return True

    @staticmethod
    def _cycle_cell_value(entity: PCRAREntity, axis: str) -> Optional[Any]:
        leaves = entity.get_leaves()
        if axis == CYCLE_AXIS_DENSITY:
            return int(entity.obs.density_preset_idx)
        if axis == CYCLE_AXIS_SIZE:
            if not leaves:
                return None
            return str(leaves[0].size_level.value)
        if axis == CYCLE_AXIS_SHAPE:
            if not leaves:
                return None
            return str(leaves[0].prim_type.value)
        if axis == CYCLE_AXIS_COLOR:
            return int(entity.obs.color_preset_idx)
        return None

    @classmethod
    def _cycle_distribute_three_ok(cls, grid: List[List[PCRAREntity]], params: RuleParams) -> bool:
        axis = params.axis
        if axis not in CYCLE_AXES:
            return True
        values: List[List[Any]] = []
        for r in range(3):
            row_values: List[Any] = []
            for c in range(3):
                v = cls._cycle_cell_value(grid[r][c], axis)
                if v is None:
                    return False
                row_values.append(v)
            values.append(row_values)
        for r in range(3):
            if len(set(values[r])) != 3:
                return False
        for c in range(3):
            if len({values[r][c] for r in range(3)}) != 3:
                return False
        return True

    def _sample_missing_positions(self) -> List[Tuple[int, int]]:
        target_pos = (2, 2)
        if not self.config.matrix_missing_one_per_row:
            return [target_pos]
        # Keep (2,1) and (1,2) visible because candidate checks depend on them.
        row0_col = int(self.rng.integers(0, 3))
        row1_col = int(self.rng.integers(0, 2))
        return [(0, row0_col), (1, row1_col), target_pos]

    @staticmethod
    def _normalize_missing_positions(missing_positions: Sequence[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        missing_set: Set[Tuple[int, int]] = set()
        for r, c in missing_positions:
            rr = int(r)
            cc = int(c)
            if rr < 0 or rr > 2 or cc < 0 or cc > 2:
                raise RuntimeError(f"Invalid missing position ({rr}, {cc})")
            missing_set.add((rr, cc))
        return missing_set

    @staticmethod
    def _validate_missing_positions(missing_positions: Set[Tuple[int, int]]) -> None:
        if (2, 2) not in missing_positions:
            raise RuntimeError("target position (2,2) must be missing")
        blocked = {(2, 1), (1, 2)}.intersection(missing_positions)
        if blocked:
            raise RuntimeError(
                f"Missing positions {sorted(blocked)} break target anchors and candidate consistency checks"
            )

    @staticmethod
    def _build_observation_mask(missing_positions: Set[Tuple[int, int]]) -> List[List[bool]]:
        return [[(r, c) not in missing_positions for c in range(3)] for r in range(3)]

    @staticmethod
    def _safe_bbox_diag(points: np.ndarray) -> float:
        if len(points) == 0:
            return 1.0
        span = np.max(points, axis=0) - np.min(points, axis=0)
        diag = float(np.linalg.norm(span))
        return max(diag, 1e-6)

    @staticmethod
    def _rule_uses_density(params: Optional[RuleParams]) -> bool:
        if params is None:
            return False
        if params.template == RuleTemplate.PROGRESSION and params.axis == "d":
            return True
        if params.template == RuleTemplate.SYMMETRY and params.axis == "d":
            return True
        if params.template == RuleTemplate.CYCLE and params.axis == CYCLE_AXIS_DENSITY:
            return True
        if params.template == RuleTemplate.COPY and params.axis == "copy_density_cycle":
            return True
        return False

    def _allow_density_perturbation(
        self,
        params_h: RuleParams,
        params_v: Optional[RuleParams],
    ) -> bool:
        if self.config.apply_density_perturb_on_density_rules:
            return True
        return not (self._rule_uses_density(params_h) or self._rule_uses_density(params_v))

    def _sample_single_perturbation_type(
        self,
        params_h: RuleParams,
        params_v: Optional[RuleParams],
        preferred_type: Optional[str] = None,
        strict: bool = False,
    ) -> str:
        choices = list(ROW_PERTURB_TYPES)
        if not self._allow_density_perturbation(params_h, params_v):
            choices = [x for x in choices if x != "density"]
        if not choices:
            return "jitter"
        if preferred_type:
            target = str(preferred_type).strip().lower()
            if target in choices:
                return target
            if strict:
                raise RuntimeError(f"preferred_perturbation_type_unavailable:{target}")
        return str(self.rng.choice(choices))

    def _apply_point_cloud_perturbation(
        self,
        points: np.ndarray,
        row_idx: int,
        params_h: RuleParams,
        params_v: Optional[RuleParams],
        perturbation_type: str,
    ) -> np.ndarray:
        if not self.config.enable_row_perturbation:
            return points
        if row_idx <= 0:
            return points

        if row_idx == 1:
            keep_ratio = float(self.config.row2_density_keep_ratio)
            jitter_ratio = float(self.config.row2_jitter_std_ratio)
            quantize_ratio = float(self.config.row2_quantize_step_ratio)
            outlier_ratio = float(self.config.row2_outlier_replace_ratio)
        else:
            keep_ratio = float(self.config.row3_density_keep_ratio)
            jitter_ratio = float(self.config.row3_jitter_std_ratio)
            quantize_ratio = float(self.config.row3_quantize_step_ratio)
            outlier_ratio = float(self.config.row3_outlier_replace_ratio)

        out = points.copy()
        n_src = len(out)
        if n_src == 0:
            return out

        bbox_min = np.min(out, axis=0)
        bbox_max = np.max(out, axis=0)
        span = bbox_max - bbox_min
        diag = self._safe_bbox_diag(out)

        if perturbation_type == "density":
            keep_ratio = float(np.clip(keep_ratio, 0.05, 1.0))
            if keep_ratio < 0.999 and self._allow_density_perturbation(params_h, params_v):
                keep_n = max(32, int(round(len(out) * keep_ratio)))
                if keep_n < len(out):
                    keep_idx = self.rng.choice(len(out), size=keep_n, replace=False)
                    out = out[keep_idx]
        elif perturbation_type == "jitter":
            jitter_std = max(0.0, jitter_ratio) * diag
            if jitter_std > 0.0:
                out = out + self.rng.normal(loc=0.0, scale=jitter_std, size=out.shape)
        elif perturbation_type == "quantize":
            quant_step = max(0.0, quantize_ratio) * diag
            if quant_step > 0.0:
                out = np.round(out / quant_step) * quant_step
        elif perturbation_type == "outlier":
            outlier_ratio = float(np.clip(outlier_ratio, 0.0, 0.9))
            outlier_n = min(len(out), int(round(len(out) * outlier_ratio)))
            if outlier_n > 0:
                outlier_idx = self.rng.choice(len(out), size=outlier_n, replace=False)
                expand = 0.15 * span + 0.05 * diag
                low = bbox_min - expand
                high = bbox_max + expand
                high = np.maximum(high, low + 1e-6)
                out[outlier_idx] = self.rng.uniform(low=low, high=high, size=(outlier_n, 3))

        # Keep row perturbation count stable unless density downsample is explicitly enabled.
        if len(out) > n_src:
            idx = self.rng.choice(len(out), size=n_src, replace=False)
            out = out[idx]
        return out

    @staticmethod
    def _build_symmetry_vertical_params(params_h: RuleParams) -> RuleParams:
        if params_h.axis == "d":
            return RuleParams(
                template=RuleTemplate.SYMMETRY,
                axis="d",
                direction=-int(params_h.direction),
            )
        return RuleParams(
            template=RuleTemplate.SYMMETRY,
            axis=params_h.axis,
            leaf_idx=int(params_h.direction),
            direction=int(params_h.leaf_idx) if params_h.leaf_idx is not None else int(params_h.direction),
        )

    def _prepare_entity_for_symmetry_dual(
        self,
        entity: PCRAREntity,
        params_h: RuleParams,
    ) -> PCRAREntity:
        out = entity.copy()
        leaves = out.get_leaves()
        if len(leaves) != 2:
            return out
        if params_h.axis == "r":
            mid = len(self.level_cfg.size_levels) // 2
            center = self.level_cfg.size_levels[mid]
            for leaf in leaves:
                leaf.size_level = center
        elif params_h.axis == "d":
            out.obs.part_sampling_weights = [0.5, 0.5]
        return out

    def _sample_matrix_problem(
        self,
        k_h: int,
        k_v: int,
        preferred_axis: Optional[str] = None,
        max_attempts: int = 200,
    ) -> Tuple[List[List[PCRAREntity]], int, PCRARRule, RuleParams, PCRARRule, RuleParams]:
        k_max = 2 * k_h + 2 * k_v
        normalized_filter = (
            {self._normalize_template_name(t) for t in self.config.rule_filter}
            if self.config.rule_filter
            else None
        )
        if normalized_filter == {RuleTemplate.SYMMETRY} and preferred_axis == "R":
            # Current matrix constraints only admit size-axis symmetry ("r").
            # Pose-axis symmetry ("R") is not reachable and should fail fast
            # instead of burning all retry attempts.
            raise RuntimeError("symmetry_pose_axis_unavailable_in_matrix_mode")
        for _ in range(max_attempts):
            if normalized_filter == {RuleTemplate.CYCLE}:
                leaf_count = int(self.rng.integers(2, 4))
            elif normalized_filter == {RuleTemplate.PERMUTATION}:
                leaf_count = 3
            elif normalized_filter == {RuleTemplate.CONSERVATION}:
                leaf_count = 3
            elif normalized_filter == {RuleTemplate.SYMMETRY}:
                leaf_count = 2
            else:
                leaf_count = int(self.rng.integers(self.config.leaf_count_min, self.config.leaf_count_max + 1))
            e00 = sample_random_entity(
                self.rng,
                leaf_count=leaf_count,
                allowed_ops=self.config.allowed_ops,
            )
            e00 = normalize_entity_levels(e00, self.level_cfg, self.rng)
            # 归一化后再次拉开部件，避免出现完全包裹/严重重叠。
            enforce_leaf_separation(e00.get_leaves())
            try:
                rule, params = self._sample_matrix_rule(e00, preferred_axis=preferred_axis)
            except RuntimeError:
                continue

            # Ensure shape-cycle questions start from a homogeneous primitive type.
            # Preconditions sampled before params may have used a non-shape axis.
            if rule.template == RuleTemplate.CYCLE and params.axis == CYCLE_AXIS_SHAPE:
                self._enforce_cycle_preconditions(e00, preferred_axis=CYCLE_AXIS_SHAPE)

            rule_v = rule
            params_v = params
            dual_symmetry = rule.template == RuleTemplate.SYMMETRY
            if dual_symmetry:
                params_v = self._build_symmetry_vertical_params(params)
                e00 = self._prepare_entity_for_symmetry_dual(e00, params)
            else:
                e00 = prepare_entity_for_rule_path(e00, rule, params, self.level_cfg)
            skip_visibility_checks = rule.template in {RuleTemplate.CONSERVATION, RuleTemplate.SYMMETRY}
            if (not skip_visibility_checks) and (not self._entity_visibility_ok(e00)):
                continue
            if dual_symmetry:
                if not can_apply_k(e00, rule, params, 2 * k_h):
                    continue
                if not can_apply_k(e00, rule_v, params_v, 2 * k_v):
                    continue
            else:
                if not can_apply_k(e00, rule, params, k_max):
                    continue

            try:
                grid, _, _ = generate_grid(
                    e00,
                    rule,
                    params,
                    k_h=k_h,
                    k_v=k_v,
                    vertical_rule=rule_v if dual_symmetry else None,
                    vertical_params=params_v if dual_symmetry else None,
                )
            except RuntimeError:
                continue

            if (not skip_visibility_checks) and (not self._grid_visibility_ok(grid)):
                continue
            if rule.template == RuleTemplate.CYCLE and not self._cycle_distribute_three_ok(grid, params):
                continue

            ok, reason = grid_quality_checks(
                grid,
                rule,
                params,
                k_h=k_h,
                k_v=k_v,
                min_unique=self._min_unique_threshold(rule),
                vertical_rule=rule_v if dual_symmetry else None,
                vertical_params=params_v if dual_symmetry else None,
            )
            if not ok:
                if reason in {"adjacent_cell_collision", "low_global_diversity", "path_consistency_failed"}:
                    continue
            return grid, k_max, rule, params, rule_v, params_v

        raise RuntimeError("Failed to sample a valid 3x3 matrix problem")

    @staticmethod
    def _build_grid_context(
        grid: List[List[PCRAREntity]],
        missing_positions: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> List[List[Optional[PCRAREntity]]]:
        missing_set = {(2, 2)} if not missing_positions else {
            (int(r), int(c)) for r, c in missing_positions
        }
        context: List[List[Optional[PCRAREntity]]] = []
        for r in range(3):
            row: List[Optional[PCRAREntity]] = []
            for c in range(3):
                if (r, c) in missing_set:
                    row.append(None)
                else:
                    row.append(grid[r][c].copy())
            context.append(row)
        return context

    @staticmethod
    def _conservation_leaf_steps(params: RuleParams) -> List[Tuple[int, int]]:
        if params.leaf_indices is not None and params.directions is not None:
            leaf_steps: List[Tuple[int, int]] = []
            for leaf_idx, step in zip(params.leaf_indices, params.directions):
                leaf_steps.append((int(leaf_idx), int(step)))
            return leaf_steps
        if params.leaf_idx is not None and params.direction is not None:
            return [(int(params.leaf_idx), 1), (int(params.direction), -1)]
        return []

    @staticmethod
    def _summarize_rule_semantics(params: RuleParams) -> Dict[str, Any]:
        template = params.template
        axis = params.axis
        summary = {
            "template": template.value,
            "axis": axis,
            "changed_attribute": "unknown",
            "description": "fixed rule instance",
            "direction": int(params.direction),
        }
        if template == RuleTemplate.PROGRESSION:
            attr_map = {"r": "size_level", "R": "global_pose", "p": "slot_position", "d": "density_level"}
            summary["changed_attribute"] = attr_map.get(axis, "progression_axis")
            if axis == "R":
                summary["description"] = f"Progression on global pose (rot_axis={params.rot_axis}, step={params.direction})"
            else:
                summary["description"] = f"Progression on {summary['changed_attribute']} (step={params.direction})"
        elif template == RuleTemplate.CYCLE:
            cycle_attr_map = {
                CYCLE_AXIS_DENSITY: "density_level",
                CYCLE_AXIS_SIZE: "size_level",
                CYCLE_AXIS_SHAPE: "primitive_type",
                CYCLE_AXIS_COLOR: "color",
            }
            summary["changed_attribute"] = cycle_attr_map.get(axis, "cycle_distribute3_axis")
            summary["description"] = (
                f"Cycle distribute-three on {summary['changed_attribute']} (direction={int(params.direction)})"
            )
        elif template == RuleTemplate.COPY:
            summary["changed_attribute"] = axis or "copy_pattern"
            summary["description"] = f"Copy pattern ({axis}) with direction={params.direction}"
        elif template == RuleTemplate.COUNT:
            summary["changed_attribute"] = "leaf_count"
            summary["description"] = f"Count transition with direction={params.direction}"
        elif template == RuleTemplate.CONSERVATION:
            steps = PCRARDatasetGenerator._conservation_leaf_steps(params)
            step_desc = ", ".join([f"leaf{leaf_idx} {step:+d}" for leaf_idx, step in steps])
            summary["changed_attribute"] = "multi_leaf_size_levels"
            summary["description"] = f"Conservation: {step_desc} per T" if step_desc else "Conservation on size levels"
        elif template == RuleTemplate.PERMUTATION:
            summary["changed_attribute"] = "slot_permutation"
            summary["description"] = f"Permutation over slots with direction={params.direction}"
        elif template == RuleTemplate.SYMMETRY:
            sym_attr_map = {"R": "local_pose", "r": "size_level", "d": "density_weights"}
            summary["changed_attribute"] = sym_attr_map.get(axis, "symmetry_axis")
            summary["description"] = f"Symmetry on {summary['changed_attribute']}"
        return summary

    @staticmethod
    def _build_attribute_change_detail(params: RuleParams, k_h: int, k_v: int) -> Dict[str, Any]:
        template = params.template
        axis = params.axis
        detail: Dict[str, Any] = {
            "template": template.value,
            "axis": axis,
            "per_T_change": {},
            "horizontal_effective_change": {},
            "vertical_effective_change": {},
        }
        if template == RuleTemplate.PROGRESSION:
            if axis == "r":
                detail["per_T_change"] = {"attribute": "size_level", "step": int(params.direction)}
            elif axis == "R":
                detail["per_T_change"] = {
                    "attribute": f"global_pose_deg.{(params.rot_axis or 'x').lower()}",
                    "step_degrees": int(params.direction * 60),
                }
            elif axis == "p":
                detail["per_T_change"] = {"attribute": "slot", "step": int(params.direction)}
            elif axis == "d":
                detail["per_T_change"] = {"attribute": "density_preset_idx", "step": int(params.direction)}
        elif template == RuleTemplate.CYCLE:
            attr_map = {
                CYCLE_AXIS_DENSITY: "density_preset_idx",
                CYCLE_AXIS_SIZE: "size_level(all_leaves)",
                CYCLE_AXIS_SHAPE: "primitive_type(all_leaves)",
                CYCLE_AXIS_COLOR: "color_preset_idx",
            }
            detail["per_T_change"] = {
                "attribute": attr_map.get(axis, axis),
                "direction": int(params.direction),
                "distribution": "distribute-three",
            }
        elif template == RuleTemplate.COPY:
            detail["per_T_change"] = {"attribute": params.axis, "direction": int(params.direction)}
        elif template == RuleTemplate.COUNT:
            detail["per_T_change"] = {
                "attribute": "leaf_count",
                "direction": int(params.direction),
                "cycle": "1->2->3->1 (direction=+1) or reverse (direction=-1)",
            }
        elif template == RuleTemplate.CONSERVATION:
            steps = PCRARDatasetGenerator._conservation_leaf_steps(params)
            detail["per_T_change"] = {
                "attribute": "multi_leaf_size_levels",
                "leaf_steps": [{"leaf": int(leaf_idx), "step": int(step)} for leaf_idx, step in steps],
            }
        elif template == RuleTemplate.PERMUTATION:
            detail["per_T_change"] = {"attribute": "slot_permutation", "direction": int(params.direction)}
        elif template == RuleTemplate.SYMMETRY:
            if axis == "d":
                detail["per_T_change"] = {
                    "attribute": "part_sampling_weights(left/right)",
                    "step": float(SYMMETRY_DENSITY_WEIGHT_STEP),
                    "direction": int(params.direction),
                }
            else:
                detail["per_T_change"] = {"attribute": axis, "direction_or_pair": int(params.direction)}

        detail["horizontal_effective_change"] = {"applied_power": int(k_h)}
        detail["vertical_effective_change"] = {"applied_power": int(k_v)}
        return detail

    @classmethod
    def _build_matrix_relation_spec(
        cls,
        params: RuleParams,
        k_h: int,
        k_v: int,
        params_v: Optional[RuleParams] = None,
    ) -> Dict[str, Any]:
        if params_v is not None and params_v.to_dict() != params.to_dict():
            semantics_h = cls._summarize_rule_semantics(params)
            semantics_v = cls._summarize_rule_semantics(params_v)
            return {
                "formula": "E[r,c] = T_v^(r*k_v)(T_h^(c*k_h)(E[0,0]))",
                "dual_rule": True,
                "horizontal_rule_instance": semantics_h,
                "vertical_rule_instance": semantics_v,
                "horizontal_relation": {
                    "step_power": int(k_h),
                    "formula": f"E[r,c+1] = T_h^{int(k_h)}(E[r,c])",
                    "changed_attribute": semantics_h["changed_attribute"],
                    "description": semantics_h["description"],
                },
                "vertical_relation": {
                    "step_power": int(k_v),
                    "formula": f"E[r+1,c] = T_v^{int(k_v)}(E[r,c])",
                    "changed_attribute": semantics_v["changed_attribute"],
                    "description": semantics_v["description"],
                },
            }
        semantics = cls._summarize_rule_semantics(params)
        return {
            "formula": "E[r,c] = T^(r*k_v + c*k_h)(E[0,0])",
            "rule_instance": semantics,
            "attribute_change_detail": cls._build_attribute_change_detail(params, k_h=k_h, k_v=k_v),
            "horizontal_relation": {
                "step_power": int(k_h),
                "formula": f"E[r,c+1] = T^{int(k_h)}(E[r,c])",
                "changed_attribute": semantics["changed_attribute"],
                "description": semantics["description"],
            },
            "vertical_relation": {
                "step_power": int(k_v),
                "formula": f"E[r+1,c] = T^{int(k_v)}(E[r,c])",
                "changed_attribute": semantics["changed_attribute"],
                "description": semantics["description"],
            },
        }

    @staticmethod
    def _build_matrix_focus(
        params: RuleParams,
        k_h: int,
        k_v: int,
        params_v: Optional[RuleParams] = None,
    ) -> str:
        template = params.template
        axis = params.axis or "none"

        def _shift_word(step: int) -> str:
            if step > 0:
                return f"+{step}"
            return str(step)

        if params_v is not None and params_v.to_dict() != params.to_dict() and template == RuleTemplate.SYMMETRY:
            axis_v = params_v.axis or "none"
            return (
                "3x3 matrix completion using dual symmetry rules: horizontal uses T_h and vertical uses inverse-like T_v. "
                f"Horizontal axis={axis}, vertical axis={axis_v}. "
                f"Horizontal relation E[r,c+1]=T_h^{int(k_h)}(E[r,c]); "
                f"vertical relation E[r+1,c]=T_v^{int(k_v)}(E[r,c])."
            )
        if template == RuleTemplate.PROGRESSION:
            if axis == "r":
                per_t = f"all leaves change size_level by {_shift_word(int(params.direction))} per T"
                stride = (
                    f"horizontal net size_level {_shift_word(int(params.direction) * int(k_h))}, "
                    f"vertical net size_level {_shift_word(int(params.direction) * int(k_v))}"
                )
            elif axis == "R":
                rot_axis = (params.rot_axis or "x").lower()
                step_deg = int(params.direction) * 60
                per_t = f"global_pose_deg.{rot_axis} rotates {_shift_word(step_deg)} degrees per T"
                stride = (
                    f"horizontal net rotation {_shift_word(step_deg * int(k_h))} degrees on {rot_axis}, "
                    f"vertical net rotation {_shift_word(step_deg * int(k_v))} degrees on {rot_axis}"
                )
            elif axis == "p":
                per_t = f"all leaves shift slot by {_shift_word(int(params.direction))} per T"
                stride = (
                    f"horizontal net slot shift {_shift_word(int(params.direction) * int(k_h))}, "
                    f"vertical net slot shift {_shift_word(int(params.direction) * int(k_v))}"
                )
            elif axis == "d":
                per_t = f"density_preset_idx changes by {_shift_word(int(params.direction))} per T"
                stride = (
                    f"horizontal net density preset shift {_shift_word(int(params.direction) * int(k_h))}, "
                    f"vertical net density preset shift {_shift_word(int(params.direction) * int(k_v))}"
                )
            else:
                per_t = "one progression attribute changes per T"
                stride = "horizontal and vertical both apply repeated progression steps"
        elif template == RuleTemplate.CYCLE:
            direction = int(params.direction)
            cycle_dir = "forward" if direction >= 0 else "reverse"
            if axis == CYCLE_AXIS_DENSITY:
                triplet = " -> ".join(str(int(x)) for x in CYCLE_DENSITY_INDICES)
                per_t = f"density level cycles in distribute-three order ({triplet}), {cycle_dir}"
            elif axis == CYCLE_AXIS_SIZE:
                triplet = " -> ".join(x.value for x in CYCLE_SIZE_LEVELS)
                per_t = f"all leaves share one size level and cycle in distribute-three order ({triplet}), {cycle_dir}"
            elif axis == CYCLE_AXIS_SHAPE:
                triplet = " -> ".join(x.value for x in CYCLE_SHAPE_LEVELS)
                per_t = f"all leaves share one primitive type and cycle in distribute-three order ({triplet}), {cycle_dir}"
            elif axis == CYCLE_AXIS_COLOR:
                per_t = "entity color cycles in distribute-three order (red -> green -> blue), " + cycle_dir
            else:
                per_t = f"cycle distribute-three on axis {axis}, {cycle_dir}"
            stride = (
                f"horizontal applies T^{int(k_h)} cycle steps; "
                f"vertical applies T^{int(k_v)} cycle steps"
            )
        elif template == RuleTemplate.COPY:
            direction = int(params.direction)
            copy_dir = "forward (copy from right neighbor in slot order)" if direction > 0 else "reverse (copy from left neighbor in slot order)"
            if axis == "copy_size_cycle":
                per_t = f"size_level is copied cyclically in slot order, {copy_dir}"
            elif axis == "copy_density_cycle":
                per_t = f"part_sampling_weights are copied cyclically in slot order with all leaves kept at the same size, {copy_dir}"
            elif axis == "copy_shape_cycle":
                per_t = f"prim_type is copied cyclically in slot order, {copy_dir}"
            else:
                per_t = f"copy pattern {axis} is applied in slot order, {copy_dir}"
            stride = (
                f"horizontal applies T^{int(k_h)} copy iterations; "
                f"vertical applies T^{int(k_v)} copy iterations"
            )
        elif template == RuleTemplate.COUNT:
            direction = int(params.direction)
            cycle_desc = "1->2->3->1" if direction > 0 else "1->3->2->1"
            per_t = f"leaf_count follows cycle {cycle_desc}"
            stride = (
                f"horizontal applies T^{int(k_h)} count transitions; "
                f"vertical applies T^{int(k_v)} count transitions"
            )
        elif template == RuleTemplate.CONSERVATION:
            steps = PCRARDatasetGenerator._conservation_leaf_steps(params)
            step_desc = ", ".join([f"leaf{leaf_idx}:{_shift_word(int(step))}" for leaf_idx, step in steps])
            per_t = f"size conservation triplet: {step_desc}" if step_desc else "size conservation over selected leaves"
            stride = (
                f"horizontal applies T^{int(k_h)} conservation transitions; "
                f"vertical applies T^{int(k_v)} conservation transitions"
            )
        elif template == RuleTemplate.PERMUTATION:
            direction = int(params.direction)
            perm_dir = "right shift" if direction > 0 else "left shift"
            per_t = f"slot assignment is permuted by one-step {perm_dir} per T"
            stride = (
                f"horizontal applies T^{int(k_h)} permutations; "
                f"vertical applies T^{int(k_v)} permutations"
            )
        elif template == RuleTemplate.SYMMETRY:
            if axis == "d":
                direction = int(params.direction)
                density_step = float(SYMMETRY_DENSITY_WEIGHT_STEP)
                if direction >= 0:
                    per_t = f"symmetry pair density weights: left +{density_step:.1f}, right -{density_step:.1f} per T"
                else:
                    per_t = f"symmetry pair density weights: left -{density_step:.1f}, right +{density_step:.1f} per T"
                stride = (
                    f"horizontal applies T^{int(k_h)} symmetry transitions; "
                    f"vertical applies T^{int(k_v)} symmetry transitions"
                )
            else:
                left_idx = int(params.leaf_idx) if params.leaf_idx is not None else None
                right_idx = int(params.direction) if params.direction is not None else None
                if axis == "R":
                    per_t = f"symmetry pair: leaf{left_idx} local_pose_deg.x +90, leaf{right_idx} local_pose_deg.x -90"
                elif axis == "r":
                    per_t = f"symmetry pair: leaf{left_idx} size_level +1, leaf{right_idx} size_level -1"
                else:
                    per_t = f"symmetry transform on axis {axis}"
                stride = (
                    f"horizontal applies T^{int(k_h)} symmetry transitions; "
                    f"vertical applies T^{int(k_v)} symmetry transitions"
                )
        else:
            per_t = f"template {template.value} applies one fixed transform per T"
            stride = (
                f"horizontal applies T^{int(k_h)} transitions; "
                f"vertical applies T^{int(k_v)} transitions"
            )

        return (
            f"3x3 matrix completion using one fixed rule instance T (template={template.value}, axis={axis}). "
            f"Per-T attribute change: {per_t}. "
            f"Horizontal relation E[r,c+1]=T^{int(k_h)}(E[r,c]); vertical relation E[r+1,c]=T^{int(k_v)}(E[r,c]). "
            f"Effective stride detail: {stride}."
        )

    def _save_matrix_sample(
        self,
        output_root: Path,
        sample_index: int,
        grid: List[List[PCRAREntity]],
        missing_positions: Sequence[Tuple[int, int]],
        k_h: int,
        k_v: int,
        k_max: int,
        rule: PCRARRule,
        params: RuleParams,
        rule_v: Optional[PCRARRule],
        params_v: Optional[RuleParams],
        candidates: List[PCRAREntity],
        gt_index: int,
        distractor_types: List[str],
        candidate_notes: List[str],
        alt_relations: List[Dict[str, Any]],
        preferred_perturbation_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        sample_id = f"sample_{sample_index:06d}"
        sample_dir = output_root / sample_id
        ensure_dir(sample_dir)
        missing_set = self._normalize_missing_positions(missing_positions)
        self._validate_missing_positions(missing_set)
        missing_positions_sorted = sorted(missing_set)
        selected_perturbation_type = self._sample_single_perturbation_type(
            params,
            params_v,
            preferred_type=preferred_perturbation_type,
            strict=preferred_perturbation_type is not None,
        )

        grid_paths: List[List[Optional[str]]] = [[None for _ in range(3)] for _ in range(3)]
        grid_clouds: List[List[Optional[np.ndarray]]] = [[None for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                if (r, c) in missing_set:
                    continue
                filename = f"grid_{r}_{c}.ply"
                path = sample_dir / filename
                points = grid[r][c].sample_point_cloud(self.rng, n_points=None)
                points = self._apply_point_cloud_perturbation(
                    points,
                    row_idx=r,
                    params_h=params,
                    params_v=params_v,
                    perturbation_type=selected_perturbation_type,
                )
                write_ply(path, points, color=color_rgb(grid[r][c].obs.color_preset_idx))
                grid_paths[r][c] = f"{sample_id}/{filename}"
                grid_clouds[r][c] = points

        candidate_paths: List[str] = []
        candidate_clouds: List[np.ndarray] = []
        for i, entity in enumerate(candidates):
            filename = f"cand_{i}.ply"
            path = sample_dir / filename
            points = entity.sample_point_cloud(self.rng, n_points=None)
            write_ply(path, points, color=color_rgb(entity.obs.color_preset_idx))
            candidate_paths.append(f"{sample_id}/{filename}")
            candidate_clouds.append(points)

        gt_label = chr(ord("A") + gt_index)
        grid_entities: List[List[Optional[Dict[str, Any]]]] = [[None for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                grid_entities[r][c] = grid[r][c].to_dict()
        matrix_relation = self._build_matrix_relation_spec(params, k_h=k_h, k_v=k_v, params_v=params_v)
        meta = {
            "id": sample_id,
            "task_type": "matrix_3x3",
            "focus": self._build_matrix_focus(params, k_h=k_h, k_v=k_v, params_v=params_v),
            "target_position": [2, 2],
            "missing_positions": [[r, c] for r, c in missing_positions_sorted],
            "empty_grid_positions": [[r, c] for r, c in missing_positions_sorted],
            "grid_observation_mask": self._build_observation_mask(missing_set),
            "grid_paths": grid_paths,
            "candidate_paths": candidate_paths,
            "gt_index": gt_index,
            "gt_label": gt_label,
            "distractor_types": distractor_types,
            "rule_template": rule.template.value,
            "rule_params": params.to_dict(),
            "rule_params_vertical": params_v.to_dict() if params_v is not None else None,
            "matrix_relation": matrix_relation,
            "k_h": int(k_h),
            "k_v": int(k_v),
            "K_max": int(k_max),
            "matrix_level_config": self.level_cfg.to_dict(),
            "n_points": self.config.n_points,
            "point_cloud_row_perturbation": {
                "enabled": bool(self.config.enable_row_perturbation),
                "single_type_per_sample": True,
                "selected_type": selected_perturbation_type,
                "apply_density_perturb_on_density_rules": bool(self.config.apply_density_perturb_on_density_rules),
                "rows": {
                    "index_0_row1": {
                        "density_keep_ratio": 1.0,
                        "jitter_std_ratio": 0.0,
                        "quantize_step_ratio": 0.0,
                        "outlier_replace_ratio": 0.0,
                    },
                    "index_1_row2": {
                        "density_keep_ratio": float(self.config.row2_density_keep_ratio),
                        "jitter_std_ratio": float(self.config.row2_jitter_std_ratio),
                        "quantize_step_ratio": float(self.config.row2_quantize_step_ratio),
                        "outlier_replace_ratio": float(self.config.row2_outlier_replace_ratio),
                    },
                    "index_2_row3": {
                        "density_keep_ratio": float(self.config.row3_density_keep_ratio),
                        "jitter_std_ratio": float(self.config.row3_jitter_std_ratio),
                        "quantize_step_ratio": float(self.config.row3_quantize_step_ratio),
                        "outlier_replace_ratio": float(self.config.row3_outlier_replace_ratio),
                    },
                },
            },
            "rule": {
                "template": rule.template.value,
                "source_align": RULE_SOURCE_ALIGN.get(rule.template, []),
                "params": params.to_dict(),
                "vertical_template": rule_v.template.value if rule_v is not None else None,
                "vertical_params": params_v.to_dict() if params_v is not None else None,
            },
            "entities": {
                "grid": grid_entities,
                "candidates": [cand.to_dict() for cand in candidates],
            },
            "notes": {
                "candidate_notes": candidate_notes,
                "alt_relations": alt_relations,
            },
        }

        if self.config.generate_confusing_view:
            from .render_confusing_view import generate_confusing_view_for_matrix_sample

            entities_for_view: List[PCRAREntity] = []
            for r in range(3):
                for c in range(3):
                    if (r, c) in missing_set:
                        continue
                    entities_for_view.append(grid[r][c])
            entities_for_view.extend(candidates)
            confusing_view = generate_confusing_view_for_matrix_sample(
                rule_template=rule.template,
                params=params,
                entities=entities_for_view,
                grid_point_clouds=grid_clouds,
                candidate_point_clouds=candidate_clouds,
                output_dir=sample_dir,
                rng=self.rng,
                image_size=self.config.view_image_size,
            )
            meta["confusing_view"] = confusing_view

        write_meta(sample_dir / "meta.json", meta)
        return meta

    def generate_sample(
        self,
        output_root: Path,
        sample_index: int,
        mode: str = "matrix",
        task_type: Optional[str] = None,
        correct_idx: Optional[int] = None,
        preferred_axis: Optional[str] = None,
        preferred_perturbation_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        del task_type
        output_root = Path(output_root)
        if mode == "legacy":
            if self._legacy_generator is None:
                raise RuntimeError("Legacy mode requested but legacy generator is disabled")
            return self._legacy_generator.generate_sample(
                output_root=output_root,
                sample_index=sample_index,
                task_type=None,
                correct_idx=correct_idx,
                preferred_axis=None,
            )

        if self.config.rule_filter == {RuleTemplate.SYMMETRY}:
            # 对称规则按相邻格一步递推：E[r,c+1] = T(E[r,c]), E[r+1,c] = T(E[r,c])
            k_pairs = [(1, 1)]
        else:
            # Enforce complementary stride granularity:
            # horizontal one-step copy <-> vertical two-step copy, or vice versa.
            h_choices = {int(x) for x in self.config.matrix_k_h_choices}
            v_choices = {int(x) for x in self.config.matrix_k_v_choices}
            k_pairs: List[Tuple[int, int]] = []
            if 1 in h_choices and 2 in v_choices:
                k_pairs.append((1, 2))
            if 2 in h_choices and 1 in v_choices:
                k_pairs.append((2, 1))
            if not k_pairs:
                raise RuntimeError(
                    "matrix_k_h_choices/matrix_k_v_choices must support complementary strides (1,2) or (2,1)"
                )
        self.rng.shuffle(k_pairs)
        last_err: Optional[Exception] = None
        for k_h, k_v in k_pairs:
            try:
                grid, k_max, rule, params, rule_v, params_v = self._sample_matrix_problem(
                    k_h=k_h,
                    k_v=k_v,
                    preferred_axis=preferred_axis,
                )
                break
            except RuntimeError as exc:
                last_err = exc
                continue
        else:
            raise RuntimeError(f"Failed to sample matrix problem across all k pairs: {last_err}")
        missing_positions = self._sample_missing_positions()
        missing_set = self._normalize_missing_positions(missing_positions)
        self._validate_missing_positions(missing_set)
        grid_context = self._build_grid_context(grid, missing_positions=missing_positions)
        gt_entity = grid[2][2].copy()

        cand_payload = generate_candidates(
            grid_context=grid_context,
            gt_entity=gt_entity,
            true_rule=rule,
            true_params=params,
            k_h=k_h,
            k_v=k_v,
            num_options=self.config.num_options,
            mix_cfg=CandidateMixConfig(num_options=self.config.num_options),
            level_cfg=self.level_cfg,
            rng=self.rng,
            rule_whitelist=list(self.config.rule_filter) if self.config.rule_filter else None,
            true_rule_v=rule_v,
            true_params_v=params_v,
        )

        candidates = cand_payload["candidates"]
        if rule.template not in {RuleTemplate.CONSERVATION, RuleTemplate.SYMMETRY} and not all(
            self._entity_visibility_ok(cand) for cand in candidates
        ):
            raise RuntimeError("candidate_visibility_failed")
        gt_index = int(cand_payload["gt_index"]) if correct_idx is None else int(correct_idx)

        if correct_idx is not None and gt_index != int(correct_idx):
            # Reorder to respect forced correct index for compatibility callers.
            target = int(correct_idx)
            current = int(cand_payload["gt_index"])
            candidates[target], candidates[current] = candidates[current], candidates[target]
            cand_payload["distractor_types"][target], cand_payload["distractor_types"][current] = (
                cand_payload["distractor_types"][current],
                cand_payload["distractor_types"][target],
            )
            cand_payload["candidate_notes"][target], cand_payload["candidate_notes"][current] = (
                cand_payload["candidate_notes"][current],
                cand_payload["candidate_notes"][target],
            )
            gt_index = target

        return self._save_matrix_sample(
            output_root=output_root,
            sample_index=sample_index,
            grid=grid,
            missing_positions=missing_positions,
            k_h=k_h,
            k_v=k_v,
            k_max=k_max,
            rule=rule,
            params=params,
            rule_v=rule_v,
            params_v=params_v,
            candidates=candidates,
            gt_index=gt_index,
            distractor_types=cand_payload["distractor_types"],
            candidate_notes=cand_payload["candidate_notes"],
            alt_relations=cand_payload["alt_relations"],
            preferred_perturbation_type=preferred_perturbation_type,
        )

    def generate_dataset(
        self,
        output_root: str | Path,
        num_samples: int,
        mode: str = "matrix",
    ) -> None:
        output_root = Path(output_root)
        ensure_dir(output_root)

        if mode == "legacy":
            if self._legacy_generator is None:
                raise RuntimeError("Legacy mode requested but legacy generator is disabled")
            self._legacy_generator.generate_dataset(output_root, num_samples)
            return

        entries: List[Dict[str, Any]] = []
        preferred_axis_plan: Optional[List[str]] = None
        preferred_perturbation_plan: Optional[List[str]] = None
        planned_axes: Optional[List[str]] = None
        single_template: Optional[RuleTemplate] = None
        if mode == "matrix" and self.config.enable_row_perturbation:
            perturb_types = list(ROW_PERTURB_TYPES)
            # When density perturbation is globally disabled on density-based rules,
            # remove it from the forced-balance plan to avoid impossible allocations.
            if not self.config.apply_density_perturb_on_density_rules:
                perturb_types = [x for x in perturb_types if x != "density"]
            if perturb_types:
                base = num_samples // len(perturb_types)
                remainder = num_samples % len(perturb_types)
                preferred_perturbation_plan = []
                for i, perturb_type in enumerate(perturb_types):
                    count = base + (1 if i < remainder else 0)
                    preferred_perturbation_plan.extend([perturb_type] * count)
                self.rng.shuffle(preferred_perturbation_plan)

        if mode == "matrix" and self.config.rule_filter and len(self.config.rule_filter) == 1:
            template = self._normalize_template_name(next(iter(self.config.rule_filter)))
            single_template = template
            axes: Optional[List[str]] = None
            if template == RuleTemplate.CYCLE:
                axes = [CYCLE_AXIS_DENSITY, CYCLE_AXIS_SIZE, CYCLE_AXIS_SHAPE, CYCLE_AXIS_COLOR]
            elif template == RuleTemplate.PROGRESSION:
                # Keep progression exams balanced between size and pose questions.
                axes = ["r", "R"]
            elif template == RuleTemplate.SYMMETRY:
                # Matrix-path symmetry with k_h=k_v=1 is currently feasible on size axis only.
                axes = ["r"]

            if axes:
                planned_axes = list(axes)
                base = num_samples // len(axes)
                remainder = num_samples % len(axes)
                preferred_axis_plan = []
                for i, axis in enumerate(axes):
                    count = base + (1 if i < remainder else 0)
                    preferred_axis_plan.extend([axis] * count)
                self.rng.shuffle(preferred_axis_plan)

        max_attempts = 30
        for idx in range(num_samples):
            last_err: Optional[Exception] = None
            preferred_axis = preferred_axis_plan[idx] if preferred_axis_plan else None
            preferred_perturbation_type = (
                preferred_perturbation_plan[idx] if preferred_perturbation_plan else None
            )
            generated = False
            attempts_used = 0
            preferred_attempts = 120 if preferred_axis else max_attempts
            for _ in range(preferred_attempts):
                attempts_used += 1
                try:
                    entry = self.generate_sample(
                        output_root=output_root,
                        sample_index=idx,
                        mode="matrix",
                        preferred_axis=preferred_axis,
                        preferred_perturbation_type=preferred_perturbation_type,
                    )
                    entries.append(entry)
                    generated = True
                    break
                except RuntimeError as exc:
                    last_err = exc
                    continue

            # Fallback: keep the same preferred axis and continue retrying.
            # This preserves per-attribute balance for each rule-specific exam.
            if (not generated) and preferred_axis:
                for _ in range(max_attempts):
                    attempts_used += 1
                    try:
                        entry = self.generate_sample(
                            output_root=output_root,
                            sample_index=idx,
                            mode="matrix",
                            preferred_axis=preferred_axis,
                            preferred_perturbation_type=preferred_perturbation_type,
                        )
                        entries.append(entry)
                        generated = True
                        break
                    except RuntimeError as exc:
                        last_err = exc
                        continue

            if not generated:
                raise RuntimeError(
                    f"Failed to generate matrix sample {idx} after {preferred_attempts + (max_attempts if preferred_axis else 0)} attempts: {last_err}"
                )
            if idx == 0 or (idx + 1) % 100 == 0 or (idx + 1) == num_samples:
                template_name = single_template.value if single_template is not None else "mixed"
                print(
                    f"[matrix:{template_name}] progress {idx + 1}/{num_samples} "
                    f"(attempts_for_last={attempts_used})"
                )

        if mode == "matrix" and single_template is not None and planned_axes:
            axis_counts: Dict[str, int] = {axis: 0 for axis in planned_axes}
            for entry in entries:
                params = entry.get("rule_params", {})
                axis = params.get("axis") if isinstance(params, dict) else None
                if single_template == RuleTemplate.CYCLE:
                    axis = self._normalize_cycle_axis_name(axis)
                if axis in axis_counts:
                    axis_counts[str(axis)] += 1
            counts = [axis_counts[a] for a in planned_axes]
            if counts and (max(counts) - min(counts) > 1):
                raise RuntimeError(
                    f"Axis balance check failed for {single_template.value}: {axis_counts}"
                )

        if mode == "matrix" and preferred_perturbation_plan:
            expected_perturb_counts: Dict[str, int] = {}
            for perturb_type in preferred_perturbation_plan:
                expected_perturb_counts[perturb_type] = expected_perturb_counts.get(perturb_type, 0) + 1
            actual_perturb_counts: Dict[str, int] = {k: 0 for k in expected_perturb_counts}
            for entry in entries:
                selected = (
                    entry.get("point_cloud_row_perturbation", {}).get("selected_type")
                    if isinstance(entry, dict)
                    else None
                )
                if selected in actual_perturb_counts:
                    actual_perturb_counts[str(selected)] += 1
            if actual_perturb_counts != expected_perturb_counts:
                raise RuntimeError(
                    "Perturbation balance check failed: "
                    f"expected={expected_perturb_counts}, actual={actual_perturb_counts}"
                )

        write_meta(output_root / "meta.json", entries)
        print(f"Generated {num_samples} PCRAR matrix samples in {output_root}")
