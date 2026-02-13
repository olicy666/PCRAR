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
from .pcrar_entity import DEFAULT_N_POINTS, PCRAREntity, sample_random_entity
from .pcrar_rules import (
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
    matrix_size_levels: int = 7
    matrix_density_levels: int = 5
    matrix_delta_levels: int = 5
    matrix_slot_levels: Tuple[int, ...] = (-1, 0, 1)
    matrix_missing_one_per_row: bool = True
    generate_confusing_view: bool = True
    view_image_size: Tuple[int, int] = (512, 512)
    rule_filter: Optional[Set[RuleTemplate]] = None

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
        templates = list(self.config.rule_filter) if self.config.rule_filter else list(RuleTemplate)
        self.rng.shuffle(templates)
        for template in templates:
            rule = get_rule(template)
            if template == RuleTemplate.COPY:
                self._enforce_copy_preconditions(entity, preferred_axis=preferred_axis)
            for _ in range(40):
                params = rule.sample_params(self.rng, entity)
                if template == RuleTemplate.COPY and preferred_axis and params.axis != preferred_axis:
                    continue
                if rule.can_apply(entity, params):
                    return rule, params
        raise RuntimeError("No applicable matrix rule found for current entity")

    def _enforce_copy_preconditions(self, entity: PCRAREntity, preferred_axis: Optional[str] = None) -> None:
        leaves = entity.get_leaves()
        if len(leaves) != 3:
            return
        mode_map = {
            "copy_shape_cycle": 0,
            "copy_size_cycle": 1,
            "copy_density_cycle": 2,
        }
        mode = mode_map.get(preferred_axis, int(self.rng.integers(3)))
        if mode == 0:
            # copy_shape_cycle: 3 distinct shapes
            for i, leaf in enumerate(leaves):
                leaf.prim_type = PRIM_TYPE_CYCLE[i % len(PRIM_TYPE_CYCLE)]
        elif mode == 1:
            # copy_size_cycle: same shape, all sizes different
            base = leaves[0].prim_type
            for leaf in leaves:
                leaf.prim_type = base
            size_levels = self.level_cfg.size_levels
            if len(size_levels) >= 3:
                # 使用中间三档而非极端档位，降低包含/重叠概率，提升 size 题通过率
                mid = len(size_levels) // 2
                lo = max(0, mid - 1)
                hi = min(len(size_levels) - 1, lo + 2)
                lo = max(0, hi - 2)
                chosen = [size_levels[lo], size_levels[lo + 1], size_levels[hi]]
                for i, leaf in enumerate(leaves):
                    leaf.size_level = chosen[i]
            # 明确左中右槽位并重新分离，避免尺寸差导致遮蔽
            slots = list(self.level_cfg.slot_levels) if self.level_cfg.slot_levels else [-1, 0, 1]
            slots = sorted(slots)
            if len(slots) >= 3:
                for i, leaf in enumerate(leaves):
                    leaf.slot = int(slots[i])
            enforce_leaf_separation(leaves)
        else:
            # copy_density_cycle: same shape, same size, distinguishable weights
            base = leaves[0].prim_type
            for leaf in leaves:
                leaf.prim_type = base
            base_size = leaves[0].size_level
            for leaf in leaves:
                leaf.size_level = base_size
            entity.obs.part_sampling_weights = [0.5, 0.3125, 0.1875]

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
        for _ in range(max_attempts):
            if self.config.rule_filter == {RuleTemplate.COPY}:
                leaf_count = 3
            elif self.config.rule_filter == {RuleTemplate.CYCLE}:
                leaf_count = 3
            elif self.config.rule_filter == {RuleTemplate.PERMUTATION}:
                leaf_count = 3
            elif self.config.rule_filter == {RuleTemplate.SYMMETRY}:
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

            rule_v = rule
            params_v = params
            dual_symmetry = rule.template == RuleTemplate.SYMMETRY
            if dual_symmetry:
                params_v = self._build_symmetry_vertical_params(params)
                e00 = self._prepare_entity_for_symmetry_dual(e00, params)
            else:
                e00 = prepare_entity_for_rule_path(e00, rule, params, self.level_cfg)
            if not self._entity_visibility_ok(e00):
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

            if not self._grid_visibility_ok(grid):
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
            summary["changed_attribute"] = "primitive_type"
            summary["description"] = (
                f"Cycle over primitive types on leaves={params.leaf_indices if params.leaf_indices is not None else [params.leaf_idx]}"
            )
        elif template == RuleTemplate.COPY:
            summary["changed_attribute"] = axis or "copy_pattern"
            summary["description"] = f"Copy pattern ({axis}) with direction={params.direction}"
        elif template == RuleTemplate.COUNT:
            summary["changed_attribute"] = "leaf_count"
            summary["description"] = f"Count transition with direction={params.direction}"
        elif template == RuleTemplate.CONSERVATION:
            summary["changed_attribute"] = "paired_size_levels"
            summary["description"] = f"Conservation: leaf{params.leaf_idx} +1 and leaf{params.direction} -1"
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
            detail["per_T_change"] = {
                "attribute": "primitive_type",
                "leaf_indices": params.leaf_indices if params.leaf_indices is not None else [params.leaf_idx],
                "directions": params.directions if params.directions is not None else [params.direction],
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
            detail["per_T_change"] = {
                "attribute": "paired_size_levels",
                "plus_leaf": int(params.leaf_idx) if params.leaf_idx is not None else None,
                "minus_leaf": int(params.direction) if params.direction is not None else None,
                "plus_step": 1,
                "minus_step": -1,
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
            leaf_indices = params.leaf_indices if params.leaf_indices is not None else [params.leaf_idx]
            leaf_indices = [int(i) for i in leaf_indices if i is not None]
            directions = params.directions if params.directions is not None else [params.direction] * max(1, len(leaf_indices))
            leaf_desc = []
            for idx, leaf_idx in enumerate(leaf_indices):
                direction = int(directions[idx]) if idx < len(directions) else int(params.direction)
                leaf_desc.append(f"leaf{leaf_idx}:{_shift_word(direction)}")
            leaf_part = ", ".join(leaf_desc) if leaf_desc else "selected leaves"
            per_t = (
                "primitive_type cycles over Sphere->Box->Cylinder->Cone "
                f"on {leaf_part}"
            )
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
            plus_leaf = int(params.leaf_idx) if params.leaf_idx is not None else None
            minus_leaf = int(params.direction) if params.direction is not None else None
            per_t = f"size conservation pair: leaf{plus_leaf} +1 level, leaf{minus_leaf} -1 level"
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
    ) -> Dict[str, Any]:
        sample_id = f"sample_{sample_index:06d}"
        sample_dir = output_root / sample_id
        ensure_dir(sample_dir)
        missing_set = self._normalize_missing_positions(missing_positions)
        self._validate_missing_positions(missing_set)
        missing_positions_sorted = sorted(missing_set)

        grid_paths: List[List[Optional[str]]] = [[None for _ in range(3)] for _ in range(3)]
        grid_clouds: List[List[Optional[np.ndarray]]] = [[None for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                if (r, c) in missing_set:
                    continue
                filename = f"grid_{r}_{c}.ply"
                path = sample_dir / filename
                points = grid[r][c].sample_point_cloud(self.rng, n_points=None)
                write_ply(path, points, color=GRID_COLOR_MAP.get((r, c), (200, 200, 200)))
                grid_paths[r][c] = f"{sample_id}/{filename}"
                grid_clouds[r][c] = points

        candidate_paths: List[str] = []
        candidate_clouds: List[np.ndarray] = []
        for i, entity in enumerate(candidates):
            filename = f"cand_{i}.ply"
            path = sample_dir / filename
            points = entity.sample_point_cloud(self.rng, n_points=None)
            write_ply(path, points, color=CAND_COLOR_MAP.get(i, (180, 180, 180)))
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
            k_pairs = [
                (int(kh), int(kv))
                for kh in self.config.matrix_k_h_choices
                for kv in self.config.matrix_k_v_choices
            ]
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
        if not all(self._entity_visibility_ok(cand) for cand in candidates):
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
        copy_axis_plan: Optional[List[str]] = None
        if (
            mode == "matrix"
            and self.config.rule_filter
            and len(self.config.rule_filter) == 1
            and RuleTemplate.COPY in self.config.rule_filter
        ):
            axes = ["copy_size_cycle", "copy_density_cycle", "copy_shape_cycle"]
            base = num_samples // len(axes)
            remainder = num_samples % len(axes)
            copy_axis_plan = []
            for i, axis in enumerate(axes):
                count = base + (1 if i < remainder else 0)
                copy_axis_plan.extend([axis] * count)
            self.rng.shuffle(copy_axis_plan)

        max_attempts = 30
        for idx in range(num_samples):
            last_err: Optional[Exception] = None
            preferred_axis = copy_axis_plan[idx] if copy_axis_plan else None
            per_sample_attempts = max_attempts
            if preferred_axis:
                # 目标轴约束下适当放宽重试次数，保证三类题比例可达
                per_sample_attempts = 120
            for _ in range(per_sample_attempts):
                try:
                    entry = self.generate_sample(
                        output_root=output_root,
                        sample_index=idx,
                        mode="matrix",
                        preferred_axis=preferred_axis,
                    )
                    entries.append(entry)
                    break
                except RuntimeError as exc:
                    last_err = exc
                    continue
            else:
                raise RuntimeError(
                    f"Failed to generate matrix sample {idx} after {per_sample_attempts} attempts: {last_err}"
                )

        write_meta(output_root / "meta.json", entries)
        print(f"Generated {num_samples} PCRAR matrix samples in {output_root}")
