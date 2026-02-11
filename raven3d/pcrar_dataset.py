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
from .csg import OpType
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
from .pcrar_rules import PCRARRule, RuleParams, RuleTemplate, RULE_SOURCE_ALIGN, get_rule


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

    def _sample_matrix_rule(self, entity: PCRAREntity) -> Tuple[PCRARRule, RuleParams]:
        templates = list(self.config.rule_filter) if self.config.rule_filter else list(RuleTemplate)
        self.rng.shuffle(templates)
        for template in templates:
            rule = get_rule(template)
            for _ in range(40):
                params = rule.sample_params(self.rng, entity)
                if rule.can_apply(entity, params):
                    return rule, params
        raise RuntimeError("No applicable matrix rule found for current entity")

    def _sample_matrix_problem(
        self,
        k_h: int,
        k_v: int,
        max_attempts: int = 200,
    ) -> Tuple[List[List[PCRAREntity]], int, PCRARRule, RuleParams]:
        k_max = 2 * k_h + 2 * k_v
        for _ in range(max_attempts):
            leaf_count = int(self.rng.integers(self.config.leaf_count_min, self.config.leaf_count_max + 1))
            e00 = sample_random_entity(
                self.rng,
                leaf_count=leaf_count,
                allowed_ops=self.config.allowed_ops,
            )
            e00 = normalize_entity_levels(e00, self.level_cfg, self.rng)
            try:
                rule, params = self._sample_matrix_rule(e00)
            except RuntimeError:
                continue

            e00 = prepare_entity_for_rule_path(e00, rule, params, self.level_cfg)
            if not can_apply_k(e00, rule, params, k_max):
                continue

            try:
                grid, _, _ = generate_grid(e00, rule, params, k_h=k_h, k_v=k_v)
            except RuntimeError:
                continue

            ok, reason = grid_quality_checks(grid, rule, params, k_h=k_h, k_v=k_v)
            if not ok:
                if reason in {"adjacent_cell_collision", "low_global_diversity", "path_consistency_failed"}:
                    continue
            return grid, k_max, rule, params

        raise RuntimeError("Failed to sample a valid 3x3 matrix problem")

    @staticmethod
    def _build_grid_context(grid: List[List[PCRAREntity]]) -> List[List[Optional[PCRAREntity]]]:
        context: List[List[Optional[PCRAREntity]]] = []
        for r in range(3):
            row: List[Optional[PCRAREntity]] = []
            for c in range(3):
                if (r, c) == (2, 2):
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
            sym_attr_map = {"p": "slot_position", "R": "local_pose", "r": "size_level", "d": "density_level"}
            summary["changed_attribute"] = sym_attr_map.get(axis, "symmetry_axis")
            summary["description"] = f"Symmetry on {summary['changed_attribute']}"
        return summary

    @classmethod
    def _build_matrix_relation_spec(cls, params: RuleParams, k_h: int, k_v: int) -> Dict[str, Any]:
        semantics = cls._summarize_rule_semantics(params)
        return {
            "formula": "E[r,c] = T^(r*k_v + c*k_h)(E[0,0])",
            "rule_instance": semantics,
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

    def _save_matrix_sample(
        self,
        output_root: Path,
        sample_index: int,
        grid: List[List[PCRAREntity]],
        k_h: int,
        k_v: int,
        k_max: int,
        rule: PCRARRule,
        params: RuleParams,
        candidates: List[PCRAREntity],
        gt_index: int,
        distractor_types: List[str],
        candidate_notes: List[str],
        alt_relations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        sample_id = f"sample_{sample_index:06d}"
        sample_dir = output_root / sample_id
        ensure_dir(sample_dir)

        grid_paths: List[List[Optional[str]]] = [[None for _ in range(3)] for _ in range(3)]
        grid_clouds: List[List[Optional[np.ndarray]]] = [[None for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                if (r, c) == (2, 2):
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
        meta = {
            "id": sample_id,
            "task_type": "matrix_3x3",
            "focus": "3x3 matrix completion with a fixed rule instance and dual-step strides.",
            "target_position": [2, 2],
            "grid_paths": grid_paths,
            "candidate_paths": candidate_paths,
            "gt_index": gt_index,
            "gt_label": gt_label,
            "distractor_types": distractor_types,
            "rule_template": rule.template.value,
            "rule_params": params.to_dict(),
            "matrix_relation": self._build_matrix_relation_spec(params, k_h=k_h, k_v=k_v),
            "k_h": int(k_h),
            "k_v": int(k_v),
            "K_max": int(k_max),
            "matrix_level_config": self.level_cfg.to_dict(),
            "n_points": self.config.n_points,
            "rule": {
                "template": rule.template.value,
                "source_align": RULE_SOURCE_ALIGN.get(rule.template, []),
                "params": params.to_dict(),
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
                    if (r, c) == (2, 2):
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
        del task_type, preferred_axis
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

        k_h = int(self.rng.choice(self.config.matrix_k_h_choices))
        k_v = int(self.rng.choice(self.config.matrix_k_v_choices))
        grid, k_max, rule, params = self._sample_matrix_problem(k_h=k_h, k_v=k_v)
        grid_context = self._build_grid_context(grid)
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
        )

        candidates = cand_payload["candidates"]
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
            k_h=k_h,
            k_v=k_v,
            k_max=k_max,
            rule=rule,
            params=params,
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
        max_attempts = 30
        for idx in range(num_samples):
            last_err: Optional[Exception] = None
            for _ in range(max_attempts):
                try:
                    entry = self.generate_sample(output_root=output_root, sample_index=idx, mode="matrix")
                    entries.append(entry)
                    break
                except RuntimeError as exc:
                    last_err = exc
                    continue
            else:
                raise RuntimeError(
                    f"Failed to generate matrix sample {idx} after {max_attempts} attempts: {last_err}"
                )

        write_meta(output_root / "meta.json", entries)
        print(f"Generated {num_samples} PCRAR matrix samples in {output_root}")
