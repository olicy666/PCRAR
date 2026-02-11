"""Matrix grid generation and relation checks for PCRAR matrix tasks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .csg import DeltaLevel, SizeLevel
from .pcrar_entity import PCRAREntity, DENSITY_POINT_PRESETS, density_point_count, entities_equal
from .pcrar_rules import PCRARRule, RuleParams, RuleTemplate


Grid3x3 = List[List[PCRAREntity]]


@dataclass(frozen=True)
class MatrixLevelConfig:
    size_levels: List[SizeLevel]
    delta_levels: List[DeltaLevel]
    density_indices: List[int]
    slot_levels: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "size_levels": [x.value for x in self.size_levels],
            "delta_levels": [x.value for x in self.delta_levels],
            "density_levels": [int(DENSITY_POINT_PRESETS[i]) for i in self.density_indices],
            "density_indices": [int(i) for i in self.density_indices],
            "slot_levels": [int(x) for x in self.slot_levels],
        }


def _pick_discrete_subset(items: Sequence[Any], count: int) -> List[Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count >= len(items):
        return list(items)
    idx = np.linspace(0, len(items) - 1, count)
    chosen = sorted({int(round(x)) for x in idx})
    while len(chosen) < count:
        for i in range(len(items)):
            if i not in chosen:
                chosen.append(i)
            if len(chosen) == count:
                break
    chosen.sort()
    return [items[i] for i in chosen[:count]]


def build_matrix_level_config(
    matrix_size_levels: int = 7,
    matrix_density_levels: int = 5,
    matrix_delta_levels: int = 5,
    matrix_slot_levels: Optional[Sequence[int]] = None,
) -> MatrixLevelConfig:
    all_size = list(SizeLevel)
    all_delta = list(DeltaLevel)
    size_levels = _pick_discrete_subset(all_size, matrix_size_levels)
    delta_levels = _pick_discrete_subset(all_delta, matrix_delta_levels)

    density_candidates = list(range(len(DENSITY_POINT_PRESETS)))
    density_indices = _pick_discrete_subset(density_candidates, matrix_density_levels)

    slots = list(matrix_slot_levels) if matrix_slot_levels is not None else [-1, 0, 1]
    if not slots:
        raise ValueError("matrix_slot_levels cannot be empty")
    return MatrixLevelConfig(
        size_levels=size_levels,
        delta_levels=delta_levels,
        density_indices=[int(x) for x in density_indices],
        slot_levels=[int(x) for x in sorted(set(slots))],
    )


def normalize_entity_levels(
    entity: PCRAREntity,
    level_cfg: MatrixLevelConfig,
    rng: np.random.Generator,
) -> PCRAREntity:
    out = entity.copy()
    leaves = out.get_leaves()
    for leaf in leaves:
        leaf.size_level = level_cfg.size_levels[int(rng.integers(len(level_cfg.size_levels)))]
        leaf.delta_level = level_cfg.delta_levels[int(rng.integers(len(level_cfg.delta_levels)))]
        if leaf.slot not in level_cfg.slot_levels:
            leaf.slot = int(level_cfg.slot_levels[int(rng.integers(len(level_cfg.slot_levels)))])

    out.obs.density_preset_idx = int(level_cfg.density_indices[int(rng.integers(len(level_cfg.density_indices)))])
    out.obs.n_points = density_point_count(out.obs.density_preset_idx)
    return out


def _set_density_at_boundary(entity: PCRAREntity, direction: int, level_cfg: MatrixLevelConfig) -> None:
    idx = min(level_cfg.density_indices) if direction >= 0 else max(level_cfg.density_indices)
    entity.obs.density_preset_idx = int(idx)
    entity.obs.n_points = density_point_count(int(idx))


def _set_size_at_boundary(entity: PCRAREntity, direction: int, level_cfg: MatrixLevelConfig) -> None:
    target = level_cfg.size_levels[0] if direction >= 0 else level_cfg.size_levels[-1]
    for leaf in entity.get_leaves():
        leaf.size_level = target


def _set_slot_at_boundary(entity: PCRAREntity, direction: int, level_cfg: MatrixLevelConfig) -> None:
    target = level_cfg.slot_levels[0] if direction >= 0 else level_cfg.slot_levels[-1]
    for leaf in entity.get_leaves():
        leaf.slot = int(target)


def prepare_entity_for_rule_path(
    entity: PCRAREntity,
    rule: PCRARRule,
    params: RuleParams,
    level_cfg: MatrixLevelConfig,
) -> PCRAREntity:
    out = entity.copy()
    if params.template == RuleTemplate.PROGRESSION:
        if params.axis == "r":
            _set_size_at_boundary(out, params.direction, level_cfg)
        elif params.axis == "d":
            _set_density_at_boundary(out, params.direction, level_cfg)
        elif params.axis == "p":
            _set_slot_at_boundary(out, params.direction, level_cfg)
    elif params.template == RuleTemplate.SYMMETRY:
        leaves = out.get_leaves()
        if params.axis == "r" and params.leaf_idx is not None and params.direction is not None:
            li, ri = int(params.leaf_idx), int(params.direction)
            if 0 <= li < len(leaves) and 0 <= ri < len(leaves):
                leaves[li].size_level = level_cfg.size_levels[0]
                leaves[ri].size_level = level_cfg.size_levels[-1]
        elif params.axis == "d":
            _set_density_at_boundary(out, params.direction, level_cfg)
    elif params.template == RuleTemplate.CONSERVATION:
        leaves = out.get_leaves()
        li, ri = params.leaf_idx, params.direction
        if li is not None and ri is not None and 0 <= li < len(leaves) and 0 <= ri < len(leaves):
            leaves[li].size_level = level_cfg.size_levels[0]
            leaves[ri].size_level = level_cfg.size_levels[-1]
    return out


def can_apply_k(entity: PCRAREntity, rule: PCRARRule, params: RuleParams, k: int) -> bool:
    if k < 0:
        return False
    current = entity.copy()
    for _ in range(k):
        try:
            if not rule.can_apply(current, params):
                return False
            current = rule.apply(current, params)
        except Exception:
            return False
    return True


def apply_k(entity: PCRAREntity, rule: PCRARRule, params: RuleParams, k: int) -> PCRAREntity:
    if k < 0:
        raise ValueError("k must be non-negative")
    current = entity.copy()
    for _ in range(k):
        try:
            if not rule.can_apply(current, params):
                raise RuntimeError("Rule cannot be applied for required number of steps")
            current = rule.apply(current, params)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Rule application failed") from exc
    return current


def generate_grid(
    e00: PCRAREntity,
    rule: PCRARRule,
    params: RuleParams,
    k_h: int,
    k_v: int,
) -> Tuple[Grid3x3, int, List[PCRAREntity]]:
    if k_h <= 0 or k_v <= 0:
        raise ValueError("k_h and k_v must be positive integers")

    k_max = 2 * k_h + 2 * k_v
    states: List[PCRAREntity] = [e00.copy()]
    for _ in range(k_max):
        nxt = apply_k(states[-1], rule, params, 1)
        states.append(nxt)

    grid: Grid3x3 = []
    for r in range(3):
        row: List[PCRAREntity] = []
        for c in range(3):
            exp = r * k_v + c * k_h
            row.append(states[exp].copy())
        grid.append(row)
    return grid, k_max, states


def check_path_consistency(
    grid: Grid3x3,
    rule: PCRARRule,
    params: RuleParams,
    k_h: int,
    k_v: int,
) -> bool:
    e10 = grid[1][0]
    e01 = grid[0][1]
    e11 = grid[1][1]
    from_right = apply_k(e10, rule, params, k_h)
    from_down = apply_k(e01, rule, params, k_v)
    return entities_equal(from_right, e11, check_obs=True) and entities_equal(from_down, e11, check_obs=True)


def grid_quality_checks(
    grid: Grid3x3,
    rule: PCRARRule,
    params: RuleParams,
    k_h: int,
    k_v: int,
    min_unique: int = 5,
) -> Tuple[bool, str]:
    if not check_path_consistency(grid, rule, params, k_h, k_v):
        return False, "path_consistency_failed"

    pairs: List[Tuple[PCRAREntity, PCRAREntity]] = []
    for r in range(3):
        for c in range(2):
            pairs.append((grid[r][c], grid[r][c + 1]))
    for c in range(3):
        for r in range(2):
            pairs.append((grid[r][c], grid[r + 1][c]))

    for a, b in pairs:
        if entities_equal(a, b, check_obs=True):
            return False, "adjacent_cell_collision"

    unique_cells: List[PCRAREntity] = []
    for r in range(3):
        for c in range(3):
            item = grid[r][c]
            if not any(entities_equal(item, seen, check_obs=True) for seen in unique_cells):
                unique_cells.append(item)

    if len(unique_cells) < min_unique:
        return False, "low_global_diversity"
    return True, "ok"


def check_consistent_with_true_relation(
    candidate: PCRAREntity,
    grid_context: List[List[Optional[PCRAREntity]]],
    rule: PCRARRule,
    params: RuleParams,
    k_h: int,
    k_v: int,
) -> bool:
    left = grid_context[2][1]
    up = grid_context[1][2]
    if left is None or up is None:
        return False
    try:
        from_left = apply_k(left, rule, params, k_h)
        from_up = apply_k(up, rule, params, k_v)
    except RuntimeError:
        return False

    return (
        entities_equal(candidate, from_left, check_obs=True)
        and entities_equal(candidate, from_up, check_obs=True)
        and entities_equal(from_left, from_up, check_obs=True)
    )


def check_consistent_with_alt_relation(
    candidate: PCRAREntity,
    grid_context: List[List[Optional[PCRAREntity]]],
    alt_rule: PCRARRule,
    alt_params: RuleParams,
    k_h: int,
    k_v: int,
) -> bool:
    # 至少在最后一行或最后一列上形成一致的替代递推。
    try:
        row_ok = False
        if grid_context[2][1] is not None:
            row_target = apply_k(grid_context[2][1], alt_rule, alt_params, k_h)
            row_ok = entities_equal(row_target, candidate, check_obs=True)

        col_ok = False
        if grid_context[1][2] is not None:
            col_target = apply_k(grid_context[1][2], alt_rule, alt_params, k_v)
            col_ok = entities_equal(col_target, candidate, check_obs=True)
        return row_ok or col_ok
    except RuntimeError:
        return False
