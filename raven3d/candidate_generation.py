"""Candidate generation for matrix-style PCRAR tasks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .csg import OpType, PRIM_TYPE_CYCLE
from .matrix_grid import (
    MatrixLevelConfig,
    apply_k,
    can_apply_k,
    check_consistent_with_alt_relation,
    check_consistent_with_true_relation,
)
from .pcrar_entity import PCRAREntity, density_point_count, entities_equal, sample_random_entity
from .pcrar_rules import (
    CYCLE_AXIS_COLOR,
    CYCLE_AXIS_DENSITY,
    CYCLE_AXIS_SHAPE,
    CYCLE_AXIS_SIZE,
    CYCLE_DENSITY_INDICES,
    CYCLE_SIZE_LEVELS,
    PCRARRule,
    RuleParams,
    RuleTemplate,
    get_rule,
)


@dataclass
class CandidateMixConfig:
    num_options: int = 4
    min_analogical_wrong_relation: int = 1
    min_perceptual_plausible: int = 1


def _is_duplicate(entity: PCRAREntity, pool: Sequence[PCRAREntity]) -> bool:
    return any(entities_equal(entity, other, check_obs=True) for other in pool)


def _is_structure_duplicate(entity: PCRAREntity, pool: Sequence[PCRAREntity]) -> bool:
    return any(entities_equal(entity, other, check_obs=False) for other in pool)


def _neighbor_index(idx: int, n: int, direction: int = 1) -> int:
    nxt = idx + direction
    if nxt < 0:
        return 0
    if nxt >= n:
        return n - 1
    return nxt


def _perturb_from_gt(
    gt: PCRAREntity,
    level_cfg: MatrixLevelConfig,
    rng: np.random.Generator,
) -> PCRAREntity:
    out = gt.copy()
    leaves = out.get_leaves()
    methods = ["size", "delta", "pose", "slot", "shape"]
    rng.shuffle(methods)
    for method in methods:
        if method == "size" and leaves:
            idx = int(rng.integers(len(leaves)))
            cur = level_cfg.size_levels.index(leaves[idx].size_level)
            nxt = _neighbor_index(cur, len(level_cfg.size_levels), int(rng.choice([-1, 1])))
            if nxt != cur:
                leaves[idx].size_level = level_cfg.size_levels[nxt]
                return out
        elif method == "delta" and leaves:
            idx = int(rng.integers(len(leaves)))
            cur = level_cfg.delta_levels.index(leaves[idx].delta_level)
            nxt = _neighbor_index(cur, len(level_cfg.delta_levels), int(rng.choice([-1, 1])))
            if nxt != cur:
                leaves[idx].delta_level = level_cfg.delta_levels[nxt]
                return out
        elif method == "pose":
            pose = list(out.obs.global_pose_deg)
            axis = int(rng.integers(3))
            pose[axis] = int((pose[axis] + int(rng.choice([-60, 60]))) % 360)
            out.obs.global_pose_deg = tuple(pose)
            return out
        elif method == "slot" and leaves:
            idx = int(rng.integers(len(leaves)))
            cur = leaves[idx].slot
            candidates = [x for x in level_cfg.slot_levels if x != cur]
            if candidates:
                leaves[idx].slot = int(candidates[int(rng.integers(len(candidates)))])
                return out
        elif method == "shape" and leaves:
            idx = int(rng.integers(len(leaves)))
            cur = leaves[idx].prim_type
            cycle_idx = PRIM_TYPE_CYCLE.index(cur)
            leaves[idx].prim_type = PRIM_TYPE_CYCLE[(cycle_idx + 1) % len(PRIM_TYPE_CYCLE)]
            return out
    return out


def _perturb_cycle_shape_only(
    base: PCRAREntity,
    rng: np.random.Generator,
    step_choices: Sequence[int],
    max_changes: int,
) -> PCRAREntity:
    out = base.copy()
    leaves = out.get_leaves()
    if not leaves:
        return out

    cycle_len = len(PRIM_TYPE_CYCLE)
    valid_steps = [int(step) for step in step_choices if int(step) % cycle_len != 0]
    if not valid_steps:
        valid_steps = [1]

    limit = max(1, min(int(max_changes), len(leaves)))
    change_count = int(rng.integers(1, limit + 1))
    change_indices = rng.choice(len(leaves), size=change_count, replace=False)

    for idx in change_indices:
        leaf = leaves[int(idx)]
        cur = PRIM_TYPE_CYCLE.index(leaf.prim_type)
        step = int(rng.choice(valid_steps))
        leaf.prim_type = PRIM_TYPE_CYCLE[(cur + step) % cycle_len]
    return out


def _make_irrelevant(
    context_anchor: PCRAREntity,
    rng: np.random.Generator,
) -> PCRAREntity:
    leaf_count = max(1, context_anchor.leaf_count())
    entity = sample_random_entity(rng, leaf_count=leaf_count, allowed_ops=[OpType.UNION])
    leaves = entity.get_leaves()
    if leaves:
        leaves[0].prim_type = PRIM_TYPE_CYCLE[(PRIM_TYPE_CYCLE.index(leaves[0].prim_type) + 2) % len(PRIM_TYPE_CYCLE)]
    return entity


def _align_candidate_color(candidate: PCRAREntity, gt: PCRAREntity) -> None:
    """Keep candidate color aligned with GT when color is not the tested rule factor."""
    candidate.obs.color_preset_idx = int(gt.obs.color_preset_idx)


def _shares_gt_count(candidate: PCRAREntity, gt: PCRAREntity, true_rule: PCRARRule) -> bool:
    return true_rule.template == RuleTemplate.COUNT and candidate.leaf_count() == gt.leaf_count()


def _perturb_cycle_distribute_three(
    gt: PCRAREntity,
    params: RuleParams,
    level_cfg: MatrixLevelConfig,
    rng: np.random.Generator,
    strong: bool,
) -> PCRAREntity:
    out = gt.copy()
    leaves = out.get_leaves()
    axis = params.axis
    if axis == CYCLE_AXIS_SHAPE:
        steps = (2,) if strong else (-1, 1)
        max_changes = len(leaves) if strong else 1
        return _perturb_cycle_shape_only(out, rng, step_choices=steps, max_changes=max_changes)

    if axis == CYCLE_AXIS_SIZE and leaves:
        choices = [x for x in CYCLE_SIZE_LEVELS if x != leaves[0].size_level]
        if choices:
            target = choices[int(rng.integers(len(choices)))]
            for leaf in leaves:
                leaf.size_level = target
        return out

    if axis == CYCLE_AXIS_DENSITY:
        choices = [int(x) for x in CYCLE_DENSITY_INDICES if int(x) != int(out.obs.density_preset_idx)]
        if choices:
            target = int(choices[int(rng.integers(len(choices)))])
            out.obs.density_preset_idx = target
            out.obs.n_points = density_point_count(target)
        return out

    if axis == CYCLE_AXIS_COLOR:
        cur = int(out.obs.color_preset_idx)
        choices = [x for x in (0, 1, 2) if x != cur]
        if choices:
            if strong and len(choices) > 1:
                out.obs.color_preset_idx = int(choices[-1])
            else:
                out.obs.color_preset_idx = int(choices[int(rng.integers(len(choices)))])
        return out

    return _perturb_from_gt(gt, level_cfg, rng)


def _sample_count_only_candidate(
    grid_context: List[List[Optional[PCRAREntity]]],
    gt: PCRAREntity,
    rng: np.random.Generator,
) -> Optional[PCRAREntity]:
    """Generate a candidate using only Count-rule transitions."""
    count_rule = get_rule(RuleTemplate.COUNT)

    sources: List[PCRAREntity] = [gt]
    for row in grid_context:
        for item in row:
            if item is not None:
                sources.append(item)
    rng.shuffle(sources)

    for src in sources:
        for _ in range(12):
            direction = int(rng.choice([-1, 1]))
            steps = int(rng.choice([1, 2]))
            params = RuleParams(
                template=RuleTemplate.COUNT,
                axis="count",
                direction=direction,
            )
            if not can_apply_k(src, count_rule, params, steps):
                continue
            try:
                return apply_k(src, count_rule, params, steps)
            except RuntimeError:
                continue
    return None


def _enumerate_count_only_candidates(
    grid_context: List[List[Optional[PCRAREntity]]],
    gt: PCRAREntity,
    rng: np.random.Generator,
) -> List[PCRAREntity]:
    """Enumerate unique candidates produced only by Count transitions."""
    count_rule = get_rule(RuleTemplate.COUNT)
    sources: List[PCRAREntity] = [gt]
    for row in grid_context:
        for item in row:
            if item is not None:
                sources.append(item)

    out: List[PCRAREntity] = []
    for src in sources:
        for direction in (-1, 1):
            for steps in (1, 2):
                params = RuleParams(
                    template=RuleTemplate.COUNT,
                    axis="count",
                    direction=direction,
                )
                if not can_apply_k(src, count_rule, params, steps):
                    continue
                try:
                    cand = apply_k(src, count_rule, params, steps)
                except RuntimeError:
                    continue
                if _is_duplicate(cand, out):
                    continue
                out.append(cand)
    rng.shuffle(out)
    return out


def _sample_alt_rule_candidate(
    grid_context: List[List[Optional[PCRAREntity]]],
    true_rule: PCRARRule,
    true_params: RuleParams,
    gt: PCRAREntity,
    k_h: int,
    k_v: int,
    rng: np.random.Generator,
    template_whitelist: Optional[Sequence[RuleTemplate]] = None,
    true_rule_v: Optional[PCRARRule] = None,
    true_params_v: Optional[RuleParams] = None,
) -> Optional[Tuple[PCRAREntity, PCRARRule, RuleParams]]:
    anchor = grid_context[2][1]
    if anchor is None:
        return None

    templates = list(RuleTemplate)
    if template_whitelist:
        preferred = [t for t in template_whitelist if t in templates]
        others = [t for t in templates if t not in preferred]
        templates = preferred + others
    if true_rule.template in templates:
        templates = [true_rule.template] + [t for t in templates if t != true_rule.template]

    for tpl in templates:
        alt_rule = get_rule(tpl)
        for _ in range(180):
            probe = anchor
            if probe is None:
                continue
            alt_params = alt_rule.sample_params(rng, probe)
            if tpl == true_rule.template and alt_params.to_dict() == true_params.to_dict():
                continue

            if not can_apply_k(anchor, alt_rule, alt_params, k_h):
                continue
            try:
                candidate = apply_k(anchor, alt_rule, alt_params, k_h)
            except RuntimeError:
                continue
            if entities_equal(candidate, gt, check_obs=True):
                continue
            if check_consistent_with_true_relation(
                candidate,
                grid_context,
                true_rule,
                true_params,
                k_h,
                k_v,
                vertical_rule=true_rule_v,
                vertical_params=true_params_v,
            ):
                continue
            if not check_consistent_with_alt_relation(candidate, grid_context, alt_rule, alt_params, k_h, k_v):
                continue
            return candidate, alt_rule, alt_params

    return None


def _parse_alt_spec(spec: Dict[str, Any]) -> Tuple[PCRARRule, RuleParams]:
    template = RuleTemplate(spec["rule_template"])
    params_dict = dict(spec["rule_params"])
    params_dict["template"] = template
    return get_rule(template), RuleParams(**params_dict)


def _matches_any_alt(
    candidate: PCRAREntity,
    grid_context: List[List[Optional[PCRAREntity]]],
    alt_specs: Sequence[Dict[str, Any]],
    k_h: int,
    k_v: int,
) -> bool:
    for spec in alt_specs:
        alt_rule, alt_params = _parse_alt_spec(spec)
        if check_consistent_with_alt_relation(candidate, grid_context, alt_rule, alt_params, k_h, k_v):
            return True
    return False


def generate_candidates(
    grid_context: List[List[Optional[PCRAREntity]]],
    gt_entity: PCRAREntity,
    true_rule: PCRARRule,
    true_params: RuleParams,
    k_h: int,
    k_v: int,
    num_options: int,
    mix_cfg: CandidateMixConfig,
    level_cfg: MatrixLevelConfig,
    rng: np.random.Generator,
    rule_whitelist: Optional[Sequence[RuleTemplate]] = None,
    true_rule_v: Optional[PCRARRule] = None,
    true_params_v: Optional[RuleParams] = None,
) -> Dict[str, Any]:
    if num_options < 4:
        raise ValueError("num_options must be >= 4 for semantic layering")

    is_cycle_color_task = true_rule.template == RuleTemplate.CYCLE and true_params.axis == CYCLE_AXIS_COLOR
    lock_color_for_analogical = true_rule.template != RuleTemplate.CYCLE or is_cycle_color_task
    lock_color_for_perceptual = true_rule.template != RuleTemplate.CYCLE
    lock_color_for_irrelevant = true_rule.template != RuleTemplate.CYCLE or is_cycle_color_task

    candidates: List[PCRAREntity] = []
    candidate_types: List[str] = []
    distractor_notes: List[str] = []
    alt_specs: List[Dict[str, Any]] = []
    enforce_structure_diversity = true_rule.template == RuleTemplate.PERMUTATION

    # 1) analogical-but-wrong-relation
    analogical_added = 0
    attempts = 0
    alt_template_whitelist = list(rule_whitelist) if rule_whitelist is not None else None
    if true_rule.template == RuleTemplate.COUNT:
        alt_template_whitelist = [RuleTemplate.COUNT]
    elif true_rule.template == RuleTemplate.CYCLE:
        alt_template_whitelist = [RuleTemplate.CYCLE]
    while analogical_added < mix_cfg.min_analogical_wrong_relation and attempts < 200:
        attempts += 1
        sampled = _sample_alt_rule_candidate(
            grid_context,
            true_rule,
            true_params,
            gt_entity,
            k_h,
            k_v,
            rng,
            template_whitelist=alt_template_whitelist,
            true_rule_v=true_rule_v,
            true_params_v=true_params_v,
        )
        if sampled is None:
            continue
        cand, alt_rule, alt_params = sampled
        if lock_color_for_analogical:
            _align_candidate_color(cand, gt_entity)
        if entities_equal(cand, gt_entity, check_obs=True):
            continue
        if _shares_gt_count(cand, gt_entity, true_rule):
            continue
        if _is_duplicate(cand, candidates):
            continue
        if enforce_structure_diversity and (
            entities_equal(cand, gt_entity, check_obs=False)
            or _is_structure_duplicate(cand, candidates)
        ):
            continue
        candidates.append(cand)
        candidate_types.append("analogical_wrong_relation")
        if is_cycle_color_task:
            distractor_notes.append("符合替代关系 T'（忽略颜色维度），但不符合真实关系 T")
        else:
            distractor_notes.append("符合替代关系 T'，但不符合真实关系 T")
        alt_specs.append(
            {
                "rule_template": alt_rule.template.value,
                "rule_params": alt_params.to_dict(),
            }
        )
        analogical_added += 1

    if analogical_added < mix_cfg.min_analogical_wrong_relation:
        raise RuntimeError("Failed to generate analogical_wrong_relation candidate")

    # 2) perceptual-plausible
    perceptual_added = 0
    for _ in range(120):
        if perceptual_added >= mix_cfg.min_perceptual_plausible:
            break
        perceptual_is_count = False
        cand: Optional[PCRAREntity] = None

        # Add a count-based perceptual distractor path for non-Count tasks.
        if true_rule.template != RuleTemplate.COUNT:
            cand = _sample_count_only_candidate(grid_context, gt_entity, rng)
            perceptual_is_count = cand is not None

        if cand is None:
            if true_rule.template == RuleTemplate.COUNT:
                cand = _sample_count_only_candidate(grid_context, gt_entity, rng)
                if cand is None:
                    continue
                perceptual_is_count = True
            elif true_rule.template == RuleTemplate.CYCLE:
                cand = _perturb_cycle_distribute_three(
                    gt_entity,
                    true_params,
                    level_cfg,
                    rng,
                    strong=False,
                )
            else:
                cand = _perturb_from_gt(gt_entity, level_cfg, rng)
        if lock_color_for_perceptual:
            _align_candidate_color(cand, gt_entity)
        if entities_equal(cand, gt_entity, check_obs=True) or _is_duplicate(cand, candidates):
            continue
        if _shares_gt_count(cand, gt_entity, true_rule):
            continue
        if enforce_structure_diversity and (
            entities_equal(cand, gt_entity, check_obs=False)
            or _is_structure_duplicate(cand, candidates)
        ):
            continue
        if check_consistent_with_true_relation(
            cand,
            grid_context,
            true_rule,
            true_params,
            k_h,
            k_v,
            vertical_rule=true_rule_v,
            vertical_params=true_params_v,
        ):
            continue
        if _matches_any_alt(cand, grid_context, alt_specs, k_h, k_v):
            continue
        candidates.append(cand)
        candidate_types.append("perceptual_plausible")
        if perceptual_is_count:
            distractor_notes.append("数量干扰项：仅改变几何体数量，不满足真实/替代关系")
        elif true_rule.template == RuleTemplate.CYCLE:
            axis = true_params.axis
            if axis == CYCLE_AXIS_DENSITY:
                distractor_notes.append("密度干扰项：仅改变密度档位，不满足真实/替代关系")
            elif axis == CYCLE_AXIS_SIZE:
                distractor_notes.append("尺寸干扰项：仅改变尺寸档位，不满足真实/替代关系")
            elif axis == CYCLE_AXIS_COLOR:
                distractor_notes.append("颜色干扰项：仅改变颜色档位，不满足真实/替代关系")
            else:
                distractor_notes.append("形状干扰项：仅改变 primitive_type，不满足真实/替代关系")
        else:
            distractor_notes.append("外观上接近目标格，但不满足真实/替代关系")
        perceptual_added += 1

    if perceptual_added < mix_cfg.min_perceptual_plausible:
        raise RuntimeError("Failed to generate perceptual_plausible candidate")

    # 3) irrelevant
    max_attempts = 300
    attempts = 0
    while len(candidates) < num_options - 1 and attempts < max_attempts:
        attempts += 1
        if true_rule.template == RuleTemplate.COUNT:
            cand = _sample_count_only_candidate(grid_context, gt_entity, rng)
            if cand is None:
                continue
        elif true_rule.template == RuleTemplate.CYCLE:
            if is_cycle_color_task:
                # For color-cycle tasks, keep non-perceptual distractor colors aligned with GT.
                cand = _perturb_from_gt(gt_entity, level_cfg, rng)
            else:
                cand = _perturb_cycle_distribute_three(
                    gt_entity,
                    true_params,
                    level_cfg,
                    rng,
                    strong=True,
                )
        else:
            cand = _make_irrelevant(grid_context[0][0] or gt_entity, rng)
        if lock_color_for_irrelevant:
            _align_candidate_color(cand, gt_entity)
        if _is_duplicate(cand, candidates) or entities_equal(cand, gt_entity, check_obs=True):
            continue
        if _shares_gt_count(cand, gt_entity, true_rule):
            continue
        if enforce_structure_diversity and (
            entities_equal(cand, gt_entity, check_obs=False)
            or _is_structure_duplicate(cand, candidates)
        ):
            continue
        if check_consistent_with_true_relation(
            cand,
            grid_context,
            true_rule,
            true_params,
            k_h,
            k_v,
            vertical_rule=true_rule_v,
            vertical_params=true_params_v,
        ):
            continue
        if _matches_any_alt(cand, grid_context, alt_specs, k_h, k_v):
            continue
        candidates.append(cand)
        candidate_types.append("irrelevant")
        if true_rule.template == RuleTemplate.COUNT:
            distractor_notes.append("数量干扰项：仅改变几何体数量")
        elif true_rule.template == RuleTemplate.CYCLE:
            axis = true_params.axis
            if axis == CYCLE_AXIS_DENSITY:
                distractor_notes.append("密度干扰项：仅改变密度档位")
            elif axis == CYCLE_AXIS_SIZE:
                distractor_notes.append("尺寸干扰项：仅改变尺寸档位")
            elif axis == CYCLE_AXIS_COLOR:
                if is_cycle_color_task:
                    distractor_notes.append("非颜色干扰项：颜色与 GT 一致")
                else:
                    distractor_notes.append("颜色干扰项：仅改变颜色档位")
            else:
                distractor_notes.append("形状干扰项：仅改变 primitive_type")
        else:
            distractor_notes.append("与目标域风格或关系明显不一致")

    if len(candidates) < num_options - 1:
        if true_rule.template == RuleTemplate.COUNT:
            for cand in _enumerate_count_only_candidates(grid_context, gt_entity, rng):
                if len(candidates) >= num_options - 1:
                    break
                if lock_color_for_irrelevant:
                    _align_candidate_color(cand, gt_entity)
                if _is_duplicate(cand, candidates) or entities_equal(cand, gt_entity, check_obs=True):
                    continue
                if _shares_gt_count(cand, gt_entity, true_rule):
                    continue
                if check_consistent_with_true_relation(
                    cand,
                    grid_context,
                    true_rule,
                    true_params,
                    k_h,
                    k_v,
                    vertical_rule=true_rule_v,
                    vertical_params=true_params_v,
                ):
                    continue
                candidates.append(cand)
                candidate_types.append("irrelevant")
                distractor_notes.append("数量干扰项：仅改变几何体数量")
        if len(candidates) < num_options - 1:
            raise RuntimeError("Failed to generate enough irrelevant candidates")

    # insert GT and shuffle
    gt_index = int(rng.integers(num_options))
    all_candidates: List[PCRAREntity] = []
    all_types: List[str] = []
    all_notes: List[str] = []
    d = 0
    for i in range(num_options):
        if i == gt_index:
            all_candidates.append(gt_entity)
            all_types.append("gt")
            all_notes.append("真实关系 T 的唯一正确补全")
        else:
            all_candidates.append(candidates[d])
            all_types.append(candidate_types[d])
            all_notes.append(distractor_notes[d])
            d += 1

    # hard uniqueness/consistency check
    true_hits = [
        i
        for i, cand in enumerate(all_candidates)
        if check_consistent_with_true_relation(
            cand,
            grid_context,
            true_rule,
            true_params,
            k_h,
            k_v,
            vertical_rule=true_rule_v,
            vertical_params=true_params_v,
        )
    ]
    if true_hits != [gt_index]:
        raise RuntimeError(f"Expected exactly one true-relation candidate, got hits={true_hits}")

    if is_cycle_color_task:
        analogical_count = sum(1 for ctype in all_types if ctype == "analogical_wrong_relation")
        if analogical_count < mix_cfg.min_analogical_wrong_relation:
            raise RuntimeError("Not enough analogical_wrong_relation distractors")
    else:
        alt_hit_count = 0
        for i, cand in enumerate(all_candidates):
            if i == gt_index:
                continue
            if _matches_any_alt(cand, grid_context, alt_specs, k_h, k_v):
                alt_hit_count += 1
        if alt_hit_count < mix_cfg.min_analogical_wrong_relation:
            raise RuntimeError("Not enough alt-relation-consistent distractors")

    if enforce_structure_diversity:
        for i in range(len(all_candidates)):
            for j in range(i + 1, len(all_candidates)):
                if entities_equal(all_candidates[i], all_candidates[j], check_obs=False):
                    raise RuntimeError("Permutation candidates have duplicate CSG structures")

    return {
        "candidates": all_candidates,
        "gt_index": gt_index,
        "distractor_types": all_types,
        "candidate_notes": all_notes,
        "alt_relations": alt_specs,
    }
