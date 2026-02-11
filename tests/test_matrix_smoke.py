from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from raven3d.matrix_grid import (
    apply_k,
    check_consistent_with_alt_relation,
    check_consistent_with_true_relation,
    check_path_consistency,
    grid_quality_checks,
)
from raven3d.pcrar_entity import PCRAREntity, entities_equal
from raven3d.pcrar_rules import RuleParams, RuleTemplate, get_rule


def _load_entries(dataset_dir: Path) -> List[Dict[str, Any]]:
    meta_path = dataset_dir / "meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def _to_rule_params(d: Dict[str, Any]) -> RuleParams:
    data = dict(d)
    data["template"] = RuleTemplate(data["template"])
    return RuleParams(**data)


def _to_grid(entry: Dict[str, Any]) -> List[List[PCRAREntity]]:
    rows = entry["entities"]["grid"]
    out: List[List[PCRAREntity]] = []
    for row in rows:
        out.append([PCRAREntity.from_dict(cell) for cell in row])
    return out


def _to_candidates(entry: Dict[str, Any]) -> List[PCRAREntity]:
    return [PCRAREntity.from_dict(x) for x in entry["entities"]["candidates"]]


def _build_context(grid: List[List[PCRAREntity]]) -> List[List[Optional[PCRAREntity]]]:
    ctx: List[List[Optional[PCRAREntity]]] = []
    for r in range(3):
        row: List[Optional[PCRAREntity]] = []
        for c in range(3):
            if (r, c) == (2, 2):
                row.append(None)
            else:
                row.append(grid[r][c].copy())
        ctx.append(row)
    return ctx


def validate_entry(entry: Dict[str, Any]) -> None:
    rule = get_rule(RuleTemplate(entry["rule_template"]))
    params = _to_rule_params(entry["rule_params"])
    k_h = int(entry["k_h"])
    k_v = int(entry["k_v"])

    grid = _to_grid(entry)
    candidates = _to_candidates(entry)
    gt_index = int(entry["gt_index"])
    distractor_types = entry.get("distractor_types", [])

    # 1) Exponent formula
    e00 = grid[0][0]
    for r in range(3):
        for c in range(3):
            expected = apply_k(e00, rule, params, r * k_v + c * k_h)
            if not entities_equal(expected, grid[r][c], check_obs=True):
                raise AssertionError(f"Exponent formula failed at ({r},{c})")

    # 2) Path consistency
    if not check_path_consistency(grid, rule, params, k_h, k_v):
        raise AssertionError("Path consistency failed")

    # 3) Adjacent differences and global diversity
    min_unique = 3 if rule.template in {RuleTemplate.CYCLE, RuleTemplate.COPY} else 5
    ok, reason = grid_quality_checks(grid, rule, params, k_h, k_v, min_unique=min_unique)
    if not ok:
        raise AssertionError(f"Grid quality check failed: {reason}")

    # 4) Unique true relation candidate
    grid_context = _build_context(grid)
    true_hits = [
        i
        for i, cand in enumerate(candidates)
        if check_consistent_with_true_relation(cand, grid_context, rule, params, k_h, k_v)
    ]
    if true_hits != [gt_index]:
        raise AssertionError(f"Expected unique true candidate at {gt_index}, got {true_hits}")

    # 5) At least one alt relation candidate
    alt_specs = entry.get("notes", {}).get("alt_relations", [])
    if not alt_specs:
        raise AssertionError("No alt_relations found")

    alt_hit_count = 0
    for i, cand in enumerate(candidates):
        if i == gt_index:
            continue
        matched = False
        for spec in alt_specs:
            alt_rule = get_rule(RuleTemplate(spec["rule_template"]))
            alt_params = _to_rule_params(spec["rule_params"])
            if check_consistent_with_alt_relation(cand, grid_context, alt_rule, alt_params, k_h, k_v):
                matched = True
                break
        if matched:
            alt_hit_count += 1

        ctype = distractor_types[i] if i < len(distractor_types) else ""
        if ctype == "analogical_wrong_relation" and not matched:
            raise AssertionError("analogical_wrong_relation candidate does not satisfy alt relation")
        if ctype == "perceptual_plausible" and matched:
            raise AssertionError("perceptual_plausible candidate unexpectedly satisfies alt relation")

    if alt_hit_count < 1:
        raise AssertionError("No distractor satisfies alt relation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for PCRAR matrix dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    entries = _load_entries(dataset_dir)
    for idx, entry in enumerate(entries):
        validate_entry(entry)
        print(f"[OK] {idx + 1}/{len(entries)} {entry.get('id', idx)}")

    print(f"Validated {len(entries)} matrix samples from {dataset_dir}")


if __name__ == "__main__":
    main()
