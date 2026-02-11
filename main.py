from __future__ import annotations

import argparse
from typing import List, Optional, Set, Tuple

from raven3d.dataset import DatasetGenerator, GenerationConfig
from raven3d.factory import create_default_registry
from raven3d.rules.base import RuleDifficulty
from raven3d.rules.groups import list_available_modes, rules_for_mode, validate_rule_ids


def get_all_modes() -> List[str]:
    modes = list_available_modes()
    modes.extend(["pcrar", "pcrar-legacy"])
    return modes


def _parse_csv_ints(text: str) -> Tuple[int, ...]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3D reasoning datasets.")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--points", type=int, default=8192, help="Point count per cloud")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument(
        "--mode",
        type=str.lower,
        default="pcrar",
        choices=get_all_modes(),
        help="pcrar (matrix default) / pcrar-legacy / legacy modes (main, r1-only, ...)",
    )

    # Legacy (non-PCRAR) options.
    parser.add_argument("--simple-prob", type=float, default=0.7, help="Legacy difficulty prob")
    parser.add_argument("--medium-prob", type=float, default=0.2, help="Legacy difficulty prob")
    parser.add_argument("--complex-prob", type=float, default=0.1, help="Legacy difficulty prob")
    parser.add_argument(
        "--rules",
        type=str,
        default=None,
        help="Legacy rule IDs, comma-separated (overrides --mode for legacy mode)",
    )

    # PCRAR matrix defaults.
    parser.add_argument("--num-options", type=int, default=4, help="PCRAR matrix option count")
    parser.add_argument("--k-h-choices", type=str, default="1,2", help="PCRAR matrix horizontal step choices")
    parser.add_argument("--k-v-choices", type=str, default="1,2", help="PCRAR matrix vertical step choices")
    parser.add_argument("--matrix-size-levels", type=int, default=7, help="PCRAR matrix size level count")
    parser.add_argument("--matrix-density-levels", type=int, default=5, help="PCRAR matrix density level count")
    parser.add_argument("--matrix-delta-levels", type=int, default=5, help="PCRAR matrix delta level count")
    parser.add_argument("--matrix-slot-levels", type=str, default="-1,0,1", help="PCRAR matrix slot levels")
    parser.add_argument(
        "--matrix-missing-one-per-row",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PCRAR matrix: hide one cell per row (3 missing total, including target at (2,2))",
    )
    parser.add_argument(
        "--generate-confusing-view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PCRAR: generate rule-aware rendered views including view_combined.png",
    )
    parser.add_argument(
        "--pcrar-rules",
        type=str,
        default=None,
        help="PCRAR rule templates, comma-separated (e.g., Progression,Cycle)",
    )

    return parser.parse_args()


def _parse_rule_filter(raw: Optional[str]):
    if not raw:
        return None
    from raven3d.pcrar_rules import RuleTemplate

    rule_filter: Set[RuleTemplate] = set()
    for name in [x.strip() for x in raw.split(",") if x.strip()]:
        try:
            rule_filter.add(RuleTemplate(name))
        except ValueError as exc:
            valid = ", ".join(t.value for t in RuleTemplate)
            raise SystemExit(f"Invalid PCRAR rule '{name}'. Valid rules: {valid}") from exc
    return rule_filter


def main_pcrar(args: argparse.Namespace, legacy: bool = False) -> None:
    from raven3d.pcrar_dataset import PCRARConfig, PCRARDatasetGenerator

    try:
        k_h_choices = _parse_csv_ints(args.k_h_choices)
        k_v_choices = _parse_csv_ints(args.k_v_choices)
        slot_levels = _parse_csv_ints(args.matrix_slot_levels)
    except ValueError as exc:
        raise SystemExit(f"Invalid matrix choices: {exc}") from exc

    config = PCRARConfig(
        n_points=args.points,
        num_options=args.num_options,
        matrix_k_h_choices=k_h_choices,
        matrix_k_v_choices=k_v_choices,
        matrix_size_levels=args.matrix_size_levels,
        matrix_density_levels=args.matrix_density_levels,
        matrix_delta_levels=args.matrix_delta_levels,
        matrix_slot_levels=slot_levels,
        matrix_missing_one_per_row=args.matrix_missing_one_per_row,
        generate_confusing_view=args.generate_confusing_view,
        rule_filter=_parse_rule_filter(args.pcrar_rules),
        legacy_enabled=legacy,
    )
    generator = PCRARDatasetGenerator(config=config, seed=args.seed)
    generator.generate_dataset(args.output, args.num_samples, mode="legacy" if legacy else "matrix")


def main_legacy(args: argparse.Namespace) -> None:
    probs = {
        RuleDifficulty.SIMPLE: args.simple_prob,
        RuleDifficulty.MEDIUM: args.medium_prob,
        RuleDifficulty.COMPLEX: args.complex_prob,
    }
    try:
        rule_filter = validate_rule_ids(args.rules.split(",")) if args.rules else rules_for_mode(args.mode)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = GenerationConfig(n_points=args.points, difficulty_probs=probs, rule_filter=rule_filter)
    registry = create_default_registry()
    generator = DatasetGenerator(registry, config=config, seed=args.seed)
    generator.generate_dataset(args.output, args.num_samples)
    mode_info = f"custom rules [{', '.join(sorted(rule_filter))}]" if args.rules else f"mode '{args.mode}'"
    print(f"Generated {args.num_samples} samples in {args.output} with {mode_info}")


def main() -> None:
    args = parse_args()
    if args.mode == "pcrar":
        main_pcrar(args, legacy=False)
    elif args.mode == "pcrar-legacy":
        main_pcrar(args, legacy=True)
    else:
        main_legacy(args)


if __name__ == "__main__":
    main()
