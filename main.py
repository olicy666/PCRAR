from __future__ import annotations

import argparse
from typing import List

from raven3d.dataset import DatasetGenerator, GenerationConfig
from raven3d.factory import create_default_registry
from raven3d.rules.groups import list_available_modes, rules_for_mode, validate_rule_ids
from raven3d.rules.base import RuleDifficulty


def get_all_modes() -> List[str]:
    """获取所有可用模式，包括 pcrar"""
    modes = list_available_modes()
    modes.append("pcrar")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate three-step 3D reasoning samples.")
    parser.add_argument("--output", type=str, default="output", help="Output directory for generated samples")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--points", type=int, default=8192, help="Number of points per point cloud (default: 8192 for pcrar, 4096 for legacy)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--simple-prob", type=float, default=0.7, help="Deprecated (rule sampling is uniform)")
    parser.add_argument("--medium-prob", type=float, default=0.2, help="Deprecated (rule sampling is uniform)")
    parser.add_argument("--complex-prob", type=float, default=0.1, help="Deprecated (rule sampling is uniform)")
    parser.add_argument(
        "--mode",
        type=str.lower,
        default="pcrar",
        choices=get_all_modes(),
        help="Mode: pcrar (default, CSG-based) / main / r1-only / r2-only / r3-only / r4-only / all-minus-r1 / etc.",
    )
    parser.add_argument(
        "--rules",
        type=str,
        default=None,
        help="Comma-separated rule IDs (e.g., R1-1,R2-3,R3-2). When provided, overrides --mode. For legacy mode only.",
    )
    # PCRAR 专用参数
    parser.add_argument(
        "--task-mix",
        type=float,
        default=0.5,
        help="PCRAR: Ratio of relational tasks (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--leaf-count-min",
        type=int,
        default=2,
        help="PCRAR: Minimum leaf count (default: 2)",
    )
    parser.add_argument(
        "--leaf-count-max",
        type=int,
        default=3,
        help="PCRAR: Maximum leaf count (default: 3)",
    )
    parser.add_argument(
        "--pcrar-rules",
        type=str,
        default=None,
        help="PCRAR: Comma-separated rule templates (e.g., Progression,Cycle,Copy). When provided, filters rules.",
    )
    return parser.parse_args()


def main_pcrar(args: argparse.Namespace) -> None:
    """运行 PCRAR 模式"""
    from raven3d.pcrar_dataset import PCRARDatasetGenerator, PCRARConfig
    from raven3d.pcrar_rules import RuleTemplate
    
    # 解析规则过滤
    rule_filter = None
    if args.pcrar_rules:
        rule_names = [r.strip() for r in args.pcrar_rules.split(",")]
        rule_filter = set()
        for name in rule_names:
            try:
                rule_filter.add(RuleTemplate(name))
            except ValueError:
                valid_rules = [t.value for t in RuleTemplate]
                raise SystemExit(f"Invalid PCRAR rule '{name}'. Valid rules: {', '.join(valid_rules)}")
    
    config = PCRARConfig(
        n_points=args.points,
        task_mix=args.task_mix,
        leaf_count_min=args.leaf_count_min,
        leaf_count_max=args.leaf_count_max,
        rule_filter=rule_filter,
    )
    
    generator = PCRARDatasetGenerator(config=config, seed=args.seed)
    generator.generate_dataset(args.output, args.num_samples)


def main_legacy(args: argparse.Namespace) -> None:
    """运行传统模式"""
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
        main_pcrar(args)
    else:
        main_legacy(args)


if __name__ == "__main__":
    main()
