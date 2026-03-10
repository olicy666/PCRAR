#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash generate_pcrar_final_point.sh
#   bash generate_pcrar_final_point.sh "/home/xfy/demos/PCRAR_model/final_point"
#
# Output layout (full_2000-like for each split):
#   <OUTPUT_ROOT>/easy/{Rule}/sample_******/...
#   <OUTPUT_ROOT>/easy/{Rule}/meta.json
#   <OUTPUT_ROOT>/easy/meta.json
#   <OUTPUT_ROOT>/hard/{Rule}/sample_******/...
#   <OUTPUT_ROOT>/hard/{Rule}/meta.json
#   <OUTPUT_ROOT>/hard/meta.json
#
# Generation plan:
# - rules: Progression, Distribute-three, Count, Conservation, Permutation, Symmetry
# - seeds: 1..30
# - per seed per rule: 10 samples
# - total per rule per split: 300

OUTPUT_ROOT="${1:-/home/xfy/demos/PCRAR_model/final_point}"

python - <<'PY' "$OUTPUT_ROOT"
import json
import shutil
import sys
from pathlib import Path

from raven3d.pcrar_dataset import PCRARConfig, PCRARDatasetGenerator
from raven3d.pcrar_rules import RuleTemplate

ROOT = Path(sys.argv[1])
SEEDS = list(range(1, 31))
PER_SEED = 10
MAX_TRIES_PER_SAMPLE = 200

RULES = [
    RuleTemplate.PROGRESSION,
    RuleTemplate.CYCLE,
    RuleTemplate.COUNT,
    RuleTemplate.CONSERVATION,
    RuleTemplate.PERMUTATION,
    RuleTemplate.SYMMETRY,
]

SPLITS = {
    "easy": False,  # matrix_missing_one_per_row=False -> only target cell missing
    "hard": True,   # matrix_missing_one_per_row=True  -> one missing per row (incl. target)
}

ROOT.mkdir(parents=True, exist_ok=True)

for split_name, missing_one_per_row in SPLITS.items():
    split_dir = ROOT / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    split_entries = []

    for rule in RULES:
        rule_dir = split_dir / rule.value
        rule_dir.mkdir(parents=True, exist_ok=True)

        # Rebuild this rule directory from scratch.
        for p in rule_dir.glob("sample_*"):
            if p.is_dir():
                shutil.rmtree(p)
        meta_path = rule_dir / "meta.json"
        if meta_path.exists():
            meta_path.unlink()

        entries = []
        sample_idx = 0

        for seed in SEEDS:
            cfg = PCRARConfig(
                rule_filter={rule},
                matrix_missing_one_per_row=missing_one_per_row,
                generate_confusing_view=False,
            )
            gen = PCRARDatasetGenerator(config=cfg, seed=seed)

            made = 0
            tries = 0
            while made < PER_SEED:
                tries += 1
                if tries > PER_SEED * MAX_TRIES_PER_SAMPLE:
                    raise RuntimeError(
                        f"{split_name}/{rule.value} seed={seed} generation failed: "
                        f"retries exceeded, made={made}/{PER_SEED}"
                    )
                try:
                    entry = gen.generate_sample(
                        output_root=rule_dir,
                        sample_index=sample_idx,
                        mode="matrix",
                    )
                    entries.append(entry)
                    split_entries.append(entry)
                    sample_idx += 1
                    made += 1
                except RuntimeError:
                    continue

            print(f"{split_name}/{rule.value}: seed={seed} -> {made}")

        expected_rule_total = len(SEEDS) * PER_SEED
        if sample_idx != expected_rule_total:
            raise RuntimeError(
                f"{split_name}/{rule.value} count mismatch: {sample_idx} != {expected_rule_total}"
            )

        with (rule_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

        print(f"[DONE] {split_name}/{rule.value}: {sample_idx}, dir={rule_dir}")

    expected_split_total = len(RULES) * len(SEEDS) * PER_SEED
    if len(split_entries) != expected_split_total:
        raise RuntimeError(
            f"{split_name} meta count mismatch: {len(split_entries)} != {expected_split_total}"
        )

    with (split_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(split_entries, f, ensure_ascii=False, indent=2)

    print(f"[DONE] {split_name}/meta.json: {len(split_entries)} entries")

print(f"All done. Output root: {ROOT}")
PY

echo "Quick count check:"
for split in easy hard; do
  echo "--- ${split} ---"
  for r in Progression "Distribute-three" Count Conservation Permutation Symmetry; do
    c="$(find "${OUTPUT_ROOT}/${split}/${r}" -maxdepth 1 -type d -name 'sample_*' | wc -l)"
    echo "${r}: ${c}"
  done
done
