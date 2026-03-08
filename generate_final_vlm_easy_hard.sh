#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash generate_final_vlm_easy_hard.sh
#   bash generate_final_vlm_easy_hard.sh "/path/to/final_vlm"
#
# Output layout:
#   final_vlm/
#     Progression/{easy,hard}/sample_*/
#     Cycle/{easy,hard}/sample_*/
#     Count/{easy,hard}/sample_*/
#     Conservation/{easy,hard}/sample_*/
#     Permutation/{easy,hard}/sample_*/
#     Symmetry/{easy,hard}/sample_*/

OUTPUT_ROOT="${1:-final_vlm}"

python - <<'PY' "$OUTPUT_ROOT"
import json
import shutil
import sys
from pathlib import Path

from raven3d.pcrar_dataset import PCRARConfig, PCRARDatasetGenerator
from raven3d.pcrar_rules import RuleTemplate

ROOT = Path(sys.argv[1]).resolve()
ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(1, 31))
PER_SEED = 10
MAX_TRIES_PER_SEED = PER_SEED * 200

RULES = [
    RuleTemplate.PROGRESSION,
    RuleTemplate.CYCLE,
    RuleTemplate.COUNT,
    RuleTemplate.CONSERVATION,
    RuleTemplate.PERMUTATION,
    RuleTemplate.SYMMETRY,
]

DIFFICULTIES = [
    ("easy", False),  # only target (2,2) missing
    ("hard", True),   # one missing per row (3 missing total incl. target)
]

summary = []

for rule in RULES:
    rule_dir = ROOT / rule.value
    rule_dir.mkdir(parents=True, exist_ok=True)

    for difficulty_name, missing_one_per_row in DIFFICULTIES:
        diff_dir = rule_dir / difficulty_name
        diff_dir.mkdir(parents=True, exist_ok=True)

        # Rebuild this split from scratch.
        for p in diff_dir.glob("sample_*"):
            if p.is_dir():
                shutil.rmtree(p)
        meta_path = diff_dir / "meta.json"
        if meta_path.exists():
            meta_path.unlink()

        entries = []
        sample_idx = 0

        for seed in SEEDS:
            cfg = PCRARConfig(
                rule_filter={rule},
                matrix_missing_one_per_row=missing_one_per_row,
            )
            gen = PCRARDatasetGenerator(config=cfg, seed=seed)

            made = 0
            tries = 0
            while made < PER_SEED:
                tries += 1
                if tries > MAX_TRIES_PER_SEED:
                    raise RuntimeError(
                        f"{rule.value}/{difficulty_name} seed={seed} generation failed: "
                        f"made={made}/{PER_SEED}, retries exceeded ({MAX_TRIES_PER_SEED})"
                    )
                try:
                    entry = gen.generate_sample(
                        output_root=diff_dir,
                        sample_index=sample_idx,
                        mode="matrix",
                    )
                except RuntimeError:
                    continue

                entries.append(entry)
                sample_idx += 1
                made += 1

            print(f"{rule.value}/{difficulty_name}: seed={seed} -> {made}")

        expected = len(SEEDS) * PER_SEED
        if sample_idx != expected:
            raise RuntimeError(
                f"{rule.value}/{difficulty_name} count mismatch: {sample_idx} != {expected}"
            )

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

        summary.append(
            {
                "rule": rule.value,
                "difficulty": difficulty_name,
                "samples": sample_idx,
                "seeds": [SEEDS[0], SEEDS[-1]],
                "per_seed": PER_SEED,
                "path": str(diff_dir),
            }
        )
        print(f"[DONE] {rule.value}/{difficulty_name}: {sample_idx}, dir={diff_dir}")

with (ROOT / "meta.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"[DONE] root summary: {len(summary)} splits, file={ROOT / 'meta.json'}")
print(f"All done. Output root: {ROOT}")
PY

echo "Quick count check:"
for r in Progression Cycle Count Conservation Permutation Symmetry; do
  for d in easy hard; do
    c="$(find "${OUTPUT_ROOT}/${r}/${d}" -maxdepth 1 -type d -name 'sample_*' | wc -l)"
    echo "${r}/${d}: ${c}"
  done
done
