#!/usr/bin/env bash
set -euo pipefail

# Regenerate only Cycle rule for final_vlm (easy + hard), then keep only:
# - view_combined.png
# - meta.json
#
# Usage:
#   bash regenerate_cycle_vlm.sh
#   bash regenerate_cycle_vlm.sh /home/xfy/demos/PCRAR_model/final_vlm

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FINAL_VLM_ROOT="${1:-/home/xfy/demos/PCRAR_model/final_vlm}"
CYCLE_ROOT="${FINAL_VLM_ROOT%/}/Cycle"

python - "$CYCLE_ROOT" <<'PY'
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

DIFFICULTIES = [
    ("easy", False),  # only target (2,2) missing
    ("hard", True),   # one missing per row (3 missing total incl. target)
]

for difficulty_name, missing_one_per_row in DIFFICULTIES:
    diff_dir = ROOT / difficulty_name
    diff_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild split from scratch.
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
            rule_filter={RuleTemplate.CYCLE},
            matrix_missing_one_per_row=missing_one_per_row,
        )
        gen = PCRARDatasetGenerator(config=cfg, seed=seed)

        made = 0
        tries = 0
        while made < PER_SEED:
            tries += 1
            if tries > MAX_TRIES_PER_SEED:
                raise RuntimeError(
                    f"Cycle/{difficulty_name} seed={seed} generation failed: "
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

        print(f"Cycle/{difficulty_name}: seed={seed} -> {made}")

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Cycle/{difficulty_name}: {sample_idx}, dir={diff_dir}")
PY

# Keep only view_combined.png and meta.json under Cycle.
find "$CYCLE_ROOT" -type f ! \( -name 'view_combined.png' -o -name 'meta.json' \) -delete

echo "Quick count check:"
for d in easy hard; do
  c="$(find "${CYCLE_ROOT}/${d}" -maxdepth 1 -type d -name 'sample_*' | wc -l)"
  bad="$(find "${CYCLE_ROOT}/${d}" -type f ! \( -name 'view_combined.png' -o -name 'meta.json' \) | wc -l)"
  echo "Cycle/${d}: samples=${c}, non_target_files=${bad}"
done

echo "Done: ${CYCLE_ROOT}"
