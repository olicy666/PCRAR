#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash generate_pcrar_6rules.sh
#   bash generate_pcrar_6rules.sh "/home/xfy/demos/PCRAR_model/final 1.8k"
#   bash generate_pcrar_6rules.sh "/home/xfy/demos/PCRAR_model/final 1.8k" Conservation
#
# This script generates:
# - seeds: 1..30
# - per seed per rule: 10 samples
# - total per rule: 300 samples
# - total rules: 6 (Progression, Cycle, Count, Conservation, Permutation, Symmetry)

OUTPUT_ROOT="${1:-/home/xfy/demos/PCRAR_model/final 1.8k}"
START_RULE="${2:-Progression}"

python - <<'PY' "$OUTPUT_ROOT" "$START_RULE"
import json
import shutil
import sys
from pathlib import Path

from raven3d.pcrar_dataset import PCRARConfig, PCRARDatasetGenerator
from raven3d.pcrar_rules import RuleTemplate

ROOT = Path(sys.argv[1])
START_RULE = str(sys.argv[2]).strip()
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

ROOT.mkdir(parents=True, exist_ok=True)
all_entries = []
rule_names = [r.value for r in RULES]
if START_RULE not in rule_names:
    raise RuntimeError(f"Invalid START_RULE={START_RULE}, valid={rule_names}")
start_idx = rule_names.index(START_RULE)

for idx, rule in enumerate(RULES):
    rule_dir = ROOT / rule.value
    rule_dir.mkdir(parents=True, exist_ok=True)
    meta_path = rule_dir / "meta.json"

    if idx < start_idx:
        # Keep finished rules untouched and only read existing meta for root summary.
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                keep_entries = json.load(f)
            if isinstance(keep_entries, list):
                all_entries.extend(keep_entries)
                print(f"[SKIP] {rule.value}: keep existing {len(keep_entries)}")
            else:
                raise RuntimeError(f"{meta_path} is not a list")
        else:
            print(f"[SKIP] {rule.value}: meta.json not found")
        continue

    # Clean old outputs under current rule dir only.
    for p in rule_dir.glob("sample_*"):
        if p.is_dir():
            shutil.rmtree(p)
    if meta_path.exists():
        meta_path.unlink()

    entries = []
    sample_idx = 0

    for seed in SEEDS:
        cfg = PCRARConfig(rule_filter={rule})
        gen = PCRARDatasetGenerator(config=cfg, seed=seed)

        made = 0
        tries = 0
        while made < PER_SEED:
            tries += 1
            if tries > PER_SEED * MAX_TRIES_PER_SAMPLE:
                raise RuntimeError(
                    f"{rule.value} seed={seed} generation failed: retries exceeded, made={made}/{PER_SEED}"
                )
            try:
                entry = gen.generate_sample(
                    output_root=rule_dir,
                    sample_index=sample_idx,
                    mode="matrix",
                )
                entries.append(entry)
                all_entries.append(entry)
                sample_idx += 1
                made += 1
            except RuntimeError:
                continue

        print(f"{rule.value}: seed={seed} -> {made}")

    expected = len(SEEDS) * PER_SEED
    if sample_idx != expected:
        raise RuntimeError(f"{rule.value} count mismatch: {sample_idx} != {expected}")

    with (rule_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"[DONE] {rule.value}: {sample_idx}, dir={rule_dir}")

expected_total = len(RULES) * len(SEEDS) * PER_SEED
if START_RULE == rule_names[0] and len(all_entries) != expected_total:
    raise RuntimeError(f"root meta count mismatch: {len(all_entries)} != {expected_total}")

with (ROOT / "meta.json").open("w", encoding="utf-8") as f:
    json.dump(all_entries, f, ensure_ascii=False, indent=2)

print(f"[DONE] root meta: {len(all_entries)}, file={ROOT / 'meta.json'}")
print(f"All done. Output root: {ROOT}")
PY

echo "Quick count check:"
for r in Progression Cycle Count Conservation Permutation Symmetry; do
  c="$(find "${OUTPUT_ROOT}/${r}" -maxdepth 1 -type d -name 'sample_*' | wc -l)"
  echo "${r}: ${c}"
done
