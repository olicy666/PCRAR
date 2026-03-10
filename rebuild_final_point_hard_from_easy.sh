#!/usr/bin/env bash
set -euo pipefail

# Rebuild final_point/hard from final_point/easy while preserving puzzle logic:
# - same underlying puzzle entities/candidates/answers
# - hard only adds two extra masked grid cells (row0 one + row1 one) plus target (2,2)
#
# Usage:
#   bash rebuild_final_point_hard_from_easy.sh
#   bash rebuild_final_point_hard_from_easy.sh /home/xfy/demos/PCRAR_model/final_point
#
# Optional:
#   RULES_CSV="Distribute-three,Count" bash rebuild_final_point_hard_from_easy.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ROOT="${1:-/home/xfy/demos/PCRAR_model/final_point}"

python - "$ROOT" <<'PY'
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def choose_rule_names(easy_root: Path) -> List[str]:
    raw = os.environ.get("RULES_CSV", "").strip()
    if raw:
        wanted = {x.strip() for x in raw.split(",") if x.strip()}
        names = [p.name for p in sorted(easy_root.iterdir()) if p.is_dir() and p.name in wanted]
    else:
        names = [p.name for p in sorted(easy_root.iterdir()) if p.is_dir()]
    return names


def choose_extra_missing(rule_name: str, sample_id: str) -> Set[Tuple[int, int]]:
    # Deterministic per rule/sample. Keep (1,2) and (2,1) visible.
    digest = hashlib.sha256(f"{rule_name}/{sample_id}".encode("utf-8")).digest()
    row0_col = int(digest[0] % 3)   # 0,1,2
    row1_col = int(digest[1] % 2)   # 0,1 only
    return {(0, row0_col), (1, row1_col), (2, 2)}


def build_obs_mask(missing: Set[Tuple[int, int]]) -> List[List[bool]]:
    return [[(r, c) not in missing for c in range(3)] for r in range(3)]


def null_grid_slots(grid_like, missing: Set[Tuple[int, int]]) -> None:
    if not isinstance(grid_like, list) or len(grid_like) < 3:
        return
    for r, c in missing:
        if 0 <= r < len(grid_like):
            row = grid_like[r]
            if isinstance(row, list) and 0 <= c < len(row):
                row[c] = None


def should_skip_grid_file(filename: str, missing: Set[Tuple[int, int]]) -> bool:
    m = re.fullmatch(r"grid_(\d)_(\d)\.ply", filename)
    if not m:
        return False
    r = int(m.group(1))
    c = int(m.group(2))
    return (r, c) in missing


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    easy_root = root / "easy"
    hard_root = root / "hard"
    if not easy_root.is_dir():
        raise SystemExit(f"Missing easy root: {easy_root}")

    rule_names = choose_rule_names(easy_root)
    if not rule_names:
        raise SystemExit(f"No rules found under: {easy_root}")

    if hard_root.exists():
        shutil.rmtree(hard_root)
    hard_root.mkdir(parents=True, exist_ok=True)

    hard_meta_by_key: Dict[Tuple[str, str], dict] = {}
    total_samples = 0

    for rule_name in rule_names:
        easy_rule_dir = easy_root / rule_name
        hard_rule_dir = hard_root / rule_name
        hard_rule_dir.mkdir(parents=True, exist_ok=True)

        easy_rule_meta_path = easy_rule_dir / "meta.json"
        if not easy_rule_meta_path.exists():
            raise RuntimeError(f"Missing easy rule meta: {easy_rule_meta_path}")
        easy_entries = json.loads(easy_rule_meta_path.read_text(encoding="utf-8"))
        if not isinstance(easy_entries, list):
            raise RuntimeError(f"Expected list meta: {easy_rule_meta_path}")

        hard_entries: List[dict] = []
        for entry in easy_entries:
            sample_id = str(entry.get("id", ""))
            if not re.fullmatch(r"sample_\d{6}", sample_id):
                raise RuntimeError(f"Invalid sample id in {easy_rule_meta_path}: {sample_id!r}")

            missing = choose_extra_missing(rule_name, sample_id)
            missing_sorted = [[r, c] for r, c in sorted(missing)]

            easy_sample_dir = easy_rule_dir / sample_id
            hard_sample_dir = hard_rule_dir / sample_id
            hard_sample_dir.mkdir(parents=True, exist_ok=True)

            # Copy point clouds except newly masked grid cells.
            for src in easy_sample_dir.iterdir():
                if src.name == "meta.json":
                    continue
                if src.is_file() and should_skip_grid_file(src.name, missing):
                    continue
                if src.is_file():
                    shutil.copy2(src, hard_sample_dir / src.name)

            meta = copy.deepcopy(entry)
            meta["missing_positions"] = missing_sorted
            meta["empty_grid_positions"] = missing_sorted
            meta["grid_observation_mask"] = build_obs_mask(missing)
            null_grid_slots(meta.get("grid_paths"), missing)

            (hard_sample_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            hard_entries.append(meta)
            hard_meta_by_key[(rule_name, sample_id)] = meta

        (hard_rule_dir / "meta.json").write_text(
            json.dumps(hard_entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        total_samples += len(hard_entries)
        print(f"[DONE] hard/{rule_name}: {len(hard_entries)} samples")

    # Build hard/meta.json with the same ordering as easy/meta.json when possible.
    easy_global_meta_path = easy_root / "meta.json"
    if easy_global_meta_path.exists():
        easy_global = json.loads(easy_global_meta_path.read_text(encoding="utf-8"))
        if isinstance(easy_global, list):
            ordered = []
            for e in easy_global:
                rule = str(e.get("rule_template", ""))
                sid = str(e.get("id", ""))
                item = hard_meta_by_key.get((rule, sid))
                if item is None:
                    raise RuntimeError(f"Missing generated hard entry for ({rule}, {sid})")
                ordered.append(item)
            (hard_root / "meta.json").write_text(
                json.dumps(ordered, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            raise RuntimeError(f"Expected list in {easy_global_meta_path}")
    else:
        # Fallback: concatenate by rule order.
        all_entries = []
        for rule_name in rule_names:
            p = hard_root / rule_name / "meta.json"
            all_entries.extend(json.loads(p.read_text(encoding="utf-8")))
        (hard_root / "meta.json").write_text(
            json.dumps(all_entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"All done. Total hard samples: {total_samples}")


if __name__ == "__main__":
    main()
PY

echo "Quick count check:"
for rule_dir in "$ROOT/hard"/*; do
  [[ -d "$rule_dir" ]] || continue
  c="$(find "$rule_dir" -maxdepth 1 -type d -name 'sample_*' | wc -l)"
  echo "hard/$(basename "$rule_dir"): $c"
done
