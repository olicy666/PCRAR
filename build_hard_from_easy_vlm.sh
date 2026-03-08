#!/usr/bin/env bash
set -euo pipefail

# Build hard split from easy split for final_vlm, without changing puzzle logic:
# - Keep the same underlying puzzle/candidates/answer
# - Only add two extra masked grid cells in rows 0 and 1
#
# Usage:
#   bash build_hard_from_easy_vlm.sh
#   bash build_hard_from_easy_vlm.sh /home/xfy/demos/PCRAR_model/final_vlm
#
# Optional:
#   RULES_CSV="Cycle,Count" bash build_hard_from_easy_vlm.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ROOT="${1:-/home/xfy/demos/PCRAR_model/final_vlm}"

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
from typing import Iterable, List, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def choose_rules(root: Path) -> List[Path]:
    raw = os.environ.get("RULES_CSV", "").strip()
    if raw:
        wanted = {x.strip() for x in raw.split(",") if x.strip()}
        out = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name in wanted]
    else:
        out = [p for p in sorted(root.iterdir()) if p.is_dir()]
    return [p for p in out if (p / "easy").is_dir()]


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


def remove_masked_from_rendered_paths(rendered_paths: List[str], missing: Set[Tuple[int, int]]) -> List[str]:
    if not isinstance(rendered_paths, list):
        return rendered_paths
    blocked = {f"view_grid_{r}_{c}.png" for r, c in missing}
    out = []
    for name in rendered_paths:
        if isinstance(name, str) and name in blocked:
            continue
        out.append(name)
    return out


def infer_tile_size(width: int, height: int) -> int:
    # from render_confusing_view.py
    # canvas_h = 4*tile + 2*gap + 3*pad + cand_label_h = 4*tile + 104
    t = (height - 104) / 4.0
    if abs(t - round(t)) > 1e-6:
        raise RuntimeError(f"Unexpected image height {height}, cannot infer tile size")
    tile = int(round(t))
    expected_w = 4 * tile + 62  # n_cand=4 in final_vlm
    if expected_w != width:
        raise RuntimeError(
            f"Unexpected image size ({width},{height}), expected width {expected_w} for tile={tile}"
        )
    return tile


def apply_hard_mask_to_combined(src: Path, dst: Path, missing: Set[Tuple[int, int]]) -> None:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    tile = infer_tile_size(w, h)
    pad = 16
    gap = 10
    matrix_w = 3 * tile + 2 * gap
    mx = (w - matrix_w) // 2
    my = pad

    draw = ImageDraw.Draw(img)
    try:
        q_font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(48, int(tile * 0.45)))
    except Exception:
        q_font = ImageFont.load_default()

    for r, c in sorted(missing):
        x0 = mx + c * (tile + gap)
        y0 = my + r * (tile + gap)
        blank = Image.new("RGB", (tile, tile), (245, 245, 245))
        bdraw = ImageDraw.Draw(blank)
        for yy in range(0, tile, 8):
            bdraw.line((0, yy, tile, yy), fill=(220, 220, 220), width=1)
        for xx in range(0, tile, 8):
            bdraw.line((xx, 0, xx, tile), fill=(220, 220, 220), width=1)
        img.paste(blank, (x0, y0))

        q_text = "?"
        q_bbox = draw.textbbox((0, 0), q_text, font=q_font)
        qw = q_bbox[2] - q_bbox[0]
        qh = q_bbox[3] - q_bbox[1]
        qx = x0 + (tile - qw) // 2
        qy = y0 + (tile - qh) // 2
        draw.text((qx, qy), q_text, fill=(70, 70, 70), font=q_font)

    img.save(dst)


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    rule_dirs = choose_rules(root)
    if not rule_dirs:
        raise SystemExit(f"No rule dirs with easy split found under: {root}")

    total_samples = 0
    for rule_dir in rule_dirs:
        easy_dir = rule_dir / "easy"
        hard_dir = rule_dir / "hard"

        if hard_dir.exists():
            shutil.rmtree(hard_dir)
        hard_dir.mkdir(parents=True, exist_ok=True)

        easy_meta_path = easy_dir / "meta.json"
        if not easy_meta_path.exists():
            raise RuntimeError(f"Missing easy meta: {easy_meta_path}")

        easy_entries = json.loads(easy_meta_path.read_text(encoding="utf-8"))
        if not isinstance(easy_entries, list):
            raise RuntimeError(f"Expected list meta in: {easy_meta_path}")

        hard_entries = []
        for entry in easy_entries:
            sample_id = str(entry.get("id", ""))
            if not re.match(r"^sample_\d{6}$", sample_id):
                raise RuntimeError(f"Invalid sample id in {easy_meta_path}: {sample_id!r}")

            easy_sample_dir = easy_dir / sample_id
            hard_sample_dir = hard_dir / sample_id
            hard_sample_dir.mkdir(parents=True, exist_ok=True)

            src_img = easy_sample_dir / "view_combined.png"
            if not src_img.exists():
                raise RuntimeError(f"Missing easy image: {src_img}")
            dst_img = hard_sample_dir / "view_combined.png"

            missing = choose_extra_missing(rule_dir.name, sample_id)
            apply_hard_mask_to_combined(src_img, dst_img, missing)

            meta = copy.deepcopy(entry)
            missing_sorted = [[r, c] for r, c in sorted(missing)]
            meta["missing_positions"] = missing_sorted
            meta["empty_grid_positions"] = missing_sorted
            meta["grid_observation_mask"] = build_obs_mask(missing)
            null_grid_slots(meta.get("grid_paths"), missing)

            conf = meta.get("confusing_view")
            if isinstance(conf, dict):
                null_grid_slots(conf.get("grid_view_paths"), missing)
                conf["rendered_paths"] = remove_masked_from_rendered_paths(conf.get("rendered_paths", []), missing)

            (hard_sample_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            hard_entries.append(meta)

        (hard_dir / "meta.json").write_text(
            json.dumps(hard_entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        total_samples += len(hard_entries)
        print(f"[DONE] {rule_dir.name}/hard: {len(hard_entries)} samples")

    print(f"All done. Total hard samples built: {total_samples}")


if __name__ == "__main__":
    main()
PY

echo "Quick check:"
for rule_dir in "$ROOT"/*; do
  [[ -d "$rule_dir/easy" ]] || continue
  h="$rule_dir/hard"
  c="$(find "$h" -maxdepth 1 -type d -name 'sample_*' | wc -l)"
  echo "$(basename "$rule_dir")/hard: $c"
done
