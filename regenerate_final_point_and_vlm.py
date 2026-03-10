#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

from raven3d.pcrar_dataset import PCRARConfig, PCRARDatasetGenerator
from raven3d.pcrar_rules import RuleTemplate


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate / overwrite final_point and final_vlm with 300 easy + 300 hard "
            "samples per rule, where hard is derived from easy by adding extra PCRAR masks."
        )
    )
    parser.add_argument(
        "--point-root",
        type=Path,
        default=Path("/home/xfy/demos/PCRAR_model/final_point"),
        help="Output root for point-cloud-only dataset.",
    )
    parser.add_argument(
        "--vlm-root",
        type=Path,
        default=Path("/home/xfy/demos/PCRAR_model/final_vlm"),
        help="Output root for image-only dataset.",
    )
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=None,
        help="Optional parent directory for temporary full samples.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from already completed rules instead of clearing both output roots first.",
    )
    return parser.parse_args()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def remove_path_if_exists(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def choose_missing_positions(rule_name: str, sample_id: str) -> Set[Tuple[int, int]]:
    # Match current hard-from-easy behavior while staying deterministic across reruns.
    digest = hashlib.sha256(f"{rule_name}/{sample_id}".encode("utf-8")).digest()
    row0_col = int(digest[0] % 3)
    row1_col = int(digest[1] % 2)
    return {(0, row0_col), (1, row1_col), (2, 2)}


def build_observation_mask(missing: Set[Tuple[int, int]]) -> List[List[bool]]:
    return [[(r, c) not in missing for c in range(3)] for r in range(3)]


def null_grid_slots(grid_like: Any, missing: Set[Tuple[int, int]]) -> None:
    if not isinstance(grid_like, list) or len(grid_like) < 3:
        return
    for r, c in missing:
        if 0 <= r < len(grid_like):
            row = grid_like[r]
            if isinstance(row, list) and 0 <= c < len(row):
                row[c] = None


def remove_masked_from_rendered_paths(rendered_paths: Any, missing: Set[Tuple[int, int]]) -> Any:
    if not isinstance(rendered_paths, list):
        return rendered_paths
    blocked = {f"view_grid_{r}_{c}.png" for r, c in missing}
    return [name for name in rendered_paths if not (isinstance(name, str) and name in blocked)]


def infer_tile_size(width: int, height: int) -> int:
    tile = int(round((height - 104) / 4.0))
    expected_width = 4 * tile + 62
    if 4 * tile + 104 != height or expected_width != width:
        raise RuntimeError(f"Unexpected combined image size: width={width}, height={height}")
    return tile


def apply_hard_mask_to_combined(src: Path, dst: Path, missing: Set[Tuple[int, int]]) -> None:
    image = Image.open(src).convert("RGB")
    width, height = image.size
    tile = infer_tile_size(width, height)
    pad = 16
    gap = 10
    matrix_width = 3 * tile + 2 * gap
    matrix_x = (width - matrix_width) // 2
    matrix_y = pad

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(48, int(tile * 0.45)))
    except Exception:
        font = ImageFont.load_default()

    for r, c in sorted(missing):
        x0 = matrix_x + c * (tile + gap)
        y0 = matrix_y + r * (tile + gap)

        blank = Image.new("RGB", (tile, tile), (245, 245, 245))
        blank_draw = ImageDraw.Draw(blank)
        for yy in range(0, tile, 8):
            blank_draw.line((0, yy, tile, yy), fill=(220, 220, 220), width=1)
        for xx in range(0, tile, 8):
            blank_draw.line((xx, 0, xx, tile), fill=(220, 220, 220), width=1)
        image.paste(blank, (x0, y0))

        text = "?"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x0 + (tile - text_w) // 2
        text_y = y0 + (tile - text_h) // 2
        draw.text((text_x, text_y), text, fill=(70, 70, 70), font=font)

    image.save(dst)


def generate_easy_full_samples(tmp_rule_easy_dir: Path, rule: RuleTemplate) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    sample_idx = 0

    for seed in SEEDS:
        print(f"[start] {rule.value}: seed={seed}", flush=True)
        cfg = PCRARConfig(
            rule_filter={rule.value},
            matrix_missing_one_per_row=False,
            generate_confusing_view=True,
        )
        generator = PCRARDatasetGenerator(config=cfg, seed=seed)

        made = 0
        tries = 0
        while made < PER_SEED:
            tries += 1
            if tries > MAX_TRIES_PER_SEED:
                raise RuntimeError(
                    f"{rule.value}/easy seed={seed}: retries exceeded, made={made}/{PER_SEED}"
                )
            try:
                entry = generator.generate_sample(
                    output_root=tmp_rule_easy_dir,
                    sample_index=sample_idx,
                    mode="matrix",
                )
            except RuntimeError:
                continue

            entries.append(entry)
            sample_idx += 1
            made += 1

        print(f"[easy] {rule.value}: seed={seed} -> {made}", flush=True)

    expected = len(SEEDS) * PER_SEED
    if len(entries) != expected:
        raise RuntimeError(f"{rule.value}/easy count mismatch: {len(entries)} != {expected}")

    write_json(tmp_rule_easy_dir / "meta.json", entries)
    return entries


def copy_point_files(src_sample_dir: Path, dst_sample_dir: Path, missing: Optional[Set[Tuple[int, int]]] = None) -> None:
    dst_sample_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_sample_dir.iterdir()):
        if not src.is_file() or src.suffix != ".ply":
            continue
        if missing is not None and src.name.startswith("grid_"):
            parts = src.stem.split("_")
            if len(parts) == 3:
                pos = (int(parts[1]), int(parts[2]))
                if pos in missing:
                    continue
        shutil.copy2(src, dst_sample_dir / src.name)


def export_easy_point_rule(
    tmp_rule_easy_dir: Path,
    point_rule_easy_dir: Path,
    easy_entries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ensure_empty_dir(point_rule_easy_dir)
    exported: List[Dict[str, Any]] = []
    for entry in easy_entries:
        sample_id = str(entry["id"])
        src_sample_dir = tmp_rule_easy_dir / sample_id
        dst_sample_dir = point_rule_easy_dir / sample_id
        copy_point_files(src_sample_dir, dst_sample_dir)

        meta = copy.deepcopy(entry)
        meta.pop("confusing_view", None)
        write_json(dst_sample_dir / "meta.json", meta)
        exported.append(meta)

    write_json(point_rule_easy_dir / "meta.json", exported)
    return exported


def export_easy_vlm_rule(
    tmp_rule_easy_dir: Path,
    vlm_rule_easy_dir: Path,
    easy_entries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ensure_empty_dir(vlm_rule_easy_dir)
    exported: List[Dict[str, Any]] = []
    for entry in easy_entries:
        sample_id = str(entry["id"])
        src_sample_dir = tmp_rule_easy_dir / sample_id
        dst_sample_dir = vlm_rule_easy_dir / sample_id
        dst_sample_dir.mkdir(parents=True, exist_ok=True)

        src_image = src_sample_dir / "view_combined.png"
        if not src_image.exists():
            raise RuntimeError(f"Missing combined image: {src_image}")
        shutil.copy2(src_image, dst_sample_dir / "view_combined.png")

        meta = copy.deepcopy(entry)
        write_json(dst_sample_dir / "meta.json", meta)
        exported.append(meta)

    write_json(vlm_rule_easy_dir / "meta.json", exported)
    return exported


def build_hard_meta_from_easy(
    easy_entry: Dict[str, Any],
    missing: Set[Tuple[int, int]],
    keep_confusing_view: bool,
) -> Dict[str, Any]:
    meta = copy.deepcopy(easy_entry)
    if not keep_confusing_view:
        meta.pop("confusing_view", None)

    missing_sorted = [[r, c] for r, c in sorted(missing)]
    meta["missing_positions"] = missing_sorted
    meta["empty_grid_positions"] = missing_sorted
    meta["grid_observation_mask"] = build_observation_mask(missing)
    null_grid_slots(meta.get("grid_paths"), missing)

    confusing_view = meta.get("confusing_view")
    if isinstance(confusing_view, dict):
        null_grid_slots(confusing_view.get("grid_view_paths"), missing)
        confusing_view["rendered_paths"] = remove_masked_from_rendered_paths(
            confusing_view.get("rendered_paths"),
            missing,
        )

    return meta


def export_hard_point_rule(
    tmp_rule_easy_dir: Path,
    point_rule_hard_dir: Path,
    easy_entries: Sequence[Dict[str, Any]],
    rule: RuleTemplate,
) -> List[Dict[str, Any]]:
    ensure_empty_dir(point_rule_hard_dir)
    exported: List[Dict[str, Any]] = []
    for entry in easy_entries:
        sample_id = str(entry["id"])
        missing = choose_missing_positions(rule.value, sample_id)
        src_sample_dir = tmp_rule_easy_dir / sample_id
        dst_sample_dir = point_rule_hard_dir / sample_id
        copy_point_files(src_sample_dir, dst_sample_dir, missing=missing)

        meta = build_hard_meta_from_easy(entry, missing, keep_confusing_view=False)
        write_json(dst_sample_dir / "meta.json", meta)
        exported.append(meta)

    write_json(point_rule_hard_dir / "meta.json", exported)
    return exported


def export_hard_vlm_rule(
    tmp_rule_easy_dir: Path,
    vlm_rule_hard_dir: Path,
    easy_entries: Sequence[Dict[str, Any]],
    rule: RuleTemplate,
) -> List[Dict[str, Any]]:
    ensure_empty_dir(vlm_rule_hard_dir)
    exported: List[Dict[str, Any]] = []
    for entry in easy_entries:
        sample_id = str(entry["id"])
        missing = choose_missing_positions(rule.value, sample_id)
        src_sample_dir = tmp_rule_easy_dir / sample_id
        dst_sample_dir = vlm_rule_hard_dir / sample_id
        dst_sample_dir.mkdir(parents=True, exist_ok=True)

        src_image = src_sample_dir / "view_combined.png"
        if not src_image.exists():
            raise RuntimeError(f"Missing combined image: {src_image}")
        apply_hard_mask_to_combined(src_image, dst_sample_dir / "view_combined.png", missing)

        meta = build_hard_meta_from_easy(entry, missing, keep_confusing_view=True)
        write_json(dst_sample_dir / "meta.json", meta)
        exported.append(meta)

    write_json(vlm_rule_hard_dir / "meta.json", exported)
    return exported


def prepare_output_roots(point_root: Path, vlm_root: Path, resume: bool) -> None:
    point_root.mkdir(parents=True, exist_ok=True)
    vlm_root.mkdir(parents=True, exist_ok=True)

    if not resume:
        ensure_empty_dir(point_root / "easy")
        ensure_empty_dir(point_root / "hard")
        remove_path_if_exists(vlm_root / "meta.json")
        for rule in RULES:
            remove_path_if_exists(vlm_root / rule.value)
            (vlm_root / rule.value).mkdir(parents=True, exist_ok=True)
        return

    (point_root / "easy").mkdir(parents=True, exist_ok=True)
    (point_root / "hard").mkdir(parents=True, exist_ok=True)
    for rule in RULES:
        (vlm_root / rule.value).mkdir(parents=True, exist_ok=True)


def build_vlm_root_summary(vlm_root: Path) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for rule in RULES:
        for difficulty in ("easy", "hard"):
            summary.append(
                {
                    "rule": rule.value,
                    "difficulty": difficulty,
                    "samples": len(SEEDS) * PER_SEED,
                    "seeds": [SEEDS[0], SEEDS[-1]],
                    "per_seed": PER_SEED,
                    "path": str((vlm_root / rule.value / difficulty).resolve()),
                }
            )
    return summary


def count_sample_dirs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_dir() and p.name.startswith("sample_"))


def rule_is_complete(point_root: Path, vlm_root: Path, rule: RuleTemplate) -> bool:
    expected_rule = len(SEEDS) * PER_SEED
    paths = [
        point_root / "easy" / rule.value,
        point_root / "hard" / rule.value,
        vlm_root / rule.value / "easy",
        vlm_root / rule.value / "hard",
    ]
    for path in paths:
        meta_path = path / "meta.json"
        if not meta_path.exists():
            return False
        if count_sample_dirs(path) != expected_rule:
            return False
        try:
            entries = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(entries, list) or len(entries) != expected_rule:
            return False
    return True


def load_rule_meta(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected list meta: {path}")
    return data


def verify_counts(point_root: Path, vlm_root: Path) -> None:
    expected_rule = len(SEEDS) * PER_SEED
    expected_split = expected_rule * len(RULES)

    for split in ("easy", "hard"):
        split_meta = json.loads((point_root / split / "meta.json").read_text(encoding="utf-8"))
        if len(split_meta) != expected_split:
            raise RuntimeError(f"point/{split}/meta.json count mismatch: {len(split_meta)} != {expected_split}")
        for rule in RULES:
            rule_dir = point_root / split / rule.value
            if count_sample_dirs(rule_dir) != expected_rule:
                raise RuntimeError(f"point/{split}/{rule.value} sample count mismatch")

    for rule in RULES:
        for difficulty in ("easy", "hard"):
            rule_dir = vlm_root / rule.value / difficulty
            rule_meta = json.loads((rule_dir / "meta.json").read_text(encoding="utf-8"))
            if len(rule_meta) != expected_rule:
                raise RuntimeError(f"vlm/{rule.value}/{difficulty}/meta.json count mismatch")
            if count_sample_dirs(rule_dir) != expected_rule:
                raise RuntimeError(f"vlm/{rule.value}/{difficulty} sample count mismatch")


def main() -> None:
    args = parse_args()
    point_root = args.point_root.resolve()
    vlm_root = args.vlm_root.resolve()
    tmp_parent = args.tmp_root.resolve() if args.tmp_root else None

    prepare_output_roots(point_root, vlm_root, resume=args.resume)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pcrar_regen_", dir=str(tmp_parent) if tmp_parent else None))
    try:
        for rule in RULES:
            if args.resume and rule_is_complete(point_root, vlm_root, rule):
                print(f"[skip] {rule.value}: already complete", flush=True)
                continue

            print(f"=== {rule.value} ===", flush=True)
            tmp_rule_easy_dir = tmp_dir / rule.value / "easy_full"
            tmp_rule_easy_dir.mkdir(parents=True, exist_ok=True)

            easy_entries = generate_easy_full_samples(tmp_rule_easy_dir, rule)

            point_easy_entries = export_easy_point_rule(
                tmp_rule_easy_dir=tmp_rule_easy_dir,
                point_rule_easy_dir=point_root / "easy" / rule.value,
                easy_entries=easy_entries,
            )
            point_hard_entries = export_hard_point_rule(
                tmp_rule_easy_dir=tmp_rule_easy_dir,
                point_rule_hard_dir=point_root / "hard" / rule.value,
                easy_entries=easy_entries,
                rule=rule,
            )
            export_easy_vlm_rule(
                tmp_rule_easy_dir=tmp_rule_easy_dir,
                vlm_rule_easy_dir=vlm_root / rule.value / "easy",
                easy_entries=easy_entries,
            )
            export_hard_vlm_rule(
                tmp_rule_easy_dir=tmp_rule_easy_dir,
                vlm_rule_hard_dir=vlm_root / rule.value / "hard",
                easy_entries=easy_entries,
                rule=rule,
            )

            print(
                f"[done] {rule.value}: easy={len(easy_entries)} "
                f"hard={len(point_hard_entries)}"
            ,
                flush=True,
            )

        point_easy_all: List[Dict[str, Any]] = []
        point_hard_all: List[Dict[str, Any]] = []
        for rule in RULES:
            point_easy_all.extend(load_rule_meta(point_root / "easy" / rule.value / "meta.json"))
            point_hard_all.extend(load_rule_meta(point_root / "hard" / rule.value / "meta.json"))

        write_json(point_root / "easy" / "meta.json", point_easy_all)
        write_json(point_root / "hard" / "meta.json", point_hard_all)
        write_json(vlm_root / "meta.json", build_vlm_root_summary(vlm_root))

        verify_counts(point_root, vlm_root)

        print("All done.", flush=True)
        print(f"point_root={point_root}", flush=True)
        print(f"vlm_root={vlm_root}", flush=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
