"""
visualize_coco.py — Draw COCO export bboxes on train/val images for QA.

Usage:
    python -m vlm_annotation.visualize_coco --dataset-dir data/run_20260520_034326/synthetic_clothing_v3
    python -m vlm_annotation.visualize_coco --dataset-dir data/run_20260520_034326/synthetic_clothing_v3 --split train --samples 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from vlm_annotation.taxonomy import (
    CLOTHING_CLASSES,
    GLASSES_OD_CLASSES,
    HEADWEAR_OD_CLASS,
    OD_CLASSES,
)

CLASS_COLORS: dict[str, tuple[int, int, int]] = {}
for name in CLOTHING_CLASSES:
    CLASS_COLORS[name] = (0, 200, 0)
for name in GLASSES_OD_CLASSES:
    CLASS_COLORS[name] = (255, 200, 0)
CLASS_COLORS[HEADWEAR_OD_CLASS] = (220, 0, 220)


def _load_font(size: int = 16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _index_coco(coco: dict[str, Any]) -> tuple[dict[int, str], dict[str, int], dict[int, list[dict]]]:
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    file_to_image_id = {img["file_name"]: img["id"] for img in coco["images"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    return id_to_name, file_to_image_id, anns_by_image


def visualize_image(
    image_path: Path,
    annotations: list[dict],
    id_to_name: dict[int, str],
    out_path: Path,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(16)

    for ann in annotations:
        cat_id = ann["category_id"]
        name = id_to_name.get(cat_id, str(cat_id))
        color = CLASS_COLORS.get(name, (255, 255, 255))
        x, y, w, h = ann["bbox"]
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0, max(0, y0 - 18)), name, fill=color, font=font)

    draw.text((10, 10), image_path.name, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def visualize_split(
    dataset_dir: Path,
    split: str,
    samples: int | None,
    out_dir: Path,
) -> dict[str, int]:
    coco_path = dataset_dir / "annotations" / f"{split}.json"
    images_dir = dataset_dir / split
    if not coco_path.is_file():
        raise FileNotFoundError(f"Missing {coco_path}")

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    id_to_name, file_to_image_id, anns_by_image = _index_coco(coco)

    image_id_to_file = {v: k for k, v in file_to_image_id.items()}
    all_image_ids = sorted(anns_by_image.keys())

    if samples is not None and samples < len(all_image_ids):
        step = max(1, len(all_image_ids) // samples)
        chosen_ids = [all_image_ids[i] for i in range(0, len(all_image_ids), step)][:samples]
    else:
        chosen_ids = all_image_ids

    stats = {"visualized": 0, "skipped": 0, "boxes": 0}
    for image_id in chosen_ids:
        file_name = image_id_to_file.get(image_id)
        if not file_name:
            stats["skipped"] += 1
            continue
        src = images_dir / file_name
        if not src.is_file():
            stats["skipped"] += 1
            continue
        anns = anns_by_image.get(image_id, [])
        visualize_image(src, anns, id_to_name, out_dir / f"{file_name.rsplit('.', 1)[0]}_coco.png")
        stats["visualized"] += 1
        stats["boxes"] += len(anns)

    return stats


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Visualize COCO v3 export bboxes")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=root / "data" / "run_20260520_034326" / "synthetic_clothing_v3",
    )
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--samples", type=int, default=20, help="Per split; use 0 for all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <dataset-dir>/_preview",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    out_root = args.output_dir or (dataset_dir / "_preview")
    sample_n = None if args.samples == 0 else args.samples

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        stats = visualize_split(dataset_dir, split, sample_n, out_root / split)
        print(f"{split}: {stats['visualized']} images, {stats['boxes']} boxes -> {out_root / split}")
        if stats["skipped"]:
            print(f"  skipped: {stats['skipped']}")


if __name__ == "__main__":
    main()
