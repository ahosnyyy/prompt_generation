"""
export_synthetic_clothing_v2.py - Export passed per-image annotations to COCO format.

Creates synthetic_clothing_v2/ with flat train/ and val/ image folders plus
annotations/train.json and annotations/val.json.

Usage:
    python scripts/export_synthetic_clothing_v2.py
    python scripts/export_synthetic_clothing_v2.py --run-dir data/run_20260520_034326
    python scripts/export_synthetic_clothing_v2.py --val-ratio 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bbox_annotation import OD_CLASSES  # noqa: E402

DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42
DATASET_NAME = "synthetic_clothing_v2"


def _build_categories() -> list[dict[str, Any]]:
    return [
        {
            "id": idx,
            "name": name,
            "supercategory": "clothing",
        }
        for idx, name in enumerate(OD_CLASSES)
    ]


def _coco_info() -> dict[str, Any]:
    return {
        "description": "Synthetic Clothing Dataset v2",
        "version": "2.0",
        "year": date.today().year,
        "contributor": "",
        "date_created": date.today().isoformat(),
    }


def _load_passed_records(annotations_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ann_path in sorted(annotations_dir.glob("*.json")):
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        if not data.get("quality", {}).get("passed"):
            continue

        image_path = Path(data["image_path"])
        file_name = image_path.name
        boxes: list[dict[str, Any]] = []
        for person in data.get("persons", []):
            for ann in person.get("annotations", []):
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                category_id = ann.get("category_id")
                if category_id is None:
                    continue
                coco_category_id = int(category_id)
                if not 0 <= coco_category_id < len(OD_CLASSES):
                    continue
                boxes.append(
                    {
                        "category_id": coco_category_id,
                        "bbox": [float(v) for v in bbox],
                    }
                )

        if not boxes:
            continue

        records.append(
            {
                "file_name": file_name,
                "source_image_path": image_path,
                "width": int(data["width"]),
                "height": int(data["height"]),
                "boxes": boxes,
            }
        )
    return records


def _resolve_source_image(run_dir: Path, source_image_path: Path) -> Path:
    if source_image_path.is_file():
        return source_image_path
    candidate = run_dir / source_image_path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Image not found: {source_image_path}")


def _build_split_coco(records: list[dict[str, Any]]) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1

    for image_id, record in enumerate(records, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": record["file_name"],
                "width": record["width"],
                "height": record["height"],
                "license": 1,
                "date_captured": "",
            }
        )
        for box in record["boxes"]:
            x, y, w, h = box["bbox"]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": box["category_id"],
                    "bbox": box["bbox"],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {
        "info": _coco_info(),
        "licenses": [{"id": 1, "name": "MIT", "url": ""}],
        "categories": _build_categories(),
        "images": images,
        "annotations": annotations,
    }


def _copy_split_images(
    records: list[dict[str, Any]],
    run_dir: Path,
    split_dir: Path,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        src = _resolve_source_image(run_dir, record["source_image_path"])
        dst = split_dir / record["file_name"]
        shutil.copy2(src, dst)


def export_dataset(
    run_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    annotations_dir = run_dir / "annotations"
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Missing annotations directory: {annotations_dir}")

    records = _load_passed_records(annotations_dir)
    if not records:
        raise RuntimeError("No passed annotation records with boxes found.")

    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)

    val_count = max(1, round(len(shuffled) * val_ratio))
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    _copy_split_images(train_records, run_dir, train_dir)
    _copy_split_images(val_records, run_dir, val_dir)

    train_json = _build_split_coco(train_records)
    val_json = _build_split_coco(val_records)
    (ann_dir / "train.json").write_text(
        json.dumps(train_json, indent=2),
        encoding="utf-8",
    )
    (ann_dir / "val.json").write_text(
        json.dumps(val_json, indent=2),
        encoding="utf-8",
    )

    return {
        "dataset_name": DATASET_NAME,
        "output_dir": str(output_dir),
        "total_passed_records": len(records),
        "train_images": len(train_records),
        "val_images": len(val_records),
        "train_annotations": len(train_json["annotations"]),
        "val_annotations": len(val_json["annotations"]),
        "val_ratio": val_ratio,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export synthetic clothing annotations to COCO v2 dataset."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "data" / "run_20260520_034326",
        help="Run directory containing annotations/ and images/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output dataset directory (default: <run-dir>/{DATASET_NAME})",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for train/val split (default: 42)",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must be between 0 and 1.")

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / DATASET_NAME).resolve()

    summary = export_dataset(
        run_dir=run_dir,
        output_dir=output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
