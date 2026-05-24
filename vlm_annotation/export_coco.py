"""
export_coco.py — Export aggregated VLM annotations to COCO format.

Reads passed records from annotations_vlm/ (output of run_aggregate.py).
Uses Option A taxonomy from vlm_annotation.taxonomy (10 OD classes).

Creates synthetic_clothing_v3/ with flat train/ and val/ image folders plus
annotations/train.json and annotations/val.json.

Usage:
    python -m vlm_annotation.export_coco --run-dir data/run_20260520_034326
    python -m vlm_annotation.export_coco --run-dir data/run_20260520_034326 --ref-id dash_ref_01
    python -m vlm_annotation.export_coco --run-dir data/run_20260520_034326 --val-ratio 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import date
from pathlib import Path
from typing import Any

from vlm_annotation.taxonomy import (
    CLOTHING_CLASSES,
    GLASSES_OD_CLASSES,
    HEADWEAR_OD_CLASS,
    HEADWEAR_OD_CLASSES,
    OD_CLASSES,
    category_id_for,
    category_id_from_legacy,
)

DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42
DATASET_NAME = "synthetic_clothing_v3"
DEFAULT_ANNOTATIONS_DIR = "annotations_vlm"


def _supercategory(class_name: str) -> str:
    if class_name in CLOTHING_CLASSES:
        return "clothing"
    if class_name in GLASSES_OD_CLASSES:
        return "glasses"
    if class_name in HEADWEAR_OD_CLASSES or class_name == HEADWEAR_OD_CLASS:
        return "headwear"
    return "clothing"


def build_categories() -> list[dict[str, Any]]:
    return [
        {
            "id": idx,
            "name": name,
            "supercategory": _supercategory(name),
        }
        for idx, name in enumerate(OD_CLASSES)
    ]


def coco_info() -> dict[str, Any]:
    return {
        "description": "Synthetic Clothing Dataset v3 (VLM labels, Option A taxonomy)",
        "version": "3.0",
        "year": date.today().year,
        "contributor": "",
        "date_created": date.today().isoformat(),
        "taxonomy": "option_a_v2",
        "od_classes": OD_CLASSES,
    }


def load_passed_records(
    annotations_dir: Path,
    ref_id: str | None = None,
    min_confidence: str | None = None,
) -> list[dict[str, Any]]:
    """Load passed per-image records with at least one bbox."""
    records: list[dict[str, Any]] = []
    paths = sorted(annotations_dir.glob("*.json"))

    for ann_path in paths:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        if ref_id and data.get("ref_id") != ref_id:
            continue
        if not data.get("quality", {}).get("passed"):
            continue

        image_path = Path(data["image_path"])
        file_name = image_path.name
        boxes: list[dict[str, Any]] = []

        for person in data.get("persons", []):
            for ann in person.get("annotations", []):
                if min_confidence:
                    conf = ann.get("vlm_confidence", "medium")
                    if min_confidence == "high" and conf != "high":
                        continue
                    if min_confidence == "medium" and conf == "low":
                        continue

                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                category_name = ann.get("category")
                category_id = category_id_for(category_name) if category_name else None
                if category_id is None:
                    raw_id = ann.get("category_id")
                    if raw_id is not None:
                        category_id = category_id_from_legacy(int(raw_id), category_name)
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
                "ref_id": data.get("ref_id"),
                "image_name": data.get("image_name"),
                "boxes": boxes,
            }
        )

    return records


def resolve_source_image(run_dir: Path, source_image_path: Path) -> Path:
    if source_image_path.is_file():
        return source_image_path
    candidate = run_dir / source_image_path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Image not found: {source_image_path}")


def build_split_coco(records: list[dict[str, Any]]) -> dict[str, Any]:
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
        "info": coco_info(),
        "licenses": [{"id": 1, "name": "MIT", "url": ""}],
        "categories": build_categories(),
        "images": images,
        "annotations": annotations,
    }


def copy_split_images(
    records: list[dict[str, Any]],
    run_dir: Path,
    split_dir: Path,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        src = resolve_source_image(run_dir, record["source_image_path"])
        dst = split_dir / record["file_name"]
        shutil.copy2(src, dst)


def export_dataset(
    run_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    annotations_dir_name: str = DEFAULT_ANNOTATIONS_DIR,
    ref_id: str | None = None,
    min_confidence: str | None = None,
) -> dict[str, Any]:
    annotations_dir = run_dir / annotations_dir_name
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Missing annotations directory: {annotations_dir}")

    records = load_passed_records(
        annotations_dir,
        ref_id=ref_id,
        min_confidence=min_confidence,
    )
    if not records:
        raise RuntimeError(
            f"No passed records with boxes in {annotations_dir}"
            + (f" for ref_id={ref_id}" if ref_id else "")
        )

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

    copy_split_images(train_records, run_dir, train_dir)
    copy_split_images(val_records, run_dir, val_dir)

    train_json = build_split_coco(train_records)
    val_json = build_split_coco(val_records)
    (ann_dir / "train.json").write_text(
        json.dumps(train_json, indent=2),
        encoding="utf-8",
    )
    (ann_dir / "val.json").write_text(
        json.dumps(val_json, indent=2),
        encoding="utf-8",
    )

    class_counts: dict[str, int] = {name: 0 for name in OD_CLASSES}
    for record in records:
        for box in record["boxes"]:
            class_counts[OD_CLASSES[box["category_id"]]] += 1

    return {
        "dataset_name": DATASET_NAME,
        "output_dir": str(output_dir),
        "annotations_source": str(annotations_dir),
        "ref_id_filter": ref_id,
        "min_confidence": min_confidence,
        "total_passed_records": len(records),
        "train_images": len(train_records),
        "val_images": len(val_records),
        "train_annotations": len(train_json["annotations"]),
        "val_annotations": len(val_json["annotations"]),
        "class_counts": class_counts,
        "val_ratio": val_ratio,
        "seed": seed,
        "od_classes": OD_CLASSES,
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Export VLM annotations (annotations_vlm/) to COCO v3 dataset."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=root / "data" / "run_20260520_034326",
        help="Run directory containing annotations_vlm/ and images/",
    )
    parser.add_argument(
        "--annotations-dir",
        default=DEFAULT_ANNOTATIONS_DIR,
        help=f"Annotations folder under run-dir (default: {DEFAULT_ANNOTATIONS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output dataset directory (default: <run-dir>/{DATASET_NAME})",
    )
    parser.add_argument("--ref-id", default=None, help="Export one ref only")
    parser.add_argument(
        "--min-confidence",
        choices=("high", "medium"),
        default=None,
        help="Filter boxes by vlm_confidence (high=high only, medium=exclude low)",
    )
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
        annotations_dir_name=args.annotations_dir,
        ref_id=args.ref_id,
        min_confidence=args.min_confidence,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
