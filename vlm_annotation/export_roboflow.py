"""
export_roboflow.py — Roboflow-style COCO layout (annotations co-located with images).

Creates:
    <output-dir>/
        train/
            _annotations.coco.json
            image1.jpg
        valid/
            _annotations.coco.json
            image1.jpg

Can run standalone from annotations_vlm/, convert an existing synthetic_clothing_v3/
export, or be triggered via export_coco --also-roboflow.

Usage:
    python -m vlm_annotation.export_roboflow --run-dir data/run_20260520_034326
    python -m vlm_annotation.export_roboflow --from-coco-dir data/run_20260520_034326/synthetic_clothing_v3
    python -m vlm_annotation.export_coco --run-dir data/run_20260520_034326 --also-roboflow
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Literal

from vlm_annotation.export_coco import (
    DATASET_NAME,
    DEFAULT_ANNOTATIONS_DIR,
    DEFAULT_SEED,
    DEFAULT_SPLIT_BY,
    DEFAULT_VAL_RATIO,
    SPLIT_BY_CHOICES,
    build_split_coco,
    copy_split_images,
    load_passed_records,
    split_records,
)

ROBOFLOW_DATASET_NAME = f"{DATASET_NAME}_roboflow"
ROBOFLOW_ANNOTATIONS_FILE = "_annotations.coco.json"
ROBOFLOW_VAL_SPLIT = "valid"


def write_roboflow_split(
    records: list[dict[str, Any]],
    coco_json: dict[str, Any],
    run_dir: Path,
    split_dir: Path,
    *,
    copy_images: bool = True,
    source_images_dir: Path | None = None,
) -> dict[str, int]:
    """Write one split folder with _annotations.coco.json and images."""
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / ROBOFLOW_ANNOTATIONS_FILE).write_text(
        json.dumps(coco_json, indent=2),
        encoding="utf-8",
    )

    if copy_images:
        if source_images_dir is not None:
            for record in records:
                src = source_images_dir / record["file_name"]
                if not src.is_file():
                    raise FileNotFoundError(f"Missing image: {src}")
                shutil.copy2(src, split_dir / record["file_name"])
        else:
            copy_split_images(records, run_dir, split_dir)

    return {
        "images": len(records),
        "annotations": len(coco_json["annotations"]),
    }


def write_roboflow_layout(
    run_dir: Path,
    output_dir: Path,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    train_json: dict[str, Any],
    val_json: dict[str, Any],
    *,
    copy_images: bool = True,
    source_train_dir: Path | None = None,
    source_val_dir: Path | None = None,
) -> dict[str, Any]:
    train_stats = write_roboflow_split(
        train_records,
        train_json,
        run_dir,
        output_dir / "train",
        copy_images=copy_images,
        source_images_dir=source_train_dir,
    )
    val_stats = write_roboflow_split(
        val_records,
        val_json,
        run_dir,
        output_dir / ROBOFLOW_VAL_SPLIT,
        copy_images=copy_images,
        source_images_dir=source_val_dir,
    )

    return {
        "format": "roboflow_coco",
        "output_dir": str(output_dir),
        "train": train_stats,
        "valid": val_stats,
        "annotation_file": ROBOFLOW_ANNOTATIONS_FILE,
    }


def export_roboflow_dataset(
    run_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    annotations_dir_name: str = DEFAULT_ANNOTATIONS_DIR,
    ref_id: str | None = None,
    min_confidence: str | None = None,
    split_by: Literal["ref", "global"] = DEFAULT_SPLIT_BY,
) -> dict[str, Any]:
    """Export Roboflow layout directly from annotations_vlm/."""
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

    train_records, val_records = split_records(records, val_ratio, seed, split_by=split_by)
    train_json = build_split_coco(train_records)
    val_json = build_split_coco(val_records)

    summary = write_roboflow_layout(
        run_dir,
        output_dir,
        train_records,
        val_records,
        train_json,
        val_json,
    )
    summary.update(
        {
            "dataset_name": ROBOFLOW_DATASET_NAME,
            "annotations_source": str(annotations_dir),
            "ref_id_filter": ref_id,
            "min_confidence": min_confidence,
            "val_ratio": val_ratio,
            "seed": seed,
            "split_by": split_by,
        }
    )
    return summary


def convert_from_coco_export(
    coco_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Build Roboflow layout from an existing synthetic_clothing_v3/ export."""
    coco_dir = coco_dir.resolve()
    if not (coco_dir / "annotations" / "train.json").is_file():
        raise FileNotFoundError(f"Missing COCO export at {coco_dir}")

    output_dir = (output_dir or coco_dir.parent / ROBOFLOW_DATASET_NAME).resolve()

    train_json = json.loads((coco_dir / "annotations" / "train.json").read_text(encoding="utf-8"))
    val_json = json.loads((coco_dir / "annotations" / "val.json").read_text(encoding="utf-8"))

    train_records = [{"file_name": img["file_name"]} for img in train_json["images"]]
    val_records = [{"file_name": img["file_name"]} for img in val_json["images"]]

    summary = write_roboflow_layout(
        run_dir=coco_dir,
        output_dir=output_dir,
        train_records=train_records,
        val_records=val_records,
        train_json=train_json,
        val_json=val_json,
        copy_images=True,
        source_train_dir=coco_dir / "train",
        source_val_dir=coco_dir / "val",
    )
    summary.update(
        {
            "dataset_name": ROBOFLOW_DATASET_NAME,
            "source_coco_dir": str(coco_dir),
        }
    )
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Export Roboflow-style COCO dataset (_annotations.coco.json per split)."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=root / "data" / "run_20260520_034326",
        help="Run directory containing annotations_vlm/ and images/",
    )
    parser.add_argument(
        "--from-coco-dir",
        type=Path,
        default=None,
        help="Convert existing synthetic_clothing_v3/ export instead of annotations_vlm/",
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
        help=f"Output directory (default: <run-dir>/{ROBOFLOW_DATASET_NAME})",
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
    parser.add_argument(
        "--split-by",
        choices=SPLIT_BY_CHOICES,
        default=DEFAULT_SPLIT_BY,
        help="Train/val split strategy (default: ref = ~val-ratio per ref_id)",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must be between 0 and 1.")

    if args.from_coco_dir is not None:
        output_dir = (args.output_dir or args.from_coco_dir.parent / ROBOFLOW_DATASET_NAME).resolve()
        summary = convert_from_coco_export(args.from_coco_dir.resolve(), output_dir)
    else:
        run_dir = args.run_dir.resolve()
        output_dir = (args.output_dir or run_dir / ROBOFLOW_DATASET_NAME).resolve()
        summary = export_roboflow_dataset(
            run_dir=run_dir,
            output_dir=output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            annotations_dir_name=args.annotations_dir,
            ref_id=args.ref_id,
            min_confidence=args.min_confidence,
            split_by=args.split_by,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
