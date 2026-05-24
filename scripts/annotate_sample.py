"""
annotate_sample.py - Annotate a sample of images and visualize bboxes.

Usage:
    python scripts/annotate_sample.py --run-dir data/run_20260520_034326 --ref-id dash_ref_01
    python scripts/annotate_sample.py --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --samples 10
    python scripts/annotate_sample.py --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --all --no-visualize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bbox_annotation import (  # noqa: E402
    AREA_COLORS,
    annotate_image,
    load_ref_steering_side,
    save_annotation,
)


def _pick_sample_images(
    run_dir: Path,
    ref_id: str,
    samples: int,
    prompts_dir: Path,
) -> list[str]:
    """Pick a diverse sample: spread + at least one open_outer if available."""
    all_names = sorted(
        p.stem for p in prompts_dir.glob(f"{ref_id}_*.json")
    )
    if not all_names:
        return []

    if samples >= len(all_names):
        return all_names

    # Evenly spaced base sample
    step = max(1, len(all_names) // samples)
    chosen = [all_names[i] for i in range(0, len(all_names), step)][:samples]

    # Swap in one open_outer example if not already included
    for name in all_names[:500]:
        with open(prompts_dir / f"{name}.json", encoding="utf-8") as f:
            data = json.load(f)
        has_open = any(
            p.get("clothing", {}).get("layering_mode") == "open_outer"
            for p in data.get("persons", [])
        )
        if has_open and name not in chosen:
            chosen[-1] = name
            break

    return chosen[:samples]


def visualize_annotation(
    image_path: Path,
    annotation: dict,
    out_path: Path,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width = annotation["width"]
    height = annotation["height"]

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for person in annotation.get("persons", []):
        role = person["role"]
        for ann in person.get("annotations", []):
            x, y, w, h = ann["bbox"]
            color = AREA_COLORS.get(ann["area"], (255, 255, 255))
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            label = f"{role}: {ann['category']}"
            draw.text((x, max(0, y - 20)), label, fill=color, font=font)

    quality = annotation.get("quality", {})
    status = "PASS" if quality.get("passed") else "FAIL"
    header = f"{annotation['image_name']} [{status}]"
    if quality.get("reason"):
        header += f" ({quality['reason']})"
    draw.text((10, 10), header, fill=(255, 80, 80), font=font)

    for i, warning in enumerate(quality.get("warnings", [])[:3]):
        draw.text((10, 35 + i * 22), warning, fill=(255, 180, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def process_images(
    run_dir: Path,
    ref_id: str,
    image_names: list[str],
    manifest_path: Path,
    visualize: bool,
) -> dict[str, int]:
    prompts_dir = run_dir / "individual_prompts"
    images_dir = run_dir / "images" / ref_id
    annotations_dir = run_dir / "annotations"
    preview_dir = annotations_dir / "_preview"

    steering_side = load_ref_steering_side(str(manifest_path), ref_id)

    stats = {"passed": 0, "failed": 0, "total": 0}

    for name in image_names:
        stats["total"] += 1
        prompt_path = prompts_dir / f"{name}.json"
        pose_path = images_dir / f"{name}_pose.json"
        image_path = images_dir / f"{name}.png"

        if not prompt_path.exists():
            print(f"  SKIP {name}: missing prompt")
            stats["failed"] += 1
            continue
        if not pose_path.exists():
            print(f"  SKIP {name}: missing pose")
            stats["failed"] += 1
            continue
        if not image_path.exists():
            print(f"  SKIP {name}: missing image")
            stats["failed"] += 1
            continue

        result = annotate_image(
            prompt_path=prompt_path,
            pose_path=pose_path,
            image_path=image_path,
            steering_side=steering_side,
        )

        out_json = annotations_dir / f"{name}.json"
        save_annotation(result, out_json)

        if result.passed:
            stats["passed"] += 1
            n_boxes = sum(len(p.annotations) for p in result.persons)
            print(f"  OK   {name}: {n_boxes} bboxes")
        else:
            stats["failed"] += 1
            print(f"  FAIL {name}: {result.fail_reason}")

        if visualize and image_path.exists():
            with open(out_json, encoding="utf-8") as f:
                ann_dict = json.load(f)
            visualize_annotation(
                image_path,
                ann_dict,
                preview_dir / f"{name}_preview.png",
            )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate sample images with OpenPose bboxes")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "data" / "run_20260520_034326",
        help="Run directory containing images/ and individual_prompts/",
    )
    parser.add_argument("--ref-id", default="dash_ref_01", help="Reference ID to process")
    parser.add_argument("--samples", type=int, default=10, help="Number of sample images")
    parser.add_argument("--all", action="store_true", help="Process all images for ref")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "refs" / "manifest.yaml",
        help="Manifest YAML for steering_side lookup",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip preview PNG generation",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        help="Explicit image names (without extension) to process",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    prompts_dir = run_dir / "individual_prompts"

    if args.images:
        image_names = args.images
    elif args.all:
        image_names = sorted(p.stem for p in prompts_dir.glob(f"{args.ref_id}_*.json"))
    else:
        image_names = _pick_sample_images(run_dir, args.ref_id, args.samples, prompts_dir)

    if not image_names:
        print(f"No images found for ref {args.ref_id} in {run_dir}")
        sys.exit(1)

    print(f"Annotating {len(image_names)} images from {args.ref_id}...")
    print(f"  Run dir:    {run_dir}")
    print(f"  Output:     {run_dir / 'annotations'}")
    if not args.no_visualize:
        print(f"  Previews:   {run_dir / 'annotations' / '_preview'}")

    stats = process_images(
        run_dir=run_dir,
        ref_id=args.ref_id,
        image_names=image_names,
        manifest_path=args.manifest,
        visualize=not args.no_visualize,
    )

    print()
    print(f"Done: {stats['passed']} passed, {stats['failed']} failed, {stats['total']} total")


if __name__ == "__main__":
    main()
