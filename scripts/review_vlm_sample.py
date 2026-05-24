"""
review_vlm_sample.py — Crop envelope samples + visualize bboxes on full frames.

Builds geometry (envelope + refined export slots), saves envelope JPEG crops,
and writes full-frame preview PNGs for manual QA. No VLM API calls.

Usage:
    python scripts/review_vlm_sample.py --run-dir data/run_20260520_034326 --ref-id dash_ref_01
    python scripts/review_vlm_sample.py --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --samples 100
    python scripts/review_vlm_sample.py --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --samples 20 --no-vlm-labels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bbox_annotation import AREA_COLORS  # noqa: E402
from vlm_annotation.crops import save_crop  # noqa: E402
from vlm_annotation.geometry import (  # noqa: E402
    PersonGeometry,
    build_geometry_for_image,
    load_steering_for_ref,
)
from vlm_annotation.pipeline import discover_images  # noqa: E402

ENVELOPE_COLOR = (255, 80, 255)
ENVELOPE_WIDTH = 2


def _pick_sample_images(
    prompts_dir: Path,
    ref_id: str,
    samples: int,
) -> list[str]:
    all_names = sorted(p.stem for p in prompts_dir.glob(f"{ref_id}_*.json"))
    if not all_names:
        return []
    if samples >= len(all_names):
        return all_names

    step = max(1, len(all_names) // samples)
    chosen = [all_names[i] for i in range(0, len(all_names), step)][:samples]

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


def _load_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _load_vlm_labels(vlm_output_path: Path) -> dict[str, dict[str, Any]]:
    if not vlm_output_path.is_file():
        return {}
    data = json.loads(vlm_output_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for person in data.get("persons", []):
        role = person.get("role")
        parsed = person.get("vlm_parsed")
        if role and parsed:
            out[role] = parsed
    return out


def _norm_to_pixels(bbox_norm: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox_norm
    x0 = int(round(x * width))
    y0 = int(round(y * height))
    x1 = int(round((x + w) * width))
    y1 = int(round((y + h) * height))
    return x0, y0, x1, y1


def _draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    width: int = 2,
    dash: int = 10,
) -> None:
    x0, y0, x1, y1 = box
    for x in range(x0, x1, dash * 2):
        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=color, width=width)
        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=color, width=width)
    for y in range(y0, y1, dash * 2):
        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=color, width=width)
        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=color, width=width)


def _class_for_area(vlm: dict[str, Any] | None, area: str) -> str | None:
    if not vlm:
        return None
    if area == "clothing_outer":
        return vlm.get("clothing_outer")
    if area == "clothing_inner":
        return vlm.get("clothing_inner")
    if area == "glasses":
        g = vlm.get("glasses")
        return None if g in (None, "no_glasses") else g
    if area == "headwear":
        h = vlm.get("headwear")
        return None if h in (None, "no_headwear", "bare_scalp") else h
    return None


def visualize_full_frame(
    image_path: Path,
    image_name: str,
    width: int,
    height: int,
    persons: list[PersonGeometry],
    vlm_by_role: dict[str, dict[str, Any]],
    out_path: Path,
    show_vlm_labels: bool,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(16)
    small = _load_font(13)

    draw.text((10, 10), image_name, fill=(255, 255, 255), font=font)
    draw.text((10, 32), "magenta dashed = VLM envelope", fill=ENVELOPE_COLOR, font=small)

    for person in persons:
        role = person.role
        vlm = vlm_by_role.get(role)

        env = person.envelope.to_coco_normalized()
        ex0, ey0, ex1, ey1 = _norm_to_pixels(env, width, height)
        _draw_dashed_rect(draw, (ex0, ey0, ex1, ey1), ENVELOPE_COLOR, width=ENVELOPE_WIDTH)
        draw.text((ex0, max(0, ey0 - 18)), f"{role} ENV", fill=ENVELOPE_COLOR, font=small)

        for slot in person.export_slots:
            norm = slot.bbox.to_coco_normalized()
            x0, y0, x1, y1 = _norm_to_pixels(norm, width, height)
            color = AREA_COLORS.get(slot.area, (255, 255, 255))
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

            parts = [f"{role}", slot.area]
            if show_vlm_labels:
                vlm_cls = _class_for_area(vlm, slot.area)
                if vlm_cls:
                    parts.append(f"vlm:{vlm_cls}")
            if slot.prompt_hint:
                parts.append(f"hint:{slot.prompt_hint}")
            label = " | ".join(parts)
            draw.text((x0, max(0, y0 - 18)), label, fill=color, font=small)

        if show_vlm_labels and vlm:
            summary = (
                f"{role} VLM: {vlm.get('clothing_outer')} / "
                f"{vlm.get('glasses')} / {vlm.get('headwear')}"
            )
            draw.text((ex0, ey1 + 4), summary, fill=(255, 255, 255), font=small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def process_sample(
    run_dir: Path,
    ref_id: str,
    image_names: list[str],
    manifest_path: Path,
    out_root: Path,
    show_vlm_labels: bool,
) -> dict[str, int]:
    prompts_dir = run_dir / "individual_prompts"
    images_dir = run_dir / "images" / ref_id
    vlm_outputs_dir = run_dir / "vlm_outputs" / ref_id
    crops_dir = out_root / "crops" / ref_id
    previews_dir = out_root / "previews" / ref_id

    steering = load_steering_for_ref(manifest_path, ref_id)
    stats = {"images": 0, "persons": 0, "crops": 0, "skipped": 0}

    index_rows: list[dict[str, Any]] = []

    for name in image_names:
        prompt_path = prompts_dir / f"{name}.json"
        pose_path = images_dir / f"{name}_pose.json"
        image_path = images_dir / f"{name}.png"

        if not prompt_path.is_file() or not pose_path.is_file() or not image_path.is_file():
            stats["skipped"] += 1
            continue

        persons, meta = build_geometry_for_image(prompt_path, pose_path, steering)
        if not meta.get("passed") or not persons:
            stats["skipped"] += 1
            print(f"  SKIP {name}: {meta.get('fail_reason', 'geometry failed')}")
            continue

        vlm_by_role = _load_vlm_labels(vlm_outputs_dir / f"{name}.json")

        image = Image.open(image_path)
        for person in persons:
            crop_path = crops_dir / f"{person.person_id}.jpg"
            save_crop(image, person.envelope, crop_path)
            stats["crops"] += 1
        image.close()

        visualize_full_frame(
            image_path=image_path,
            image_name=name,
            width=meta["width"],
            height=meta["height"],
            persons=persons,
            vlm_by_role=vlm_by_role,
            out_path=previews_dir / f"{name}_review.png",
            show_vlm_labels=show_vlm_labels,
        )

        stats["images"] += 1
        stats["persons"] += len(persons)
        index_rows.append(
            {
                "image_name": name,
                "persons": len(persons),
                "preview": str(previews_dir / f"{name}_review.png"),
                "crops": [str(crops_dir / f"{p.person_id}.jpg") for p in persons],
                "has_vlm": bool(vlm_by_role),
            }
        )
        print(f"  OK   {name}: {len(persons)} persons, {sum(len(p.export_slots) for p in persons)} export boxes")

    index_path = out_root / f"review_index_{ref_id}.json"
    index_path.write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop VLM envelope samples and visualize bboxes on full frames"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "data" / "run_20260520_034326",
    )
    parser.add_argument("--ref-id", default="dash_ref_01")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--images", nargs="*")
    parser.add_argument("--manifest", type=Path, default=ROOT / "refs" / "manifest.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/vlm_review",
    )
    parser.add_argument(
        "--no-vlm-labels",
        action="store_true",
        help="Do not overlay labels from vlm_outputs/ (geometry only)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_root = args.output_dir or (run_dir / "vlm_review")
    prompts_dir = run_dir / "individual_prompts"

    if args.images:
        image_names = args.images
    elif args.all:
        image_names = discover_images(run_dir, args.ref_id, None)
    else:
        image_names = _pick_sample_images(prompts_dir, args.ref_id, args.samples)

    if not image_names:
        print(f"No images found for {args.ref_id}")
        sys.exit(1)

    print(f"Review sample: {len(image_names)} images from {args.ref_id}")
    print(f"  Output: {out_root}")

    stats = process_sample(
        run_dir=run_dir,
        ref_id=args.ref_id,
        image_names=image_names,
        manifest_path=args.manifest,
        out_root=out_root,
        show_vlm_labels=not args.no_vlm_labels,
    )

    print()
    print(
        f"Done: {stats['images']} images, {stats['persons']} persons, "
        f"{stats['crops']} envelope crops, {stats['skipped']} skipped"
    )
    print(f"  Previews: {out_root / 'previews' / args.ref_id}")
    print(f"  Crops:    {out_root / 'crops' / args.ref_id}")
    print(f"  Index:    {out_root / f'review_index_{args.ref_id}.json'}")


if __name__ == "__main__":
    main()
