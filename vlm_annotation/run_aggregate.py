"""
run_aggregate.py — Step 2: Map VLM labels → refined pose bboxes (full frame).

Reads vlm_geometry/ + vlm_cache/vlm_person.jsonl and writes annotations_vlm/.

Usage:
    python -m vlm_annotation.run_aggregate --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --all
    python -m vlm_annotation.run_aggregate --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_annotation.merge import merge_image_annotations
from vlm_annotation.pipeline import discover_images, load_geometry_sidecar, load_vlm_cache


def _pick_sample_names(all_names: list[str], samples: int) -> list[str]:
    if samples >= len(all_names):
        return all_names
    step = max(1, len(all_names) // samples)
    return [all_names[i] for i in range(0, len(all_names), step)][:samples]


def aggregate_run(
    run_dir: Path,
    ref_id: str,
    image_names: list[str],
    out_dir: Path,
) -> int:
    geometry_dir = run_dir / "vlm_geometry" / ref_id
    cache_path = run_dir / "vlm_cache" / "vlm_person.jsonl"
    vlm_cache = load_vlm_cache(cache_path)

    parsed_by_person: dict[str, dict] = {}
    for pid, record in vlm_cache.items():
        parsed = record.get("parsed")
        if parsed:
            parsed_by_person[pid] = parsed

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    for name in image_names:
        sidecar_path = geometry_dir / f"{name}.json"
        loaded = load_geometry_sidecar(sidecar_path)
        if not loaded:
            skipped += 1
            continue

        meta, persons = loaded
        ann = merge_image_annotations(meta, persons, parsed_by_person)
        out_path = out_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ann, f, indent=2)
        written += 1

    if skipped:
        print(f"  Skipped {skipped} images (no geometry sidecar — run run_batch first)")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate VLM labels onto refined pose bboxes (step 2 of 2)"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ref-id", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--images", nargs="*")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/annotations_vlm",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.output_dir or (run_dir / "annotations_vlm")

    names = discover_images(run_dir, args.ref_id, args.images)
    if args.all or args.images:
        selected = names
    else:
        selected = _pick_sample_names(names, args.samples)

    n = aggregate_run(run_dir, args.ref_id, selected, out_dir)
    print(f"Aggregated {n} images -> {out_dir}")


if __name__ == "__main__":
    main()
