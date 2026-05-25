"""
rebuild_geometry.py — Recompute vlm_geometry/ from OpenPose only (no VLM API).

Use after changing torso/headwear/glasses bbox logic in src/bbox_annotation.py.

Usage:
    python -m vlm_annotation.rebuild_geometry --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_annotation.pipeline import (
    _save_geometry_sidecars,
    discover_images,
    prepare_work_items,
)


def _pick_sample_names(all_names: list[str], samples: int) -> list[str]:
    if samples >= len(all_names):
        return all_names
    step = max(1, len(all_names) // samples)
    return [all_names[i] for i in range(0, len(all_names), step)][:samples]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild vlm_geometry sidecars from OpenPose (no VLM calls)"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ref-id", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--images", nargs="*")
    parser.add_argument("--manifest", type=Path, default=ROOT / "refs" / "manifest.yaml")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    geometry_dir = run_dir / "vlm_geometry"

    names = discover_images(run_dir, args.ref_id, args.images)
    if args.all or args.images:
        selected = names
    else:
        selected = _pick_sample_names(names, args.samples)

    items = prepare_work_items(
        run_dir, args.ref_id, selected, args.manifest, progress=None
    )
    if not items:
        print("No images with valid pose geometry found.")
        sys.exit(1)

    _save_geometry_sidecars(items, geometry_dir, progress=None)
    n_persons = sum(len(i.persons) for i in items)
    print(
        f"Rebuilt geometry for {len(items)} images ({n_persons} persons) "
        f"-> {geometry_dir / args.ref_id}/"
    )
    print(
        "Next:\n"
        f"  python -m vlm_annotation.run_aggregate --run-dir {run_dir} "
        f"--ref-id {args.ref_id} --all\n"
        f"  python -m vlm_annotation.export_coco --run-dir {run_dir} "
        f"--ref-id {args.ref_id}"
    )


if __name__ == "__main__":
    main()
