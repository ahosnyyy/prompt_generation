"""
normalize_images.py - Remove ComfyUI duplicates and rename outputs to match individual_prompts.

Duplicate removal (per prompt index):
    dash_ref_02_000000_00001_.png   keep
    dash_ref_02_000000_00002_.png   remove
    dash_ref_02_000000_00002.json   keep
    dash_ref_02_000000_00003.json   remove

Normalization (only when all expected files are present after duplicate removal):
    dash_ref_01_000000_00001_.png   -> dash_ref_01_000000.png
    dash_ref_01_000000_00002.json   -> dash_ref_01_000000_pose.json

Usage:
    python scripts/normalize_images.py --run-dir data/run_20260520_034326
    python scripts/normalize_images.py --run-dir data/run_20260520_034326 --ref-id dash_ref_01
    python scripts/normalize_images.py --run-dir data/run_20260520_034326 --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# dash_ref_01_000000_00001_.png  or  dash_ref_01_000000_00002.json
COMFYUI_PATTERN = re.compile(
    r"^(?P<image_name>[a-z0-9_]+_\d{6})_(?P<counter>\d{5})_?\.(?P<ext>png|json)$",
    re.IGNORECASE,
)

NORMALIZED_PNG = re.compile(r"^(?P<image_name>[a-z0-9_]+_\d{6})\.png$", re.IGNORECASE)
NORMALIZED_POSE = re.compile(r"^(?P<image_name>[a-z0-9_]+_\d{6})_pose\.json$", re.IGNORECASE)


def parse_comfyui_name(source_name: str) -> tuple[str, int, str] | None:
    match = COMFYUI_PATTERN.match(source_name)
    if not match:
        return None
    return (
        match.group("image_name"),
        int(match.group("counter")),
        match.group("ext").lower(),
    )


def target_name(source_name: str) -> str | None:
    """Return normalized filename, or None if already normalized / unrecognized."""
    if NORMALIZED_PNG.match(source_name) or NORMALIZED_POSE.match(source_name):
        return None

    parsed = parse_comfyui_name(source_name)
    if parsed is None:
        return None

    image_name, _, ext = parsed
    if ext == "png":
        return f"{image_name}.png"
    return f"{image_name}_pose.json"


def collect_ref_dirs(images_dir: Path, ref_id: str | None) -> list[Path]:
    if ref_id:
        ref_dir = images_dir / ref_id
        if not ref_dir.is_dir():
            raise FileNotFoundError(f"Ref image directory not found: {ref_dir}")
        return [ref_dir]

    return sorted(p for p in images_dir.iterdir() if p.is_dir())


def expected_prompts_for_ref(prompts_dir: Path, ref_id: str) -> set[str]:
    return {
        path.stem
        for path in prompts_dir.glob("*.json")
        if path.stem.startswith(f"{ref_id}_")
    }


def find_duplicate_paths(ref_dir: Path) -> list[Path]:
    """Return duplicate ComfyUI files that would be removed (lowest counter kept)."""
    groups: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)

    for path in ref_dir.iterdir():
        if not path.is_file():
            continue

        parsed = parse_comfyui_name(path.name)
        if parsed is None:
            continue

        image_name, counter, ext = parsed
        groups[(image_name, ext)].append((counter, path))

    to_remove: list[Path] = []
    for entries in groups.values():
        if len(entries) <= 1:
            continue

        entries.sort(key=lambda item: item[0])
        for _, path in entries[1:]:
            to_remove.append(path)

    return to_remove


def scan_present_indices(ref_dir: Path, excluded: set[Path] | None = None) -> tuple[set[str], set[str]]:
    """Return image_name sets for present PNGs and pose JSON files."""
    excluded = excluded or set()
    png_indices: set[str] = set()
    pose_indices: set[str] = set()

    for path in ref_dir.iterdir():
        if not path.is_file() or path in excluded:
            continue

        png_match = NORMALIZED_PNG.match(path.name)
        if png_match:
            png_indices.add(png_match.group("image_name"))
            continue

        pose_match = NORMALIZED_POSE.match(path.name)
        if pose_match:
            pose_indices.add(pose_match.group("image_name"))
            continue

        parsed = parse_comfyui_name(path.name)
        if parsed is None:
            continue

        image_name, _, ext = parsed
        if ext == "png":
            png_indices.add(image_name)
        else:
            pose_indices.add(image_name)

    return png_indices, pose_indices


def remove_duplicates(ref_dir: Path, dry_run: bool) -> list[Path]:
    """Keep the lowest output counter per prompt index and extension."""
    to_remove = find_duplicate_paths(ref_dir)

    for path in to_remove:
        if dry_run:
            kept = next(
                p for p in ref_dir.iterdir()
                if p.is_file()
                and parse_comfyui_name(p.name) is not None
                and parse_comfyui_name(p.name)[:2] == (parse_comfyui_name(path.name)[0], parse_comfyui_name(path.name)[2])
                and p not in to_remove
            )
            # simpler: just print remove message
            print(f"  would remove duplicate: {path.name}")
        else:
            path.unlink()
            print(f"  removed duplicate: {path.name}")

    if not dry_run and to_remove:
        for path in to_remove:
            # already unlinked above; find kept name for message clarity
            pass

    return to_remove


def check_missing_after_removal(
    ref_dir: Path,
    expected: set[str],
    excluded: set[Path],
) -> tuple[list[str], list[str]]:
    present_png, present_pose = scan_present_indices(ref_dir, excluded=excluded)
    missing_png = sorted(expected - present_png)
    missing_pose = sorted(expected - present_pose)
    return missing_png, missing_pose


def print_missing_notice(ref_id: str, missing_png: list[str], missing_pose: list[str]) -> None:
    print("  Missing files after duplicate removal - skipping normalization.")
    print("  Regenerate the missing outputs in ComfyUI, then re-run this script.")

    if missing_png:
        print(f"  Missing PNG ({len(missing_png)}):")
        for name in missing_png[:10]:
            print(f"    - {name}.png")
        if len(missing_png) > 10:
            print(f"    ... and {len(missing_png) - 10} more")

    if missing_pose:
        print(f"  Missing pose JSON ({len(missing_pose)}):")
        for name in missing_pose[:10]:
            print(f"    - {name}_pose.json")
        if len(missing_pose) > 10:
            print(f"    ... and {len(missing_pose) - 10} more")


def normalize_ref_dir(ref_dir: Path, dry_run: bool) -> tuple[int, int, int, int]:
    renamed = 0
    skipped = 0
    conflicts = 0
    unrecognized = 0

    for source in sorted(ref_dir.iterdir()):
        if not source.is_file():
            continue

        new_name = target_name(source.name)
        if new_name is None:
            if COMFYUI_PATTERN.match(source.name):
                skipped += 1
                print(f"  skip (unhandled): {source.name}")
            else:
                skipped += 1
            continue

        target = source.with_name(new_name)

        if target.exists() and target.resolve() != source.resolve():
            conflicts += 1
            print(f"  conflict: {source.name} -> {new_name} (target already exists)")
            continue

        if dry_run:
            print(f"  would rename: {source.name} -> {new_name}")
        else:
            source.rename(target)
            print(f"  renamed: {source.name} -> {new_name}")
        renamed += 1

    unrecognized_in_ref = [
        p.name
        for p in ref_dir.iterdir()
        if p.is_file()
        and not NORMALIZED_PNG.match(p.name)
        and not NORMALIZED_POSE.match(p.name)
        and target_name(p.name) is None
    ]
    if unrecognized_in_ref:
        unrecognized += len(unrecognized_in_ref)
        print(f"  unrecognized ({len(unrecognized_in_ref)}):")
        for name in unrecognized_in_ref[:5]:
            print(f"    - {name}")
        if len(unrecognized_in_ref) > 5:
            print(f"    ... and {len(unrecognized_in_ref) - 5} more")

    return renamed, skipped, conflicts, unrecognized


def normalize_run(run_dir: Path, ref_id: str | None, dry_run: bool) -> int:
    images_root = run_dir / "images"
    prompts_dir = run_dir / "individual_prompts"

    if not images_root.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_root}")

    removed = 0
    renamed = 0
    skipped = 0
    conflicts = 0
    unrecognized = 0
    refs_skipped_missing = 0

    for ref_dir in collect_ref_dirs(images_root, ref_id):
        current_ref_id = ref_dir.name
        print(f"\n{ref_dir.relative_to(run_dir)}/")

        print("  Removing duplicates...")
        duplicate_paths = find_duplicate_paths(ref_dir)
        for path in duplicate_paths:
            if dry_run:
                print(f"  would remove duplicate: {path.name}")
            else:
                path.unlink()
                print(f"  removed duplicate: {path.name}")
        removed += len(duplicate_paths)

        expected = (
            expected_prompts_for_ref(prompts_dir, current_ref_id)
            if prompts_dir.is_dir()
            else set()
        )

        if expected:
            print("  Checking completeness after duplicate removal...")
            excluded = set(duplicate_paths) if dry_run else set()
            missing_png, missing_pose = check_missing_after_removal(
                ref_dir, expected, excluded=excluded
            )

            if missing_png or missing_pose:
                print_missing_notice(current_ref_id, missing_png, missing_pose)
                refs_skipped_missing += 1
                continue

            print(
                f"  complete: {len(expected)} PNG + pose JSON pairs ready "
                f"(expected {len(expected)})"
            )

        print("  Normalizing names...")
        ref_renamed, ref_skipped, ref_conflicts, ref_unrecognized = normalize_ref_dir(
            ref_dir, dry_run
        )
        renamed += ref_renamed
        skipped += ref_skipped
        conflicts += ref_conflicts
        unrecognized += ref_unrecognized

    print(
        f"\nSummary: removed={removed}, renamed={renamed}, skipped={skipped}, "
        f"conflicts={conflicts}, unrecognized={unrecognized}, "
        f"refs_skipped_missing={refs_skipped_missing}"
        + (" (dry run)" if dry_run else "")
    )

    if refs_skipped_missing:
        print(
            "\nNormalization was skipped for refs with missing files. "
            "Regenerate the listed outputs, then re-run this script."
        )
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove ComfyUI duplicates and normalize image outputs to match individual_prompts naming."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory, e.g. data/run_20260520_034326",
    )
    parser.add_argument(
        "--ref-id",
        default=None,
        help="Only normalize one ref subfolder, e.g. dash_ref_01",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without changing files",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        exit_code = normalize_run(run_dir, args.ref_id, args.dry_run)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
