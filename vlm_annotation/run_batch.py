"""
run_batch.py — Step 1: VLM classification on person envelope crops.

Does NOT write final annotations. Run run_aggregate.py after.

Usage:
    python -m vlm_annotation.run_batch --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --samples 100 --sync
    python -m vlm_annotation.run_batch --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --all
    python -m vlm_annotation.run_batch --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --dry-run --samples 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_annotation.batch_api import (
    DEFAULT_BATCH_CHUNK_MAX_BYTES,
    DEFAULT_BATCH_CHUNK_REQUESTS,
    build_batch_request_line,
    chunk_batch_requests,
    chunk_byte_size,
)
from vlm_annotation.batch_cost import estimate_requests_cost, format_cost_line, sum_cost_estimates
from vlm_annotation.progress import RunProgress, configure_stdio
from vlm_annotation.pipeline import (
    discover_images,
    prepare_work_items,
    run_batch_classification,
    run_sync_classification,
)


def _plan_uniform_chunks(
    total: int,
    line_bytes: float,
    batch_max_requests: int,
    batch_max_bytes: int,
) -> list[tuple[int, float]]:
    chunks: list[tuple[int, float]] = []
    remaining = total
    while remaining > 0:
        by_count = min(remaining, batch_max_requests)
        by_bytes = max(1, int(batch_max_bytes / line_bytes))
        count = min(by_count, by_bytes)
        chunks.append((count, count * line_bytes / 1024 / 1024))
        remaining -= count
    return chunks


def _estimate_batch_chunks(
    items,
    crops_dir: Path,
    ref_id: str,
    batch_max_requests: int,
    batch_max_bytes: int,
) -> tuple[int, list[tuple[int, float]]]:
    """Return (total_requests, [(count, mb), ...]) for all persons in items."""
    total = sum(len(item.persons) for item in items)
    crop_dir = crops_dir / ref_id
    sample_requests: list[dict] = []

    for item in items:
        for person in item.persons:
            crop_path = crop_dir / f"{person.person_id}.jpg"
            if crop_path.is_file():
                sample_requests.append(
                    build_batch_request_line(
                        person.person_id,
                        person.role,
                        person.layering_mode,
                        crop_path.read_bytes(),
                    )
                )

    if len(sample_requests) == total:
        chunks = chunk_batch_requests(
            sample_requests,
            max_requests=batch_max_requests,
            max_bytes=batch_max_bytes,
        )
        return total, [
            (len(chunk), chunk_byte_size(chunk) / 1024 / 1024) for chunk in chunks
        ]

    if sample_requests:
        avg_line_bytes = chunk_byte_size(sample_requests) / len(sample_requests)
    else:
        avg_line_bytes = 105 * 1024  # measured dash_ref_01 average

    return total, _plan_uniform_chunks(
        total, avg_line_bytes, batch_max_requests, batch_max_bytes
    )


def _pick_sample_names(all_names: list[str], samples: int) -> list[str]:
    if samples >= len(all_names):
        return all_names
    step = max(1, len(all_names) // samples)
    return [all_names[i] for i in range(0, len(all_names), step)][:samples]


def main() -> None:
    configure_stdio()
    print("vlm_annotation.run_batch: starting...", file=sys.stderr, flush=True)

    parser = argparse.ArgumentParser(
        description="VLM classify person envelope crops (step 1 of 2)"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ref-id", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--images", nargs="*")
    parser.add_argument("--manifest", type=Path, default=ROOT / "refs" / "manifest.yaml")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--batch-max-requests",
        type=int,
        default=DEFAULT_BATCH_CHUNK_REQUESTS,
        help=f"Max requests per Batch API upload (default: {DEFAULT_BATCH_CHUNK_REQUESTS})",
    )
    parser.add_argument(
        "--batch-max-mb",
        type=float,
        default=DEFAULT_BATCH_CHUNK_MAX_BYTES / 1024 / 1024,
        help=f"Max JSONL size per batch chunk in MB (default: {DEFAULT_BATCH_CHUNK_MAX_BYTES / 1024 / 1024:.0f})",
    )
    parser.add_argument(
        "--parallel-chunks",
        action="store_true",
        help="Submit all batch chunks at once and poll them in parallel (faster for multi-chunk refs)",
    )
    parser.add_argument(
        "--no-wait-for-queue",
        action="store_true",
        help="Do not wait for other in-flight OpenAI batches before submitting (not recommended)",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Also write envelope JPEGs to vlm_crops/ (for QA; default is in-memory only)",
    )
    args = parser.parse_args()

    if args.batch_max_requests < 1:
        parser.error("--batch-max-requests must be >= 1")
    if args.batch_max_mb <= 0:
        parser.error("--batch-max-mb must be > 0")
    batch_max_bytes = int(args.batch_max_mb * 1024 * 1024)

    run_dir = args.run_dir.resolve()
    cache_dir = run_dir / "vlm_cache"
    crops_dir = run_dir / "vlm_crops"
    vlm_outputs_dir = run_dir / "vlm_outputs"
    geometry_dir = run_dir / "vlm_geometry"
    cache_path = cache_dir / "vlm_person.jsonl"
    disagreements_path = cache_dir / "disagreements.jsonl"

    names = discover_images(run_dir, args.ref_id, args.images)
    if args.all or args.images:
        selected = names
    else:
        selected = _pick_sample_names(names, args.samples)

    if args.dry_run:
        dry_items = prepare_work_items(
            run_dir, args.ref_id, selected, args.manifest, progress=None
        )
        n_images = len(dry_items)
        n_persons = sum(len(item.persons) for item in dry_items)
        total_req, chunk_plan = _estimate_batch_chunks(
            dry_items,
            crops_dir,
            args.ref_id,
            args.batch_max_requests,
            batch_max_bytes,
        )
        print(
            f"Dry run: {n_images} images, {n_persons} person envelope VLM calls "
            f"(1 call per person)"
        )
        print(
            f"Batch plan: {total_req} pending requests -> {len(chunk_plan)} chunk(s) "
            f"(max {args.batch_max_requests} req, {args.batch_max_mb:.0f} MB each)"
        )
        if len(chunk_plan) > 1:
            mode = "parallel" if args.parallel_chunks else "sequential"
            print(f"  Submission mode: {mode}")
        for i, (count, mb) in enumerate(chunk_plan, start=1):
            print(f"  Chunk {i}: {count} requests, ~{mb:.1f} MB")

        dry_requests: list[dict] = []
        crop_dir = crops_dir / args.ref_id
        for item in dry_items:
            for person in item.persons:
                crop_path = crop_dir / f"{person.person_id}.jpg"
                if crop_path.is_file():
                    dry_requests.append(
                        build_batch_request_line(
                            person.person_id,
                            person.role,
                            person.layering_mode,
                            crop_path.read_bytes(),
                        )
                    )
        if len(dry_requests) == total_req:
            chunks = chunk_batch_requests(
                dry_requests,
                max_requests=args.batch_max_requests,
                max_bytes=batch_max_bytes,
            )
            cost_chunks = [estimate_requests_cost(chunk) for chunk in chunks]
            print("Cost estimate (batch gpt-5.2, 50% batch discount):")
            for i, (est, (count, mb)) in enumerate(
                zip(cost_chunks, chunk_plan), start=1
            ):
                print(f"  Chunk {i}: {count} req, ~{mb:.1f} MB -> about USD {est.total_usd:.2f}")
            if len(cost_chunks) > 1:
                print(format_cost_line("  Total", sum_cost_estimates(cost_chunks)))
        return

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env or environment.")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package required. Install with: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    total_phases = 4 if args.sync else 8
    progress = RunProgress(args.ref_id, total_phases=total_phases)
    progress.banner(f"=== VLM run_batch: {args.ref_id} ===")

    items = prepare_work_items(
        run_dir, args.ref_id, selected, args.manifest, progress=progress
    )
    if not items:
        print("No images with valid pose geometry found.")
        sys.exit(1)

    n_persons = sum(len(i.persons) for i in items)
    progress.note(f"{len(items)} images ready, {n_persons} persons")
    crop_out = crops_dir if args.save_crops else None

    if args.sync:
        run_sync_classification(
            items,
            cache_path,
            vlm_outputs_dir,
            geometry_dir,
            disagreements_path,
            client,
            force=args.force,
            crops_dir=crop_out,
            progress=progress,
        )
    else:
        run_batch_classification(
            items,
            cache_dir,
            vlm_outputs_dir,
            geometry_dir,
            disagreements_path,
            client,
            force=args.force,
            batch_max_requests=args.batch_max_requests,
            batch_max_bytes=batch_max_bytes,
            parallel_chunks=args.parallel_chunks,
            wait_for_queue=not args.no_wait_for_queue,
            crops_dir=crop_out,
            progress=progress,
        )

    progress.banner(f"=== VLM run_batch: {args.ref_id} finished ===")
    print(f"  VLM outputs:  {vlm_outputs_dir / args.ref_id}/", file=sys.stderr, flush=True)
    print(f"  VLM cache:    {cache_path}", file=sys.stderr, flush=True)
    print(f"  Geometry:     {geometry_dir / args.ref_id}/", file=sys.stderr, flush=True)
    if args.save_crops:
        print(f"  Envelope crops: {crops_dir / args.ref_id}/", file=sys.stderr, flush=True)
    print("Next: python -m vlm_annotation.run_aggregate --run-dir ... --ref-id ...", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
