"""
estimate_batch_cost.py — Estimate or report OpenAI Batch API cost.

Usage:
    python -m vlm_annotation.estimate_batch_cost --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --all
    python -m vlm_annotation.estimate_batch_cost --jsonl data/run_.../vlm_cache/batch_input_001.jsonl
    python -m vlm_annotation.estimate_batch_cost --batch-id batch_abc...   # actual usage if completed
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
from vlm_annotation.batch_cost import (
    DEFAULT_OUTPUT_TOKENS_EST,
    CostEstimate,
    batch_pricing,
    estimate_batch_jsonl,
    estimate_requests_cost,
    format_cost_line,
    sum_cost_estimates,
    summarize_usage_from_batch_output,
)
from vlm_annotation.pipeline import discover_images, load_vlm_cache, prepare_work_items
from vlm_annotation.run_batch import _pick_sample_names


def _estimate_from_ref(
    run_dir: Path,
    ref_id: str,
    image_names: list[str],
    manifest: Path,
    batch_max_requests: int,
    batch_max_bytes: int,
) -> list[tuple[int, float, CostEstimate]]:
    items = prepare_work_items(run_dir, ref_id, image_names, manifest, progress=None)
    cache = load_vlm_cache(run_dir / "vlm_cache" / "vlm_person.jsonl")

    requests: list[dict] = []
    for item in items:
        for person in item.persons:
            if person.person_id in cache:
                continue
            crop_path = run_dir / "vlm_crops" / ref_id / f"{person.person_id}.jpg"
            if crop_path.is_file():
                jpeg = crop_path.read_bytes()
            else:
                from PIL import Image

                from vlm_annotation.crops import crop_to_jpeg_bytes

                image = Image.open(item.image_path)
                try:
                    jpeg = crop_to_jpeg_bytes(image, person.envelope)
                finally:
                    image.close()

            requests.append(
                build_batch_request_line(
                    person.person_id,
                    person.role,
                    person.layering_mode,
                    jpeg,
                )
            )

    if not requests:
        return []

    chunks = chunk_batch_requests(
        requests,
        max_requests=batch_max_requests,
        max_bytes=batch_max_bytes,
    )
    out: list[tuple[int, float, CostEstimate]] = []
    for chunk in chunks:
        est = estimate_requests_cost(chunk)
        mb = chunk_byte_size(chunk) / 1024 / 1024
        out.append((len(chunk), mb, est))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate VLM batch API cost")
    parser.add_argument("--run-dir", type=Path, help="Run directory")
    parser.add_argument("--ref-id", help="Reference ID for pending-person estimate")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--images", nargs="*")
    parser.add_argument("--manifest", type=Path, default=ROOT / "refs" / "manifest.yaml")
    parser.add_argument("--jsonl", type=Path, help="Estimate from a batch_input_*.jsonl file")
    parser.add_argument("--batch-id", help="Actual usage from a completed OpenAI batch")
    parser.add_argument(
        "--batch-max-requests",
        type=int,
        default=DEFAULT_BATCH_CHUNK_REQUESTS,
    )
    parser.add_argument(
        "--batch-max-mb",
        type=float,
        default=DEFAULT_BATCH_CHUNK_MAX_BYTES / 1024 / 1024,
    )
    args = parser.parse_args()

    if args.batch_id:
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not found.")
            sys.exit(1)
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        batch = client.batches.retrieve(args.batch_id)
        if batch.status != "completed" or not batch.output_file_id:
            print(f"Batch {args.batch_id} status={batch.status} (need completed + output)")
            sys.exit(1)
        text = client.files.content(batch.output_file_id).text
        est = summarize_usage_from_batch_output(text)
        print(f"Actual usage for {args.batch_id}:")
        print(format_cost_line("  Batch", est))
        return

    if args.jsonl:
        path = args.jsonl.resolve()
        est = estimate_batch_jsonl(path)
        print(f"Estimate for {path.name}:")
        print(format_cost_line("  File", est))
        return

    if not args.run_dir or not args.ref_id:
        parser.error("Provide --jsonl, --batch-id, or both --run-dir and --ref-id")

    run_dir = args.run_dir.resolve()
    names = discover_images(run_dir, args.ref_id, args.images)
    if args.all or args.images:
        selected = names
    else:
        selected = _pick_sample_names(names, args.samples)

    batch_max_bytes = int(args.batch_max_mb * 1024 * 1024)
    chunk_estimates = _estimate_from_ref(
        run_dir,
        args.ref_id,
        selected,
        args.manifest,
        args.batch_max_requests,
        batch_max_bytes,
    )

    if not chunk_estimates:
        print("No pending persons to classify (all cached).")
        return

    total_req = sum(n for n, _, _ in chunk_estimates)
    print(
        f"Cost estimate: {args.ref_id}, {total_req} pending requests, "
        f"{len(chunk_estimates)} chunk(s)"
    )
    in_rate, out_rate = batch_pricing()
    print(
        f"Pricing: batch ${in_rate}/M input, ${out_rate}/M output "
        f"(~{DEFAULT_OUTPUT_TOKENS_EST} out tok/req est.)"
    )
    for idx, (count, mb, est) in enumerate(chunk_estimates, start=1):
        print(f"  Chunk {idx}: {count} req, {mb:.1f} MB -> ${est.total_usd:.2f}")
    print(format_cost_line("  Total", sum_cost_estimates([e for _, _, e in chunk_estimates])))


if __name__ == "__main__":
    main()
