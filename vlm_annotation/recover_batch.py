"""
recover_batch.py — Pull completed OpenAI batch results into local vlm_cache.

Use when run_batch exited early but OpenAI finished the batch (e.g. parallel
chunk failure after another chunk completed).

Usage:
    python -m vlm_annotation.recover_batch \\
        --run-dir data/run_20260520_034326 \\
        --batch-id batch_6a1314bd052c8190b9df37915e61b406

    python -m vlm_annotation.recover_batch \\
        --run-dir data/run_20260520_034326 \\
        --batch-id batch_aaa --batch-id batch_bbb \\
        --ref-id dash_ref_01
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

from vlm_annotation.pipeline import recover_batch_classification
from vlm_annotation.progress import RunProgress, configure_stdio


def main() -> None:
    configure_stdio()

    parser = argparse.ArgumentParser(
        description="Recover VLM results from a completed OpenAI batch job"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-id",
        action="append",
        required=True,
        dest="batch_ids",
        metavar="BATCH_ID",
        help="Completed OpenAI batch ID (repeat for multiple batches)",
    )
    parser.add_argument(
        "--ref-id",
        help="Limit geometry lookup to one ref (optional; searches all refs if omitted)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite persons already in vlm_cache/vlm_person.jsonl",
    )
    args = parser.parse_args()

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

    run_dir = args.run_dir.resolve()
    client = OpenAI(api_key=api_key)

    label = args.ref_id or "all refs"
    progress = RunProgress(f"recover:{label}", total_phases=2)
    progress.banner(f"=== VLM recover_batch ({label}) ===")
    progress.note(f"{len(args.batch_ids)} batch job(s) to recover")

    stats = recover_batch_classification(
        args.batch_ids,
        run_dir,
        client,
        force=args.force,
        ref_id=args.ref_id,
        progress=progress,
    )

    progress.banner("=== VLM recover_batch finished ===")
    print(
        f"  Recovered:      {stats['recovered']} persons",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"  Skipped cached: {stats['skipped_cached']}",
        file=sys.stderr,
        flush=True,
    )
    if stats["skipped_unknown"]:
        print(
            f"  Skipped unknown:{stats['skipped_unknown']} (no geometry sidecar)",
            file=sys.stderr,
            flush=True,
        )
    cache_path = run_dir / "vlm_cache" / "vlm_person.jsonl"
    print(f"  VLM cache:        {cache_path}", file=sys.stderr, flush=True)
    if stats["recovered"]:
        print(
            "Next: re-run run_batch for any remaining pending persons, then run_aggregate.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "Nothing new recovered. Check batch status with list_batches.",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
