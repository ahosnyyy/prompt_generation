"""
list_batches.py — List OpenAI Batch API jobs (active or recent).

Usage:
    python -m vlm_annotation.list_batches
    python -m vlm_annotation.list_batches --active-only
    python -m vlm_annotation.list_batches --limit 30
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

ACTIVE_STATUSES = frozenset(
    {"validating", "in_progress", "finalizing", "cancelling"}
)


def _fmt_ts(unix_ts: int | None) -> str:
    if not unix_ts:
        return "-"
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _progress(batch) -> str:
    counts = getattr(batch, "request_counts", None)
    if not counts:
        return "-"
    total = getattr(counts, "total", None)
    completed = getattr(counts, "completed", None)
    failed = getattr(counts, "failed", None)
    if total is None:
        return "-"
    parts = [f"{completed or 0}/{total}"]
    if failed:
        parts.append(f"{failed} failed")
    return " ".join(parts)


def _description(batch) -> str:
    meta = getattr(batch, "metadata", None) or {}
    if isinstance(meta, dict):
        return meta.get("description", "")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="List OpenAI Batch API jobs")
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Show only in-flight batches (validating/in_progress/finalizing)",
    )
    parser.add_argument("--limit", type=int, default=20, help="Max batches to fetch")
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

    client = OpenAI(api_key=api_key)
    batches = list(client.batches.list(limit=args.limit))

    if args.active_only:
        batches = [b for b in batches if b.status in ACTIVE_STATUSES]

    if not batches:
        label = "active" if args.active_only else "recent"
        print(f"No {label} batches found.")
        return

    print(f"{'BATCH ID':<34} {'STATUS':<14} {'PROGRESS':<16} {'CREATED':<18} DESCRIPTION")
    print("-" * 110)
    for batch in batches:
        print(
            f"{batch.id:<34} {batch.status:<14} {_progress(batch):<16} "
            f"{_fmt_ts(batch.created_at):<18} {_description(batch)}"
        )


if __name__ == "__main__":
    main()
