"""
compare_models.py — Run the same envelope VLM batch on two models and compare.

Submits two sequential Batch API jobs (same persons/crops), compares parsed labels
and actual token cost from batch output usage.

Usage:
    python -m vlm_annotation.compare_models \\
        --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --samples 100

    python -m vlm_annotation.compare_models \\
        --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --samples 100 \\
        --model-a gpt-5.2 --model-b gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import VLM_MODEL
from vlm_annotation.batch_api import (
    build_batch_request_line,
    retrieve_batch_results_detailed,
    submit_batch,
    wait_for_batch,
    wait_for_batch_queue_drain,
    write_batch_jsonl,
)
from vlm_annotation.batch_cost import (
    MODEL_BATCH_PRICING,
    BatchResultDetail,
    cost_from_usage,
    format_cost_line,
)
from vlm_annotation.geometry import PersonGeometry
from vlm_annotation.parse import parse_person_response
from vlm_annotation.pipeline import (
    build_envelope_crops,
    discover_images,
    prepare_work_items,
)
from vlm_annotation.run_batch import _pick_sample_names

COMPARE_FIELDS = ("clothing_outer", "clothing_inner", "glasses", "headwear")


def _parse_results(raw_by_person: dict[str, str]) -> tuple[dict[str, dict], dict[str, str]]:
    parsed: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for person_id, raw in raw_by_person.items():
        try:
            parsed[person_id] = parse_person_response(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            failures[person_id] = str(exc)
    return parsed, failures


def _load_completed_batch(batch_id: str, client) -> BatchResultDetail:
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise SystemExit(
            f"Batch {batch_id} is {batch.status}, not completed — cannot reuse."
        )
    print(f"  reusing completed batch {batch_id} ({batch.request_counts})", flush=True)
    return retrieve_batch_results_detailed(batch, client)


def _run_or_load_batch(
    model: str,
    pending: list[tuple[PersonGeometry, bytes]],
    out_dir: Path,
    client,
    label: str,
    batch_id: str | None,
) -> tuple[BatchResultDetail, dict[str, dict], dict[str, str]]:
    if batch_id:
        detail = _load_completed_batch(batch_id, client)
    else:
        detail, _, _ = _run_model_batch(model, pending, out_dir, client, label)
    parsed, parse_failures = _parse_results(detail.results)
    if detail.errors and not detail.results:
        sample = next(iter(detail.errors.values()))
        raise SystemExit(
            f"Batch for {model} had 0 successes ({len(detail.errors)} errors). "
            f"Sample: {sample}"
        )
    return detail, parsed, parse_failures


def _run_model_batch(
    model: str,
    pending: list[tuple[PersonGeometry, bytes]],
    out_dir: Path,
    client,
    label: str,
) -> tuple[BatchResultDetail, dict[str, dict], dict[str, str]]:
    requests = [
        build_batch_request_line(
            person.person_id,
            person.role,
            person.layering_mode,
            jpeg,
            model=model,
        )
        for person, jpeg in pending
    ]
    jsonl_path = out_dir / f"batch_input_{label}.jsonl"
    write_batch_jsonl(requests, jsonl_path)

    print(f"\n=== {model} ({len(requests)} requests) ===", flush=True)
    wait_for_batch_queue_drain(client)
    submitted_id = submit_batch(jsonl_path, client, description=f"vlm compare {label} {model}")
    print(f"  submitted {submitted_id}", flush=True)
    batch = wait_for_batch(submitted_id, client)
    detail = retrieve_batch_results_detailed(batch, client)
    parsed, parse_failures = _parse_results(detail.results)
    return detail, parsed, parse_failures


def _compare_parsed(
    model_a: str,
    parsed_a: dict[str, dict],
    model_b: str,
    parsed_b: dict[str, dict],
) -> dict[str, Any]:
    common = sorted(set(parsed_a) & set(parsed_b))
    field_stats: dict[str, dict[str, Any]] = {}
    disagreements: list[dict[str, Any]] = []

    for field in COMPARE_FIELDS:
        agree = 0
        for person_id in common:
            va = parsed_a[person_id].get(field)
            vb = parsed_b[person_id].get(field)
            if va == vb:
                agree += 1
            elif len(disagreements) < 50:
                disagreements.append(
                    {
                        "person_id": person_id,
                        "field": field,
                        model_a: va,
                        model_b: vb,
                    }
                )
        field_stats[field] = {
            "agree": agree,
            "total": len(common),
            "rate": round(agree / len(common), 4) if common else None,
        }

    all_match = sum(
        1
        for pid in common
        if all(parsed_a[pid].get(f) == parsed_b[pid].get(f) for f in COMPARE_FIELDS)
    )
    return {
        "persons_compared": len(common),
        "all_fields_match": all_match,
        "all_fields_match_rate": round(all_match / len(common), 4) if common else None,
        "by_field": field_stats,
        "sample_disagreements": disagreements,
    }


def _cost_summary(model: str, detail: BatchResultDetail) -> dict[str, Any]:
    est = cost_from_usage(
        model,
        detail.input_tokens,
        detail.output_tokens,
        requests=detail.request_count,
    )
    in_rate, out_rate = MODEL_BATCH_PRICING.get(model, ("?", "?"))
    return {
        "model": model,
        "requests_ok": len(detail.results),
        "requests_failed": len(detail.errors),
        "input_tokens": detail.input_tokens,
        "output_tokens": detail.output_tokens,
        "input_usd_per_m": in_rate,
        "output_usd_per_m": out_rate,
        "input_usd": round(est.input_usd, 4),
        "output_usd": round(est.output_usd, 4),
        "total_usd": round(est.total_usd, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two VLM models on the same envelope crop batch"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ref-id", required=True)
    parser.add_argument("--samples", type=int, default=100, help="Number of images")
    parser.add_argument("--images", nargs="*")
    parser.add_argument("--manifest", type=Path, default=ROOT / "refs" / "manifest.yaml")
    parser.add_argument("--model-a", default=VLM_MODEL, help="Baseline model (default: config)")
    parser.add_argument(
        "--model-b",
        default="gpt-5-mini",
        help="Cheaper candidate (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--batch-id-a",
        help="Reuse completed batch results for model-a (skip re-submit)",
    )
    parser.add_argument(
        "--batch-id-b",
        help="Reuse completed batch results for model-b (skip re-submit)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/vlm_model_compare/<timestamp>",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("Error: pip install openai")
        sys.exit(1)

    run_dir = args.run_dir.resolve()
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (run_dir / "vlm_model_compare" / f"{args.ref_id}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    names = discover_images(run_dir, args.ref_id, args.images)
    selected = names if args.images else _pick_sample_names(names, args.samples)

    print(f"Preparing {len(selected)} images for {args.ref_id}...", flush=True)
    items = prepare_work_items(run_dir, args.ref_id, selected, args.manifest, progress=None)
    if not items:
        print("No passed images with geometry.")
        sys.exit(1)

    person_ids = {p.person_id for item in items for p in item.persons}
    crop_bytes = build_envelope_crops(items, person_ids, progress=None)
    pending = sorted(
        ((person, crop_bytes[person.person_id]) for item in items for person in item.persons),
        key=lambda x: x[0].person_id,
    )

    n_images = len(items)
    n_persons = len(pending)
    print(f"  {n_images} images, {n_persons} persons ({n_persons / n_images:.1f} per image)", flush=True)

    client = OpenAI(api_key=api_key)

    detail_a, parsed_a, fail_a = _run_or_load_batch(
        args.model_a, pending, out_dir, client, "a", args.batch_id_a
    )
    detail_b, parsed_b, fail_b = _run_or_load_batch(
        args.model_b, pending, out_dir, client, "b", args.batch_id_b
    )

    comparison = _compare_parsed(args.model_a, parsed_a, args.model_b, parsed_b)
    cost_a = _cost_summary(args.model_a, detail_a)
    cost_b = _cost_summary(args.model_b, detail_b)

    savings = 0.0
    if cost_a["total_usd"]:
        savings = 1.0 - (cost_b["total_usd"] / cost_a["total_usd"])

    report = {
        "ref_id": args.ref_id,
        "images": n_images,
        "persons": n_persons,
        "model_a": args.model_a,
        "model_b": args.model_b,
        "cost_a": cost_a,
        "cost_b": cost_b,
        "cost_savings_pct": round(savings * 100, 1),
        "parse_failures_a": len(fail_a),
        "parse_failures_b": len(fail_b),
        "batch_errors_a": len(detail_a.errors),
        "batch_errors_b": len(detail_b.errors),
        "comparison": comparison,
    }

    report_path = out_dir / "compare_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Cost (actual batch usage) ===", flush=True)
    print(format_cost_line(f"  {args.model_a}", cost_from_usage(
        args.model_a, detail_a.input_tokens, detail_a.output_tokens, detail_a.request_count
    )), flush=True)
    print(format_cost_line(f"  {args.model_b}", cost_from_usage(
        args.model_b, detail_b.input_tokens, detail_b.output_tokens, detail_b.request_count
    )), flush=True)
    if cost_a["total_usd"]:
        print(
            f"  {args.model_b} is {savings * 100:.1f}% cheaper "
            f"(${cost_a['total_usd']:.4f} -> ${cost_b['total_usd']:.4f})",
            flush=True,
        )

    print("\n=== Agreement (parsed labels) ===", flush=True)
    print(
        f"  All 4 fields match: {comparison['all_fields_match']}/{comparison['persons_compared']} "
        f"({comparison['all_fields_match_rate']:.1%})",
        flush=True,
    )
    for field, stats in comparison["by_field"].items():
        rate = stats["rate"]
        pct = f"{rate:.1%}" if rate is not None else "n/a"
        print(f"  {field}: {stats['agree']}/{stats['total']} ({pct})", flush=True)

    if fail_a or fail_b:
        print(
            f"\n  Parse failures: {args.model_a}={len(fail_a)}, {args.model_b}={len(fail_b)}",
            flush=True,
        )
    if detail_a.errors or detail_b.errors:
        print(
            f"  Batch API errors: {args.model_a}={len(detail_a.errors)}, "
            f"{args.model_b}={len(detail_b.errors)}",
            flush=True,
        )

    print(f"\nReport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
