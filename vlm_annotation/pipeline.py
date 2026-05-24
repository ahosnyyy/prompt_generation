"""Orchestrate envelope crops → VLM classify → save outputs."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.bbox_annotation import Bbox
from vlm_annotation.batch_api import (
    DEFAULT_BATCH_CHUNK_MAX_BYTES,
    DEFAULT_BATCH_CHUNK_REQUESTS,
    build_batch_request_line,
    chunk_batch_requests,
    chunk_byte_size,
    classify_person_sync,
    retrieve_batch_results,
    submit_batch,
    wait_for_batch,
    wait_for_batch_queue_drain,
    wait_for_batches,
    write_batch_jsonl,
)
from vlm_annotation.crops import JPEG_MIME, crop_to_jpeg_bytes, save_jpeg_bytes
from vlm_annotation.geometry import (
    ExportSlot,
    PersonGeometry,
    build_geometry_for_image,
    load_steering_for_ref,
)
from vlm_annotation.merge import log_disagreements
from vlm_annotation.progress import RunProgress
from vlm_annotation.parse import parse_person_response


def _print_progress(
    label: str,
    current: int,
    total: int,
    extra: str = "",
    progress: RunProgress | None = None,
) -> None:
    if progress is not None:
        progress.step(current, total, extra)
    else:
        suffix = f" ({extra})" if extra else ""
        import sys

        print(f"  {label}: {current}/{total}{suffix}", file=sys.stderr, flush=True)


@dataclass
class WorkItem:
    image_name: str
    ref_id: str
    prompt_path: Path
    pose_path: Path
    image_path: Path
    persons: list[PersonGeometry]
    meta: dict[str, Any]


def discover_images(
    run_dir: Path,
    ref_id: str,
    image_names: list[str] | None,
) -> list[str]:
    prompts_dir = run_dir / "individual_prompts"
    if image_names:
        return image_names
    return sorted(p.stem for p in prompts_dir.glob(f"{ref_id}_*.json"))


def _save_geometry_sidecars(
    items: list[WorkItem],
    geometry_dir: Path,
    progress: RunProgress | None = None,
) -> None:
    total = len(items)
    if total == 0:
        return
    if progress is not None:
        progress.begin_phase("Writing geometry sidecars")
    _print_progress("Geometry sidecars", 0, total, "images", progress)
    for idx, item in enumerate(items, start=1):
        save_geometry_sidecar(item, geometry_dir)
        if idx == total or idx % 50 == 0:
            _print_progress("Geometry sidecars", idx, total, "images", progress)


def prepare_work_items(
    run_dir: Path,
    ref_id: str,
    image_names: list[str],
    manifest_path: Path,
    progress_every: int = 50,
    progress: RunProgress | None = None,
) -> list[WorkItem]:
    steering = load_steering_for_ref(manifest_path, ref_id)
    prompts_dir = run_dir / "individual_prompts"
    images_dir = run_dir / "images" / ref_id
    items: list[WorkItem] = []
    total = len(image_names)

    if total > 0:
        if progress is not None:
            progress.begin_phase("Loading geometry")
        _print_progress("Loading geometry", 0, total, "images", progress)

    for idx, name in enumerate(image_names, start=1):
        prompt_path = prompts_dir / f"{name}.json"
        pose_path = images_dir / f"{name}_pose.json"
        image_path = images_dir / f"{name}.png"
        if not prompt_path.is_file() or not pose_path.is_file() or not image_path.is_file():
            continue

        persons, meta = build_geometry_for_image(prompt_path, pose_path, steering)
        if not meta.get("passed"):
            continue
        items.append(
            WorkItem(
                image_name=name,
                ref_id=ref_id,
                prompt_path=prompt_path,
                pose_path=pose_path,
                image_path=image_path,
                persons=persons,
                meta=meta,
            )
        )
        if idx == 1 or idx == total or (progress_every and idx % progress_every == 0):
            _print_progress("Loading geometry", idx, total, f"{len(items)} passed", progress)

    return items


def build_envelope_crops(
    items: list[WorkItem],
    person_ids: set[str] | None = None,
    crops_dir: Path | None = None,
    progress_every: int = 25,
    progress: RunProgress | None = None,
) -> dict[str, bytes]:
    """Crop envelope JPEGs in memory; optionally write to crops_dir for QA."""
    if person_ids is not None:
        relevant_items = [
            item
            for item in items
            if any(p.person_id in person_ids for p in item.persons)
        ]
    else:
        relevant_items = items

    total_persons = (
        len(person_ids)
        if person_ids is not None
        else sum(len(item.persons) for item in relevant_items)
    )
    total_images = len(relevant_items)
    crop_bytes: dict[str, bytes] = {}

    if total_images == 0:
        return crop_bytes

    if progress is not None:
        progress.begin_phase("Building envelope crops")
    _print_progress(
        "Envelope crops",
        0,
        total_images,
        f"0/{total_persons} persons",
        progress,
    )

    for item_idx, item in enumerate(relevant_items, start=1):
        pending_persons = [
            p for p in item.persons if person_ids is None or p.person_id in person_ids
        ]
        if not pending_persons:
            continue

        image = Image.open(item.image_path)
        try:
            for person in pending_persons:
                jpeg = crop_to_jpeg_bytes(image, person.envelope)
                crop_bytes[person.person_id] = jpeg
                if crops_dir is not None:
                    out = crops_dir / item.ref_id / f"{person.person_id}.jpg"
                    save_jpeg_bytes(jpeg, out)
        finally:
            image.close()

        if item_idx == total_images or (progress_every and item_idx % progress_every == 0):
            _print_progress(
                "Envelope crops",
                item_idx,
                total_images,
                f"{len(crop_bytes)}/{total_persons} persons",
                progress,
            )

    return crop_bytes


def load_geometry_sidecar(path: Path) -> tuple[dict[str, Any], list[PersonGeometry]] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["meta"]
    persons: list[PersonGeometry] = []
    for p in data["persons"]:
        env = p["envelope_normalized"]
        envelope = Bbox(env[0], env[1], env[0] + env[2], env[1] + env[3])
        export_slots = []
        for s in p["export_slots"]:
            bn = s["bbox_normalized"]
            export_slots.append(
                ExportSlot(
                    area=s["area"],
                    bbox=Bbox(bn[0], bn[1], bn[0] + bn[2], bn[1] + bn[3]),
                    prompt_hint=s.get("prompt_hint"),
                )
            )
        persons.append(
            PersonGeometry(
                person_id=p["person_id"],
                image_name=p["image_name"],
                role=p["role"],
                envelope=envelope,
                export_slots=export_slots,
                layering_mode=p.get("layering_mode", "single"),
                pose_index=p.get("pose_index", 0),
                pose_score=p.get("pose_score", 0),
                prompt_hints=p.get("prompt_hints", {}),
            )
        )
    return meta, persons


def build_person_lookup_from_geometry(
    geometry_dir: Path,
    ref_id: str | None = None,
) -> dict[str, tuple[dict[str, Any], PersonGeometry]]:
    """Map person_id -> (image meta, person geometry) from saved sidecars."""
    lookup: dict[str, tuple[dict[str, Any], PersonGeometry]] = {}
    if ref_id:
        ref_dirs = [geometry_dir / ref_id]
    else:
        ref_dirs = sorted(p for p in geometry_dir.iterdir() if p.is_dir())

    for ref_path in ref_dirs:
        for sidecar_path in sorted(ref_path.glob("*.json")):
            loaded = load_geometry_sidecar(sidecar_path)
            if not loaded:
                continue
            meta, persons = loaded
            for person in persons:
                lookup[person.person_id] = (meta, person)
    return lookup


def _work_item_from_geometry(meta: dict[str, Any], persons: list[PersonGeometry]) -> WorkItem:
    return WorkItem(
        image_name=meta["image_name"],
        ref_id=meta["ref_id"],
        prompt_path=Path(),
        pose_path=Path(),
        image_path=Path(),
        persons=persons,
        meta=meta,
    )


def load_vlm_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["person_id"]] = row
    return out


def append_vlm_cache(cache_path: Path, person_id: str, record: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"person_id": person_id, **record}
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def save_vlm_output_file(
    item: WorkItem,
    vlm_records: dict[str, dict[str, Any]],
    out_dir: Path,
) -> None:
    """Per-image VLM output (person-level, includes raw + parsed)."""
    out_path = out_dir / item.ref_id / f"{item.image_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    persons_payload = []
    for person in item.persons:
        rec = vlm_records.get(person.person_id, {})
        persons_payload.append(
            {
                "person_id": person.person_id,
                "role": person.role,
                "layering_mode": person.layering_mode,
                "envelope_normalized": person.envelope.to_coco_normalized(),
                "prompt_hints": person.prompt_hints,
                "vlm_parsed": rec.get("parsed"),
                "vlm_raw": rec.get("raw"),
            }
        )

    payload = {
        "image_name": item.image_name,
        "ref_id": item.ref_id,
        "image_path": item.meta["image_path"],
        "width": item.meta["width"],
        "height": item.meta["height"],
        "persons": persons_payload,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_geometry_sidecar(item: WorkItem, geometry_dir: Path) -> None:
    """Store refined export slots for aggregate step."""
    out_path = geometry_dir / item.ref_id / f"{item.image_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": item.meta,
        "persons": [p.to_dict() for p in item.persons],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_vlm_output_files(
    items: list[WorkItem],
    cache: dict[str, dict[str, Any]],
    vlm_outputs_dir: Path,
    progress: RunProgress | None = None,
) -> None:
    total = len(items)
    if progress is not None:
        progress.begin_phase("Writing VLM output files")
    _print_progress("VLM outputs", 0, total, "images", progress)
    for idx, item in enumerate(items, start=1):
        item_records = {
            p.person_id: cache[p.person_id]
            for p in item.persons
            if p.person_id in cache
        }
        save_vlm_output_file(item, item_records, vlm_outputs_dir)
        if idx == total or idx % 50 == 0:
            _print_progress("VLM outputs", idx, total, "images", progress)


def _store_classification(
    person: PersonGeometry,
    raw: str,
    disagreements_path: Path,
) -> dict[str, Any]:
    parsed = parse_person_response(raw)
    log_disagreements(person, parsed, disagreements_path)
    return {"raw": raw, "parsed": parsed}


def _cache_batch_results(
    batch,
    client,
    person_lookup: dict[str, tuple[WorkItem, PersonGeometry]],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    disagreements_path: Path,
) -> int:
    raw_results = retrieve_batch_results(batch, client)
    for person_id, raw in raw_results.items():
        entry = person_lookup.get(person_id)
        if not entry:
            continue
        _, person = entry
        stored = _store_classification(person, raw, disagreements_path)
        cache[person_id] = stored
        append_vlm_cache(cache_path, person_id, stored)
    return len(raw_results)


def run_sync_classification(
    items: list[WorkItem],
    cache_path: Path,
    vlm_outputs_dir: Path,
    geometry_dir: Path,
    disagreements_path: Path,
    client,
    force: bool = False,
    crops_dir: Path | None = None,
    progress: RunProgress | None = None,
) -> dict[str, dict[str, Any]]:
    cache = load_vlm_cache(cache_path)

    pending_ids: set[str] = set()
    for item in items:
        for person in item.persons:
            if force or person.person_id not in cache:
                pending_ids.add(person.person_id)

    _save_geometry_sidecars(items, geometry_dir, progress)

    if not pending_ids:
        if progress is not None:
            progress.note("all persons already classified (use --force to redo)")
        else:
            print("  All persons already classified (use --force to redo).")
        return cache

    crop_bytes = build_envelope_crops(
        items, pending_ids, crops_dir=crops_dir, progress=progress
    )

    all_records: dict[str, dict[str, Any]] = dict(cache)
    n_pending = len(pending_ids)
    done = 0

    if progress is not None:
        progress.begin_phase("VLM classify (sync)")

    for item in items:
        item_records: dict[str, dict[str, Any]] = {}

        for person in item.persons:
            if not force and person.person_id in cache:
                item_records[person.person_id] = cache[person.person_id]
                continue

            raw = classify_person_sync(
                person.person_id,
                person.role,
                person.layering_mode,
                crop_bytes[person.person_id],
                client,
            )
            stored = _store_classification(person, raw, disagreements_path)
            item_records[person.person_id] = stored
            all_records[person.person_id] = stored
            append_vlm_cache(cache_path, person.person_id, stored)
            done += 1
            if progress is not None and (done == n_pending or done % 25 == 0):
                progress.step(done, n_pending, "persons")
            elif progress is None:
                p = stored["parsed"]
                print(
                    f"  {person.person_id} -> {p['clothing_outer']} | "
                    f"{p['glasses']} | {p['headwear']}"
                )

        save_vlm_output_file(item, item_records, vlm_outputs_dir)

    return all_records


def run_batch_classification(
    items: list[WorkItem],
    cache_dir: Path,
    vlm_outputs_dir: Path,
    geometry_dir: Path,
    disagreements_path: Path,
    client,
    force: bool = False,
    batch_max_requests: int = DEFAULT_BATCH_CHUNK_REQUESTS,
    batch_max_bytes: int = DEFAULT_BATCH_CHUNK_MAX_BYTES,
    parallel_chunks: bool = False,
    wait_for_queue: bool = True,
    crops_dir: Path | None = None,
    progress: RunProgress | None = None,
) -> dict[str, dict[str, Any]]:
    cache_path = cache_dir / "vlm_person.jsonl"
    cache = load_vlm_cache(cache_path)

    person_lookup: dict[str, tuple[WorkItem, PersonGeometry]] = {}
    pending_ids: set[str] = set()

    for item in items:
        for person in item.persons:
            person_lookup[person.person_id] = (item, person)
            if not force and person.person_id in cache:
                continue
            pending_ids.add(person.person_id)

    _save_geometry_sidecars(items, geometry_dir, progress)

    if not pending_ids:
        if progress is not None:
            progress.note("all persons already classified (use --force to redo)")
        else:
            print("  All persons already classified (use --force to redo).")
        return cache

    n_pending = len(pending_ids)
    if progress is not None:
        progress.note(f"{n_pending} pending persons across {len(items)} images")

    crop_bytes = build_envelope_crops(
        items, pending_ids, crops_dir=crops_dir, progress=progress
    )
    pending = [
        (person_lookup[person_id][1], crop_bytes[person_id]) for person_id in sorted(pending_ids)
    ]

    if progress is not None:
        progress.begin_phase("Building batch JSONL")
    requests: list[dict] = []
    _print_progress("Batch JSONL", 0, n_pending, "requests", progress)
    for idx, (person, jpeg) in enumerate(pending, start=1):
        requests.append(
            build_batch_request_line(
                person.person_id,
                person.role,
                person.layering_mode,
                jpeg,
            )
        )
        if idx == n_pending or idx % 200 == 0:
            _print_progress("Batch JSONL", idx, n_pending, "requests", progress)
    chunks = chunk_batch_requests(
        requests,
        max_requests=batch_max_requests,
        max_bytes=batch_max_bytes,
    )

    total = len(requests)
    mode = "parallel" if parallel_chunks and len(chunks) > 1 else "sequential"
    if progress is not None:
        progress.begin_phase("Submitting OpenAI batches")
        progress.note(
            f"{total} requests in {len(chunks)} chunk(s), mode={mode}, "
            f"max {batch_max_requests} req / {batch_max_bytes // 1024 // 1024} MB"
        )
    else:
        print(
            f"  Submitting {total} VLM requests in {len(chunks)} batch chunk(s) "
            f"({mode}; max {batch_max_requests} req, "
            f"{batch_max_bytes / 1024 / 1024:.0f} MB each)...",
            flush=True,
        )

    chunk_jobs: list[tuple[int, str, list[dict]]] = []
    completed_jobs: list[tuple[int, Any, list[dict]]] = []
    cached_total = 0
    use_parallel = parallel_chunks and len(chunks) > 1

    if use_parallel:
        if wait_for_queue:
            if progress is not None:
                progress.note("checking OpenAI batch queue before submit")
            wait_for_batch_queue_drain(client, progress=progress)

        for chunk_idx, chunk in enumerate(chunks, start=1):
            chunk_mb = chunk_byte_size(chunk) / 1024 / 1024
            batch_in = cache_dir / f"batch_input_{chunk_idx:03d}.jsonl"
            write_batch_jsonl(chunk, batch_in)
            msg = (
                f"chunk {chunk_idx}/{len(chunks)}: {len(chunk)} requests, "
                f"{chunk_mb:.1f} MB -> {batch_in.name}"
            )
            if progress is not None:
                progress.note(msg)
            else:
                print(
                    f"  Chunk {chunk_idx}/{len(chunks)}: {len(chunk)} requests, "
                    f"{chunk_mb:.1f} MB -> {batch_in.name}"
                )
            batch_id = submit_batch(
                batch_in,
                client,
                description=f"vlm_annotation envelope classify {chunk_idx}/{len(chunks)}",
            )
            chunk_jobs.append((chunk_idx, batch_id, chunk))
            if progress is not None:
                progress.note(f"submitted {batch_id}")
            else:
                print(f"    Submitted batch {batch_id}")

        if progress is not None:
            progress.begin_phase("Waiting for OpenAI batches")
        batch_map = wait_for_batches(
            [batch_id for _, batch_id, _ in chunk_jobs],
            client,
            progress=progress,
        )
        completed_jobs = [
            (idx, batch_map[batch_id], chunk) for idx, batch_id, chunk in chunk_jobs
        ]
    else:
        for chunk_idx, chunk in enumerate(chunks, start=1):
            chunk_mb = chunk_byte_size(chunk) / 1024 / 1024
            batch_in = cache_dir / f"batch_input_{chunk_idx:03d}.jsonl"
            write_batch_jsonl(chunk, batch_in)
            msg = (
                f"chunk {chunk_idx}/{len(chunks)}: {len(chunk)} requests, "
                f"{chunk_mb:.1f} MB -> {batch_in.name}"
            )
            if progress is not None:
                progress.note(msg)
            else:
                print(
                    f"  Chunk {chunk_idx}/{len(chunks)}: {len(chunk)} requests, "
                    f"{chunk_mb:.1f} MB -> {batch_in.name}"
                )

            if wait_for_queue:
                if progress is not None:
                    progress.note("checking OpenAI batch queue before submit")
                wait_for_batch_queue_drain(client, progress=progress)

            batch_id = submit_batch(
                batch_in,
                client,
                description=f"vlm_annotation envelope classify {chunk_idx}/{len(chunks)}",
            )
            if progress is not None:
                progress.note(f"submitted {batch_id}")
            else:
                print(f"    Submitted batch {batch_id}")

            if progress is not None:
                progress.begin_phase("Waiting for OpenAI batches")
            batch = wait_for_batch(batch_id, client, progress=progress)

            if progress is not None:
                progress.begin_phase("Caching VLM results")
            n_cached = _cache_batch_results(
                batch,
                client,
                person_lookup,
                cache,
                cache_path,
                disagreements_path,
            )
            cached_total += n_cached
            completed_jobs.append((chunk_idx, batch, chunk))
            if progress is not None:
                progress.step(
                    cached_total,
                    n_pending,
                    f"chunk {chunk_idx}/{len(chunks)} cached",
                )
            else:
                print(f"    Cached {n_cached} results from chunk {chunk_idx}")

    if use_parallel:
        if progress is not None:
            progress.begin_phase("Caching VLM results")
        for chunk_idx, batch, chunk in completed_jobs:
            n_cached = _cache_batch_results(
                batch,
                client,
                person_lookup,
                cache,
                cache_path,
                disagreements_path,
            )
            cached_total += n_cached
            if progress is not None:
                progress.step(
                    cached_total,
                    n_pending,
                    f"chunk {chunk_idx}/{len(completed_jobs)} cached",
                )
            else:
                print(f"    Cached {n_cached} results from chunk {chunk_idx}")

    _write_vlm_output_files(items, cache, vlm_outputs_dir, progress)
    return cache


def recover_batch_classification(
    batch_ids: list[str],
    run_dir: Path,
    client,
    force: bool = False,
    ref_id: str | None = None,
    progress: RunProgress | None = None,
) -> dict[str, int]:
    """Pull completed OpenAI batch results into vlm_cache without re-submitting."""
    if not batch_ids:
        raise ValueError("batch_ids must not be empty")

    cache_dir = run_dir / "vlm_cache"
    cache_path = cache_dir / "vlm_person.jsonl"
    vlm_outputs_dir = run_dir / "vlm_outputs"
    geometry_dir = run_dir / "vlm_geometry"
    disagreements_path = cache_dir / "disagreements.jsonl"

    cache = load_vlm_cache(cache_path)
    person_lookup = build_person_lookup_from_geometry(geometry_dir, ref_id=ref_id)
    if not person_lookup:
        scope = ref_id or "any ref"
        raise SystemExit(
            f"No geometry sidecars found under {geometry_dir} ({scope}). "
            "Run run_batch first so vlm_geometry/ exists."
        )

    stats = {
        "recovered": 0,
        "skipped_cached": 0,
        "skipped_unknown": 0,
    }
    affected_images: set[tuple[str, str]] = set()

    if progress is not None:
        progress.begin_phase("Recovering batch results")

    for batch_id in batch_ids:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        if status != "completed":
            raise SystemExit(
                f"Batch {batch_id} is {status}, not completed — "
                "wait for it to finish or use list_batches to check status."
            )
        if not batch.output_file_id:
            raise SystemExit(f"Batch {batch_id} completed but has no output file.")

        if progress is not None:
            progress.note(f"downloading results from {batch_id}")
        else:
            print(f"  Downloading results from {batch_id}...", flush=True)

        raw_results = retrieve_batch_results(batch, client)
        for person_id, raw in raw_results.items():
            if not force and person_id in cache:
                stats["skipped_cached"] += 1
                continue

            entry = person_lookup.get(person_id)
            if not entry:
                stats["skipped_unknown"] += 1
                print(
                    f"    WARNING: {person_id} not in geometry sidecars — skipped",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            meta, person = entry
            stored = _store_classification(person, raw, disagreements_path)
            cache[person_id] = stored
            append_vlm_cache(cache_path, person_id, stored)
            stats["recovered"] += 1
            affected_images.add((meta["ref_id"], meta["image_name"]))

        if progress is not None:
            progress.note(
                f"{batch_id}: {len(raw_results)} lines, "
                f"{stats['recovered']} recovered so far"
            )
        else:
            print(f"    Parsed {len(raw_results)} result lines from {batch_id}", flush=True)

    if stats["recovered"] and affected_images:
        if progress is not None:
            progress.begin_phase("Writing VLM output files")
        total = len(affected_images)
        for idx, (image_ref_id, image_name) in enumerate(sorted(affected_images), start=1):
            sidecar_path = geometry_dir / image_ref_id / f"{image_name}.json"
            loaded = load_geometry_sidecar(sidecar_path)
            if not loaded:
                continue
            meta, persons = loaded
            item_records = {
                p.person_id: cache[p.person_id]
                for p in persons
                if p.person_id in cache
            }
            save_vlm_output_file(
                _work_item_from_geometry(meta, persons),
                item_records,
                vlm_outputs_dir,
            )
            if progress is not None and (idx == total or idx % 50 == 0):
                progress.step(idx, total, "images updated")

    return stats
