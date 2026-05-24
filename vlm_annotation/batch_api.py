"""OpenAI Batch API — one envelope crop per person."""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

from src.config import VLM_MODEL

from vlm_annotation.progress import RunProgress
from vlm_annotation.prompts import PERSON_SYSTEM_PROMPT, user_prompt_for_person

# OpenAI Batch API hard limits (https://platform.openai.com/docs/guides/batch)
OPENAI_BATCH_MAX_REQUESTS = 50_000
OPENAI_BATCH_MAX_FILE_BYTES = 200 * 1024 * 1024

# Conservative defaults — stay under org enqueued-token limits for vision batches
# (~102 KB JSONL line per person; 50 MB ~= 500 requests for dash_ref_01)
DEFAULT_BATCH_CHUNK_REQUESTS = 500
DEFAULT_BATCH_CHUNK_MAX_BYTES = 50 * 1024 * 1024

ACTIVE_BATCH_STATUSES = frozenset(
    {"validating", "in_progress", "finalizing", "cancelling"}
)

# gpt-5-mini/nano and o-series only accept default temperature (1)
_NO_CUSTOM_TEMPERATURE_PREFIXES = ("gpt-5-mini", "gpt-5-nano", "o1", "o3", "o4")


def supports_custom_temperature(model: str) -> bool:
    return not model.startswith(_NO_CUSTOM_TEMPERATURE_PREFIXES)


def chat_completion_body(
    model: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 400,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if supports_custom_temperature(model):
        body["temperature"] = 0.1
    return body


def build_image_message(image_b64: str, mime: str, text: str) -> list[dict]:
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
    ]


def build_batch_request_line(
    person_id: str,
    role: str,
    layering_mode: str,
    crop_jpeg: bytes,
    mime: str = "image/jpeg",
    model: str | None = None,
) -> dict[str, Any]:
    image_b64_str = base64.b64encode(crop_jpeg).decode("ascii")
    user_text = user_prompt_for_person(role, layering_mode)

    return {
        "custom_id": person_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": chat_completion_body(
            model or VLM_MODEL,
            [
                {"role": "system", "content": PERSON_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_image_message(image_b64_str, mime, user_text),
                },
            ],
        ),
    }


def write_batch_jsonl(requests: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")


def request_line_byte_size(request: dict) -> int:
    """Size of one JSONL line (json + newline) in bytes."""
    return len(json.dumps(request, ensure_ascii=False).encode("utf-8")) + 1


def chunk_byte_size(requests: list[dict]) -> int:
    return sum(request_line_byte_size(req) for req in requests)


def chunk_batch_requests(
    requests: list[dict],
    max_requests: int = DEFAULT_BATCH_CHUNK_REQUESTS,
    max_bytes: int = DEFAULT_BATCH_CHUNK_MAX_BYTES,
) -> list[list[dict]]:
    """Split requests into batches that fit OpenAI count and file-size limits."""
    if max_requests < 1:
        raise ValueError("max_requests must be >= 1")
    if max_bytes < 1:
        raise ValueError("max_bytes must be >= 1")
    if max_requests > OPENAI_BATCH_MAX_REQUESTS:
        max_requests = OPENAI_BATCH_MAX_REQUESTS
    if max_bytes > OPENAI_BATCH_MAX_FILE_BYTES:
        max_bytes = OPENAI_BATCH_MAX_FILE_BYTES

    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = 0

    for req in requests:
        line_bytes = request_line_byte_size(req)
        if line_bytes > max_bytes:
            person_id = req.get("custom_id", "?")
            raise ValueError(
                f"Request {person_id} is {line_bytes / 1024:.1f} KB — exceeds "
                f"batch chunk limit ({max_bytes / 1024 / 1024:.0f} MB). "
                "Reduce envelope crop size or raise --batch-max-mb."
            )

        exceeds_count = bool(current) and len(current) >= max_requests
        exceeds_bytes = bool(current) and current_bytes + line_bytes > max_bytes
        if exceeds_count or exceeds_bytes:
            chunks.append(current)
            current = []
            current_bytes = 0

        current.append(req)
        current_bytes += line_bytes

    if current:
        chunks.append(current)
    return chunks


def submit_batch(jsonl_path: Path, client, description: str) -> str:
    with open(jsonl_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    return batch.id


def list_active_batches(client, limit: int = 20) -> list[Any]:
    batches = list(client.batches.list(limit=limit))
    return [b for b in batches if b.status in ACTIVE_BATCH_STATUSES]


def wait_for_batch_queue_drain(
    client,
    poll_interval: int = 30,
    progress: RunProgress | None = None,
) -> None:
    """Block until no org batch jobs are validating/in_progress/finalizing."""
    while True:
        active = list_active_batches(client)
        if not active:
            return

        parts = []
        for batch in active[:5]:
            parts.append(f"{batch.id[-8:]}={batch.status}{_batch_progress(batch)}")
        summary = ", ".join(parts)
        if len(active) > 5:
            summary += f", +{len(active) - 5} more"

        msg = f"waiting for {len(active)} in-flight batch(es): {summary}"
        if progress is not None:
            progress.note(msg)
        else:
            print(f"  {msg}", flush=True)
        time.sleep(poll_interval)


def wait_for_batch(
    batch_id: str,
    client,
    poll_interval: int = 10,
    progress: RunProgress | None = None,
) -> Any:
    if progress is None:
        print(f"    Batch ID: {batch_id}")
    completed = wait_for_batches(
        [batch_id], client, poll_interval=poll_interval, progress=progress
    )
    return completed[batch_id]


_BATCH_FAILURE_STATUSES = frozenset({"failed", "expired", "cancelled"})


def _batch_progress(batch) -> str:
    counts = getattr(batch, "request_counts", None)
    if not counts:
        return ""
    total = getattr(counts, "total", None)
    completed = getattr(counts, "completed", None)
    failed = getattr(counts, "failed", None) or 0
    if total is None or completed is None:
        return ""
    pending = max(0, total - completed - failed)
    parts = [f"{completed}/{total} done"]
    if failed:
        parts.append(f"{failed} failed")
    if pending:
        parts.append(f"{pending} pending")
    return f" ({', '.join(parts)})"


def _batch_status_line(batch) -> str:
    return f"{batch.status}{_batch_progress(batch)}"


def wait_for_batches(
    batch_ids: list[str],
    client,
    poll_interval: int = 10,
    progress: RunProgress | None = None,
    stall_warn_polls: int = 6,
) -> dict[str, Any]:
    """Poll multiple batch jobs until all complete; raise SystemExit on failure."""
    if not batch_ids:
        return {}

    pending = set(batch_ids)
    completed: dict[str, Any] = {}
    last_progress_key: tuple[int, int, str] | None = None
    polls_since_change = 0

    if progress is None and len(batch_ids) > 1:
        print(f"    Polling {len(batch_ids)} batches in parallel:")
        for batch_id in batch_ids:
            print(f"      {batch_id}")
    elif progress is not None:
        progress.note(f"polling {len(batch_ids)} batch job(s)")

    while pending:
        snapshots: dict[str, Any] = {}
        for batch_id in batch_ids:
            snapshots[batch_id] = client.batches.retrieve(batch_id)

        done_requests = 0
        total_requests = 0
        failed_requests = 0
        for batch_id in batch_ids:
            counts = getattr(snapshots[batch_id], "request_counts", None)
            if counts:
                done_requests += getattr(counts, "completed", 0) or 0
                failed_requests += getattr(counts, "failed", 0) or 0
                total_requests += getattr(counts, "total", 0) or 0

        progress_key = (done_requests, failed_requests, ",".join(
            sorted(_batch_status_line(snapshots[bid]) for bid in batch_ids if bid in pending)
        ))
        if progress_key == last_progress_key:
            polls_since_change += 1
        else:
            polls_since_change = 0
            last_progress_key = progress_key

        for batch_id in list(pending):
            batch = snapshots[batch_id]
            status = batch.status
            batch_prog = _batch_progress(batch)

            if status == "completed":
                completed[batch_id] = batch
                pending.remove(batch_id)
                if progress is not None:
                    progress.note(f"{batch_id} completed{batch_prog}")
                else:
                    print(f"    {batch_id}: completed{batch_prog}")
            elif status in _BATCH_FAILURE_STATUSES:
                if progress is not None:
                    progress.note(f"{batch_id} {status}{batch_prog}")
                else:
                    print(f"    {batch_id}: {status}{batch_prog}")
                if batch.errors:
                    for err in batch.errors.data:
                        print(f"      Error: {err.message}")
                        if "Enqueued token limit" in err.message:
                            print(
                                "      Tip: run list_batches --active-only, recover any "
                                "completed batches, wait for the queue to drain, then "
                                "re-run (defaults are now 500 req / 50 MB per chunk).",
                                file=sys.stderr,
                                flush=True,
                            )
                sys.exit(1)

        if pending:
            pending_count = max(0, total_requests - done_requests - failed_requests)
            extra = "OpenAI requests"
            if total_requests and pending_count:
                primary = snapshots[next(iter(pending))]
                extra = f"OpenAI requests, {_batch_status_line(primary)}"
            elif len(pending) == 1:
                primary = snapshots[next(iter(pending))]
                extra = _batch_status_line(primary)

            if progress is not None and total_requests:
                progress.step(done_requests, total_requests, extra)
            elif progress is None and len(batch_ids) == 1:
                batch = snapshots[batch_ids[0]]
                print(
                    f"    Status: {_batch_status_line(batch)}...",
                    end="\r",
                )
            elif progress is None:
                parts = []
                for batch_id in sorted(pending):
                    batch = snapshots[batch_id]
                    parts.append(f"{batch_id[-8:]}={_batch_status_line(batch)}")
                print(f"    Waiting: {', '.join(parts)}")

            if polls_since_change >= stall_warn_polls:
                stalled_secs = polls_since_change * poll_interval
                msg = (
                    f"no progress for {stalled_secs}s on {', '.join(sorted(pending))} "
                    f"({pending_count} request(s) still pending on OpenAI — "
                    "this is normal for slow/finalizing batches; safe to leave running "
                    "or Ctrl+C and recover later with recover_batch)"
                )
                if progress is not None:
                    progress.note(msg)
                else:
                    print(f"    {msg}", flush=True)
                polls_since_change = 0

            time.sleep(poll_interval)

    return completed


def retrieve_batch_results(batch, client) -> dict[str, str]:
    detailed = retrieve_batch_results_detailed(batch, client)
    return detailed.results


def retrieve_batch_results_detailed(batch, client) -> "BatchResultDetail":
    from vlm_annotation.batch_cost import BatchResultDetail

    results: dict[str, str] = {}
    errors: dict[str, str] = {}
    input_tokens = 0
    output_tokens = 0

    def _ingest_line(entry: dict[str, Any]) -> None:
        nonlocal input_tokens, output_tokens
        custom_id = entry["custom_id"]
        if entry.get("error"):
            msg = entry["error"].get("message", str(entry["error"]))
            errors[custom_id] = msg
            print(f"    WARNING: {custom_id} failed: {msg}")
            return

        response = entry.get("response") or {}
        status_code = response.get("status_code")
        body = response.get("body") or {}
        if status_code and status_code >= 400:
            err = body.get("error") or {}
            msg = err.get("message", f"HTTP {status_code}")
            errors[custom_id] = msg
            print(f"    WARNING: {custom_id} failed: {msg}")
            return

        usage = body.get("usage", {})
        input_tokens += int(usage.get("prompt_tokens") or 0)
        output_tokens += int(usage.get("completion_tokens") or 0)
        content = body["choices"][0]["message"]["content"]
        results[custom_id] = content

    if batch.output_file_id:
        file_response = client.files.content(batch.output_file_id)
        for line in file_response.text.strip().split("\n"):
            if line.strip():
                _ingest_line(json.loads(line))
    elif batch.error_file_id:
        file_response = client.files.content(batch.error_file_id)
        for line in file_response.text.strip().split("\n"):
            if line.strip():
                _ingest_line(json.loads(line))

    return BatchResultDetail(
        results=results,
        errors=errors,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def classify_person_sync(
    person_id: str,
    role: str,
    layering_mode: str,
    crop_jpeg: bytes,
    client,
    mime: str = "image/jpeg",
) -> str:
    del person_id  # custom_id for batch only
    image_b64_str = base64.b64encode(crop_jpeg).decode("ascii")
    user_text = user_prompt_for_person(role, layering_mode)

    response = client.chat.completions.create(
        **chat_completion_body(
            VLM_MODEL,
            [
                {"role": "system", "content": PERSON_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_image_message(image_b64_str, mime, user_text),
                },
            ],
        ),
    )
    return response.choices[0].message.content
