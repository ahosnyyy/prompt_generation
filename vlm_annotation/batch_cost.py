"""Estimate OpenAI Batch API cost for VLM envelope classification."""

from __future__ import annotations

import base64
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.config import VLM_MODEL
from vlm_annotation.prompts import PERSON_SYSTEM_PROMPT, user_prompt_for_person

# Batch API pricing ($/1M tokens) — https://developers.openai.com/api/docs/pricing
# Vision-capable chat models relevant to envelope VLM classification:
MODEL_BATCH_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.2": (0.875, 7.0),
    "gpt-5.1": (0.625, 5.0),
    "gpt-5": (0.625, 5.0),
    "gpt-5-mini": (0.125, 1.0),
    "gpt-5-nano": (0.025, 0.2),
    "gpt-5.4": (1.25, 7.5),
    "gpt-5.4-mini": (0.375, 2.25),
    "gpt-5.4-nano": (0.10, 0.625),
    "gpt-4o": (1.25, 5.0),
    "gpt-4o-mini": (0.075, 0.30),
    "gpt-4.1": (1.0, 4.0),
    "gpt-4.1-mini": (0.20, 0.80),
    "gpt-4.1-nano": (0.05, 0.20),
}

# Tile-based vision tokens for gpt-5 family (detail=high default)
GPT5_VISION_BASE_TOKENS = 70
GPT5_VISION_TILE_TOKENS = 140

# Typical JSON response size when max_completion_tokens=400
DEFAULT_OUTPUT_TOKENS_EST = 150


@dataclass
class TokenEstimate:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostEstimate:
    requests: int
    input_tokens: int
    output_tokens: int
    input_usd: float
    output_usd: float
    model: str = VLM_MODEL

    @property
    def total_usd(self) -> float:
        return self.input_usd + self.output_usd

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class BatchResultDetail:
    results: dict[str, str]
    errors: dict[str, str]
    input_tokens: int
    output_tokens: int

    @property
    def request_count(self) -> int:
        return len(self.results) + len(self.errors)


def batch_pricing(model: str | None = None) -> tuple[float, float]:
    name = model or VLM_MODEL
    if name in MODEL_BATCH_PRICING:
        return MODEL_BATCH_PRICING[name]
    return MODEL_BATCH_PRICING["gpt-5.2"]


def cost_from_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    requests: int = 0,
) -> CostEstimate:
    in_rate, out_rate = batch_pricing(model)
    input_usd = input_tokens * in_rate / 1_000_000
    output_usd = output_tokens * out_rate / 1_000_000
    return CostEstimate(
        requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_usd=input_usd,
        output_usd=output_usd,
        model=model,
    )


def _usd(input_tokens: int, output_tokens: int, model: str | None = None) -> tuple[float, float]:
    in_rate, out_rate = batch_pricing(model)
    return (
        input_tokens * in_rate / 1_000_000,
        output_tokens * out_rate / 1_000_000,
    )


def estimate_text_tokens(*parts: str) -> int:
    """Rough token count without tiktoken (~4 chars per token + overhead)."""
    chars = sum(len(p) for p in parts if p)
    return max(1, chars // 4 + 20)


def estimate_gpt5_image_tokens(width: int, height: int) -> int:
    """Tile-based vision tokens (gpt-5 / gpt-5.2, detail=high)."""
    if width <= 0 or height <= 0:
        return GPT5_VISION_BASE_TOKENS

    scale = 768 / min(width, height)
    scaled_w = width * scale
    scaled_h = height * scale
    tiles = math.ceil(scaled_w / 512) * math.ceil(scaled_h / 512)
    return GPT5_VISION_BASE_TOKENS + tiles * GPT5_VISION_TILE_TOKENS


def _image_size_from_data_url(url: str) -> tuple[int, int] | None:
    match = re.match(r"data:[^;]+;base64,(.+)", url)
    if not match:
        return None
    try:
        img = Image.open(io.BytesIO(base64.b64decode(match.group(1))))
        return img.size
    except (OSError, ValueError):
        return None


def estimate_request_tokens(
    request: dict[str, Any],
    output_tokens: int = DEFAULT_OUTPUT_TOKENS_EST,
) -> TokenEstimate:
    body = request.get("body", {})
    messages = body.get("messages", [])

    text_parts: list[str] = []
    image_tokens = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                size = _image_size_from_data_url(url)
                if size:
                    image_tokens = estimate_gpt5_image_tokens(*size)

    input_tokens = estimate_text_tokens(*text_parts) + image_tokens
    return TokenEstimate(input_tokens=input_tokens, output_tokens=output_tokens)


def estimate_request_tokens_from_crop(
    role: str,
    layering_mode: str,
    crop_jpeg: bytes,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS_EST,
) -> TokenEstimate:
    img = Image.open(io.BytesIO(crop_jpeg))
    try:
        image_tokens = estimate_gpt5_image_tokens(*img.size)
    finally:
        img.close()

    text_tokens = estimate_text_tokens(
        PERSON_SYSTEM_PROMPT,
        user_prompt_for_person(role, layering_mode),
    )
    return TokenEstimate(
        input_tokens=text_tokens + image_tokens,
        output_tokens=output_tokens,
    )


def estimate_requests_cost(
    requests: list[dict[str, Any]],
    output_tokens: int = DEFAULT_OUTPUT_TOKENS_EST,
) -> CostEstimate:
    if not requests:
        return CostEstimate(0, 0, 0, 0.0, 0.0)

    per_request = [estimate_request_tokens(req, output_tokens) for req in requests]
    input_tokens = sum(t.input_tokens for t in per_request)
    output_tokens_total = sum(t.output_tokens for t in per_request)
    input_usd, output_usd = _usd(input_tokens, output_tokens_total)
    return CostEstimate(
        requests=len(requests),
        input_tokens=input_tokens,
        output_tokens=output_tokens_total,
        input_usd=input_usd,
        output_usd=output_usd,
    )


def estimate_batch_jsonl(
    path: Path,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS_EST,
) -> CostEstimate:
    requests: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            requests.append(json.loads(line))
    return estimate_requests_cost(requests, output_tokens=output_tokens)


def summarize_usage_from_batch_output(text: str) -> CostEstimate:
    """Sum prompt/completion tokens from a completed batch output JSONL file."""
    input_tokens = 0
    output_tokens = 0
    requests = 0

    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("error"):
            continue
        usage = entry.get("response", {}).get("body", {}).get("usage", {})
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if prompt is None or completion is None:
            continue
        requests += 1
        input_tokens += int(prompt)
        output_tokens += int(completion)

    input_usd, output_usd = _usd(input_tokens, output_tokens)
    return CostEstimate(
        requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_usd=input_usd,
        output_usd=output_usd,
    )


def format_cost_line(label: str, est: CostEstimate) -> str:
    return (
        f"{label}: {est.requests} req, "
        f"{est.input_tokens:,} in + {est.output_tokens:,} out tok, "
        f"${est.input_usd:.2f} in + ${est.output_usd:.2f} out = ${est.total_usd:.2f} "
        f"(batch {VLM_MODEL})"
    )


def format_cost_summary(chunks: list[CostEstimate]) -> list[str]:
    lines: list[str] = []
    total = sum_cost_estimates(chunks)
    for idx, est in enumerate(chunks, start=1):
        lines.append(format_cost_line(f"  Chunk {idx}", est))
    if len(chunks) > 1:
        lines.append(format_cost_line("  Total", total))
    return lines


def sum_cost_estimates(chunks: list[CostEstimate]) -> CostEstimate:
    total = CostEstimate(0, 0, 0, 0.0, 0.0)
    for est in chunks:
        total = CostEstimate(
            requests=total.requests + est.requests,
            input_tokens=total.input_tokens + est.input_tokens,
            output_tokens=total.output_tokens + est.output_tokens,
            input_usd=total.input_usd + est.input_usd,
            output_usd=total.output_usd + est.output_usd,
        )
    return total
