"""Parse and validate multi-field VLM person classification JSON."""

from __future__ import annotations

import json
import re
from typing import Any

from vlm_annotation.taxonomy import (
    CLOTHING_CLASSES,
    GLASSES_ABSENT,
    GLASSES_OD_CLASSES,
    HEADWEAR_ABSENT,
    HEADWEAR_BARE_SCALP,
    HEADWEAR_VLM_CLASSES,
)

CLOTHING_SET = set(CLOTHING_CLASSES)
GLASSES_SET = {GLASSES_ABSENT, *GLASSES_OD_CLASSES}
HEADWEAR_SET = {HEADWEAR_ABSENT, HEADWEAR_BARE_SCALP, *HEADWEAR_VLM_CLASSES}
CONFIDENCE_LEVELS = {"high", "medium", "low"}


def strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _norm_confidence(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value in CONFIDENCE_LEVELS:
        return value
    return "medium"


def parse_person_response(raw: str) -> dict[str, Any]:
    """Parse envelope VLM response; raise ValueError on invalid JSON or classes."""
    text = strip_markdown_fences(raw)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object")

    outer = data.get("clothing_outer")
    if not isinstance(outer, str) or outer not in CLOTHING_SET:
        raise ValueError(f"Invalid clothing_outer: {outer!r}")

    inner = data.get("clothing_inner")
    if inner is not None:
        if not isinstance(inner, str) or inner not in CLOTHING_SET:
            raise ValueError(f"Invalid clothing_inner: {inner!r}")

    glasses = data.get("glasses")
    if not isinstance(glasses, str) or glasses not in GLASSES_SET:
        raise ValueError(f"Invalid glasses: {glasses!r}")

    headwear = data.get("headwear")
    if not isinstance(headwear, str) or headwear not in HEADWEAR_SET:
        raise ValueError(f"Invalid headwear: {headwear!r}")

    conf_in = data.get("confidence") or {}
    if not isinstance(conf_in, dict):
        conf_in = {}

    confidence = {
        "clothing_outer": _norm_confidence(conf_in.get("clothing_outer")) or "medium",
        "clothing_inner": _norm_confidence(conf_in.get("clothing_inner")),
        "glasses": _norm_confidence(conf_in.get("glasses")) or "medium",
        "headwear": _norm_confidence(conf_in.get("headwear")) or "medium",
    }

    reason = data.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)

    return {
        "clothing_outer": outer,
        "clothing_inner": inner,
        "glasses": glasses,
        "headwear": headwear,
        "confidence": confidence,
        "reason": reason,
    }
