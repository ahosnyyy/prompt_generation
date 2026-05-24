"""
Option A taxonomy — 10 OD classes for export.

Geometry: OpenPose bboxes (src/bbox_annotation.py).
Labels: VLM classifies headwear subtypes; export merges to one headwear class.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# OD classes (export order = category_id)
# ---------------------------------------------------------------------------

CLOTHING_CLASSES = [
    "sleeveless",
    "short_sleeve",
    "long_sleeve_light",
    "long_sleeve_knit",
    "knit_hoodie",
    "light_jacket",
    "heavy_coat",
]

GLASSES_OD_CLASSES = ["eyeglasses", "sunglasses"]

# VLM returns these; export maps all to HEADWEAR_OD_CLASS
HEADWEAR_VLM_CLASSES = ["cap", "beanie", "hijab_headscarf"]
HEADWEAR_OD_CLASS = "headwear"
HEADWEAR_OD_CLASSES = [HEADWEAR_OD_CLASS]

# VLM may return these; they do not produce bboxes
GLASSES_ABSENT = "no_glasses"
HEADWEAR_ABSENT = "no_headwear"
HEADWEAR_BARE_SCALP = "bare_scalp"

OD_CLASSES = CLOTHING_CLASSES + GLASSES_OD_CLASSES + HEADWEAR_OD_CLASSES
CATEGORY_ID = {name: idx for idx, name in enumerate(OD_CLASSES)}

THERMAL_TIERS: dict[str, list[str]] = {
    "light": ["sleeveless", "short_sleeve"],
    "light_mid": ["long_sleeve_light"],
    "mid": ["long_sleeve_knit", "knit_hoodie", "light_jacket"],
    "heavy": ["heavy_coat"],
}

# ---------------------------------------------------------------------------
# Legacy mapping (old prompt JSON → VLM classes, for disagreement logs)
# ---------------------------------------------------------------------------

LEGACY_CLOTHING_MAP: dict[str, str] = {
    "sleeveless_top": "sleeveless",
    "short_sleeve_top": "short_sleeve",
    "long_sleeve_plain": "long_sleeve_light",  # ambiguous; VLM may choose knit
    "long_sleeve_collared": "long_sleeve_light",
    "hoodie_sweatshirt": "knit_hoodie",
    "light_jacket": "light_jacket",
    "heavy_jacket_coat": "heavy_coat",
}

LEGACY_GLASSES_MAP: dict[str, str] = {
    "no_glasses": GLASSES_ABSENT,
    "eyeglasses": "eyeglasses",
    "sunglasses": "sunglasses",
}

LEGACY_HEADWEAR_MAP: dict[str, str] = {
    "no_headwear": HEADWEAR_ABSENT,
    "bare_head": HEADWEAR_BARE_SCALP,
    "cap": "cap",
    "beanie": "beanie",
    "hijab_headscarf": "hijab_headscarf",
}

# Pre-merge export used separate OD ids for each headwear subtype
LEGACY_HEADWEAR_OD_IDS = {
    "cap": 9,
    "beanie": 10,
    "hijab_headscarf": 11,
}


def is_headwear_vlm_class(class_name: str) -> bool:
    return class_name in HEADWEAR_VLM_CLASSES


def headwear_od_class(vlm_class: str) -> str | None:
    """Map VLM headwear subtype (or headwear) to export OD class."""
    if vlm_class in HEADWEAR_VLM_CLASSES or vlm_class == HEADWEAR_OD_CLASS:
        return HEADWEAR_OD_CLASS
    return None


def normalize_od_class(class_name: str) -> str | None:
    """Map any known class name to a current OD class, or None if not OD."""
    hw = headwear_od_class(class_name)
    if hw:
        return hw
    if class_name in CATEGORY_ID:
        return class_name
    return None


def category_id_for(class_name: str) -> int | None:
    """Return COCO category_id, or None for absent/manifest-only classes."""
    normalized = normalize_od_class(class_name)
    if normalized is None:
        return None
    return CATEGORY_ID[normalized]


def category_id_from_legacy(category_id: int, category_name: str | None = None) -> int | None:
    """Resolve category_id from pre-merge annotations (cap/beanie/hijab ids 9–11)."""
    if category_name:
        resolved = category_id_for(category_name)
        if resolved is not None:
            return resolved
    if category_id in LEGACY_HEADWEAR_OD_IDS.values():
        return CATEGORY_ID[HEADWEAR_OD_CLASS]
    if 0 <= category_id < len(OD_CLASSES):
        return category_id
    return None


def is_od_class(class_name: str) -> bool:
    return normalize_od_class(class_name) is not None
