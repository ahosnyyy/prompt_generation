"""
utils.py - Sampling, constraint logic, and output utilities for prompt generation.

Handles:
- Manifest loading and template parsing
- Stratified sampling with weather-clothing coupling
- Layering constraint building
- Color contrast soft constraints
- Structured JSON record creation
- File output
"""

import csv
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from src.config import (
    AGES,
    CABIN_COLORS,
    CLOTHING_TAXONOMY,
    COLOR_REROLL_PROBABILITY,
    ETHNICITIES,
    GARMENT_COLORS,
    GENDERS,
    GLASSES_DISTRIBUTION,
    GLASSES_TAXONOMY,
    HEADWEAR_DISTRIBUTION,
    HEADWEAR_TAXONOMY,
    LAYERING_DISTRIBUTION,
    LIGHTING_CONDITIONS,
    MIN_COLOR_CONTRAST_RATIO,
    SAMPLE_SIZE,
    SEATBELT_OPTIONS,
    SEAT_TYPES,
    SLEEVELESS_SHARE,
    THERMAL_TIERS,
    TIMES_OF_DAY,
    VALID_INNER_CLASSES,
    VALID_OUTER_CLASSES,
    WEATHER_CLOTHING_BIAS,
    WEATHER_CONDITIONS,
    WEATHER_LAYERING_BIAS,
    WINDOW_TINTS,
)


# ---------------------------------------------------------------------------
# Manifest & template loading
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: str = "refs/manifest.yaml") -> list[dict]:
    """Load reference profiles from the YAML manifest."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("refs", [])


def load_bboxes(ref: dict) -> dict:
    """Load bboxes and image_size from the per-ref bbox file.

    Looks for refs/{id}_bboxes.yaml. Returns dict with 'image_size' and 'bboxes'.
    """
    ref_id = ref["id"]
    bbox_path = os.path.join("refs", f"{ref_id}_bboxes.yaml")

    if not os.path.exists(bbox_path):
        return {"image_size": None, "bboxes": {}}

    with open(bbox_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return {
        "image_size": data.get("image_size"),
        "bboxes": data.get("bboxes", {}),
    }


def parse_templates(template_path: str) -> dict[str, list[str]]:
    """Parse a .txt template file into sections (single/dual).

    Returns dict like {"single": [...], "dual": [...]}.
    """
    if not os.path.exists(template_path):
        return {"single": [], "dual": []}

    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    sections: dict[str, list[str]] = {"single": [], "dual": []}
    current_section = None

    for block in content.split("\n["):
        block = block.strip()
        if not block:
            continue

        if block.startswith("single]") or block.startswith("[single]"):
            current_section = "single"
            block = block.split("]", 1)[1].strip()
        elif block.startswith("dual]") or block.startswith("[dual]"):
            current_section = "dual"
            block = block.split("]", 1)[1].strip()

        if current_section and block:
            templates = [t.strip() for t in block.split("\n---\n") if t.strip()]
            sections[current_section].extend(templates)

    # Validate: templates must have all required slots
    REQUIRED_SINGLE_SLOTS = [
        "{ethnicity}", "{gender}", "{age}",
        "{clothing_phrase}", "{glasses_phrase}", "{headwear_phrase}",
        "{seatbelt}",
        "{lighting}", "{weather}", "{time}",
        "{cabin_color}", "{seat_type}", "{window_tint}",
    ]
    REQUIRED_DUAL_SLOTS = [
        "{ethnicity_1}", "{gender_1}", "{age_1}",
        "{clothing_phrase_1}", "{glasses_phrase_1}", "{headwear_phrase_1}",
        "{seatbelt_1}",
        "{ethnicity_2}", "{gender_2}", "{age_2}",
        "{clothing_phrase_2}", "{glasses_phrase_2}", "{headwear_phrase_2}",
        "{seatbelt_2}",
        "{lighting}", "{weather}", "{time}",
        "{cabin_color}", "{seat_type}", "{window_tint}",
    ]

    sections["single"] = [
        t for t in sections["single"]
        if all(slot in t for slot in REQUIRED_SINGLE_SLOTS)
    ]
    sections["dual"] = [
        t for t in sections["dual"]
        if all(slot in t for slot in REQUIRED_DUAL_SLOTS)
    ]

    return sections


# ---------------------------------------------------------------------------
# Weighted random sampling utilities
# ---------------------------------------------------------------------------


def weighted_choice(distribution: dict[str, float]) -> str:
    """Pick a key from a {key: weight} dict using weighted random."""
    keys = list(distribution.keys())
    weights = [distribution[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


def sample_from_whitelist(whitelist: list | None, global_default: list) -> str:
    """Sample uniformly from per-ref whitelist, or global default."""
    pool = whitelist if whitelist else global_default
    return random.choice(pool)


# ---------------------------------------------------------------------------
# Clothing sampling with weather coupling
# ---------------------------------------------------------------------------


def get_thermal_tier(det_class: str) -> str:
    """Return the thermal tier for a given clothing det_class."""
    return CLOTHING_TAXONOMY[det_class]["thermal"]


def sample_clothing_det_class(
    weather: str, arms_visible: bool
) -> str:
    """Sample a clothing det_class respecting weather bias and arm visibility.

    If arms_visible is False, excludes sleeveless_top and short_sleeve_top.
    """
    bias = WEATHER_CLOTHING_BIAS.get(weather, {"heavy": 0.33, "mid": 0.34, "light": 0.33})

    # Pick a thermal tier first
    tier = weighted_choice(bias)

    # Get candidate classes for that tier
    candidates = THERMAL_TIERS[tier][:]

    if not arms_visible:
        candidates = [c for c in candidates if c not in ("sleeveless_top", "short_sleeve_top")]

    # Handle sleeveless share constraint within the light tier
    if tier == "light" and arms_visible and len(candidates) > 1:
        if random.random() < SLEEVELESS_SHARE:
            return "sleeveless_top"
        else:
            candidates = [c for c in candidates if c != "sleeveless_top"]

    if not candidates:
        # Fallback: re-pick from mid tier
        candidates = THERMAL_TIERS["mid"][:]

    return random.choice(candidates)


def sample_layering_mode(weather: str) -> str:
    """Sample layering mode with weather-dependent bias."""
    bias = WEATHER_LAYERING_BIAS.get(weather, LAYERING_DISTRIBUTION)
    return weighted_choice(bias)


# ---------------------------------------------------------------------------
# Layering logic
# ---------------------------------------------------------------------------


def sample_layered_clothing(
    weather: str, arms_visible: bool
) -> dict:
    """Sample clothing attributes including layering.

    Returns a dict matching the output record clothing schema.
    """
    layering_mode = sample_layering_mode(weather)

    if layering_mode == "single":
        det_class = sample_clothing_det_class(weather, arms_visible)
        gen_variant = random.choice(CLOTHING_TAXONOMY[det_class]["gen_variants"])
        color = random.choice(GARMENT_COLORS)
        return {
            "layering_mode": "single",
            "det_class": det_class,
            "gen_variant": gen_variant,
            "color": color,
        }

    # For layered modes, pick outer first then inner
    outer_det_class = random.choice(VALID_OUTER_CLASSES)
    # Apply weather bias: prefer heavier outer in cold weather
    if weather in ("snowy", "rainy") and random.random() < 0.5:
        outer_det_class = "heavy_jacket_coat"

    outer_gen_variant = random.choice(
        CLOTHING_TAXONOMY[outer_det_class]["gen_variants"]
    )

    inner_det_class = random.choice(VALID_INNER_CLASSES)
    if not arms_visible:
        inner_det_class = random.choice(
            [c for c in VALID_INNER_CLASSES if c not in ("sleeveless_top", "short_sleeve_top")]
        )
    inner_gen_variant = random.choice(
        CLOTHING_TAXONOMY[inner_det_class]["gen_variants"]
    )

    outer_color = random.choice(GARMENT_COLORS)
    inner_color = random.choice(GARMENT_COLORS)

    if layering_mode == "open_outer":
        return {
            "layering_mode": "open_outer",
            "inner_det_class": inner_det_class,
            "inner_gen_variant": inner_gen_variant,
            "inner_color": inner_color,
            "outer_det_class": outer_det_class,
            "outer_gen_variant": outer_gen_variant,
            "outer_color": outer_color,
            "color": f"{inner_color} and {outer_color}",
        }
    else:  # closed_outer
        return {
            "layering_mode": "closed_outer",
            "inner_det_class": inner_det_class,
            "outer_det_class": outer_det_class,
            "outer_gen_variant": outer_gen_variant,
            "outer_color": outer_color,
            "color": outer_color,
        }


# ---------------------------------------------------------------------------
# Glasses & headwear sampling
# ---------------------------------------------------------------------------


def sample_glasses() -> dict:
    """Sample glasses attributes."""
    det_class = weighted_choice(GLASSES_DISTRIBUTION)
    gen_variant = random.choice(GLASSES_TAXONOMY[det_class]["gen_variants"])
    is_present = GLASSES_TAXONOMY[det_class]["is_od_class"]

    result = {
        "det_class": det_class,
        "gen_variant": gen_variant,
        "has_glasses": is_present,
    }
    if is_present:
        result["od_bbox_class"] = det_class
    return result


def sample_headwear() -> dict:
    """Sample headwear attributes."""
    det_class = weighted_choice(HEADWEAR_DISTRIBUTION)
    gen_variant = random.choice(HEADWEAR_TAXONOMY[det_class]["gen_variants"])
    is_od = HEADWEAR_TAXONOMY[det_class]["is_od_class"]
    has_bare_scalp = HEADWEAR_TAXONOMY[det_class]["has_bare_scalp"]

    result = {
        "det_class": det_class,
        "gen_variant": gen_variant,
        "has_headwear": is_od,
        "has_bare_scalp": has_bare_scalp,
    }
    if is_od:
        result["od_bbox_class"] = det_class
    return result


# ---------------------------------------------------------------------------
# Color contrast soft constraint (§7.2)
# ---------------------------------------------------------------------------

# Colors considered "same" for contrast check
_COLOR_GROUPS = {
    "black": "dark",
    "navy": "dark",
    "brown": "dark",
    "olive": "dark",
    "gray": "neutral",
    "beige": "neutral",
    "white": "light",
    "yellow": "light",
    "red": "warm",
    "orange": "warm",
    "blue": "cool",
    "green": "cool",
    "purple": "cool",
}


def has_low_contrast(garment_color: str, cabin_color: str) -> bool:
    """Check if garment and cabin colors have low contrast."""
    g_group = _COLOR_GROUPS.get(garment_color, "")
    c_group = _COLOR_GROUPS.get(cabin_color, "")
    if not g_group or not c_group:
        return False
    return g_group == c_group


def apply_color_contrast(clothing: dict, cabin_color: str) -> dict:
    """Re-roll garment color if it clashes with cabin color (soft constraint)."""
    color_key = "color" if clothing["layering_mode"] == "single" else "outer_color"
    garment_color = clothing.get(color_key, "")

    if has_low_contrast(garment_color, cabin_color):
        if random.random() < COLOR_REROLL_PROBABILITY:
            new_color = random.choice(
                [c for c in GARMENT_COLORS if not has_low_contrast(c, cabin_color)]
            )
            clothing[color_key] = new_color
            if clothing["layering_mode"] == "single":
                clothing["color"] = new_color
            else:
                clothing["color"] = f"{clothing.get('inner_color', new_color)} and {new_color}"

    return clothing


# ---------------------------------------------------------------------------
# Person record sampling
# ---------------------------------------------------------------------------


def sample_person(
    weather: str, arms_visible: bool, cabin_color: str
) -> dict:
    """Sample all attributes for one person."""
    gender = random.choice(GENDERS)
    ethnicity = random.choice(ETHNICITIES)
    age = random.choice(AGES)
    seatbelt = random.choice(SEATBELT_OPTIONS)

    clothing = sample_layered_clothing(weather, arms_visible)
    clothing = apply_color_contrast(clothing, cabin_color)

    glasses = sample_glasses()
    headwear = sample_headwear()

    return {
        "gender": gender,
        "ethnicity": ethnicity,
        "age": age,
        "seatbelt": seatbelt,
        "clothing": clothing,
        "glasses": glasses,
        "headwear": headwear,
    }


# ---------------------------------------------------------------------------
# Clothing phrase construction (§9.3)
# ---------------------------------------------------------------------------


def build_clothing_phrase(clothing: dict) -> str:
    """Build the natural-language clothing phrase for a prompt template."""
    mode = clothing["layering_mode"]

    if mode == "single":
        return f"a {clothing['color']} {clothing['gen_variant']}"
    elif mode == "open_outer":
        inner = f"a {clothing['inner_color']} {clothing['inner_gen_variant']}"
        outer = f"an open {clothing['outer_color']} {clothing['outer_gen_variant']}"
        return f"{inner} under {outer}"
    else:  # closed_outer
        return f"a {clothing['outer_color']} {clothing['outer_gen_variant']}"


def build_glasses_phrase(glasses: dict) -> str:
    """Build the glasses phrase for a prompt template."""
    if glasses["det_class"] == "no_glasses":
        return "no glasses"
    return f"wearing {glasses['gen_variant']}"


def build_headwear_phrase(headwear: dict) -> str:
    """Build the headwear phrase for a prompt template."""
    if headwear["det_class"] == "no_headwear":
        return "no headwear"
    if headwear["det_class"] == "bare_head":
        return f"a {headwear['gen_variant']}"
    return f"a {headwear['gen_variant']}"


def build_seatbelt_phrase(seatbelt: str) -> str:
    """Build seatbelt sentence fragment."""
    if seatbelt == "wearing seatbelt":
        return "They are wearing a seatbelt."
    return "The seatbelt is not visible."


# ---------------------------------------------------------------------------
# Scene sampling
# ---------------------------------------------------------------------------


def sample_scene(ref: dict) -> dict:
    """Sample shared scene variables from ref whitelist (or global defaults)."""
    conditions = ref.get("conditions", {})

    return {
        "lighting": sample_from_whitelist(
            conditions.get("lighting"), LIGHTING_CONDITIONS
        ),
        "weather": sample_from_whitelist(
            conditions.get("weather"), WEATHER_CONDITIONS
        ),
        "time": sample_from_whitelist(conditions.get("time"), TIMES_OF_DAY),
        "cabin_color": sample_from_whitelist(
            conditions.get("cabin_color"), CABIN_COLORS
        ),
        "seat_type": sample_from_whitelist(
            conditions.get("seat_type"), SEAT_TYPES
        ),
        "window_tint": sample_from_whitelist(
            conditions.get("window_tint"), WINDOW_TINTS
        ),
    }


# ---------------------------------------------------------------------------
# Full prompt record creation
# ---------------------------------------------------------------------------


def create_prompt_record(
    ref: dict,
    scene: dict,
    persons: list[dict],
    prompt_text: str,
    image_name: str,
) -> dict:
    """Create a full structured output record for one generated image."""
    occupancy = len(persons)

    if occupancy == 1:
        roles = [ref.get("person_role", "driver")]
    else:
        roles = ["driver", "passenger"]

    person_records = []
    for i, person in enumerate(persons):
        record = {
            "role": roles[i],
            "gender": person["gender"],
            "ethnicity": person["ethnicity"],
            "age": person["age"],
            "seatbelt": person["seatbelt"],
            "clothing": person["clothing"],
            "glasses": person["glasses"],
            "headwear": person["headwear"],
        }
        person_records.append(record)

    return {
        "image_name": image_name,
        "ref_id": ref["id"],
        "occupancy": occupancy,
        "scene": scene,
        "persons": person_records,
        "prompt": prompt_text,
    }


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------


def save_prompts_to_file(records: list[dict], base_dir: str = "data"):
    """Save generated prompt records to JSON files.

    Creates:
      data/run_YYYYMMDD_HHMMSS/
        prompts.json          - filename + prompt text only
        prompts_full.json     - all records with full metadata
        individual_prompts/   - one JSON per image
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    prompts_dir = os.path.join(run_dir, "individual_prompts")
    os.makedirs(prompts_dir, exist_ok=True)

    main_prompts = []

    for record in records:
        filename = f"{record['image_name']}.json"
        filepath = os.path.join(prompts_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        main_prompts.append({
            "image_name": record["image_name"],
            "prompt": record["prompt"],
        })

    # Simple prompts list
    main_filepath = os.path.join(run_dir, "prompts.json")
    with open(main_filepath, "w", encoding="utf-8") as f:
        json.dump(main_prompts, f, indent=2)

    # Full records
    full_filepath = os.path.join(run_dir, "prompts_full.json")
    with open(full_filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # Per-ref CSV files matching example_batch.csv format
    CSV_COLUMNS = [
        "shot_id", "order_number", "shot_name", "colour_scheme",
        "scene_context", "dialogue", "lora_1", "lora_2", "lora_3",
        "ref_image_1", "ref_image_2", "ref_image_3", "video_file",
        "audio_vo", "positive_image", "negative_image",
        "positive_video", "negative_video", "info",
    ]

    by_ref: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_ref[record["ref_id"]].append(record)

    csv_dir = os.path.join(run_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    global_shot_id = 1
    for ref_id, ref_records in by_ref.items():
        csv_path = os.path.join(csv_dir, f"{ref_id}.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for record in ref_records:
                row = {col: "" for col in CSV_COLUMNS}
                row["shot_id"] = global_shot_id
                row["shot_name"] = record["image_name"]
                row["positive_image"] = record["prompt"]
                row["info"] = f"{record['image_name']}.json"
                writer.writerow(row)
                global_shot_id += 1

    print(f"{len(records)} prompts saved in: {run_dir}")
    print(f"  prompts.json:      {main_filepath}")
    print(f"  prompts_full.json: {full_filepath}")
    print(f"  individual:        {prompts_dir}/")
    print(f"  csv (per ref):     {csv_dir}/")

    return run_dir
