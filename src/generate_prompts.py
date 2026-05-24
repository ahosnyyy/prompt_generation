"""
generate_prompts.py - Main prompt generation pipeline.

Loads manifest + templates, samples structured attributes per ref,
fills prompt templates, and produces output records.

Flow (§9.1):
  1. Pick ref_id → allowed occupancy → pick 1 or 2.
  2. Sample scene vars uniformly from ref whitelist.
  3. Apply global soft constraints (weather-clothing coupling, color contrast).
  4. For each person: sample det_classes (stratified), gen_variants; apply layering.
  5. Pick template for (ref_id, occupancy).
  6. Fill slots → natural-language prompt.
  7. Generate image_name = {ref_id}_{sequence:06d}.
"""

import random

from src.config import SAMPLE_SIZE
from src.utils import (
    build_clothing_phrase,
    build_glasses_phrase,
    build_headwear_phrase,
    build_seatbelt_phrase,
    create_prompt_record,
    load_manifest,
    parse_templates,
    sample_person,
    sample_scene,
)


# ---------------------------------------------------------------------------
# Template filling
# ---------------------------------------------------------------------------

# Fallback templates (used when no .txt template file is found for a ref)
FALLBACK_SINGLE_DRIVER = (
    "A {ethnicity} {gender} {age} driver is captured by the dashboard camera. "
    "The image focuses on the upper body, showing them wearing {clothing_phrase}, "
    "{glasses_phrase}, {headwear_phrase}. {seatbelt} The car interior is "
    "{cabin_color} with {seat_type} and {window_tint}. The scene is set in "
    "{weather} weather at {time}, under {lighting} lighting."
)

FALLBACK_SINGLE_PASSENGER = (
    "A {ethnicity} {gender} {age} front passenger is captured by the dashboard camera. "
    "The image focuses on the upper body, showing them wearing {clothing_phrase}, "
    "{glasses_phrase}, {headwear_phrase}. {seatbelt} The car interior is "
    "{cabin_color} with {seat_type} and {window_tint}. The scene is set in "
    "{weather} weather at {time}, under {lighting} lighting."
)

FALLBACK_DUAL = (
    "A dashboard camera captures two people in a {cabin_color} car cabin with "
    "{seat_type} and {window_tint}. The driver is a {ethnicity_1} {gender_1} "
    "{age_1}, wearing {clothing_phrase_1}, {glasses_phrase_1}, "
    "{headwear_phrase_1}. {seatbelt_1} The front passenger is a {ethnicity_2} "
    "{gender_2} {age_2}, wearing {clothing_phrase_2}, {glasses_phrase_2}, "
    "{headwear_phrase_2}. {seatbelt_2} The scene is set in {weather} weather "
    "at {time}, under {lighting} lighting."
)


def fill_single_template(
    template: str, person: dict, scene: dict
) -> str:
    """Fill a single-person template with sampled attributes."""
    slots = {
        "ethnicity": person["ethnicity"],
        "gender": person["gender"],
        "age": person["age"],
        "clothing_phrase": build_clothing_phrase(person["clothing"]),
        "glasses_phrase": build_glasses_phrase(person["glasses"]),
        "headwear_phrase": build_headwear_phrase(person["headwear"]),
        "seatbelt": build_seatbelt_phrase(person["seatbelt"]),
        "cabin_color": scene["cabin_color"],
        "seat_type": scene["seat_type"],
        "window_tint": scene["window_tint"],
        "weather": scene["weather"],
        "time": scene["time"],
        "lighting": scene["lighting"],
    }

    try:
        return template.format(**slots)
    except KeyError:
        # Template may have extra/missing slots — fill what we can
        for key, val in slots.items():
            template = template.replace(f"{{{key}}}", val)
        return template


def fill_dual_template(
    template: str, persons: list[dict], scene: dict
) -> str:
    """Fill a dual-person template with indexed slots."""
    slots = {
        "cabin_color": scene["cabin_color"],
        "seat_type": scene["seat_type"],
        "window_tint": scene["window_tint"],
        "weather": scene["weather"],
        "time": scene["time"],
        "lighting": scene["lighting"],
    }

    for i, person in enumerate(persons, 1):
        slots[f"ethnicity_{i}"] = person["ethnicity"]
        slots[f"gender_{i}"] = person["gender"]
        slots[f"age_{i}"] = person["age"]
        slots[f"clothing_phrase_{i}"] = build_clothing_phrase(person["clothing"])
        slots[f"glasses_phrase_{i}"] = build_glasses_phrase(person["glasses"])
        slots[f"headwear_phrase_{i}"] = build_headwear_phrase(person["headwear"])
        slots[f"seatbelt_{i}"] = build_seatbelt_phrase(person["seatbelt"])

    try:
        return template.format(**slots)
    except KeyError:
        for key, val in slots.items():
            template = template.replace(f"{{{key}}}", val)
        return template


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def generate_prompts(
    manifest_path: str = "refs/manifest.yaml",
    sample_size: int | None = None,
) -> list[dict]:
    """Generate prompt records across all refs in the manifest.

    sample_size is the number of prompts PER REF (not total).
    Total output = sample_size × number of refs.
    """
    if sample_size is None:
        sample_size = SAMPLE_SIZE

    refs = load_manifest(manifest_path)
    if not refs:
        raise FileNotFoundError(
            f"No refs found in manifest: {manifest_path}. "
            "Ensure refs/manifest.yaml exists with at least one ref entry."
        )

    # Load templates for all refs
    ref_templates: dict[str, dict[str, list[str]]] = {}
    for ref in refs:
        tpath = ref.get("template_file", f"refs/{ref['id']}.txt")
        ref_templates[ref["id"]] = parse_templates(tpath)

    records: list[dict] = []
    sequence_counters: dict[str, int] = {}

    for ref_idx, ref in enumerate(refs):
        ref_id = ref["id"]
        ref_count = sample_size
        sequence_counters.setdefault(ref_id, 0)

        arms_visible = ref.get("arms_visible", True)
        occupancy = ref.get("occupancy", 1)
        templates = ref_templates[ref_id]

        for _ in range(ref_count):

            # Step 2: sample scene
            scene = sample_scene(ref)

            # Step 3-4: sample persons
            persons = []
            for _ in range(occupancy):
                person = sample_person(
                    weather=scene["weather"],
                    arms_visible=arms_visible,
                    cabin_color=scene["cabin_color"],
                )
                persons.append(person)

            # Step 5-6: pick template and fill
            if occupancy == 1:
                pool = templates.get("single", [])
                if pool:
                    tmpl = random.choice(pool)
                else:
                    role = ref.get("person_role", "driver")
                    tmpl = FALLBACK_SINGLE_DRIVER if role == "driver" else FALLBACK_SINGLE_PASSENGER
                prompt_text = fill_single_template(tmpl, persons[0], scene)
            else:
                pool = templates.get("dual", [])
                tmpl = random.choice(pool) if pool else FALLBACK_DUAL
                prompt_text = fill_dual_template(tmpl, persons, scene)

            # Step 7: generate image_name
            seq = sequence_counters[ref_id]
            image_name = f"{ref_id}_{seq:06d}"
            sequence_counters[ref_id] = seq + 1

            # Create record
            record = create_prompt_record(
                ref=ref,
                scene=scene,
                persons=persons,
                prompt_text=prompt_text,
                image_name=image_name,
            )
            records.append(record)

    return records
