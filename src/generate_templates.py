"""
generate_templates.py - Generate per-ref prompt templates and scene metadata using a VLM.

One-time setup step: sends reference images to an OpenAI vision model and receives:
  1. arms_visible (bool) — whether upper arms are clearly visible
  2. conditions whitelist — what lighting/weather/time/seat/window/cabin values fit
  3. prompt templates with placeholder slots

Uses OpenAI Batch API (50% cost discount) when multiple refs need processing.
Falls back to synchronous for a single ref.

Usage:
    python -m src.generate_templates [--force] [--sync]

    --force   Regenerate even if already populated.
    --sync    Use synchronous API instead of Batch API (faster, costs more).
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.config import VLM_MODEL

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """\
You analyze car cabin reference images and extract structured metadata.

Given a reference image of a car cabin (dashboard camera perspective), respond with
a JSON object containing:

{
  "arms_visible": true/false,
  "cabin_description": "...",
  "conditions": {
    "lighting": [...],
    "weather": [...],
    "time": [...],
    "seat_type": [...],
    "window_tint": [...],
    "cabin_color": [...]
  }
}

Rules for each field:

arms_visible:
- true if the person's upper arms (deltoid/bicep area) are clearly visible in frame
- false if only torso/chest is visible or arms are cropped out

cabin_description:
- 1-2 sentence description of the car cabin as seen in the image.
- Include: vehicle type (sedan, SUV, truck, etc.), camera mount position,
  visible interior elements (dashboard, center console, rearview mirror, steering wheel),
  seat style, and any distinctive features.
- Do NOT describe the person, their clothing, lighting, or weather.
- This is a fixed physical description of the cabin itself.

conditions — pick ONLY values that are realistic/compatible with this specific image:

lighting (pick from): "bright sunlight", "overcast", "night with interior lights", "dim cabin"
- Choose based on what lighting scenarios would look natural with this cabin/window setup.

weather (pick from): "rainy", "snowy", "sunny", "foggy"
- Choose what weather conditions could plausibly be shown through the windows.

time (pick from): "morning", "noon", "evening", "night"
- Choose what times of day match the lighting you selected.

seat_type (pick from): "leather seats", "fabric seats", "sports seats"
- Identify what's actually in the image.

window_tint (pick from): "clear windows", "lightly tinted windows", "heavily tinted windows"
- Identify what's visible in the image.

cabin_color (pick from): "black", "beige", "gray", "brown", "white"
- Identify the actual interior color.

Respond ONLY with the JSON object, no markdown fences or extra text.
"""

TEMPLATE_SYSTEM_PROMPT = """\
You describe car cabin images for use as synthetic data prompt templates.

For each reference image, output prompt templates with placeholder slots.

CRITICAL RULES:
- Every template MUST include ALL required placeholder slots for each person.
- Do NOT describe the person's specific clothing, accessories, age, gender,
  ethnicity, or headwear — use ONLY the placeholders listed below.
- Do NOT describe specific lighting, weather, time, cabin color, seat material,
  or window tint — use ONLY the placeholders.
- Write 2-3 phrasing variants of the same scene.
- Keep each variant to 2-3 sentences.
- Focus on what makes this camera angle unique (what's visible, framing).
- Describe camera angle, person position/pose, what body parts are visible.

REQUIRED SLOTS FOR occupancy=1 (single person):
Every template MUST contain ALL of these exactly:
  {ethnicity}, {gender}, {age}, {clothing_phrase}, {glasses_phrase},
  {headwear_phrase}, {seatbelt}, {lighting}, {weather}, {time},
  {cabin_color}, {seat_type}, {window_tint}

REQUIRED SLOTS FOR occupancy=2 (two people):
Every template MUST contain ALL of these exactly:
  Person 1 (driver): {ethnicity_1}, {gender_1}, {age_1}, {clothing_phrase_1},
    {glasses_phrase_1}, {headwear_phrase_1}, {seatbelt_1}
  Person 2 (passenger): {ethnicity_2}, {gender_2}, {age_2}, {clothing_phrase_2},
    {glasses_phrase_2}, {headwear_phrase_2}, {seatbelt_2}
  Scene: {lighting}, {weather}, {time}, {cabin_color}, {seat_type}, {window_tint}

Output format:
- For occupancy=1: start with [single] header.
- For occupancy=2: start with [dual] header.
- Separate variants with --- on its own line.
- Do NOT write templates without the required slots — they will be rejected.

EXAMPLE (occupancy=2):
[dual]
A dashboard camera captures two people in a {cabin_color} car with {seat_type}
and {window_tint}. The driver is a {ethnicity_1} {gender_1} {age_1}, wearing
{clothing_phrase_1}, {glasses_phrase_1}, {headwear_phrase_1}. {seatbelt_1}
The front passenger is a {ethnicity_2} {gender_2} {age_2}, wearing
{clothing_phrase_2}, {glasses_phrase_2}, {headwear_phrase_2}. {seatbelt_2}
The scene is {weather} at {time}, under {lighting} lighting.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: str = "refs/manifest.yaml") -> dict:
    """Load the full manifest YAML (preserving structure for write-back)."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def save_manifest(data: dict, manifest_path: str = "refs/manifest.yaml"):
    """Write manifest back to YAML."""
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def encode_image_base64(image_path: str) -> str:
    """Read and base64-encode an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(image_path: str) -> str:
    ext = Path(image_path).suffix.lstrip(".").lower()
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    return f"image/{ext}"


def build_image_message(image_b64: str, mime: str, text: str) -> list[dict]:
    """Build a user message with image + text."""
    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
        },
    ]


# ---------------------------------------------------------------------------
# Synchronous VLM calls (single ref or --sync mode)
# ---------------------------------------------------------------------------


def analyze_ref_sync(ref: dict, client) -> dict:
    """Analyze a single ref image synchronously."""
    image_b64 = encode_image_base64(ref["image_path"])
    mime = get_image_mime(ref["image_path"])

    user_text = (
        f"Reference ID: {ref['id']}\n"
        f"Occupancy: {ref.get('occupancy', 1)} person(s)\n\n"
        "Analyze this car cabin reference image and return the JSON metadata."
    )

    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": build_image_message(image_b64, mime, user_text)},
        ],
        max_completion_tokens=500,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
    return json.loads(raw)


def generate_template_sync(ref: dict, client) -> str:
    """Generate templates for a single ref image synchronously."""
    image_b64 = encode_image_base64(ref["image_path"])
    mime = get_image_mime(ref["image_path"])

    occupancy = ref.get("occupancy", 1)
    section = "single" if occupancy == 1 else "dual"

    user_text = (
        f"Reference ID: {ref['id']}\n"
        f"Occupancy: {occupancy} person(s)\n"
        f"Arms visible: {ref.get('arms_visible', 'unknown')}\n\n"
        f"Generate prompt templates (section: [{section}]) for this reference image."
    )

    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {"role": "system", "content": TEMPLATE_SYSTEM_PROMPT},
            {"role": "user", "content": build_image_message(image_b64, mime, user_text)},
        ],
        max_completion_tokens=2000,
        temperature=0.7,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Batch API (50% cost discount, async)
# ---------------------------------------------------------------------------


def prepare_batch_requests(refs: list[dict], task: str) -> list[dict]:
    """Prepare .jsonl batch request lines for analysis or template generation.

    task: "analysis" or "templates"
    """
    requests = []

    for ref in refs:
        image_b64 = encode_image_base64(ref["image_path"])
        mime = get_image_mime(ref["image_path"])
        ref_id = ref["id"]

        if task == "analysis":
            user_text = (
                f"Reference ID: {ref_id}\n"
                f"Occupancy: {ref.get('occupancy', 1)} person(s)\n\n"
                "Analyze this car cabin reference image and return the JSON metadata."
            )
            system_prompt = ANALYSIS_SYSTEM_PROMPT
            max_tokens = 500
            temperature = 0.2
        else:  # templates
            occupancy = ref.get("occupancy", 1)
            section = "single" if occupancy == 1 else "dual"
            user_text = (
                f"Reference ID: {ref_id}\n"
                f"Occupancy: {occupancy} person(s)\n"
                f"Arms visible: {ref.get('arms_visible', 'unknown')}\n\n"
                f"Generate prompt templates (section: [{section}]) for this reference image."
            )
            system_prompt = TEMPLATE_SYSTEM_PROMPT
            max_tokens = 2000
            temperature = 0.7

        user_content = build_image_message(image_b64, mime, user_text)

        request_line = {
            "custom_id": f"{task}_{ref_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": VLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_completion_tokens": max_tokens,
                "temperature": temperature,
            },
        }
        requests.append(request_line)

    return requests


def submit_batch(requests: list[dict], client, description: str) -> str:
    """Write .jsonl, upload to OpenAI, create batch job. Returns batch ID."""
    jsonl_path = "refs/_batch_input.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    batch_input_file = client.files.create(
        file=open(jsonl_path, "rb"),
        purpose="batch",
    )

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )

    # Clean up local file
    os.remove(jsonl_path)

    return batch.id


def wait_for_batch(batch_id: str, client, poll_interval: int = 10) -> dict:
    """Poll batch status until complete. Returns the batch object."""
    print(f"    Batch ID: {batch_id}")
    print(f"    Polling every {poll_interval}s...")

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status

        if status == "completed":
            print(f"    Batch completed.")
            return batch
        elif status in ("failed", "expired", "cancelled"):
            print(f"    Batch {status}.")
            if batch.errors:
                for err in batch.errors.data:
                    print(f"      Error: {err.message}")
            sys.exit(1)
        else:
            print(f"    Status: {status}...", end="\r")
            time.sleep(poll_interval)


def retrieve_batch_results(batch, client) -> dict[str, str]:
    """Download batch output and parse into {custom_id: response_content}."""
    file_response = client.files.content(batch.output_file_id)
    results = {}

    for line in file_response.text.strip().split("\n"):
        entry = json.loads(line)
        custom_id = entry["custom_id"]
        if entry.get("error"):
            print(f"    WARNING: {custom_id} failed: {entry['error']['message']}")
            continue
        content = entry["response"]["body"]["choices"][0]["message"]["content"]
        results[custom_id] = content

    return results


def run_batch_pipeline(
    refs_to_analyze: list[dict],
    refs_to_template: list[dict],
    client,
) -> tuple[dict, dict]:
    """Run analysis and template generation via Batch API.

    Returns (analysis_results, template_results) dicts keyed by ref_id.
    """
    all_requests = []

    if refs_to_analyze:
        all_requests.extend(prepare_batch_requests(refs_to_analyze, "analysis"))
    if refs_to_template:
        all_requests.extend(prepare_batch_requests(refs_to_template, "templates"))

    if not all_requests:
        return {}, {}

    n_analysis = len(refs_to_analyze)
    n_templates = len(refs_to_template)
    desc = f"ref analysis ({n_analysis}) + templates ({n_templates})"

    print(f"\n  [batch] Submitting {len(all_requests)} requests via Batch API (50% discount)...")
    batch_id = submit_batch(all_requests, client, desc)

    batch = wait_for_batch(batch_id, client)
    raw_results = retrieve_batch_results(batch, client)

    # Split results by task prefix
    analysis_results = {}
    template_results = {}

    for custom_id, content in raw_results.items():
        if custom_id.startswith("analysis_"):
            ref_id = custom_id[len("analysis_"):]
            # Parse JSON response
            raw = content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
            try:
                analysis_results[ref_id] = json.loads(raw)
            except json.JSONDecodeError:
                print(f"    WARNING: Could not parse analysis for {ref_id}")
        elif custom_id.startswith("templates_"):
            ref_id = custom_id[len("templates_"):]
            template_results[ref_id] = content

    return analysis_results, template_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate prompt templates and scene metadata per reference image using VLM."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if template/conditions already exist.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous API (faster but no 50%% discount). "
             "Default: uses Batch API for multiple refs.",
    )
    parser.add_argument(
        "--manifest",
        default="refs/manifest.yaml",
        help="Path to manifest YAML file.",
    )
    args = parser.parse_args()

    load_dotenv()

    manifest_data = load_manifest(args.manifest)
    refs = manifest_data.get("refs", [])
    if not refs:
        print("No references found in manifest.")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package required. Install with: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env or environment.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # --- Collect refs that need work ---
    refs_needing_analysis = []
    refs_needing_templates = []

    for ref in refs:
        if not os.path.exists(ref["image_path"]):
            print(f"  [skip] {ref['id']} — image not found: {ref['image_path']}")
            continue

        needs_analysis = (
            "arms_visible" not in ref
            or "conditions" not in ref
            or args.force
        )
        if needs_analysis:
            refs_needing_analysis.append(ref)
        else:
            print(f"  [skip] {ref['id']} — metadata already present")

        template_path = ref.get("template_file", f"refs/{ref['id']}.txt")
        ref["template_file"] = template_path
        if not os.path.exists(template_path) or args.force:
            refs_needing_templates.append(ref)
        else:
            print(f"  [skip] {ref['id']} — template exists: {template_path}")

    total_work = len(refs_needing_analysis) + len(refs_needing_templates)
    if total_work == 0:
        print("\nNothing to do. Use --force to regenerate.")
        return

    # --- Choose sync vs batch ---
    use_batch = not args.sync and total_work > 1

    if use_batch:
        # Batch API path (50% discount)
        analysis_results, template_results = run_batch_pipeline(
            refs_needing_analysis, refs_needing_templates, client
        )

        # Apply analysis results
        for ref in refs_needing_analysis:
            metadata = analysis_results.get(ref["id"], {})
            ref["arms_visible"] = metadata.get("arms_visible", True)
            ref["cabin_description"] = metadata.get("cabin_description", "")
            ref["conditions"] = metadata.get("conditions", {})
            print(f"  {ref['id']}: arms_visible={ref['arms_visible']}, "
                  f"cabin=\"{ref['cabin_description'][:60]}...\"")

        # Save templates
        for ref in refs_needing_templates:
            template_path = ref["template_file"]
            template_text = template_results.get(ref["id"], "")
            if template_text:
                os.makedirs(os.path.dirname(template_path) or ".", exist_ok=True)
                with open(template_path, "w", encoding="utf-8") as f:
                    f.write(template_text)
                print(f"  {ref['id']} -> saved: {template_path}")
            else:
                print(f"  {ref['id']} — WARNING: no template in batch response")

    else:
        # Synchronous path (single ref or --sync flag)
        print(f"\n  [sync] Processing {total_work} request(s) synchronously...")

        for ref in refs_needing_analysis:
            print(f"  [analyze] {ref['id']}...")
            metadata = analyze_ref_sync(ref, client)
            ref["arms_visible"] = metadata.get("arms_visible", True)
            ref["cabin_description"] = metadata.get("cabin_description", "")
            ref["conditions"] = metadata.get("conditions", {})
            print(f"            arms_visible={ref['arms_visible']}, "
                  f"cabin=\"{ref['cabin_description'][:60]}...\"")

        for ref in refs_needing_templates:
            template_path = ref["template_file"]
            print(f"  [gen] {ref['id']}...")
            template_text = generate_template_sync(ref, client)
            os.makedirs(os.path.dirname(template_path) or ".", exist_ok=True)
            with open(template_path, "w", encoding="utf-8") as f:
                f.write(template_text)
            print(f"        -> saved: {template_path}")

    # Write back updated manifest
    if refs_needing_analysis:
        save_manifest(manifest_data, args.manifest)
        print(f"\nManifest updated: {args.manifest}")

    print("\nDone.")


if __name__ == "__main__":
    main()
