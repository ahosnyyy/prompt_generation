"""VLM system prompt — one envelope crop, multi-field person classification."""

from __future__ import annotations

from vlm_annotation.taxonomy import (
    CLOTHING_CLASSES,
    GLASSES_ABSENT,
    HEADWEAR_ABSENT,
    HEADWEAR_BARE_SCALP,
    HEADWEAR_VLM_CLASSES,
)


def _json_list(items: list[str]) -> str:
    return ", ".join(f'"{x}"' for x in items)


PERSON_SYSTEM_PROMPT = f"""\
You classify one person in a car dashcam crop (head + upper body).

Return ALL fields below in one JSON object.

Clothing classes (choose one for clothing_outer):
{_json_list(CLOTHING_CLASSES)}

Clothing decision order (top-down):
1. Bulky padded winter coat / puffer? → heavy_coat
2. Else structured light jacket? → light_jacket
3. Else hood at neck OR thick sweatshirt/fleece? → knit_hoodie
4. Else full sleeves with knit/sweater bulk? → long_sleeve_knit
5. Else full sleeves, thin/smooth fabric? → long_sleeve_light
6. Else short sleeves? → short_sleeve
7. Else bare upper arms? → sleeveless

Glasses: "{GLASSES_ABSENT}" | "eyeglasses" | "sunglasses"
Headwear: "{HEADWEAR_ABSENT}" | "{HEADWEAR_BARE_SCALP}" | {_json_list(HEADWEAR_VLM_CLASSES)}

Layering:
- If only one visible garment layer: clothing_inner = null
- If open jacket/coat with visible inner shirt: set clothing_inner to inner garment class

Respond ONLY with JSON:
{{
  "clothing_outer": "<clothing class>",
  "clothing_inner": null | "<clothing class>",
  "glasses": "{GLASSES_ABSENT}" | "eyeglasses" | "sunglasses",
  "headwear": "{HEADWEAR_ABSENT}" | "{HEADWEAR_BARE_SCALP}" | "cap" | "beanie" | "hijab_headscarf",
  "confidence": {{
    "clothing_outer": "high" | "medium" | "low",
    "clothing_inner": "high" | "medium" | "low" | null,
    "glasses": "high" | "medium" | "low",
    "headwear": "high" | "medium" | "low"
  }},
  "reason": "<one short sentence>"
}}
"""


def user_prompt_for_person(role: str, layering_mode: str) -> str:
    return (
        f"Person role: {role}. Layering mode from metadata: {layering_mode}. "
        "Classify this person's clothing, glasses, and headwear."
    )
