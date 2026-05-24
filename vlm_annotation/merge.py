"""Map person-level VLM output → refined pose bboxes on full frame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vlm_annotation.geometry import ExportSlot, PersonGeometry
from vlm_annotation.taxonomy import (
    GLASSES_ABSENT,
    HEADWEAR_ABSENT,
    HEADWEAR_BARE_SCALP,
    category_id_for,
    headwear_od_class,
    is_headwear_vlm_class,
    is_od_class,
)


def _annotation_record(
    slot: ExportSlot,
    class_name: str,
    width: int,
    height: int,
    confidence: str,
    reason: str,
    prompt_hint: str | None,
    vlm_subtype: str | None = None,
) -> dict[str, Any]:
    cat_id = category_id_for(class_name)
    if cat_id is None:
        raise ValueError(f"Cannot build OD record for absent class: {class_name}")

    record: dict[str, Any] = {
        "category": class_name,
        "category_id": cat_id,
        "area": slot.area,
        "bbox": slot.bbox.to_coco_pixels(width, height),
        "bbox_normalized": slot.bbox.to_coco_normalized(),
        "label_source": "vlm",
        "vlm_confidence": confidence,
        "vlm_reason": reason,
        "prompt_hint": prompt_hint,
    }
    if vlm_subtype:
        record["vlm_subtype"] = vlm_subtype
    return record


def _slot_for_area(person: PersonGeometry, area: str) -> ExportSlot | None:
    for slot in person.export_slots:
        if slot.area == area:
            return slot
    return None


def merge_person_annotations(
    person: PersonGeometry,
    vlm: dict[str, Any],
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[str], bool]:
    """Return (annotations, warnings, has_bare_scalp)."""
    annotations: list[dict[str, Any]] = []
    warnings: list[str] = []
    has_bare_scalp = False
    reason = vlm.get("reason", "")
    conf = vlm.get("confidence", {})

    # Clothing outer
    outer_slot = _slot_for_area(person, "clothing_outer")
    if outer_slot:
        cls = vlm["clothing_outer"]
        annotations.append(
            _annotation_record(
                outer_slot,
                cls,
                width,
                height,
                conf.get("clothing_outer", "medium"),
                reason,
                person.prompt_hints.get("clothing_outer"),
            )
        )
    else:
        warnings.append("missing_export_slot:clothing_outer")

    # Clothing inner (open_outer only)
    inner_cls = vlm.get("clothing_inner")
    inner_slot = _slot_for_area(person, "clothing_inner")
    if person.layering_mode == "open_outer":
        if inner_cls and inner_slot:
            annotations.append(
                _annotation_record(
                    inner_slot,
                    inner_cls,
                    width,
                    height,
                    conf.get("clothing_inner") or "medium",
                    reason,
                    person.prompt_hints.get("clothing_inner"),
                )
            )
        elif inner_cls and not inner_slot:
            warnings.append("vlm_inner_but_no_bbox")
    elif inner_cls:
        warnings.append("vlm_inner_ignored_not_open_outer")

    # Glasses
    glasses_cls = vlm["glasses"]
    glasses_slot = _slot_for_area(person, "glasses")
    if glasses_cls != GLASSES_ABSENT:
        if glasses_slot:
            if is_od_class(glasses_cls):
                annotations.append(
                    _annotation_record(
                        glasses_slot,
                        glasses_cls,
                        width,
                        height,
                        conf.get("glasses", "medium"),
                        reason,
                        person.prompt_hints.get("glasses"),
                    )
                )
        else:
            warnings.append("vlm_glasses_but_bbox_failed")

    # Headwear
    head_cls = vlm["headwear"]
    head_slot = _slot_for_area(person, "headwear")
    if head_cls == HEADWEAR_BARE_SCALP:
        has_bare_scalp = True
    elif head_cls != HEADWEAR_ABSENT:
        if head_slot:
            od_cls = headwear_od_class(head_cls)
            if od_cls:
                annotations.append(
                    _annotation_record(
                        head_slot,
                        od_cls,
                        width,
                        height,
                        conf.get("headwear", "medium"),
                        reason,
                        person.prompt_hints.get("headwear"),
                        vlm_subtype=head_cls if is_headwear_vlm_class(head_cls) else None,
                    )
                )
        else:
            warnings.append("vlm_headwear_but_bbox_failed")

    return annotations, warnings, has_bare_scalp


def merge_image_annotations(
    meta: dict[str, Any],
    persons: list[PersonGeometry],
    vlm_by_person: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    width = meta["width"]
    height = meta["height"]
    persons_out: list[dict[str, Any]] = []

    for person in persons:
        vlm = vlm_by_person.get(person.person_id)
        if not vlm:
            persons_out.append(
                {
                    "role": person.role,
                    "pose_index": person.pose_index,
                    "pose_score": person.pose_score,
                    "has_bare_scalp": False,
                    "warnings": ["vlm_missing"],
                    "annotations": [],
                }
            )
            continue

        anns, warnings, has_bare_scalp = merge_person_annotations(
            person, vlm, width, height
        )
        persons_out.append(
            {
                "role": person.role,
                "pose_index": person.pose_index,
                "pose_score": person.pose_score,
                "has_bare_scalp": has_bare_scalp,
                "warnings": warnings,
                "annotations": anns,
            }
        )

    passed = meta.get("passed", False) and bool(
        any(p["annotations"] for p in persons_out)
    )

    return {
        "image_name": meta["image_name"],
        "image_path": meta["image_path"],
        "width": width,
        "height": height,
        "ref_id": meta["ref_id"],
        "label_source": "vlm",
        "taxonomy_version": "option_a_v2",
        "quality": {
            "passed": passed,
            "reason": meta.get("fail_reason"),
            "warnings": meta.get("warnings", []),
            "checks": meta.get("checks", {}),
        },
        "persons": persons_out,
        "ignored_pose_indices": meta.get("ignored_pose_indices", []),
    }


def log_disagreements(
    person: PersonGeometry,
    vlm: dict[str, Any],
    out_path: Path,
) -> None:
    checks = [
        ("clothing_outer", vlm.get("clothing_outer"), person.prompt_hints.get("clothing_outer")),
        ("clothing_inner", vlm.get("clothing_inner"), person.prompt_hints.get("clothing_inner")),
        ("glasses", vlm.get("glasses"), person.prompt_hints.get("glasses")),
        ("headwear", vlm.get("headwear"), person.prompt_hints.get("headwear")),
    ]
    rows = []
    for field, vlm_val, hint in checks:
        if hint and vlm_val and hint != vlm_val:
            rows.append(
                {
                    "person_id": person.person_id,
                    "field": field,
                    "prompt_hint": hint,
                    "vlm_class": vlm_val,
                }
            )
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
