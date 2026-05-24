"""Person envelope (VLM crop) + refined export bboxes from OpenPose."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.bbox_annotation import (
    Bbox,
    FACE_CONTOUR,
    Point,
    PosePerson,
    _collect_points,
    _points_bbox,
    _valid_bbox,
    compute_glasses_bbox,
    compute_headwear_bbox,
    compute_inner_bbox,
    compute_torso_bbox,
    load_ref_steering_side,
    match_poses_to_roles,
    parse_pose_file,
)

from vlm_annotation.taxonomy import (
    LEGACY_CLOTHING_MAP,
    LEGACY_GLASSES_MAP,
    LEGACY_HEADWEAR_MAP,
)

# Envelope padding (VLM crop only — not exported)
ENVELOPE_PAD_X = 0.18
ENVELOPE_PAD_Y_BOTTOM = 0.10
ENVELOPE_PAD_Y_TOP_MIN = 0.12  # min upward pad as fraction of body keypoint span
# Face contour tracks jaw, not crown — extend above forehead for hats/hair
ENVELOPE_HEAD_ROOM_FACE_H = 0.55


@dataclass
class ExportSlot:
    """Refined pose bbox — used in COCO export, not sent to VLM."""

    area: str
    bbox: Bbox
    prompt_hint: str | None = None


@dataclass
class PersonGeometry:
    """One person: envelope for VLM + refined slots for export."""

    person_id: str
    image_name: str
    role: str
    envelope: Bbox
    export_slots: list[ExportSlot] = field(default_factory=list)
    layering_mode: str = "single"
    pose_index: int = 0
    pose_score: int = 0
    prompt_hints: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "image_name": self.image_name,
            "role": self.role,
            "layering_mode": self.layering_mode,
            "pose_index": self.pose_index,
            "pose_score": self.pose_score,
            "envelope_normalized": self.envelope.to_coco_normalized(),
            "prompt_hints": self.prompt_hints,
            "export_slots": [
                {
                    "area": s.area,
                    "bbox_normalized": s.bbox.to_coco_normalized(),
                    "prompt_hint": s.prompt_hint,
                }
                for s in self.export_slots
            ],
        }


def person_id(image_name: str, role: str) -> str:
    return f"{image_name}__{role}"


def _legacy_hint(domain: str, raw: str | None) -> str | None:
    if not raw:
        return None
    maps = {
        "clothing": LEGACY_CLOTHING_MAP,
        "glasses": LEGACY_GLASSES_MAP,
        "headwear": LEGACY_HEADWEAR_MAP,
    }
    m = maps.get(domain, {})
    return m.get(raw, raw)


def _collect_pose_points(pose: PosePerson) -> list[Point]:
    points: list[Point] = []
    for p in pose.body:
        if p is not None:
            points.append(p)
    for p in pose.face:
        if p is not None:
            points.append(p)
    return points


def _face_bbox_and_height(pose: PosePerson) -> tuple[Bbox | None, float]:
    contour = _collect_points(FACE_CONTOUR, pose.face)
    if not contour:
        return None, 0.0
    face_bbox = _points_bbox(contour)
    if face_bbox is None:
        return None, 0.0
    face_h = max(face_bbox.y_max - face_bbox.y_min, 0.02)
    return face_bbox, face_h


def compute_person_envelope(
    pose: PosePerson,
    pad_x: float = ENVELOPE_PAD_X,
    pad_y_bottom: float = ENVELOPE_PAD_Y_BOTTOM,
) -> Bbox | None:
    """Union of body + face keypoints, extra head room, union with headwear region.

    Asymmetric padding: more space above the face (caps/hijabs) than below hips.
    Clamped to [0, 1]. Used for VLM crops only.
    """
    points = _collect_pose_points(pose)
    if len(points) < 3:
        return None

    raw = _points_bbox(points)
    if raw is None:
        return None

    w = raw.x_max - raw.x_min
    h = raw.y_max - raw.y_min
    if w <= 0 or h <= 0:
        return None

    pad_x_amount = w * pad_x
    pad_bottom = h * pad_y_bottom
    pad_top_base = h * ENVELOPE_PAD_Y_TOP_MIN

    face_bbox, face_h = _face_bbox_and_height(pose)
    if face_bbox is not None:
        head_room = face_h * ENVELOPE_HEAD_ROOM_FACE_H
        y_min = min(raw.y_min - pad_top_base, face_bbox.y_min - head_room)
    else:
        y_min = raw.y_min - pad_top_base * 1.5

    envelope = Bbox(
        x_min=raw.x_min - pad_x_amount,
        y_min=y_min,
        x_max=raw.x_max + pad_x_amount,
        y_max=raw.y_max + pad_bottom,
    ).clamp()

    # Ensure envelope covers the export headwear region when geometry is available
    headwear = compute_headwear_bbox(pose, "cap")
    if headwear:
        hw = headwear.clamp()
        envelope = Bbox(
            x_min=min(envelope.x_min, hw.x_min),
            y_min=min(envelope.y_min, hw.y_min),
            x_max=max(envelope.x_max, hw.x_max),
            y_max=max(envelope.y_max, hw.y_max),
        ).clamp()

    return envelope


def _export_slots_for_person(
    clothing: dict,
    pose: PosePerson,
    glasses_hint: str | None,
    headwear_hint: str | None,
) -> list[ExportSlot]:
    slots: list[ExportSlot] = []
    mode = clothing.get("layering_mode", "single")

    torso = compute_torso_bbox(pose)
    if torso is None:
        return slots

    if mode == "single":
        bbox = torso.clamp()
        if _valid_bbox(bbox, area="clothing_outer"):
            slots.append(
                ExportSlot(
                    area="clothing_outer",
                    bbox=bbox,
                    prompt_hint=_legacy_hint("clothing", clothing.get("det_class")),
                )
            )
    elif mode == "closed_outer":
        bbox = torso.clamp()
        if _valid_bbox(bbox, area="clothing_outer"):
            slots.append(
                ExportSlot(
                    area="clothing_outer",
                    bbox=bbox,
                    prompt_hint=_legacy_hint("clothing", clothing.get("outer_det_class")),
                )
            )
    elif mode == "open_outer":
        outer = torso.clamp()
        inner = compute_inner_bbox(torso).clamp()
        if _valid_bbox(outer, area="clothing_outer"):
            slots.append(
                ExportSlot(
                    area="clothing_outer",
                    bbox=outer,
                    prompt_hint=_legacy_hint("clothing", clothing.get("outer_det_class")),
                )
            )
        if _valid_bbox(inner, area="clothing_inner"):
            slots.append(
                ExportSlot(
                    area="clothing_inner",
                    bbox=inner,
                    prompt_hint=_legacy_hint("clothing", clothing.get("inner_det_class")),
                )
            )

    glasses_bbox = compute_glasses_bbox(pose)
    if glasses_bbox and _valid_bbox(glasses_bbox.clamp(), area="glasses"):
        slots.append(
            ExportSlot(
                area="glasses",
                bbox=glasses_bbox.clamp(),
                prompt_hint=glasses_hint,
            )
        )

    head_bbox = compute_headwear_bbox(pose, "cap")
    if head_bbox and _valid_bbox(head_bbox.clamp(), area="headwear"):
        slots.append(
            ExportSlot(
                area="headwear",
                bbox=head_bbox.clamp(),
                prompt_hint=headwear_hint,
            )
        )

    return slots


def build_geometry_for_image(
    prompt_path: Path,
    pose_path: Path,
    steering_side: str,
) -> tuple[list[PersonGeometry], dict[str, Any]]:
    with open(prompt_path, encoding="utf-8") as f:
        prompt = json.load(f)

    image_name = prompt["image_name"]
    occupancy = prompt["occupancy"]
    pose_people, width, height = parse_pose_file(pose_path)

    meta: dict[str, Any] = {
        "image_name": image_name,
        "ref_id": prompt["ref_id"],
        "width": width,
        "height": height,
        "image_path": f"images/{prompt['ref_id']}/{image_name}.png",
        "occupancy": occupancy,
        "passed": False,
        "fail_reason": None,
        "warnings": [],
        "checks": {
            "expected_occupancy": occupancy,
            "detected_pose_count": len(pose_people),
        },
        "ignored_pose_indices": [],
    }

    matched, ignored, warnings = match_poses_to_roles(
        pose_people, occupancy, steering_side
    )
    meta["warnings"] = warnings
    meta["ignored_pose_indices"] = ignored

    if matched is None:
        meta["fail_reason"] = "pose_count_mismatch"
        meta["checks"]["matched_person_count"] = 0
        return [], meta

    meta["checks"]["matched_person_count"] = len(matched)
    prompt_persons = prompt["persons"]
    if len(matched) != len(prompt_persons):
        meta["fail_reason"] = "prompt_pose_count_mismatch"
        return [], meta

    persons: list[PersonGeometry] = []

    for prompt_person, pose in zip(prompt_persons, matched):
        role = prompt_person["role"]
        clothing = prompt_person["clothing"]
        mode = clothing.get("layering_mode", "single")

        envelope = compute_person_envelope(pose)
        if envelope is None:
            meta["warnings"].append(f"envelope_failed:{role}")
            continue

        hints = {
            "clothing_outer": None,
            "clothing_inner": None,
            "glasses": _legacy_hint(
                "glasses", prompt_person["glasses"].get("det_class")
            ),
            "headwear": _legacy_hint(
                "headwear", prompt_person["headwear"].get("det_class")
            ),
        }

        export_slots = _export_slots_for_person(
            clothing, pose, hints["glasses"], hints["headwear"]
        )
        if not export_slots:
            meta["warnings"].append(f"export_slots_failed:{role}")
            continue

        hints["clothing_outer"] = next(
            (s.prompt_hint for s in export_slots if s.area == "clothing_outer"),
            None,
        )
        hints["clothing_inner"] = next(
            (s.prompt_hint for s in export_slots if s.area == "clothing_inner"),
            None,
        )

        persons.append(
            PersonGeometry(
                person_id=person_id(image_name, role),
                image_name=image_name,
                role=role,
                envelope=envelope,
                export_slots=export_slots,
                layering_mode=mode,
                pose_index=pose.index,
                pose_score=pose.torso_score,
                prompt_hints=hints,
            )
        )

    if not persons:
        meta["fail_reason"] = "no_valid_person_geometry"
        return [], meta

    meta["passed"] = True
    return persons, meta


def load_steering_for_ref(manifest_path: Path, ref_id: str) -> str:
    return load_ref_steering_side(str(manifest_path), ref_id)
