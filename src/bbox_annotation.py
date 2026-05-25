"""
bbox_annotation.py - Derive COCO-style bboxes from OpenPose keypoints + prompt metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils import load_manifest

# ---------------------------------------------------------------------------
# OpenPose indices
# ---------------------------------------------------------------------------

BODY_NOSE = 0
BODY_NECK = 1
BODY_R_SHOULDER = 2
BODY_R_ELBOW = 3
BODY_R_WRIST = 4
BODY_L_SHOULDER = 5
BODY_L_ELBOW = 6
BODY_L_WRIST = 7
BODY_MID_HIP = 8

TORSO_BODY_INDICES = [
    BODY_NECK,
    BODY_R_SHOULDER,
    BODY_R_ELBOW,
    BODY_R_WRIST,
    BODY_L_SHOULDER,
    BODY_L_ELBOW,
    BODY_L_WRIST,
    BODY_MID_HIP,
]

FACE_CONTOUR = list(range(0, 17))
FACE_EYEBROWS = list(range(17, 27))
FACE_EYES = list(range(36, 48))
FACE_MOUTH = list(range(48, 68))
FACE_GLASSES = FACE_EYEBROWS + FACE_EYES
FACE_CHIN_INDEX = 8

# ---------------------------------------------------------------------------
# OD taxonomy
# ---------------------------------------------------------------------------

OD_CLASSES = [
    "sleeveless_top",
    "short_sleeve_top",
    "long_sleeve_plain",
    "long_sleeve_collared",
    "hoodie_sweatshirt",
    "light_jacket",
    "heavy_jacket_coat",
    "eyeglasses",
    "sunglasses",
    "cap",
    "beanie",
    "hijab_headscarf",
]

CATEGORY_ID = {name: idx for idx, name in enumerate(OD_CLASSES)}

AREA_COLORS = {
    "clothing_outer": (0, 200, 0),
    "clothing_inner": (0, 220, 220),
    "glasses": (255, 200, 0),
    "headwear": (220, 0, 220),
}

MIN_TORSO_KEYPOINTS = 2  # relaxed: e.g. neck + one shoulder
MIN_BBOX_AREA_RATIO = 0.0015
MIN_KEYPOINT_CONF = 0.01

# Headwear bbox — face contour sits on jaw/cheeks; extend up for cap/hijab crown
HEADWEAR_TOP_PAD_FACE_H = 0.45  # lift top edge above face contour (was 0.40, briefly 0.55)
HEADWEAR_BOX_HEIGHT_FACE_H = 0.85  # box height vs face height (was 0.78)
HEADWEAR_FINAL_PAD_X = 0.18
HEADWEAR_FINAL_PAD_Y_TOP = 0.36  # extra upward pad after box build (was 0.32, briefly 0.48)
HEADWEAR_FINAL_PAD_Y_BOTTOM = 0.01


@dataclass
class Point:
    x: float
    y: float
    index: int


@dataclass
class PosePerson:
    index: int
    body: list[Point | None]
    face: list[Point | None]
    neck_x: float | None = None
    torso_score: int = 0


@dataclass
class Bbox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def clamp(self) -> Bbox:
        return Bbox(
            x_min=max(0.0, min(1.0, self.x_min)),
            y_min=max(0.0, min(1.0, self.y_min)),
            x_max=max(0.0, min(1.0, self.x_max)),
            y_max=max(0.0, min(1.0, self.y_max)),
        )

    def area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def to_coco_normalized(self) -> list[float]:
        w = self.x_max - self.x_min
        h = self.y_max - self.y_min
        return [self.x_min, self.y_min, w, h]

    def to_coco_pixels(self, width: int, height: int) -> list[int]:
        n = self.to_coco_normalized()
        return [
            int(round(n[0] * width)),
            int(round(n[1] * height)),
            int(round(n[2] * width)),
            int(round(n[3] * height)),
        ]


@dataclass
class AnnotationRecord:
    category: str
    area: str
    bbox: Bbox
    category_id: int = 0

    def to_dict(self, width: int, height: int) -> dict[str, Any]:
        return {
            "category": self.category,
            "category_id": self.category_id,
            "area": self.area,
            "bbox": self.bbox.to_coco_pixels(width, height),
            "bbox_normalized": self.bbox.to_coco_normalized(),
        }


@dataclass
class PersonAnnotation:
    role: str
    pose_index: int
    pose_score: int
    annotations: list[AnnotationRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ImageAnnotationResult:
    image_name: str
    image_path: str
    width: int
    height: int
    ref_id: str
    passed: bool
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    persons: list[PersonAnnotation] = field(default_factory=list)
    ignored_pose_indices: list[int] = field(default_factory=list)
    fail_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_name": self.image_name,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "ref_id": self.ref_id,
            "quality": {
                "passed": self.passed,
                "reason": self.fail_reason,
                "warnings": self.warnings,
                "checks": self.checks,
            },
            "persons": [
                {
                    "role": p.role,
                    "pose_index": p.pose_index,
                    "pose_score": p.pose_score,
                    "warnings": p.warnings,
                    "annotations": [
                        a.to_dict(self.width, self.height) for a in p.annotations
                    ],
                }
                for p in self.persons
            ],
            "ignored_pose_indices": self.ignored_pose_indices,
        }


# ---------------------------------------------------------------------------
# Keypoint parsing
# ---------------------------------------------------------------------------


def _parse_keypoint_array(flat: list[float] | None, count: int) -> list[Point | None]:
    if not flat:
        return [None] * count

    points: list[Point | None] = []
    for idx in range(count):
        base = idx * 3
        if base + 2 >= len(flat):
            points.append(None)
            continue
        x, y, conf = flat[base], flat[base + 1], flat[base + 2]
        if conf <= MIN_KEYPOINT_CONF or x <= 0 or y <= 0:
            points.append(None)
        else:
            points.append(Point(x=x, y=y, index=idx))
    return points


def parse_pose_file(pose_path: Path) -> tuple[list[PosePerson], int, int]:
    with open(pose_path, encoding="utf-8") as f:
        data = json.load(f)

    frame = data[0] if isinstance(data, list) else data
    width = int(frame.get("canvas_width", 0))
    height = int(frame.get("canvas_height", 0))

    people: list[PosePerson] = []
    for i, person in enumerate(frame.get("people", [])):
        body = _parse_keypoint_array(person.get("pose_keypoints_2d"), 18)
        face = _parse_keypoint_array(person.get("face_keypoints_2d"), 70)

        neck = body[BODY_NECK]
        torso_score = sum(1 for idx in TORSO_BODY_INDICES if body[idx] is not None)
        if body[BODY_NOSE] is not None:
            torso_score += 1

        people.append(
            PosePerson(
                index=i,
                body=body,
                face=face,
                neck_x=neck.x if neck else (
                    body[BODY_NOSE].x if body[BODY_NOSE] else None
                ),
                torso_score=torso_score,
            )
        )

    return people, width, height


def _collect_points(indices: list[int], points: list[Point | None]) -> list[Point]:
    return [points[i] for i in indices if i < len(points) and points[i] is not None]


def _points_bbox(points: list[Point]) -> Bbox | None:
    if not points:
        return None
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return Bbox(min(xs), min(ys), max(xs), max(ys))


def _pad_bbox(bbox: Bbox, pad_x: float, pad_y_top: float, pad_y_bottom: float) -> Bbox:
    w = bbox.x_max - bbox.x_min
    h = bbox.y_max - bbox.y_min
    return Bbox(
        x_min=bbox.x_min - w * pad_x,
        y_min=bbox.y_min - h * pad_y_top,
        x_max=bbox.x_max + w * pad_x,
        y_max=bbox.y_max + h * pad_y_bottom,
    ).clamp()


# ---------------------------------------------------------------------------
# Bbox builders
# ---------------------------------------------------------------------------


def _estimate_wrist(body: list[Point | None], side: str) -> Point | None:
    if side == "right":
        shoulder, elbow, wrist = (
            body[BODY_R_SHOULDER],
            body[BODY_R_ELBOW],
            body[BODY_R_WRIST],
        )
    else:
        shoulder, elbow, wrist = (
            body[BODY_L_SHOULDER],
            body[BODY_L_ELBOW],
            body[BODY_L_WRIST],
        )

    if wrist:
        return wrist
    if shoulder and elbow:
        return Point(
            x=shoulder.x + 1.3 * (elbow.x - shoulder.x),
            y=shoulder.y + 1.3 * (elbow.y - shoulder.y),
            index=-1,
        )
    if shoulder:
        return Point(x=shoulder.x, y=shoulder.y + 0.12, index=-1)
    return None


def _mouth_bottom_y(pose: PosePerson) -> float | None:
    mouth = _collect_points(FACE_MOUTH, pose.face)
    if mouth:
        return max(p.y for p in mouth)

    if len(pose.face) > FACE_CHIN_INDEX and pose.face[FACE_CHIN_INDEX]:
        return pose.face[FACE_CHIN_INDEX].y

    return None


def compute_torso_bbox(pose: PosePerson) -> Bbox | None:
    body = pose.body
    base_points = _collect_points(TORSO_BODY_INDICES, body)

    for side in ("right", "left"):
        est = _estimate_wrist(body, side)
        if est and body[BODY_R_WRIST if side == "right" else BODY_L_WRIST] is None:
            base_points.append(est)

    bbox = _points_bbox(base_points)
    if bbox is None:
        return None

    if body[BODY_NECK] is None and body[BODY_MID_HIP] is None:
        shoulders = [body[BODY_R_SHOULDER], body[BODY_L_SHOULDER]]
        if sum(1 for s in shoulders if s is not None) < 1:
            return None

    h = max(bbox.y_max - bbox.y_min, 0.05)
    padded = _pad_bbox(bbox, pad_x=0.12, pad_y_top=0.3, pad_y_bottom=0.06)

    mouth_bottom = _mouth_bottom_y(pose)
    if mouth_bottom is not None:
        min_top = mouth_bottom + 0.003
        if padded.y_min < min_top:
            padded = Bbox(
                x_min=padded.x_min,
                y_min=min_top,
                x_max=padded.x_max,
                y_max=padded.y_max,
            ).clamp()

    return padded


def compute_inner_bbox(outer: Bbox) -> Bbox:
    w = outer.x_max - outer.x_min
    h = outer.y_max - outer.y_min
    cx = (outer.x_min + outer.x_max) / 2
    cy = (outer.y_min + outer.y_max) / 2
    inner_w = w * 0.30
    inner_h = h * 0.88
    return Bbox(
        x_min=cx - inner_w / 2,
        y_min=cy - inner_h / 2,
        x_max=cx + inner_w / 2,
        y_max=cy + inner_h / 2,
    ).clamp()


def compute_glasses_bbox(pose: PosePerson) -> Bbox | None:
    points = _collect_points(FACE_GLASSES, pose.face)
    bbox = _points_bbox(points)
    if bbox is None:
        return None
    return _pad_bbox(bbox, pad_x=0.15, pad_y_top=0.08, pad_y_bottom=0.72)


def compute_headwear_bbox(pose: PosePerson, headwear_class: str) -> Bbox | None:
    """Unified cap-style geometry for all headwear classes (class label comes from prompt)."""
    del headwear_class  # geometry is shared; category differs per prompt

    contour = _collect_points(FACE_CONTOUR, pose.face)
    eyebrows = _collect_points(FACE_EYEBROWS, pose.face)
    if not contour:
        return None

    face_bbox = _points_bbox(contour)
    if face_bbox is None:
        return None

    face_h = max(face_bbox.y_max - face_bbox.y_min, 0.02)

    if eyebrows:
        y_bottom = max(p.y for p in eyebrows)
    else:
        y_bottom = face_bbox.y_min + face_h * 0.35

    box_h = face_h * HEADWEAR_BOX_HEIGHT_FACE_H
    y_top = face_bbox.y_min - face_h * HEADWEAR_TOP_PAD_FACE_H
    y_top = min(y_top, y_bottom - box_h)
    y_bottom = max(y_bottom, y_top + box_h * 0.5)
    y_top = y_bottom - box_h

    y_min = min(y_top, y_bottom)
    y_max = max(y_top, y_bottom)

    return _pad_bbox(
        Bbox(face_bbox.x_min, y_min, face_bbox.x_max, y_max),
        pad_x=HEADWEAR_FINAL_PAD_X,
        pad_y_top=HEADWEAR_FINAL_PAD_Y_TOP,
        pad_y_bottom=HEADWEAR_FINAL_PAD_Y_BOTTOM,
    )


def _valid_bbox(bbox: Bbox, area: str = "") -> bool:
    if bbox.area() < MIN_BBOX_AREA_RATIO:
        return False
    w = bbox.x_max - bbox.x_min
    h = bbox.y_max - bbox.y_min
    if w <= 0 or h <= 0:
        return False
    ratio = w / h
    if area in ("glasses", "headwear"):
        return 0.05 <= ratio <= 10.0
    return 0.15 <= ratio <= 6.0


def _make_record(category: str, area: str, bbox: Bbox) -> AnnotationRecord | None:
    bbox = bbox.clamp()
    if not _valid_bbox(bbox, area=area):
        return None
    return AnnotationRecord(
        category=category,
        area=area,
        bbox=bbox,
        category_id=CATEGORY_ID[category],
    )


# ---------------------------------------------------------------------------
# Person matching
# ---------------------------------------------------------------------------


def _driver_passenger_sort_key(pose: PosePerson, steering_side: str) -> float:
    x = pose.neck_x
    if x is None:
        neck = pose.body[BODY_NECK]
        x = neck.x if neck else 0.5
    return x if steering_side == "left" else -x


def match_poses_to_roles(
    pose_people: list[PosePerson],
    occupancy: int,
    steering_side: str,
) -> tuple[list[PosePerson] | None, list[int], list[str]]:
    """Return matched poses in prompt person order, ignored indices, warnings."""
    warnings: list[str] = []

    valid = [p for p in pose_people if p.torso_score >= MIN_TORSO_KEYPOINTS]
    if len(valid) < occupancy:
        return None, [], []

    if len(valid) > occupancy:
        ignored_count = len(valid) - occupancy
        warnings.append(
            f"extra_pose_person: {ignored_count} ignored "
            f"(detected={len(valid)}, expected={occupancy})"
        )

    if occupancy == 1:
        best = max(valid, key=lambda p: (p.torso_score, p.index))
        ignored = [p.index for p in pose_people if p.index != best.index]
        return [best], ignored, warnings

    # Dual: driver = max x for LHD (left), min x for RHD (right)
    by_driver_x = sorted(
        valid,
        key=lambda p: _driver_passenger_sort_key(p, steering_side),
        reverse=True,
    )
    driver = by_driver_x[0]
    others = [p for p in valid if p.index != driver.index]
    if not others:
        return None, [], []

    if steering_side == "left":
        passenger = min(others, key=lambda p: p.neck_x if p.neck_x is not None else 1.0)
    else:
        passenger = max(others, key=lambda p: p.neck_x if p.neck_x is not None else 0.0)

    assigned_indices = {driver.index, passenger.index}
    ignored = [p.index for p in pose_people if p.index not in assigned_indices]

    # Order matches prompt persons[]: driver first, passenger second
    return [driver, passenger], ignored, warnings


# ---------------------------------------------------------------------------
# Prompt → annotations
# ---------------------------------------------------------------------------


def _clothing_records(clothing: dict, torso: Bbox) -> list[AnnotationRecord]:
    records: list[AnnotationRecord] = []
    mode = clothing.get("layering_mode", "single")

    if mode == "single":
        rec = _make_record(clothing["det_class"], "clothing_outer", torso)
        if rec:
            records.append(rec)
    elif mode == "closed_outer":
        rec = _make_record(clothing["outer_det_class"], "clothing_outer", torso)
        if rec:
            records.append(rec)
    elif mode == "open_outer":
        outer = _make_record(clothing["outer_det_class"], "clothing_outer", torso)
        if outer:
            records.append(outer)
        inner = _make_record(
            clothing["inner_det_class"], "clothing_inner", compute_inner_bbox(torso)
        )
        if inner:
            records.append(inner)

    return records


def _glasses_record(glasses: dict, pose: PosePerson) -> AnnotationRecord | None:
    if not glasses.get("has_glasses"):
        return None
    bbox = compute_glasses_bbox(pose)
    if bbox is None:
        return None
    return _make_record(glasses["od_bbox_class"], "glasses", bbox)


def _headwear_record(headwear: dict, pose: PosePerson) -> AnnotationRecord | None:
    if not headwear.get("has_headwear"):
        return None
    cls = headwear["od_bbox_class"]
    bbox = compute_headwear_bbox(pose, cls)
    if bbox is None:
        return None
    return _make_record(cls, "headwear", bbox)


def build_person_annotations(
    prompt_person: dict,
    pose: PosePerson,
) -> PersonAnnotation:
    person = PersonAnnotation(
        role=prompt_person["role"],
        pose_index=pose.index,
        pose_score=pose.torso_score,
    )

    torso = compute_torso_bbox(pose)
    if torso is None:
        person.warnings.append("torso_bbox_failed")
        return person

    person.annotations.extend(_clothing_records(prompt_person["clothing"], torso))

    glasses = _glasses_record(prompt_person["glasses"], pose)
    if glasses:
        person.annotations.append(glasses)
    elif prompt_person["glasses"].get("has_glasses"):
        person.warnings.append("glasses_bbox_failed")

    headwear = _headwear_record(prompt_person["headwear"], pose)
    if headwear:
        person.annotations.append(headwear)
    elif prompt_person["headwear"].get("has_headwear"):
        person.warnings.append("headwear_bbox_failed")

    return person


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def load_ref_steering_side(manifest_path: str, ref_id: str) -> str:
    refs = load_manifest(manifest_path)
    for ref in refs:
        if ref["id"] == ref_id:
            return ref.get("steering_side", "left")
    return "left"


def annotate_image(
    prompt_path: Path,
    pose_path: Path,
    image_path: Path,
    steering_side: str,
) -> ImageAnnotationResult:
    with open(prompt_path, encoding="utf-8") as f:
        prompt = json.load(f)

    image_name = prompt["image_name"]
    ref_id = prompt["ref_id"]
    occupancy = prompt["occupancy"]

    pose_people, width, height = parse_pose_file(pose_path)

    rel_image_path = f"images/{ref_id}/{image_name}.png"

    result = ImageAnnotationResult(
        image_name=image_name,
        image_path=rel_image_path,
        width=width,
        height=height,
        ref_id=ref_id,
        passed=False,
        checks={
            "expected_occupancy": occupancy,
            "detected_pose_count": len(pose_people),
        },
    )

    matched_poses, ignored, warnings = match_poses_to_roles(
        pose_people, occupancy, steering_side
    )
    if matched_poses is None:
        result.fail_reason = "pose_count_mismatch"
        result.checks["matched_person_count"] = 0
        return result

    result.warnings.extend(warnings)
    result.ignored_pose_indices = ignored
    result.checks["matched_person_count"] = len(matched_poses)

    prompt_persons = prompt["persons"]
    if len(matched_poses) != len(prompt_persons):
        result.fail_reason = "prompt_pose_count_mismatch"
        return result

    for prompt_person, pose in zip(prompt_persons, matched_poses):
        person_ann = build_person_annotations(prompt_person, pose)
        result.warnings.extend(person_ann.warnings)
        if person_ann.annotations:
            result.persons.append(person_ann)

    if not result.persons:
        result.fail_reason = "no_valid_annotations"
        return result

    result.passed = True
    return result


def save_annotation(result: ImageAnnotationResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
