# OpenPose Bbox Annotation Pipeline

**Status:** Implemented (sample script)  
**Run target:** `data/run_20260520_034326`  
**Scope:** Per-image COCO-compatible JSON annotations derived from OpenPose keypoints + prompt metadata.

---

## 1. Overview

Each generated image has:

| Source | Provides |
|--------|----------|
| `individual_prompts/{image_name}.json` | **Classes** — clothing, glasses, headwear, layering per person |
| `images/{ref_id}/{image_name}_pose.json` | **Geometry** — OpenPose body + face keypoints (normalized) |

**OpenPose → bbox geometry. Prompt JSON → class labels.**

Output: one JSON per image under `annotations/`, mergeable into a master COCO file later.

---

## 2. Detection classes (12 OD classes)

### Clothing (7) — full-torso bbox

| Class | Source field |
|-------|--------------|
| `sleeveless_top` | `clothing.det_class` or `inner_det_class` / `outer_det_class` |
| `short_sleeve_top` | same |
| `long_sleeve_plain` | same |
| `long_sleeve_collared` | same |
| `hoodie_sweatshirt` | same |
| `light_jacket` | same |
| `heavy_jacket_coat` | same |

### Glasses (2) — eye-region bbox

| Class | When |
|-------|------|
| `eyeglasses` | `glasses.has_glasses == true` |
| `sunglasses` | same |

### Headwear (3) — head-top bbox

| Class | When |
|-------|------|
| `cap` | `headwear.has_headwear == true` |
| `beanie` | same |
| `hijab_headscarf` | same |

### Manifest-only (no bbox)

| Class | Meaning |
|-------|---------|
| `no_glasses` | Absence — zero instances in label file |
| `no_headwear` | Hair visible, no hat |
| `bare_head` | Bald/shaved scalp — metadata only (`has_bare_scalp: true`), **no bbox** |

---

## 3. Bboxes per person

| Domain | Condition | Bboxes |
|--------|-----------|--------|
| Clothing | `single` | 1 full-torso → `det_class` |
| Clothing | `closed_outer` | 1 full-torso → `outer_det_class` |
| Clothing | `open_outer` | 2 — outer full-torso + inner chest strip |
| Glasses | `has_glasses: false` | 0 |
| Glasses | `has_glasses: true` | 1 → `od_bbox_class` |
| Headwear | `has_headwear: false` | 0 |
| Headwear | `has_headwear: true` | 1 → `od_bbox_class` (class-specific shape) |

Max per dual-occupancy image: **8 bboxes** (both persons in `open_outer` with glasses + headwear).

---

## 4. Geometry from OpenPose

Coordinates in pose JSON are **normalized [0, 1]** relative to `canvas_width` × `canvas_height`.

### 4.1 Body keypoints (18-point layout)

| Index | Joint |
|-------|-------|
| 0 | Nose |
| 1 | Neck |
| 2 | Right shoulder |
| 3 | Right elbow |
| 4 | Right wrist |
| 5 | Left shoulder |
| 6 | Left elbow |
| 7 | Left wrist |
| 8 | Mid hip |
| 9–14 | Legs |
| 15–17 | Eyes / ears |

Keypoints with confidence ≤ 0 or coordinates ≤ 0 are treated as missing.

### 4.2 Clothing bbox (full torso)

Uses valid points from `{neck, shoulders, elbows, wrists, mid hip}`:

- **Top:** min y of shoulders/neck minus upward padding (~25% of torso height)
- **Mouth constraint:** top edge clamped to at least 0.8% below the mouth bottom (face keypoints 48–67, or chin fallback)
- **Bottom:** mid hip y plus downward padding (~6%)
- **Horizontal:** min/max x of arm chain plus ~12% width padding
- **Fallback:** if wrists missing, extend from elbows/shoulders by estimated arm length

### 4.3 Inner clothing bbox (`open_outer` only)

Centered inside the outer torso bbox:

- **Width:** 30% of outer width, horizontally centered
- **Height:** 88% of outer height, vertically centered

### 4.4 Glasses bbox

Face keypoints **17–26** (eyebrows) and **36–47** (eyes):

- Horizontal padding ~15%
- Top padding ~8%
- **Bottom padding ~72%** (extends downward)

### 4.5 Headwear bbox (unified cap-style)

All wearable headwear classes share the same geometry; only the **category label** differs.

- Starts above the face contour (**45%** of face height — jaw contour misses hat/hijab crown)
- Taller box (**85%** of face height) with extra upward padding (**36%** top pad)
- Bottom anchored near the eyebrow line

### 4.6 VLM envelope crop (not exported)

Per-person union of body + face keypoints for VLM classification only:

- Horizontal / bottom: **18%** / **10%** pad on keypoint span
- Top: **max**(12% body span, **55%** of face height above face contour top)
- Union with headwear bbox region when available
- Clamped to image bounds `[0, 1]`

---

## 5. Person matching (pose → prompt)

Match pose detections to prompt `persons[]` roles using manifest `steering_side`:

| `steering_side` | Driver position | Passenger position |
|-----------------|-----------------|---------------------|
| `left` (LHD) | Higher x (right in frame) | Lower x (left in frame) |
| `right` (RHD) | Lower x | Higher x |

**Algorithm:**

1. Score each pose person by count of valid torso keypoints.
2. Keep poses with score ≥ minimum (neck + at least one shoulder).
3. Apply occupancy rules (§6).
4. Assign driver = extremal x on driver side; passenger = extremal x on passenger side.

---

## 6. Quality checks

| Check | Pass | Fail |
|-------|------|------|
| Pose count vs `occupancy` | Equal, or **more** (extras ignored) | **Fewer** than expected → skip image |
| Torso keypoints | Neck + ≥1 shoulder, or ≥2 torso points | Person skipped; if zero persons → fail |
| Bbox area | ≥ 0.3% of image area | Annotation skipped for that box |
| Bbox bounds | Fully inside [0, 1] after clamp | Clamped + warning |
| Extra pose people | Logged as warning, not annotated | — |

### Occupancy mismatch policy

| Situation | Action |
|-----------|--------|
| **Match** (1/1, 2/2) | Full annotation |
| **More** (e.g. 3 vs 2) | Match expected count by role; **ignore extras**; `passed: true` + warning |
| **Fewer** (e.g. 1 vs 2) | **Skip image**; `passed: false`, no bboxes |
| Ambiguous role match | `passed: false` |

---

## 7. Output schema (per-image JSON)

Path: `data/run_YYYYMMDD_HHMMSS/annotations/{image_name}.json`

```json
{
  "image_name": "dash_ref_01_000966",
  "image_path": "images/dash_ref_01/dash_ref_01_000966.png",
  "width": 1920,
  "height": 1088,
  "ref_id": "dash_ref_01",
  "quality": {
    "passed": true,
    "warnings": ["extra_pose_person: 1 ignored (pose_index=2)"],
    "checks": {
      "expected_occupancy": 2,
      "detected_pose_count": 3,
      "matched_person_count": 2
    }
  },
  "persons": [
    {
      "role": "driver",
      "pose_index": 0,
      "pose_score": 7,
      "annotations": [
        {
          "category": "long_sleeve_collared",
          "category_id": 3,
          "area": "clothing_outer",
          "bbox": [120, 200, 400, 500],
          "bbox_normalized": [0.0625, 0.184, 0.208, 0.459]
        }
      ]
    }
  ],
  "ignored_pose_indices": [2]
}
```

**Bbox format:** COCO `[x_min, y_min, width, height]` — pixel coords in `bbox`, normalized in `bbox_normalized`.

---

## 8. Scripts

### Sample + visualize (10 images)

```bash
python scripts/annotate_sample.py \
  --run-dir data/run_20260520_034326 \
  --ref-id dash_ref_01 \
  --samples 10
```

Outputs:

- `annotations/` — per-image JSON (sample subset)
- `annotations/_preview/` — PNG overlays with drawn bboxes

### Full run (later)

```bash
python scripts/annotate_sample.py \
  --run-dir data/run_20260520_034326 \
  --ref-id dash_ref_01 \
  --all \
  --no-visualize
```

---

## 9. Category ID map (COCO merge)

| ID | Class |
|----|-------|
| 0 | `sleeveless_top` |
| 1 | `short_sleeve_top` |
| 2 | `long_sleeve_plain` |
| 3 | `long_sleeve_collared` |
| 4 | `hoodie_sweatshirt` |
| 5 | `light_jacket` |
| 6 | `heavy_jacket_coat` |
| 7 | `eyeglasses` |
| 8 | `sunglasses` |
| 9 | `cap` |
| 10 | `beanie` |
| 11 | `hijab_headscarf` |

---

## 10. Decisions log

| Decision | Choice |
|----------|--------|
| Bbox source | OpenPose per image (not fixed manifest bboxes) |
| Inner bbox | Proportional sub-crop of outer torso |
| Headwear shape | Class-specific vertical extent |
| `bare_head` | Metadata only, no bbox |
| Output format | COCO-style per-image JSON, merge later |
| Extra pose people | Ignore, log warning |
| Missing pose people | Skip entire image |
