# Prompt Generation — Design Document

**Project:** Synthetic car-cabin clothing dataset (thermal-focused)  
**Pipeline:** z-image + ControlNet over multiple reference images  
**Status:** Design finalized 2026-05-19; implementation pending  
**Total OD classes:** 12 (7 clothing + 2 glasses + 3 headwear)  

---

## 1. Goals

| Priority | Goal | Description |
|----------|------|-------------|
| **P0** | **Bbox clothing OD** | Train a bounding-box object detector to localize and classify upper-body garments in cabin dash-cam images. |
| **P0** | **Synthetic images** | Generate diverse, realistic in-cabin images guided by depth/reference (ControlNet) as **training data** for the detector. |
| **P1** | **Thermal modeling** | Garment classes and layering should support **insulation / coverage** reasoning (light vs mid vs heavy, bare arms, outer layers). |
| **P1** | **Sleeve length** | Explicit separation where it matters (sleeveless vs short vs long). |
| **P2** | **Prompt diversity** | Use **richer phrasing** (`gen_variant`) in prompts so z-image does not collapse to repetitive looks. |

### 1.1 Detection task definition

The downstream model is a **bounding-box object detector** (YOLO / Faster R-CNN family). For each person visible in a dash-cam image it should output:

- **One clothing bbox** covering the visible torso + arms (upper-body region), classified into one of the garment `det_class` values.
- **Zero or one glasses bbox** on the eye/frame region, classified as `eyeglasses` or `sunglasses`.
- **Zero or one headwear bbox** on the head region, classified as `cap`, `beanie`, or `hijab_headscarf`.

### 1.2 Bbox region conventions

| Domain | Bbox region | Notes |
|--------|-------------|-------|
| Clothing | **Full torso:** shoulder-to-waist, including visible arms/sleeves | One bbox per visible garment covering the entire upper body. Arms included — this is how sleeve length becomes spatially distinguishable. |
| Glasses | **Eye region:** tight around frames | Small box; model must learn face location. |
| Headwear | **Head-top region:** crown of head to forehead / ears | Wraps around the hat/wrap shape. |

**Why full torso (not tight garment boundary):** Consistent region regardless of garment type. A `sleeveless_top` vs `short_sleeve_top` is only distinguishable if the bbox **covers the shoulders/upper arms**. A torso-only crop makes those classes identical to the model. Full-torso convention also simplifies annotation (one predictable region per person).

### 1.3 Layering: bbox strategy

Layering annotations are **only produced when outer is open** (inner garment visible). This is the only case where the model should learn to detect an inner garment.

| Scenario | Bbox output |
|----------|-------------|
| `single` | 1 full-torso bbox → class of the single garment |
| `open_outer` (inner visible) | **2 bboxes** — outer garment (full-torso bbox) + inner garment (**smaller bbox** on the visible chest area between lapels/zipper opening). |
| `closed_outer` (inner hidden) | 1 full-torso bbox → class of the outer garment only. Inner is metadata for thermal, **not** a detection target (not visible). |

**Inner bbox definition (`open_outer` only):**
- Region: the visible strip of the inner garment between the two sides of the open jacket — typically center-chest area.
- Since ControlNet fixes pose per reference, this visible area is **roughly consistent within a ref** — the open-jacket geometry is stable, making annotation repeatable.
- Smaller than full torso; bounded by lapel/zipper edges on left/right, neckline on top, waist on bottom.

**Rationale:** If the inner garment is **visually present** in the image, it should be detectable. If it is fully occluded (`closed_outer`), it cannot be localized and should not appear in OD labels. This keeps labels honest — only label what you can see.

**Prompting:** Only `open_outer` mentions the inner garment in the prompt. `closed_outer` prompts the outer garment only (same wording as `single`) — you cannot ask the generator to render something invisible. See §4.3 for full prompt examples.

### 1.4 Annotation pipeline (how bboxes are created)

Prompts define **class**; bboxes require **pixel regions**. Since ControlNet fixes pose per ref, bboxes are **stored in the manifest as constants** (§3.1).

**Phase 1 (v1 — bboxes from manifest, no runtime detection):**

1. **Define bboxes once per ref** in `refs/manifest.yaml`: clothing (full torso), headwear (head-top), glasses (eye region), inner_clothing (chest area for `open_outer`).
2. Coordinates are **normalized [0–1]**; convert to pixels via `image_size` (auto-detected from ref).
3. All generated images from the same ref **reuse the same bboxes** — ControlNet constrains the layout.
4. **Class** comes from manifest `det_class` (ground truth by construction — we wrote the prompt).
5. **Export:** combine fixed bboxes + manifest class → COCO/YOLO annotation per image. Zero runtime detection needed.

**Phase 2 (refinement — only if needed):**

1. If ControlNet doesn't hold pose tightly enough (validation shows drift), run **person detector + pose estimator** (YOLOv8-pose / MediaPipe) to refine bboxes.
2. For 2-person images: match detected persons to manifest `persons[]` by spatial position.
3. For layering `open_outer`: use SAM / segmentation for tighter inner-garment region (if template bbox is too loose).

**Key advantage of fixed bboxes:** No annotation model needed at all for phase 1. The prompt generator + manifest produce a **fully labeled dataset** (class + bbox) without any post-hoc detection. This eliminates annotation noise from a secondary model.

**Separation:** Bbox coordinates live in the manifest (per ref); class labels live in each prompt's JSON. The export script combines them.

### 1.5 QA / filtering (between generation and annotation)

Not every generated image faithfully follows the prompt. Before annotation:

1. **CLIP similarity check:** Compare generated image to prompt text; discard below threshold (e.g. < 0.25 cosine).
2. **Person presence check:** Run person detector; discard if expected person count doesn't match `occupancy`.
3. **Visual inspection (sample-based):** Spot-check ~5% per ref to catch systematic generator failures (wrong garment type, missing glasses, etc.).
4. **Duplicates / artifacts:** Discard images with generation artifacts (black regions, distorted faces, etc.).

Only images passing QA proceed to annotation and training.

### 1.6 Scale requirements

| Metric | Target |
|--------|--------|
| Images per clothing `det_class` | **5,000–10,000 minimum** (7 classes × 5k = 35k+ images) |
| Images per glasses/headwear class | **2,000–5,000** (less critical; fewer classes, auxiliary task) |
| Reference images | **10–15 minimum** (see §1.7) |
| Images per reference | **3,000–7,000** |
| Validation set | Hold-out **real** images for domain-gap measurement |

`SAMPLE_SIZE = 1000` in the current code is a **development placeholder**, not a production target.

### 1.7 Reference image count and diversity

| Guideline | Value | Rationale |
|-----------|-------|-----------|
| **Minimum refs** | 8–10 | Below this, model memorizes background/layout; poor generalization |
| **Recommended** | 10–15 | Good pose/layout diversity for cabin dash cams |
| **Diminishing returns** | Beyond ~20 | Each additional ref contributes less novel layout info for a fixed camera domain |
| **Images per ref** | 3,000–7,000 | Enough to cover all 7 clothing classes × variants × conditions within one layout |

**What each ref should vary:**

- Camera angle / position (slightly different dash-cam mount points)
- Cabin type / interior (different car models, seat styles)
- Person position (arm on wheel, arm on rest, hands at 10-and-2, etc.)
- 1-person vs 2-person layout
- Arm visibility (critical for sleeve-length classes — at least some refs must show arms clearly)

**Per-ref quota example:** 15 refs × 4,000 images/ref = 60,000 total images. With stratified sampling across 7 clothing classes ≈ 8,500 per class (meets the 5k+ target).

### 1.8 Domain gap awareness

Synthetic data alone rarely matches real-data performance. Plan:

- **Pre-train** on synthetic → **fine-tune** on small real labeled set.
- Track **mAP on real validation** as primary metric, not synthetic val.
- Use ControlNet depth to keep geometry realistic (reduces gap vs pure text-to-image).
- Vary lighting/weather/cabin aggressively to prevent overfitting to synthetic texture.

---

## 2. Core design pattern: two-level taxonomy

Every attribute domain uses:

| Field | Purpose | Used in |
|-------|---------|---------|
| `det_class` | Stable label in **manifest** (sampling, prompts, traceability) | `prompts_full.json`, export scripts |
| `gen_variant` | Richer surface form (more variety) | Natural-language prompt |

**Rules**

1. Variants must **not change silhouette** enough to belong in another `det_class`.
2. Sample **stratified by manifest `det_class`**; randomize `gen_variant` within the bucket.
3. Prompt text uses `gen_variant`; JSON always exports `det_class` + `gen_variant`.
4. Avoid fashion-level splits at OD bbox level — keep those as `gen_variant`s (e.g. denim vs bomber under `light_jacket`).

### 2.1 Manifest labels vs bounding-box OD

**Important:** `det_class` in the manifest is **not always** a class you localize with a bounding box.

| Concept | Meaning |
|---------|---------|
| **Present object** | Thing with a region on the person → **draw a bbox** in COCO/YOLO export (`eyeglasses`, `cap`, …). |
| **Absent object** | No instance in the label file → **no bbox**; use manifest `det_class` for prompts and dataset balance only. |

**Why:** Object detection learns to localize **visible** objects. `no_glasses` and `no_headwear` are **absence**, not objects — you do not draw boxes around “nothing.”

| Domain | Manifest `det_class` | OD bbox class? |
|--------|----------------------|----------------|
| Clothing (torso) | All 7 garment classes | Yes — torso/upper-body garment region (per pipeline) |
| Glasses | `no_glasses` | **No** — zero instances; `has_glasses: false` |
| Glasses | `eyeglasses`, `sunglasses` | **Yes** — box on frames/eye region |
| Headwear | `no_headwear` | **No** — hair visible, no hat/wrap; `has_headwear: false` |
| Headwear | `cap`, `beanie`, `hijab_headscarf` | **Yes** — box on hat/wrap region |
| Headwear | `bare_head` | **Optional** — see §6; scalp visible, no hat (classification / optional head region, not a “hat” object) |

**Export mapping (conceptual)**

```text
manifest.glasses.det_class == "no_glasses"
  → OD: (no glasses instances)
  → fields: has_glasses: false

manifest.glasses.det_class in ("eyeglasses", "sunglasses")
  → OD: one instance, class = eyeglasses | sunglasses
  → fields: has_glasses: true

manifest.headwear.det_class == "no_headwear"
  → OD: (no headwear instances)
  → fields: has_headwear: false, has_bare_scalp: false

manifest.headwear.det_class == "bare_head"
  → OD: typically no hat box; has_bare_scalp: true (see §6)

manifest.headwear.det_class in ("cap", "beanie", "hijab_headscarf")
  → OD: one instance, class = cap | beanie | hijab_headscarf
  → fields: has_headwear: true
```

---

## 3. Reference-image profiles

Generation runs over **multiple reference images** (depth / ControlNet). Each reference has its own profile.

### 3.1 Profile schema (conceptual)

```yaml
id: dash_front_driver_01
image_path: refs/dash_front_driver_01.png
template_file: refs/dash_front_driver_01.txt   # VLM-generated prompt templates
image_size: [1024, 768]       # auto-detected from ref image (width × height)
occupancy: [1, 2]             # allowed person counts for this ref
arms_visible: true            # critical: can this ref show sleeve-length differences?
steering_side: left           # LHD / RHD — for multi-person bbox assignment (§7.3)

# Fixed bounding boxes per person slot (normalized 0–1, format: [x_min, y_min, x_max, y_max])
# These are CONSTANTS for this ref — ControlNet fixes the pose.
bboxes:
  driver:
    clothing:  [0.15, 0.25, 0.85, 0.95]   # full torso (shoulders to waist, incl. arms)
    headwear:  [0.30, 0.00, 0.70, 0.22]   # head-top region
    glasses:   [0.35, 0.12, 0.65, 0.22]   # eye/frame region
    inner_clothing: [0.30, 0.35, 0.70, 0.85]  # visible chest area (only used for open_outer)
  passenger:                               # only if occupancy includes 2
    clothing:  [0.55, 0.28, 1.00, 0.95]
    headwear:  [0.65, 0.02, 0.90, 0.24]
    glasses:   [0.68, 0.14, 0.88, 0.24]
    inner_clothing: [0.62, 0.38, 0.85, 0.85]

conditions:                   # whitelist: only these values are sampled for this ref
  lighting:
    - bright sunlight
    - overcast
    - dim cabin
  weather:
    - sunny
    - rainy
  time:
    - morning
    - noon
    - evening
  seat_type:
    - leather seats
  window_tint:
    - clear windows
    - lightly tinted windows
  cabin_color:
    - black
  seatbelt:
    - wearing seatbelt
    - seatbelt not visible
```

### 3.2 Manifest fields

| Field | Source | Notes |
|-------|--------|-------|
| `image_size` | **Auto-detected** from ref image dimensions (width × height in pixels) | Used for converting normalized bboxes → pixel coords in COCO/YOLO export |
| `bboxes` | **Manually defined once** per ref (or semi-auto from pose estimation on the ref image) | Normalized [0–1] coordinates; constant for all images generated from this ref |
| `template_file` | **VLM-generated** (§9.2) | `.txt` file with prompt templates for this ref |
| `conditions` | **You define** per ref | Whitelist of allowed scene values |
| `arms_visible` | **You define** | Controls whether sleeveless/short-sleeve classes are sampled |

**Bbox coordinate format:** `[x_min, y_min, x_max, y_max]` normalized to `[0, 1]` relative to image dimensions. To get pixel coords: multiply by `image_size`.

**`inner_clothing` bbox:** Only used when `layering_mode = open_outer`. Represents the visible chest area between open jacket lapels. If a ref never uses layering, this can be omitted.

**How to define bboxes:** Open the reference image, draw boxes around the torso/head/eye regions of each person slot. Since ControlNet preserves the layout, these regions will match all generated images (within a small margin). Use generous margins (~5–10% padding).

**No per-value weights.** Sampling is **uniform within the whitelist** for each variable. Realism biases (weather↔clothing, color contrast) are handled by **global soft couplings** (§7.2, §7.4), not per-ref knobs.

### 3.3 Behavior

- Each generated row includes `ref_id` (and optionally `template_id`) for ComfyUI batching.
- **Conditions** per ref: **whitelist only** — removes impossible scene/geometry pairings for that reference. Uniform sampling within what's allowed.
- **Global soft constraints** (weather–clothing coupling, color contrast) apply **across all refs** uniformly.
- **Templates**: phrasing may differ by camera angle (driver-centric vs wider cabin).
- **Occupancy**: per ref, allow `1` and/or `2` persons; not global.
- **Arms visible**: flag per ref — only refs with `arms_visible: true` should sample `sleeveless_top` and `short_sleeve_top` (where arm presence is the distinguishing feature).

### 3.4 Multi-person (1–2 in cabin)

| Aspect | Rule |
|--------|------|
| Roles | Typically `driver` + `front_passenger` (wording depends on ref). |
| Per person | `gender`, `ethnicity`, `age`, clothing, glasses, headwear, seatbelt (if visible). |
| Shared scene | `lighting`, `weather`, `time`, `cabin_color`, `seat_type`, `window_tint`, `ref_id`. |
| Independence | Person attributes sampled independently; **do not** tie `hijab_headscarf` (or any headwear) to ethnicity. |
| Templates | Separate single-person vs dual-person template sets per ref. |

---

## 4. Clothing taxonomy

### 4.1 Detection classes (7)

Silhouette-based, **not** fashion-based. Each class should be **visually distinguishable within the clothing bbox** (shoulder-to-waist including arms).

| `det_class` | Spatial OD cue (what the model sees in bbox) | Default thermal |
|-------------|----------------------------------------------|-----------------|
| `sleeveless_top` | **Bare upper arms** visible in bbox; no sleeve fabric at shoulders | light |
| `short_sleeve_top` | **Short sleeve fabric** at upper arm; forearms bare | light |
| `long_sleeve_plain` | **Full arm coverage**, smooth torso, no collar/placket | mid |
| `long_sleeve_collared` | **Full arm coverage** + visible **collar/placket** at neckline | mid |
| `hoodie_sweatshirt` | **Hood mass** at neck/back (even down) + thicker torso volume | mid |
| `light_jacket` | **Outer layer silhouette**: lapels, zipper line, lighter drape | mid |
| `heavy_jacket_coat` | **Bulky outer silhouette**: thick volume, possibly high collar / fur | heavy |

**Separability assessment for bbox OD:**

| Pair | Distinguishable? | Key spatial cue |
|------|------------------|-----------------|
| `sleeveless_top` ↔ `short_sleeve_top` | Yes — full-torso bbox includes upper arms | Fabric at deltoid vs bare skin |
| `short_sleeve_top` ↔ `long_sleeve_plain` | Yes | Arm coverage boundary (mid-bicep) |
| `long_sleeve_plain` ↔ `long_sleeve_collared` | Yes — collar/placket is a consistent neckline feature | Collar structure + button line visible in upper-center of bbox |
| `hoodie_sweatshirt` ↔ `long_sleeve_plain` | Yes | Hood bulk at neck/back, heavier body |
| `light_jacket` ↔ `hoodie_sweatshirt` | Yes | Lapels/zipper vs hood; drape vs bulk |
| `light_jacket` ↔ `heavy_jacket_coat` | Yes | Volume, padding, length |

**On `long_sleeve_plain` vs `long_sleeve_collared`:** The collar/placket occupies a smaller fraction of the bbox than arm-coverage differences, but it is a **consistent, learnable feature** — present in every sample of that class, always in the same spatial region (neckline). With thousands of training examples the model accumulates enough signal. Keeping the split gives the detector explicit supervision to attend to neckline structure. Merging later is always possible; splitting later would require re-generation.

**Decisions captured in discussion**

- **Polo** is not a `det_class`. Short polo → `short_sleeve_top`; long polo → `long_sleeve_collared`.
- **Formal / button-up** is not a separate `det_class`; use `long_sleeve_collared` with formal `gen_variant`s.
- **Tank / sleeveless** only under `sleeveless_top`, **never** under `short_sleeve_top`.
- **Hood up** belongs to `hoodie_sweatshirt` (clothing), not headwear.

### 4.2 Generation variants (examples)

```text
sleeveless_top
  - tank top
  - sleeveless top
  - athletic tank top
  - racerback tank top
  - scoop-neck tank top

short_sleeve_top
  - t-shirt
  - casual tee
  - v-neck t-shirt
  - short sleeve polo
  - striped t-shirt
  - henley shirt
  # NO tank / sleeveless variants here

long_sleeve_plain
  - long sleeve shirt
  - long sleeve t-shirt
  - thin pullover
  - crew neck sweater
  - v-neck sweater
  - ribbed long sleeve top

long_sleeve_collared
  - long sleeve polo
  - collared dress shirt
  - white button-up shirt
  - oxford shirt
  - flannel shirt

hoodie_sweatshirt
  - hoodie
  - pullover hoodie
  - sweatshirt
  - zip-up hoodie
  - hooded sweatshirt

light_jacket
  - light jacket
  - denim jacket
  - bomber jacket
  - windbreaker
  - fleece jacket
  - unzipped light jacket

heavy_jacket_coat
  - heavy jacket
  - winter coat
  - puffer jacket
  - parka
  - wool coat
  - insulated coat
```

### 4.3 Layering

Layering is a **structured variable**, not extra top-level `det_class` values for “open jacket.”

| `layering_mode` | Description | Prompt behavior | OD annotation |
|-----------------|-------------|-----------------|---------------|
| `single` | One visible garment (default). | Describe the garment. | 1 full-torso bbox |
| `open_outer` | Inner top under **open** outer layer. | Describe **both** garments. | **2 bboxes** (outer full-torso + inner visible-area) |
| `closed_outer` | Inner top under **closed** outer (inner invisible). | Describe **outer only** (same as `single`). | 1 full-torso bbox (same as `single`) |

**Key insight:** `closed_outer` is **thermal metadata only**. It produces the same prompt and the same annotation as `single`:
- Do **not** prompt the inner garment — generator cannot render what the camera cannot see.
- Do **not** annotate an inner bbox — cannot localize what is invisible.
- **Do** record `inner_det_class` in manifest JSON for thermal modeling (P1).

Only **`open_outer`** changes the prompt and produces a different OD annotation.

**Fields when layering = `open_outer`**

| Field | Description |
|-------|-------------|
| `inner_det_class` / `inner_gen_variant` | e.g. `short_sleeve_top` + `gray t-shirt` |
| `outer_det_class` / `outer_gen_variant` | e.g. `light_jacket` + `open denim jacket` |

**Fields when layering = `closed_outer` (metadata only — not prompted)**

| Field | Description |
|-------|-------------|
| `inner_det_class` | e.g. `short_sleeve_top` (thermal metadata; not in prompt, not annotated) |
| `outer_det_class` / `outer_gen_variant` | e.g. `heavy_jacket_coat` + `closed parka` (prompted and bbox'd, same as `single`) |

**Constraints (prompt builder)**

- Outer thermal weight ≥ inner.
- `sleeveless_top` under `open_outer` is valid (tank + open light jacket).
- Open vs closed is **layering**, not a new jacket `det_class`.
- Weather–clothing coupling applies to outermost garment (see §7.4).

**OD labeling summary**

- `single` → **1 full-torso bbox**: the garment.
- `open_outer` → **2 bboxes**: outer (full-torso) + inner (visible chest area between lapels).
- `closed_outer` → **1 full-torso bbox**: outer only. Identical to `single` for OD and prompt.

**Example prompts**

| Mode | Prompt fragment |
|------|-----------------|
| `single` | *wearing a blue t-shirt* |
| `open_outer` | *wearing a gray t-shirt under an open denim jacket* |
| `closed_outer` | *wearing a heavy closed parka* (inner NOT mentioned in prompt) |

---

## 5. Glasses taxonomy

Replaces the old `ACCESSORIES` list. **No** generic accessories bucket.

### 5.1 Manifest classes (3)

Used for **prompts**, **stratified sampling**, and **manifest JSON**.

| `det_class` | Role | Example `gen_variant`s |
|-------------|------|-------------------------|
| `no_glasses` | **Absence** — explicit negative for generation & dataset balance | `no glasses`, `face without glasses` |
| `eyeglasses` | **Present** — clear lenses / visible frames | `clear prescription glasses`, `thin wire-frame glasses`, `black rectangular frame glasses`, `round metal frame glasses` |
| `sunglasses` | **Present** — dark lenses | `dark sunglasses`, `aviator sunglasses`, `large black sunglasses`, `tinted sunglasses` |

### 5.2 OD bbox classes (2)

Only **present** types get bounding boxes at export:

| OD class | When to annotate |
|----------|------------------|
| `eyeglasses` | Clear or lightly tinted lenses; frames visible |
| `sunglasses` | Dark / reflective lenses |

**`no_glasses` is not an OD class.** Export = no `eyeglasses` / `sunglasses` instances. The model learns absence from background negatives, not from a `no_glasses` box.

### 5.3 Rules

- Every person has exactly one manifest `det_class` (including `no_glasses`).
- Frame style differences stay in `gen_variant`, not new OD classes.
- Stratify **~40–60%** `no_glasses` in the manifest so many images have **zero** glasses boxes.
- Prompts for `no_glasses` should state absence explicitly (e.g. *“no glasses”*) to reduce generator adding random frames.
- Splitting **eyeglasses** vs **sunglasses** at OD level is intentional (lens appearance differs at dash-cam distance).

---

## 6. Headwear taxonomy

Replaces the flat `HEADWEAR` list with the same two-level pattern.

### 6.1 Manifest classes (5)

| `det_class` | Role | Example `gen_variant`s |
|-------------|------|-------------------------|
| `no_headwear` | **Absence** — hair visible, no hat/wrap, not bald | `no headwear`, `uncovered hair` |
| `bare_head` | **Distinct scalp signal** — bald/shaved, no hat | `bald head`, `shaved head`, `bald scalp visible` |
| `cap` | **Positive** — brim + crown | `baseball cap`, `cap worn forward`, `cap worn backward` |
| `beanie` | **Positive** — knit cap, no brim | `beanie`, `wool beanie`, `fitted beanie` |
| `hijab_headscarf` | **Positive** — head wrap / hijab (merged dash-cam silhouette) | `hijab`, `headscarf`, `wrapped headscarf` |

### 6.2 OD bbox classes (3 wearables)

Only **wearable head coverings** get bounding boxes:

| OD class | When to annotate |
|----------|------------------|
| `cap` | Baseball cap, cap with brim |
| `beanie` | Knit beanie / winter cap without brim |
| `hijab_headscarf` | Hijab, headscarf, wrapped head covering |

**`bare_head` (4th positive category in manifest, not a hat)**

- **Manifest:** separate from `no_headwear` so prompts and classifiers can distinguish **bald/scalp** vs **hair with no hat**.
- **OD export:** typically **no hat bbox**; set `has_bare_scalp: true`, `has_headwear: false`. Optional: head/scalp region box only if your tooling uses a dedicated “head” task — not a `cap`/`beanie` class.
- **Sampling:** ~5–10% of non-hat headwear samples (tunable).

**`no_headwear` is not an OD class.** Export = no `cap` / `beanie` / `hijab_headscarf` instances; `has_headwear: false`, `has_bare_scalp: false`.

### 6.3 Rules

- **Four manifest positives:** `cap`, `beanie`, `hijab_headscarf`, `bare_head` (+ `no_headwear` for absence with hair).
- **Hood up** → `hoodie_sweatshirt` (clothing), not headwear.
- Do not correlate `hijab_headscarf` with ethnicity in sampling.
- Optional v2: `bucket_hat` as fifth **wearable** positive (add OD class + variants).

---

## 7. Scene & person variables (unchanged domain, structured output)

These remain **enum lists** (global or per-ref whitelist), not two-level unless extended later:

| Variable | Notes |
|----------|--------|
| `gender` | Per person |
| `ethnicity` | Per person |
| `age` | teen, young adult, middle-aged, elderly |
| `seatbelt` | Per person if visible |
| `lighting` | Shared |
| `weather` | Shared |
| `time` | Shared |
| `seat_type` | Shared |
| `window_tint` | Shared |
| `cabin_color` | Shared |
| `color` | Garment color/pattern (applies to described clothing) |

**Removed:** `ACCESSORIES` (scarf, gloves, etc. dropped unless added later as separate domains).

### 7.1 Seatbelt and OD

The seatbelt strap crosses the torso diagonally and **partially occludes the clothing** within the full-torso bbox. This is not a problem — it is a **feature**:

- The model must learn to classify garments **despite** seatbelt occlusion (real-world images always have this).
- Balance sampling: ensure ~50% `wearing seatbelt` / ~50% `seatbelt not visible` so the model sees both cases.
- The seatbelt is **not** a separate OD class (it is always in a fixed position relative to the person and is not a wearable).

### 7.2 Garment color vs cabin color

Low contrast between garment and background (e.g. black t-shirt on black leather seat) makes the bbox region harder for the model to learn.

**Sampling rule:** Do not hard-block same color, but ensure **at least 30–40% of samples have visible contrast** between `color` and `cabin_color`. This can be a soft constraint in the prompt builder (if sampled garment color == cabin color, re-roll with 50% probability).

### 7.3 Multi-person bbox assignment

When `occupancy = 2`, each person slot has **fixed bboxes defined in the manifest** (§3.1):

- **Driver slot:** bboxes under `bboxes.driver` in `manifest.yaml`. Position is known (steering side).
- **Passenger slot:** bboxes under `bboxes.passenger`. Position is the other seat.
- **No runtime matching needed (phase 1):** since ControlNet fixes layout, driver is always in the same region, passenger always in theirs.
- **QA check:** if person-count detector finds ≠ expected occupancy in the generated image → discard (§1.5).
- **`steering_side`** in manifest determines which side is driver (left for LHD, right for RHD).

### 7.4 Weather–clothing coupling (soft constraint)

Weather and clothing thermal weight should be **loosely correlated** for realism. This is a soft sampling bias, not a hard block (people sometimes underdress for the weather).

| Weather | Clothing thermal bias |
|---------|----------------------|
| `snowy` | **70% heavy**, 20% mid, 10% light |
| `rainy` | **50% mid**, 30% heavy, 20% light |
| `foggy` | **40% mid**, 30% heavy, 30% light |
| `sunny` | Uniform (no bias) — all thermal weights equally likely |

**Implementation:** After sampling weather, apply bias to clothing `det_class` selection by thermal tier. If the sampled clothing doesn't match the bias, re-roll with the specified probability (or use weighted random from the start).

**Why soft, not hard:** People wear inappropriate clothing (tank top on a cold day, heavy coat on a mild day). The model should see these cases — just less often. Hard-blocking creates artificial gaps in the training distribution.

**Layering coupling:** Cold weather (`snowy`, `rainy`) should also bias toward layering (`open_outer` / `closed_outer`) more often than warm weather.

---

## 8. Output record shape

### 8.1 Single person, single layer

```json
{
  "image_name": "dash_front_driver_01_000147",
  "ref_id": "dash_front_driver_01",
  "occupancy": 1,
  "scene": {
    "lighting": "bright sunlight",
    "weather": "sunny",
    "time": "noon",
    "cabin_color": "black",
    "seat_type": "leather seats",
    "window_tint": "lightly tinted windows"
  },
  "persons": [
    {
      "role": "driver",
      "gender": "female",
      "ethnicity": "Middle Eastern",
      "age": "young adult",
      "seatbelt": "wearing seatbelt",
      "clothing": {
        "layering_mode": "single",
        "det_class": "short_sleeve_top",
        "gen_variant": "short sleeve polo",
        "color": "blue"
      },
      "glasses": {
        "det_class": "eyeglasses",
        "gen_variant": "thin wire-frame glasses",
        "has_glasses": true,
        "od_bbox_class": "eyeglasses"
      },
      "headwear": {
        "det_class": "hijab_headscarf",
        "gen_variant": "hijab",
        "has_headwear": true,
        "has_bare_scalp": false,
        "od_bbox_class": "hijab_headscarf"
      }
    }
  ],
  "prompt": "..."
}
```

**`image_name` convention:** `{ref_id}_{sequence_number:06d}`

- Unique per generated image across the entire run.
- Used as filename for: the generated image (`{image_name}.png`), the individual prompt JSON (`{image_name}.json`), and annotation files.
- Grouping by `ref_id` prefix makes batch processing and ComfyUI grouping trivial.

### 8.2 Layering example

```json
"clothing": {
  "layering_mode": "open_outer",
  "inner_det_class": "sleeveless_top",
  "inner_gen_variant": "tank top",
  "outer_det_class": "light_jacket",
  "outer_gen_variant": "open denim jacket",
  "color": "blue and black"
}
```

### 8.3 Dual occupancy

- `occupancy`: `2`
- `persons[]`: two entries with independent clothing / glasses / headwear
- Shared `scene` + `ref_id`
- Template describes both (e.g. driver + front passenger)

### 8.4 Manifest-only absence (no OD bbox)

```json
"glasses": {
  "det_class": "no_glasses",
  "gen_variant": "no glasses",
  "has_glasses": false
},
"headwear": {
  "det_class": "no_headwear",
  "gen_variant": "uncovered hair",
  "has_headwear": false,
  "has_bare_scalp": false
}
```

```json
"headwear": {
  "det_class": "bare_head",
  "gen_variant": "bald head",
  "has_headwear": false,
  "has_bare_scalp": true
}
```

Export scripts derive COCO/YOLO from `has_glasses`, `od_bbox_class`, and headwear flags — **do not** emit instances for `no_glasses` or `no_headwear`.

---

## 9. Prompt construction

### 9.1 Flow

1. Pick `ref_id` → allowed `occupancy` → pick `1` or `2`.
2. Sample scene vars **uniformly from ref whitelist**.
3. Apply **global soft constraints**: weather–clothing coupling (§7.4), color contrast (§7.2).
4. For each person: sample `det_class`es (stratified by class), then `gen_variant`s; apply layering if sampled.
5. Pick template for `(ref_id, occupancy)` — each ref has its own templates.
6. Fill slots: clothing phrase from variants + layering, glasses phrase, headwear phrase, demographics, scene.
7. Generate `image_name` = `{ref_id}_{sequence:06d}`.

**Style:** Natural-language paragraphs (current style); optional comma-separated tags later if z-image pipeline prefers it.

**Sleeveless prompts:** Prefer explicit sleeveless wording and visibility of upper arms when the ref/crop allows it.

### 9.2 Per-ref prompt templates

Each reference image has **its own set of templates** — the phrasing matches the camera perspective and what's visible in that ref. Templates are lists (multiple variants per ref for language diversity).

**Template slots:**

| Slot | Filled from |
|------|-------------|
| `{ethnicity}` | Person demographics |
| `{gender}` | Person demographics |
| `{age}` | Person demographics |
| `{clothing_phrase}` | Built from `gen_variant` + `color` + layering (§9.3) |
| `{glasses_phrase}` | `gen_variant` from glasses |
| `{headwear_phrase}` | `gen_variant` from headwear |
| `{seatbelt}` | Seatbelt status sentence |
| `{cabin_color}` | Scene |
| `{seat_type}` | Scene |
| `{window_tint}` | Scene |
| `{weather}` | Scene |
| `{time}` | Scene |
| `{lighting}` | Scene |

**Example templates — single person (ref: `dash_front_driver_01`)**

```text
template_1:
  "A {ethnicity} {gender} {age} driver is captured by the dashboard camera
  in front of the steering wheel. They are looking straight ahead at the road.
  The image focuses on the upper body, showing them {clothing_phrase},
  {glasses_phrase}, {headwear_phrase}. {seatbelt} The car interior is
  {cabin_color} with {seat_type} and {window_tint}. The scene is set in
  {weather} weather at {time}, under {lighting} lighting."

template_2:
  "Inside a {cabin_color} car with {seat_type}, a {ethnicity} {age} {gender}
  driver is seen from the dashboard camera. Their upper body is visible,
  wearing {clothing_phrase}. They have {glasses_phrase} and {headwear_phrase}.
  {seatbelt} The windows are {window_tint}. It is {time}, {weather},
  with {lighting} lighting."
```

**Example templates — dual person (ref: `dash_wide_cabin_02`)**

```text
template_1:
  "A dashboard camera captures two people in a {cabin_color} car cabin with
  {seat_type} and {window_tint}. The driver is a {ethnicity_1} {gender_1}
  {age_1}, wearing {clothing_phrase_1}, {glasses_phrase_1}, {headwear_phrase_1}.
  {seatbelt_1} The front passenger is a {ethnicity_2} {gender_2} {age_2},
  wearing {clothing_phrase_2}, {glasses_phrase_2}, {headwear_phrase_2}.
  {seatbelt_2} The scene is set in {weather} weather at {time}, under
  {lighting} lighting."
```

**Why per-ref templates:**
- Different refs may show different perspectives (driver-only close-up vs wide cabin with passenger).
- Some refs may not show hands/steering wheel → template shouldn't say "in front of the steering wheel."
- Dual-person templates only on refs with `occupancy: [2]`.

### 9.3 Clothing phrase construction

| `layering_mode` | Built phrase |
|-----------------|--------------|
| `single` | `"a {color} {gen_variant}"` → *"a blue short sleeve polo"* |
| `open_outer` | `"a {inner_color} {inner_gen_variant} under an open {outer_color} {outer_gen_variant}"` → *"a gray t-shirt under an open black denim jacket"* |
| `closed_outer` | `"a {outer_color} {outer_gen_variant}"` → *"a black closed parka"* (inner not mentioned) |

### 9.4 VLM template generation (`generate_templates.py`)

A one-time setup step that produces prompt templates per ref image using a Vision Language Model.

**Configuration:**

| Setting | Default | Notes |
|---------|---------|-------|
| `VLM_MODEL` | `gpt-5.2` | Configurable in `src/config.py`. Any OpenAI vision-capable model. |
| `OPENAI_API_KEY` | env var | Required. Set in `.env` or environment. |

**Flow:**

```text
refs/manifest.yaml (ids + occupancy)
       │
       ▼
[generate_templates.py]
  1. Scan manifest for refs without a .txt file (or --force to regenerate all)
  2. For each ref:
     a. Load image → base64 encode
     b. Read occupancy from manifest
     c. Call VLM with system prompt + image + occupancy
     d. Parse response → write refs/{id}.txt
```

**System prompt sent to VLM:**

```text
You describe car cabin images for use as synthetic data prompt templates.

For each reference image, output prompt templates with placeholder slots.

Rules:
- Describe: camera angle, person position/pose, what body parts are visible,
  cabin layout, seat position, visible interior elements.
- Do NOT describe the person's specific clothing, accessories, age, gender,
  ethnicity, or headwear — use these placeholders:
    {clothing_phrase}, {glasses_phrase}, {headwear_phrase},
    {ethnicity}, {gender}, {age}, {seatbelt}
- Do NOT describe specific lighting, weather, time — use:
    {lighting}, {weather}, {time}
- Do NOT describe specific cabin color, seat material, window tint — use:
    {cabin_color}, {seat_type}, {window_tint}
- Write 2-3 phrasing variants of the same scene.
- Keep each variant to 2-3 sentences.
- Focus on what makes this camera angle unique (what's visible, framing).
- If occupancy includes 2, also write dual-person variants with indexed
  slots: {ethnicity_1}, {clothing_phrase_1}, {ethnicity_2}, {clothing_phrase_2}, etc.
```

**Output format (`.txt` file per ref):**

```text
[single]
A {ethnicity} {gender} {age} driver is captured from a dashboard camera mounted
behind the steering wheel. Their upper body and arms are clearly visible, hands
resting on the wheel. They are wearing {clothing_phrase}, {glasses_phrase},
{headwear_phrase}. {seatbelt} The {cabin_color} interior has {seat_type} and
{window_tint}. The scene is {weather} at {time} with {lighting} lighting.
---
Dashboard camera view of a {ethnicity} {age} {gender} driver from chest level.
Both arms visible on the steering wheel. Wearing {clothing_phrase}, with
{glasses_phrase} and {headwear_phrase}. {seatbelt} Car interior is {cabin_color},
{seat_type}, {window_tint}. {weather} conditions, {time}, {lighting} lighting.

[dual]
A dashboard camera captures two people in a {cabin_color} car cabin with
{seat_type} and {window_tint}. The driver is a {ethnicity_1} {gender_1} {age_1},
wearing {clothing_phrase_1}, {glasses_phrase_1}, {headwear_phrase_1}. {seatbelt_1}
The front passenger is a {ethnicity_2} {gender_2} {age_2}, wearing
{clothing_phrase_2}, {glasses_phrase_2}, {headwear_phrase_2}. {seatbelt_2}
The scene is {weather} at {time}, {lighting} lighting.
```

**Caching:** Templates are generated once and committed to the repo. Re-run with `--force` only when ref images change or you want to improve wording. Editable by hand after generation.

---

## 10. Sampling strategy

| Strategy | Detail |
|----------|--------|
| **Stratify** | By `ref_id`, clothing `det_class`, and key domains (glasses/headwear presence). |
| **Scene vars** | Uniform within ref whitelist — no per-value weights. |
| **Variants** | Uniform random within `det_class` bucket. |
| **Sleeveless share** | ~8–15% of clothing samples (only on refs with `arms_visible: true`). |
| **`no_glasses` (manifest)** | ~40–60% — images with **no** glasses OD instances. |
| **`no_headwear` (manifest)** | Majority of non-hat samples — hair, no hat; not bald. |
| **`bare_head` (manifest)** | ~5–10% of non-hat samples (tunable). |
| **`eyeglasses` / `sunglasses`** | Split remaining glasses share (tunable). |
| **Layering distribution** | ~60–70% `single`, ~15–25% `open_outer`, ~10–15% `closed_outer`. Cold weather biases toward more layering (§7.4). |
| **Scale** | **Per-ref quotas** recommended (3k–7k per ref; see §1.7). |
| **Dedup** | Hash structured config; avoid duplicate tuples per run. |
| **Global soft constraints** | Weather–clothing coupling (§7.4), garment–cabin color contrast (§7.2). Applied after uniform sampling; re-roll if violated. |

---

## 11. Thermal mapping (metadata)

Thermal weight is assigned at **`det_class`** level using **3 tiers** (not per variant):

| Thermal | `det_class` |
|---------|-------------|
| **light** | `sleeveless_top`, `short_sleeve_top` |
| **mid** | `long_sleeve_plain`, `long_sleeve_collared`, `hoodie_sweatshirt`, `light_jacket` |
| **heavy** | `heavy_jacket_coat` |

**Why 3 tiers (not 4):** A 4-tier system (`light–mid`) creates ambiguity — downstream thermal models must decide if `light–mid` behaves like `light` or `mid`. Three tiers are clean: exposed arms = light; covered arms with moderate fabric = mid; insulated bulk = heavy.

**Layering adjusts effective thermal:**
- `single` light garment → light
- `open_outer` (inner light + outer mid) → mid (outer dominates)
- `closed_outer` (inner light + outer heavy) → heavy (outer dominates; inner recorded for metadata)

Collar/buttons affect **OD class**, not thermal tier (both `long_sleeve_plain` and `long_sleeve_collared` are mid).

---

## 12. Label inventory (v1)

### 12.1 Manifest `det_class` (prompts + JSON)

| Domain | Count | Values |
|--------|-------|--------|
| Clothing | 7 | §4.1 |
| Glasses | 3 | `no_glasses`, `eyeglasses`, `sunglasses` |
| Headwear | 5 | `no_headwear`, `bare_head`, `cap`, `beanie`, `hijab_headscarf` |

### 12.2 OD bbox classes (localization only)

| Domain | Count | Classes |
|--------|-------|---------|
| Clothing | 7 | Same as manifest (torso garment regions) |
| Glasses | 2 | `eyeglasses`, `sunglasses` |
| Headwear (wearables) | 3 | `cap`, `beanie`, `hijab_headscarf` |

**Total OD bbox classes: 12** (7 clothing + 2 glasses + 3 headwear).

**Not OD bbox classes:** `no_glasses`, `no_headwear`. **`bare_head`** uses manifest + `has_bare_scalp`; not a hat box.

---

## 13. Implementation plan (code)

| Step | File / area | Work | Phase |
|------|-------------|------|-------|
| 1 | `refs/manifest.yaml` | Define per-ref: id, image, occupancy, arms_visible, steering_side, **bboxes**, conditions. `image_size` auto-detected. | Setup |
| 2 | `src/generate_templates.py` | VLM call (OpenAI, default `gpt-5.2`) per ref image → `.txt` template files. One-time, cached. See §9.4. | Setup |
| 3 | `src/config.py` | `CLOTHING_TAXONOMY`, `GLASSES_TAXONOMY`, `HEADWEAR_TAXONOMY`; remove `ACCESSORIES`, `CLOTHING_TYPES`, `SLEEVE_TYPES`; layering enums, global soft constraints. | Prompt gen |
| 4 | `src/utils.py` | Stratified sampling; layering constraint builder; color-contrast soft constraint; structured `create_prompt_json`. | Prompt gen |
| 5 | `src/generate_prompts.py` | Load manifest + templates; per-ref prompt assembly; multi-person; `image_name` generation. | Prompt gen |
| 6 | `README.md` | Point to this doc; update workflow name (z-image). | Prompt gen |
| 7 | `data/` | Output JSON includes `image_name`, `ref_id`, `det_class` / `gen_variant`, `has_glasses`, `has_headwear`, `has_bare_scalp`. | Prompt gen |
| 8 | QA script | CLIP similarity filter + person-count check + artifact detection (§1.5). | Post-generation |
| 9 | Export script | Combine manifest bboxes + prompt JSON `det_class` → COCO/YOLO labels. No runtime detection needed (phase 1). | Training prep |
| 10 | Annotation (phase 2, if needed) | Person detector + pose for bbox refinement if ControlNet drift detected. | Refinement |

**Known bug to fix:** legacy `generate_clothing_combinations()` lowercases garment names incorrectly vs `CLOTHING_TYPES` casing.

### 13.1 End-to-end pipeline (generation → training)

```text
[You provide: ref images + manifest.yaml (bboxes, occupancy, conditions)]
       │
       ▼
[generate_templates.py] → VLM call → .txt template per ref (one-time, cached)
       │
       ▼
[generate_prompts.py] → loads manifest + templates → prompts_full.json
       │                  (image_name, det_class, gen_variant, scene, prompt)
       ▼
[z-image + ControlNet] → synthetic images (batched by ref_id, named by image_name)
       │
       ▼
[QA script] → CLIP filter + person count check → discard bad images
       │
       ▼
[Export script] → manifest bboxes (fixed per ref) + det_class → COCO/YOLO labels
       │             No auto-annotation model needed (phase 1)
       ▼
[OD training] → pre-train on synthetic → fine-tune on real
```

---

## 14. Open decisions (before / during implementation)

| Topic | Options | Default in this doc | Status |
|-------|---------|---------------------|--------|
| `bare_head` OD region | Scalp/head box vs manifest-only | Manifest + `has_bare_scalp`; no hat bbox | Decided |
| `long_sleeve_plain` vs `_collared` | Keep split vs merge at OD time | **Keep split** — collar is learnable; merge only if empirically needed | Decided |
| Clothing bbox region | Full torso vs tight garment | **Full torso** (shoulder-to-waist incl. arms) | Decided |
| Layering annotation | Always vs only when visible | **Only `open_outer`** gets inner bbox; `closed_outer` = metadata only | Decided |
| `closed_outer` prompting | Prompt inner vs outer only | **Outer only** — same as `single` in prompt text | Decided |
| Per-value sampling weights | Per-ref weights vs uniform | **Removed** — uniform within whitelist; global soft couplings only | Decided |
| Annotation phase 1 | Template bboxes from ControlNet-fixed positions | **Bboxes in manifest.yaml** — no runtime detection | Decided |
| Bbox storage | Where to store fixed bboxes | **`refs/manifest.yaml`** per ref, normalized [0–1] | Decided |
| Image size | Manual vs auto | **Auto-detected** from ref image dimensions | Decided |
| Template generation | Manual vs VLM | **VLM** via `generate_templates.py`; model configurable (default `gpt-5.2`); cached as `.txt` per ref | Decided |
| Reference image count | How many refs | **10–15 recommended** (§1.7) | Decided |
| Images per ref | How many per ref | **3,000–7,000** | Decided |
| Thermal tiers | 3 vs 4 | **3 tiers** (light / mid / heavy) | Decided |
| Layering distribution | % of samples | ~60–70% single, ~15–25% open_outer, ~10–15% closed_outer | Decided |
| Reference image list | User provides paths + per-ref whitelists | In `refs/manifest.yaml` | Decided |
| Scarf / gloves | Add later as domains | Dropped | Decided |
| Prompt style | Prose vs tags | Prose | Decided |
| `bucket_hat` | v2 wearable OD class | Not in v1 | Deferred |
| ComfyUI manifest | Group batches by `ref_id` | Recommended; `image_name` prefix = `ref_id` | Open |
| Auto-annotation phase 2 tool | YOLOv8-pose / MediaPipe / SAM | Only if ControlNet drift detected | Deferred |
| COCO export | Manifest bboxes + det_class | Straightforward; no auto-annotation model needed | Open |
| Real validation set | Source + size | TBD; required for domain-gap eval | Open |
| QA CLIP threshold | Similarity cutoff for filtering | TBD (~0.25 cosine as starting point) | Open |

---

## 15. Design history

All design decisions were made on **2026-05-19** through iterative discussion. Key decisions in order:

**Taxonomy & classes**
- Two-level pattern: `det_class` (OD label) + `gen_variant` (prompt diversity).
- 7 clothing classes (silhouette-based); polo/formal as variants only.
- `sleeveless_top` as own class (tanks only here, never under `short_sleeve_top`).
- `long_sleeve_plain` vs `long_sleeve_collared` split (collar is learnable).
- Glasses: 3 manifest / 2 OD bbox (`eyeglasses`, `sunglasses`).
- Headwear: 5 manifest / 3 OD bbox (`cap`, `beanie`, `hijab_headscarf`).
- Absence classes (`no_glasses`, `no_headwear`) are not OD bbox classes.

**Bbox & annotation**
- Full-torso bbox for clothing (includes arms for sleeve-length separability).
- Fixed bboxes stored in `refs/manifest.yaml` (ControlNet constrains pose).
- No runtime auto-annotation needed in phase 1.
- Layering inner bbox only for `open_outer`; `closed_outer` = metadata only.

**Layering & thermal**
- 3 layering modes: `single`, `open_outer`, `closed_outer`.
- `closed_outer` not prompted (same as `single` in text and OD).
- 3 thermal tiers: light / mid / heavy.

**Sampling & constraints**
- Uniform within ref whitelist; no per-value weights.
- Global soft couplings: weather↔clothing, garment↔cabin color contrast.
- Layering distribution: ~60–70% single, ~15–25% open_outer, ~10–15% closed_outer.

**Infrastructure**
- Per-ref prompt templates via VLM (`generate_templates.py`, default model `gpt-5.2`).
- `image_name` = `{ref_id}_{sequence:06d}` for file naming.
- `image_size` auto-detected from ref; bboxes normalized [0–1].
- 10–15 refs recommended; 3k–7k images per ref; 5k+ per clothing class.

---

## 16. Example end-to-end prompt (illustrative)

> A Middle Eastern female young adult driver is captured by the dashboard camera in front of the steering wheel. They are looking straight ahead at the road. The image focuses on the upper body, showing them in a blue short sleeve polo, wearing thin wire-frame glasses and a hijab. They are wearing a seatbelt. The car interior is black with leather seats and lightly tinted windows. The scene is set in sunny weather at noon under bright sunlight.

**OD labels for this image:**

| Instance | Class | Bbox region |
|----------|-------|-------------|
| 1 | `short_sleeve_top` | Upper body: shoulders, arms (short sleeves visible), torso to waist |
| 2 | `eyeglasses` | Eye/frame region on face |
| 3 | `hijab_headscarf` | Head covering from crown to shoulders |

**Manifest metadata (not bboxes):** `has_glasses: true`, `has_headwear: true`, `thermal: light`, scene vars, `ref_id`.

---

## 17. Design risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `long_sleeve_plain` vs `_collared` confusion | Low mAP on those two classes | Collar is learnable with volume; keep split, merge only if empirically needed |
| Sleeveless/short-sleeve same bbox if arms cropped | Classes become identical to model | Full-torso bbox includes arms; only use refs where upper arms are visible |
| Layering `open_outer` overlapping boxes | Model may suppress inner box via NMS | Use higher IoU threshold for same-person garment pairs; or multi-label head |
| Inner bbox inconsistency across images | Noisy inner-garment boxes degrade training | ControlNet fixes pose → visible inner area is consistent per ref; define template sub-region |
| Synthetic domain gap | Model overfits to synthetic textures | Pre-train synthetic → fine-tune real; vary lighting/weather; use 10–15 refs |
| Generator ignores prompt details | Wrong garment rendered | QA step (§1.5): CLIP check + person count + sample inspection |
| Seatbelt occlusion confuses garment class | Model confused by diagonal strap | Balance seatbelt on/off; model must learn through exposure (real images also have this) |
| Low garment/cabin contrast | Bbox hard to learn (black-on-black) | Soft constraint: ≥30% samples have visible color contrast (§7.2) |
| Multi-person ID mismatch | Wrong class assigned to wrong person | Map by spatial position (driver = steering side); discard if count mismatch (§7.3) |
| Annotation noise (phase 1 template bboxes) | Fixed boxes don't match slight pose variations | Template boxes with generous margins; refine in phase 2 with pose estimation |
| Small sample per class | Underfitting | Target 5k+ per clothing class; 10–15 refs × 3k–7k per ref |
