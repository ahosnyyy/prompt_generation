# Synthetic Car-Cabin Clothing Dataset — Prompt Generation

Generate structured prompts for synthetic in-cabin image generation using z-image + ControlNet over multiple reference images. The prompts drive a bounding-box object detector training pipeline for upper-body garments, glasses, and headwear (12 OD classes total).

See [docs/DESIGN.md](docs/DESIGN.md) for the full design document.

---

## How to Run

### Step 1: Add reference images

Place your reference images (depth/ControlNet source images) in `refs/`:

```
refs/
├── dash_ref_01.jpg
├── dash_ref_02.jpg
├── ...
└── dash_ref_15.jpg
```

Each ref image defines a fixed camera angle, pose, and person layout. ControlNet will preserve this geometry across all generated images.

See [Reference Image Guidelines](#reference-image-guidelines) below for what makes a good ref.

### Step 2: Update the manifest

Edit `refs/manifest.yaml` with what you know manually:

```yaml
refs:
# 2-person ref — use steering_side to identify driver vs passenger
- id: dash_ref_01
  image_path: refs/dash_ref_01.jpg
  occupancy: 2
  steering_side: left          # LHD: driver is on the left

# 1-person ref — use person_role to say who it is
- id: dash_ref_04
  image_path: refs/dash_ref_04.jpg
  occupancy: 1
  person_role: driver          # "driver" or "passenger"
```

**What you provide:**

| Field | When | Description |
|-------|------|-------------|
| `id` | Always | Unique name (becomes image_name prefix) |
| `image_path` | Always | Path to the ref image |
| `occupancy` | Always | Fixed: `1` or `2` (matches people in the ref) |
| `steering_side` | `occupancy: 2` | `left` (LHD) or `right` (RHD) — identifies driver side |
| `person_role` | `occupancy: 1` | `driver` or `passenger` — who the single person is |

**What the VLM auto-fills (step 4):**
- `arms_visible` — whether upper arms are clearly visible
- `cabin_description` — free-text description of the cabin interior
- `conditions` — whitelist of allowed scene values (lighting, weather, etc.)
- `template_file` — prompt template `.txt` file

### Step 3: Define bounding boxes

Create a file `refs/{id}_bboxes.yaml` for each ref:

```yaml
# refs/dash_ref_04_bboxes.yaml (occupancy: 1)

image_size: [1920, 1080]  # [width, height] in pixels

bboxes:
  driver:
    clothing: [0.15, 0.25, 0.85, 0.95]
    headwear: [0.30, 0.00, 0.70, 0.22]
    glasses: [0.35, 0.12, 0.65, 0.22]
    inner_clothing: [0.30, 0.35, 0.70, 0.85]
```

For 2-person refs, add both `driver` and `passenger`:

```yaml
# refs/dash_ref_01_bboxes.yaml (occupancy: 2)

image_size: [1920, 1080]

bboxes:
  driver:
    clothing: [0.05, 0.30, 0.48, 0.95]
    headwear: [0.12, 0.02, 0.42, 0.25]
    glasses: [0.18, 0.12, 0.38, 0.22]
    inner_clothing: [0.15, 0.40, 0.40, 0.85]
  passenger:
    clothing: [0.52, 0.30, 0.95, 0.95]
    headwear: [0.60, 0.02, 0.88, 0.25]
    glasses: [0.65, 0.12, 0.82, 0.22]
    inner_clothing: [0.60, 0.40, 0.85, 0.85]
```

**Bbox format:**

```
[x_min, y_min, x_max, y_max]   — all normalized 0.0 to 1.0
```

| Coordinate | Meaning |
|-----------|---------|
| `(0, 0)` | top-left corner of image |
| `(1, 1)` | bottom-right corner of image |

How to define: open the ref image, draw boxes around the regions, divide pixel coords by image width/height.

| Bbox key | What to draw around |
|----------|---------------------|
| `clothing` | Full torso: shoulders to waist, include visible arms |
| `headwear` | Head-top: crown of head down to forehead/ears |
| `glasses` | Eye region: tight around where frames would sit |
| `inner_clothing` | Chest area between open jacket lapels (for `open_outer` layering) |

Use generous margins (~5-10% padding). ControlNet preserves the layout, so these are constants for all images generated from that ref.

### Step 4: Run VLM analysis (one-time)

```bash
# Default: uses Batch API (50% cost discount, async)
python -m src.generate_templates

# Force regeneration for all refs
python -m src.generate_templates --force

# Use synchronous API instead (faster, no discount)
python -m src.generate_templates --sync
```

This does three things per ref image:
1. **Analyzes** the image → fills `arms_visible`, `cabin_description`, and `conditions` into `manifest.yaml`
2. **Generates** prompt templates → saves as `.txt` files in `refs/`
3. **Sends images** as base64 in the API requests

**Batch API vs Synchronous:**

| Mode | Cost | Speed | When to use |
|------|------|-------|-------------|
| Batch (default) | **50% discount** | Up to 24h (usually minutes) | Multiple refs, not time-sensitive |
| Sync (`--sync`) | Full price | Immediate | Single ref, quick iteration |

The script auto-selects Batch API when more than 1 ref needs processing. All requests (analysis + template generation) are bundled into a single batch job.

Requires `OPENAI_API_KEY` in your `.env` file. Uses a vision model (default: `gpt-5.2`, configurable in `src/config.py`). Results are cached — run once per ref, or `--force` to redo.

After this step, your manifest will have the full profile:

```yaml
- id: dash_ref_01
  image_path: refs/dash_ref_01.jpg
  occupancy: 2
  steering_side: left
  # --- VLM auto-filled below ---
  arms_visible: true
  cabin_description: "Modern sedan interior with black leather seats, center console
    with touchscreen, dashboard-mounted camera at eye level."
  conditions:
    lighting: [bright sunlight, overcast]
    weather: [sunny, rainy]
    time: [morning, noon, evening]
    seat_type: [leather seats]
    window_tint: [clear windows, lightly tinted windows]
    cabin_color: [black]
  template_file: refs/dash_ref_01.txt
```

You can edit the VLM output after generation if needed.

### Step 5: Generate prompts

```bash
# Default (1000 samples, from refs/manifest.yaml)
python run.py

# Custom sample size
python run.py --sample-size 5000

# Custom manifest path
python run.py --manifest path/to/manifest.yaml

# Custom output directory
python run.py --output-dir output/
```

### Step 6: Use the output

Output is saved to `data/run_YYYYMMDD_HHMMSS/`:

```
data/run_20260519_184316/
├── prompts.json              # image_name + prompt text (for ComfyUI batching)
├── prompts_full.json         # full structured records with all metadata
├── csv/                      # one CSV per ref (image_name, json_file, prompt)
│   ├── dash_ref_01.csv
│   ├── dash_ref_02.csv
│   └── ...
└── individual_prompts/       # one JSON per image
    ├── dash_ref_01_000000.json
    ├── dash_ref_01_000001.json
    └── ...
```

Feed the per-ref CSV files into your z-image + ControlNet pipeline, batched by ref.

---

## Reference Image Guidelines

### Recommended count

| Guideline | Value | Rationale |
|-----------|-------|-----------|
| Minimum refs | 8–10 | Below this, model memorizes background/layout; poor generalization |
| Recommended | 10–15 | Good pose/layout diversity for cabin dash cams |
| Diminishing returns | Beyond ~20 | Each additional ref contributes less novel layout info |

### What each ref should vary

Across your set of refs, aim for diversity in:

- **Camera angle/position** — slightly different dash-cam mount points (higher, lower, angled)
- **Cabin type/interior** — different car models, seat styles, dashboard shapes
- **Person position** — arm on wheel, arm on rest, hands at 10-and-2, one hand on wheel
- **1-person vs 2-person layout** — mix of single-driver and driver+passenger refs
- **Arm visibility** — at least some refs must clearly show upper arms (critical for sleeve-length classes)

### Quality criteria for a good ref image

| Criteria | Why it matters |
|----------|---------------|
| Upper body clearly visible | The OD model needs torso + arms in frame |
| Stable, predictable pose | ControlNet preserves this layout for all generated images |
| Natural dash-cam perspective | Matches real deployment camera angles |
| No extreme occlusion | Steering wheel can partially occlude, but torso should be mostly visible |
| Head visible (crown to chin) | Needed for headwear and glasses bbox regions |
| Neutral clothing/accessories | Ref person's actual clothing doesn't matter — it gets replaced by the prompt |
| Good resolution | At least 1024px on the shorter side; 1920x1080 recommended |
| Consistent lighting in the ref | Avoids baked-in shadows that conflict with prompted lighting |

### Refs to avoid

- Extreme close-ups where only the face is visible (no torso for clothing bbox)
- Rear-seat camera angles (not relevant for driver detection)
- Heavily cropped images where arms are fully out of frame (unless intentional — set `arms_visible: false`)
- Refs with large watermarks or artifacts
- Multiple people where it's ambiguous who is driver vs passenger

### Per-ref quota (production scale)

With 10–15 refs generating 3,000–7,000 images each:
- 15 refs × 4,000 images/ref = 60,000 total images
- Stratified across 7 clothing classes ≈ 8,500 per class (meets the 5k+ target)

---

## Project Structure

```
prompt_generation/
├── refs/
│   ├── manifest.yaml              # per-ref profiles (occupancy, role/steering)
│   ├── dash_ref_01.jpg            # reference images (gitignored)
│   ├── dash_ref_01_bboxes.yaml    # bboxes + image_size per ref
│   └── dash_ref_01.txt            # prompt templates per ref (VLM-generated)
├── src/
│   ├── __init__.py
│   ├── config.py                  # taxonomies, enums, sampling distributions
│   ├── generate_prompts.py        # main prompt generation pipeline
│   ├── generate_templates.py      # VLM analysis + template generation (Batch API)
│   └── utils.py                   # sampling logic, constraints, output
├── docs/
│   └── DESIGN.md                  # full design document
├── data/                          # generated output (gitignored)
├── run.py                         # CLI entry point
├── requirements.txt
├── .env                           # OPENAI_API_KEY (gitignored)
└── .gitignore
```

---

## Taxonomy Summary

| Domain | Manifest classes | OD bbox classes |
|--------|-----------------|-----------------|
| Clothing | 7 (silhouette-based) | 7 |
| Glasses | 3 (`no_glasses`, `eyeglasses`, `sunglasses`) | 2 |
| Headwear | 5 (`no_headwear`, `bare_head`, `cap`, `beanie`, `hijab_headscarf`) | 3 |

**Total OD bbox classes: 12**

---

## Configuration

Edit `src/config.py` to adjust:

- Clothing/glasses/headwear taxonomies and `gen_variant` lists
- Sampling distributions (glasses presence, headwear, layering)
- Weather-clothing coupling bias
- Color contrast constraints
- `SAMPLE_SIZE` (default 1000, development placeholder)
- `VLM_MODEL` for template generation (default `gpt-5.2`)

---

## Dependencies

```
python>=3.10
pyyaml
openai         # needed for generate_templates.py (VLM + Batch API)
Pillow         # only needed for auto-detecting ref image dimensions
python-dotenv  # loads OPENAI_API_KEY from .env
```

Install:

```bash
pip install -r requirements.txt
```
