# VLM Batch Annotation

Person **envelope crop** (all pose keypoints + 10% pad, clamped to image) → VLM classifies once per person → labels mapped to **refined OpenPose bboxes** for export.

## Two-step pipeline

```
Step 1 — run_batch.py
  OpenPose match + envelope crop + VLM classify
  → vlm_outputs/{ref_id}/{image}.json   (per-image VLM results)
  → vlm_cache/vlm_person.jsonl          (cache, one row per person)
  → vlm_geometry/{ref_id}/{image}.json  (refined export bboxes)
  → vlm_crops/{ref_id}/                 (envelope JPEGs)

Step 2 — run_aggregate.py
  Read geometry + VLM cache
  → annotations_vlm/{image}.json        (full-frame COCO-ready)

Step 3 — export_coco.py
  Read annotations_vlm/
  → synthetic_clothing_v3/                (train/ val/ + COCO JSON)
```

## Envelope bbox

- Union of **all body + face** OpenPose keypoints for that person
- **Asymmetric pad:** **18%** horizontal, 10% bottom; **extra head room** above face (55% of face height — face contour sits on jaw, not crown)
- **Union** with headwear export region so caps/hijabs are not clipped
- **Clamped** to `[0, 1]` — never extends outside the image
- Used **only** for VLM input; **not exported**

After changing envelope/headwear geometry, re-run with `--force` and `run_aggregate`.

## QA review (no VLM API)

```bash
python scripts/review_vlm_sample.py --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --samples 100
```

Writes envelope crops + full-frame bbox previews under `vlm_review/`.

Export uses refined bboxes: torso, inner chest (open_outer), glasses, headwear.

## Usage

```bash
# Count calls (1 VLM request per person)
python -m vlm_annotation.run_batch \
  --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --dry-run --samples 100

# Step 1: classify (sync for samples, batch for full run)
python -m vlm_annotation.run_batch \
  --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --samples 100 --sync

python -m vlm_annotation.run_batch \
  --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --all

# Step 2: aggregate to full-frame annotations
python -m vlm_annotation.run_aggregate \
  --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --samples 100

# Step 3: export COCO dataset
python -m vlm_annotation.export_coco \
  --run-dir data/run_20260520_034326 --ref-id dash_ref_01
```

## Batch recovery

If `run_batch` exits early (e.g. one parallel chunk failed) but OpenAI finished a batch, pull results without re-submitting:

```bash
# Check batch status
python -m vlm_annotation.list_batches --active-only

# Recover completed batch(es) into vlm_cache/
python -m vlm_annotation.recover_batch \
  --run-dir data/run_20260520_034326 \
  --batch-id batch_6a1314bd052c8190b9df37915e61b406 \
  --ref-id dash_ref_01

# Then re-run run_batch for any remaining pending persons (sequential, no --parallel-chunks)
python -u -m vlm_annotation.run_batch \
  --run-dir data/run_20260520_034326 --ref-id dash_ref_01 --all
```

Skips persons already in `vlm_cache/vlm_person.jsonl`. Use `--force` to overwrite cached rows.

## Batch sizing (gpt-5.2 token limit)

OpenAI enforces a **900k enqueued-token limit** per org for batch jobs. Envelope crops are large, so defaults are conservative:

- **500 requests** and **50 MB** per chunk (override with `--batch-max-requests` / `--batch-max-mb`)
- **Sequential submit** by default: submit chunk → wait → cache → next chunk
- **Waits for in-flight batches** before each submit (disable with `--no-wait-for-queue`)
- Avoid `--parallel-chunks` unless the queue is empty

For ~1476 pending persons, expect **3 chunks** (~500 each). Check queue: `python -m vlm_annotation.list_batches --active-only`

## Outputs

| Path | Contents |
|------|----------|
| `vlm_crops/{ref_id}/{image}__{role}.jpg` | Envelope crop sent to VLM |
| `vlm_outputs/{ref_id}/{image}.json` | Parsed + raw VLM per person |
| `vlm_cache/vlm_person.jsonl` | Append-only cache |
| `vlm_cache/disagreements.jsonl` | Prompt hint vs VLM mismatches |
| `vlm_geometry/{ref_id}/{image}.json` | Refined export slots (sidecar) |
| `annotations_vlm/{image}.json` | Final merged annotations |
| `synthetic_clothing_v3/` | COCO export (train/ val/ + annotations/*.json) |

## Taxonomy

Option A — 12 OD classes. See [taxonomy.py](taxonomy.py).

## Still TODO

- [x] **COCO export** — `python -m vlm_annotation.export_coco` → `synthetic_clothing_v3/`
- [ ] Update `src/config.py` for future prompt generation
- [ ] Human QA on `disagreements.jsonl` before full export
