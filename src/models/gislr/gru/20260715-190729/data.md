# Data — gislr / gru / 20260715-190729 (ME-126 subset)

Identical to the baseline run (`../20260713-213000/data.md`) in every respect
**except the landmark subset** — that is the point of the experiment.

| | |
|---|---|
| Dataset | GISLR — Kaggle competition `asl-signs` (via `kagglehub`) |
| Videos | 94,477 landmark sequences · 250 sign classes · 21 participants |
| Label map | official `sign_to_prediction_index_map.json` |
| Split | stratified-by-sign 90/10 `train_test_split(random_state=42)` → **85,029 train / 9,448 val** (identical to baseline) |

## The ME-126 landmark subset

Derived in `docs/2026-07-15.md` (§4–§6): the Kaggle-1st-place 118-landmark set
**∪** upper-body pose. Holistic row indices (543-row frame layout: face 0–467,
left hand 468–488, pose 489–521, right hand 522–542):

| group | count | holistic rows |
|---|---|---|
| lips | 40 | 1st-place `LIP` list (face-mesh indices, rows = same) |
| eyes + nose | 36 | `REYE` + `LEYE` + `NOSE` lists |
| left hand | 21 | 468–488 (all) |
| right hand | 21 | 522–542 (all) |
| pose: shoulders, elbows, wrists, hips | 8 | 500–505, 512, 513 (= pose 11–16, 23, 24) |
| **total** | **126** | exact array: [cache/landmarks.npy](cache/landmarks.npy) |

Discarded vs the baseline: 392 face landmarks (rigid-head duplicates), pose
head 0–10 (99% z-noise, xy-static), pose hand-points 17–22 (duplicate the hand
meshes), pose legs 25–32 (out of frame).

## Features fed to the model

| | |
|---|---|
| Landmarks | 126 of 543 (23.2%) |
| Coordinates | x, y, z — **z deliberately kept** so the landmark subset is the only variable vs baseline (xy-only is a separate ablation, TODO §3.1) |
| Input dim / frame | 126 × 3 = **378** |
| NaN handling | `np.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)` at cache build (identical) |
| Normalization | none in the data (model's first layer is `LayerNorm(378)`) |
| Sequence length | > 128 frames uniformly subsampled to 128 via `np.linspace` (identical) |

## Physical storage

Subset memmap cache (built in ~65 s from raw parquet, ThreadPool ×12):
`src/cache/{train,val}_me126_data.npy` + `..._offsets.npy` — 4.88 GB train +
0.54 GB val (vs ~21 GB for the full-543 cache). Loaded fully into RAM during
training (`num_workers=0`).
