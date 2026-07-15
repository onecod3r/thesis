# Data — gislr / gru / 20260713-213000

| | |
|---|---|
| Dataset | GISLR — Kaggle competition `asl-signs` (via `kagglehub`) |
| Videos | 94,477 landmark sequences · 250 sign classes · 21 participants |
| Label map | official `sign_to_prediction_index_map.json` (canonical class ordering) |
| Split | stratified-by-sign 90/10 `train_test_split(random_state=42)` → **85,029 train / 9,448 val** |
| Test set | none — Kaggle grades via submission; the val split is the only generalization monitor |

## Features fed to the model

| | |
|---|---|
| Landmarks | **all 543** MediaPipe Holistic rows (468 face, 21 left hand, 33 pose, 21 right hand) — no subset selection |
| Coordinates | x, y, z (all three) |
| Input dim / frame | 543 × 3 = **1,629** |
| NaN handling | `np.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)` at cache build |
| Normalization | none in the data (model applies `LayerNorm(1629)` as its first layer) |
| Sequence length | sequences > 128 frames are **uniformly subsampled** to 128 via `np.linspace` (not truncated); shorter sequences padded in the collate, packed for the GRU |

## Physical storage

One-time memmapped cache built by `src/gislr.1.model.gru.ipynb` cell 4b:
`src/cache/{train,val}_data.npy` (flat float32) + `{train,val}_offsets.npy`
(cumulative frame offsets), ~21 GB total. The cache holds raw cleaned values —
landmark/coordinate selection would happen downstream (none does in this run).

## Caveats

- All-543 input means the model ingests the 392 near-rigid face landmarks, the
  out-of-frame leg landmarks, and the noise-dominated z channel documented in
  `docs/2026-07-15.md` §3 — this run predates that analysis and serves as the
  full-input baseline for the subset ablations (TODO §3.1).
