# sign2speech

A sign language recognition system focused on **streaming, real-time inference** rather than offline-only accuracy. The end goal is a deployable pipeline that classifies signs frame-by-frame with low latency, trained on hand/pose/face landmark sequences extracted via MediaPipe Holistic.

## Datasets

| Dataset | Role | Raw data location | Status |
|---|---|---|---|
| **GISLR** | Fast-iteration dataset — landmarks are pre-extracted | Kaggle (via `kagglehub`) | Ready to preprocess/train immediately |
| **POPSIGN** | Primary dataset (~870GB raw video) | Kaggle (via `kagglehub`) | Requires landmark extraction first |

Raw data for both datasets is **not stored in this repository**. It's downloaded on demand via `kagglehub` into its default cache location, resolved at runtime by `modules/paths.py`. POPSIGN's extracted landmarks (the large intermediate artifact, pre-feature-caching) are written to a separate drive configured via `.env` — see [Environment setup](#environment-setup).

## Models

Three architectures are trained and compared on both datasets:

- **GRU** (unidirectional) — primary target, since it supports true causal/streaming inference
- **LSTM**
- **BiLSTM** — offline-only; requires future frames, not viable for streaming deployment, kept as an accuracy benchmark

Additional architectures (Transformer, CNN, EfficientNet-B0 on spectrogram-style input) may be added later — see [Adding a new architecture](#adding-a-new-architecture).

## Project structure

````

sign2speech/
├── .env                      # machine-specific config (drive paths, secrets) — not committed
├── pyproject.toml            # package definition; enables `modules` to be imported from anywhere
├── notebooks/
│   ├── gislr/
│   │   ├── 01_preprocess.ipynb   # raw parquet → cached feature tensors
│   │   ├── 02_train.ipynb        # trains a chosen architecture on cached features
│   │   └── 03_evaluate.ipynb     # loads a checkpoint, produces a metrics report
│   └── popsign/
│       ├── 01_extract.ipynb      # raw video → landmarks (2nd drive) via MediaPipe
│       ├── 02_preprocess.ipynb   # extracted landmarks → cached feature tensors
│       ├── 03_train.ipynb
│       └── 04_evaluate.ipynb
├── modules/
│   ├── paths.py               # single source of truth for every reused Path in the project
│   ├── datasets/              # dataset download + raw-file location helpers (via kagglehub)
│   │   ├── gislr.py
│   │   └── popsign.py
│   ├── data/                  # torch Dataset/collate_fn — reads already-cached features
│   │   ├── dataset.py
│   │   └── landmark_worker.py # MediaPipe extraction worker (used by popsign/01_extract)
│   ├── models/                # architecture definitions, one file per model
│   │   ├── gru.py
│   │   ├── lstm.py
│   │   └── bilstm.py
│   └── experiment.py          # run naming, config save/diff, report generation
├── data/
│   ├── processed/             # cached, model-ready feature tensors (small — lives in-repo)
│   │   ├── gislr/
│   │   └── popsign/
│   └── external/
│       └── mediapipe/tasks/holistic_landmarker.task
├── checkpoints/
│   └── <dataset>/<architecture>/<run_id>/
│       ├── model_best.pt
│       ├── model_latest.pt
│       └── config.json
├── metrics/
│   └── <dataset>/<architecture>/<run_id>/
│       ├── report.md          # purpose, config diff vs. previous run, outcome
│       ├── metrics.json       # per-epoch loss/accuracy/lr history
│       ├── learning_curves.png
│       └── confusion_matrix.png
└── scripts/
└── sys_disk_usage.ps1

````

### Why this layout

- **`data/raw/` doesn't exist.** GISLR and POPSIGN raw data live in `kagglehub`'s default cache and are resolved at runtime — never duplicated into the repo.
- **POPSIGN's extracted landmarks live on a separate drive**, not under `data/`, since that intermediate artifact is far too large to keep alongside the code. Only the final, small, feature-cached tensors (`data/processed/`) live in-repo.
- **`checkpoints/` and `metrics/` mirror each other path-for-path** (`<dataset>/<architecture>/<run_id>/`). Every training run gets a unique timestamped `run_id`, generated once and shared by both — so a checkpoint and its corresponding metrics report are always trivially findable from each other, by construction rather than convention.
- **Notebooks are grouped by dataset, then ordered by stage** (`01_`, `02_`, ...). Training notebooks (`0X_train.ipynb`) are architecture-agnostic — the specific model (GRU/LSTM/BiLSTM) is chosen via a config cell at the top, not the filename, so training logic isn't duplicated per architecture.
- **`modules/` is an installed editable package**, not a folder relied on via relative imports — see [Environment setup](#environment-setup) for why this matters.

## Environment setup

````bash
# Install dependencies (uv-managed)
uv sync

# Install this project itself as an editable package —
# required so `from modules.paths import ...` works from any notebook,
# regardless of that notebook's working directory
uv pip install -e .
````

Create a `.env` file at the project root (not committed) with:

````
POPSIGN_LANDMARKS_DRIVE=D:/            # or wherever your extraction-output drive is mounted
````

Make sure your Jupyter kernel is pointed at this project's `uv`-managed virtual environment, not a global Python install — otherwise `modules` won't be importable even after the editable install.

## Running the pipeline

**GISLR** (landmarks already extracted):

1. `notebooks/gislr/01_preprocess.ipynb` — builds the feature cache in `data/processed/gislr/`
2. `notebooks/gislr/02_train.ipynb` — set the architecture in the config cell, train, checkpoint auto-saves per epoch with resume support
3. `notebooks/gislr/03_evaluate.ipynb` — point at a `checkpoints/gislr/<arch>/<run_id>/` folder, generates the metrics report

**POPSIGN** (raw video, requires extraction first):

1. `notebooks/popsign/01_extract.ipynb` — runs MediaPipe Holistic over raw video, writes landmarks to the configured second drive
2. `notebooks/popsign/02_preprocess.ipynb` — builds the feature cache in `data/processed/popsign/`
3. `notebooks/popsign/03_train.ipynb`
4. `notebooks/popsign/04_evaluate.ipynb`

## Adding a new architecture

1. Add `modules/models/<name>.py` with the model class.
2. Add `notebooks/<dataset>/0X_train.ipynb` if you want a dedicated notebook per architecture — or, if using the config-cell pattern, no new notebook is needed at all; just add the architecture as a new option in the existing training notebook's config cell.
3. No changes to `modules/paths.py`, `modules/experiment.py`, or preprocessing notebooks are required — checkpoint/metrics paths are derived automatically from `<dataset>/<architecture>/<run_id>`.

## Adding a new dataset

1. Add `modules/datasets/<name>.py` with `kagglehub`-based download + raw-file location helpers.
2. Add path constants to `modules/paths.py` (processed-data dir, any dataset-specific external paths).
3. Add `notebooks/<name>/01_preprocess.ipynb` (plus `01_extract.ipynb` first, if the dataset starts as raw video rather than pre-extracted landmarks).
4. Existing training/evaluation notebooks work unchanged once the new dataset's `data/processed/<name>/` cache exists in the same format (`*_features.npy`, `*_offsets.npy`, `label_map.json`).

## Constraints & known limitations

- **BiLSTM is not streaming-viable** — kept only as an offline accuracy benchmark; production/deployment path is GRU.
- **TensorFlow GPU is not supported on native Windows** — training uses PyTorch; TFLite conversion (for eventual Kaggle-style submission format) happens post-hoc via ONNX → TensorFlow SavedModel → TFLite.
- **MediaPipe GPU delegate is Ubuntu-only** — landmark extraction runs on CPU on this Windows machine.
- Hardware: Windows 11, i7-14700K, RTX 4080 Super, 64GB DDR5 RAM.

````

A couple of things worth double-checking once you paste this in:

1. The **kagglehub dataset slugs** are still placeholders in `modules/datasets/gislr.py` / `popsign.py` from earlier — once you give me those, I can also drop the exact dataset name/slug into this README's dataset table for completeness.
2. If POPSIGN's raw structure turns out to need more setup steps than a single `kagglehub.dataset_download()` call (e.g. manual competition rules acceptance, a separate metadata file), that's worth a short note in the Environment setup section too.

Want me to now go build out `modules/models/gru.py` (porting your existing `StreamingGRU`) plus stub `lstm.py`/`bilstm.py`, so the module tree actually matches what this README describes?
````