# sign2speech

A sign language recognition system focused on **streaming, real-time inference** rather than offline-only accuracy. The end goal is a deployable pipeline that classifies signs frame-by-frame with low latency, trained on hand/pose/face landmark sequences extracted via MediaPipe Holistic.

## Datasets

| Dataset | Role | Source (via `kagglehub`) | Status |
|---|---|---|---|
| **GISLR** | Fast-iteration dataset — landmarks are pre-extracted | Kaggle competition `asl-signs` | Ready to preprocess/train immediately |
| **POPSIGN** | Primary dataset (~870GB raw video) | `mrgeislinger/popsign-asl-v1-0-game-train-{a-e,f-m,n-s,t-z}-signs` + `...-game-test` | Requires landmark extraction first |

Raw data is **not stored in this repository**. It's downloaded on demand via `kagglehub` into its default cache (`~/.cache/kagglehub/`), resolved at runtime by `src/modules/paths.py`. GISLR is a Kaggle *competition* download, so your Kaggle account must have accepted the competition rules and `kagglehub` must be authenticated.

POPSIGN's extracted landmarks (the large intermediate artifact, pre-feature-caching) are written to a separate drive configured via `.env` — see [Environment setup](#environment-setup).

## Models

- **GRU** (unidirectional `StreamingGRU`) — the implemented baseline, built in `src/gislr.1.model.gru.ipynb`. Chosen because it supports true causal/streaming inference. Trained checkpoint lives in `src/checkpoints/gislr/gru/0001/`.
- **Planned benchmarks** — LSTM, BiLSTM (offline-only accuracy reference), ST-GCN, TCN, Transformer, Conformer. See `TODO.md` §4 for scoping notes.

## Project structure

```
sign2speech/
├── README.md
├── TODO.md                   # living task list, organized by workstream
├── pyproject.toml            # uv-managed dependencies (Python >= 3.12)
├── uv.lock
├── .env                      # machine-specific config (POPSIGN_LANDMARKS_DRIVE) — not committed
├── scripts/
│   └── sys_disk_usage.ps1    # disk-usage helper (POPSIGN raw video is ~870GB)
└── src/                      # all code; notebooks assume the kernel CWD is src/
    ├── gislr.0.dataset.motion-energy.ipynb  # diagnostic: landmark motion-over-time analysis (TODO §1)
    ├── gislr.1.model.gru.ipynb              # feature cache build → GRU training → ONNX → TF → TFLite export
    ├── popsign.0.dataset.ipynb              # dataset download + video manifests (stub)
    ├── popsign.1.mediapipe.ipynb            # landmark extraction stage (currently holds early motion-energy exploration)
    ├── popsign.2.model.ipynb                # label distribution, train/val/test split, earlier TF model experiments
    ├── popsign.3.pipeline.ipynb             # end-to-end pipeline (stub)
    ├── modules/
    │   ├── paths.py                         # kagglehub downloads + shared Path constants (DATASETS, DATA_DIR, CKPT_DIR, CACHE_DIR)
    │   └── data/
    │       ├── dataset.py                   # GISLRRawDataset (memmap-backed, Windows-spawn-safe) + collate_fn
    │       └── landmark_worker.py           # MediaPipe Holistic per-video extraction worker (multiprocessing)
    ├── data/                                # gitignored — caches & external assets, never raw datasets
    │   ├── cache/dataframes/                # POPSIGN video manifests (train.csv / test.csv: file_path, label, id)
    │   └── external/mediapipe/tasks/holistic_landmarker.task
    ├── cache/                               # gitignored, created at runtime — GISLR memmapped feature cache, motion-analysis cache
    └── checkpoints/
        └── <dataset>/<architecture>/<run_id>/   # one folder per training run
            ├── gru_best.pt                      # weights (gitignored)
            └── LearningLossAccuracy.png         # metrics/plots for the same run
```

### Conventions

- **Notebooks are flat in `src/`, named `<dataset>.<stage>.<topic>.ipynb`** — dataset first, then a stage number ordering the pipeline, then what the stage does. No nested notebook folders.
- **One folder per training run** at `src/checkpoints/<dataset>/<architecture>/<run_id>/`, holding weights *and* metrics/plots together. A run's artifacts are never split across parallel trees.
- **`data/raw/` doesn't exist.** Raw GISLR/POPSIGN data lives in `kagglehub`'s cache and is resolved at runtime by `modules/paths.py` — never duplicated into the repo.
- **POPSIGN's extracted landmarks go to a separate drive** (configured via `.env`), since that intermediate artifact is far too large to keep alongside the code.
- **Everything runs from `src/`** — `modules/paths.py` and the notebooks use relative paths (`data/`, `cache/`, `checkpoints/`), and `import modules...` resolves because the notebooks sit next to `modules/`. Point your Jupyter kernel's working directory at `src/`.

## Environment setup

```bash
# Install dependencies (uv-managed, Python >= 3.12)
uv sync
```

Create a `.env` file at the project root (not committed) with:

```
POPSIGN_LANDMARKS_DRIVE=D:/    # or wherever your extraction-output drive is mounted
```

Make sure your Jupyter kernel uses this project's `uv`-managed virtual environment (`.venv`) and runs with `src/` as its working directory — both the `modules` import and every relative data/cache/checkpoint path depend on it.

## Running the pipeline

**GISLR** (landmarks already extracted by Kaggle):

1. `src/gislr.0.dataset.motion-energy.ipynb` — *optional diagnostic*: per-video / per-category / global landmark motion-energy analysis. Not part of the training pipeline.
2. `src/gislr.1.model.gru.ipynb` — end-to-end: builds the memmapped feature cache (`cache/`, one-time, resumable), trains the `StreamingGRU` with auto-resume checkpointing, plots learning curves, then exports ONNX → TensorFlow SavedModel → TFLite and validates the TFLite model with the grader's exact calling convention.

**POPSIGN** (raw video, requires extraction first — in progress):

1. `src/popsign.0.dataset.ipynb` — dataset download + video manifest generation (`data/cache/dataframes/`). Currently a stub; the manifests on disk were generated by an earlier iteration.
2. `src/popsign.1.mediapipe.ipynb` — MediaPipe Holistic landmark extraction over raw video, using `modules/data/landmark_worker.py`. Being repurposed — see `TODO.md` §2.
3. `src/popsign.2.model.ipynb` — label-distribution analysis, stratified train/val/test split, and earlier TensorFlow model experiments.
4. `src/popsign.3.pipeline.ipynb` — end-to-end pipeline (stub).

## Constraints & known limitations

- **Streaming-viability drives architecture choice** — the deployment path is the unidirectional GRU; bidirectional models (BiLSTM) can only ever be offline accuracy benchmarks.
- **TensorFlow GPU is not supported on native Windows** — training uses PyTorch (CUDA); TFLite conversion happens post-hoc via ONNX → TensorFlow SavedModel → TFLite.
- **MediaPipe GPU delegate is Ubuntu-only** — landmark extraction runs on CPU on this Windows machine, parallelized across worker processes.
- Hardware: Windows 11, i7-14700K, RTX 4080 Super, 64GB DDR5 RAM.
