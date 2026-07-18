# sign2speech

A sign language recognition system focused on **streaming, real-time inference** rather than offline-only accuracy. The end goal is a deployable pipeline that classifies signs frame-by-frame with low latency, trained on hand/pose/face landmark sequences extracted via MediaPipe Holistic.

## Datasets

| Dataset | Role | Source (via `kagglehub`) | Status |
|---|---|---|---|
| **GISLR** | Fast-iteration dataset — landmarks are pre-extracted | Kaggle competition `asl-signs` | Ready to preprocess/train immediately |
| **POPSIGN** | Primary dataset (~870GB raw video) | `mrgeislinger/popsign-asl-v1-0-game-train-{a-e,f-m,n-s,t-z}-signs` + `...-game-test` | Requires landmark extraction first |

Raw data is **not stored in this repository**. It's downloaded on demand via `kagglehub` into its default cache (`~/.cache/kagglehub/`), resolved lazily by `src/modules/paths.py` (`gislr_dir()` / `resolve_datasets()` — importing the module never downloads anything). GISLR is a Kaggle *competition* download, so the Kaggle account in use must have accepted the competition rules and `kagglehub` must be authenticated.

POPSIGN's extracted landmarks (the large intermediate artifact, pre-feature-caching) are written to a separate drive configured via `.env` — see [Environment setup](#environment-setup).

## Models

- **GRU** (unidirectional `StreamingGRU`) — the deployment baseline. Chosen because it supports true causal/streaming inference.
- **Landmark-subset ablations** — motivated by the motion-energy analysis and the Kaggle 1st-place cross-check ([docs/daily/2026-07-15.md](docs/daily/2026-07-15.md)): the **ME-126** subset (hands + upper-body pose + lips + eyes/nose) beat the full-543 baseline **73.73% vs 70.59% val accuracy with half the parameters** (v1-regime runs, pre-reset — see registry note below), independently confirmed by the discriminability probe comparison ([docs/daily/2026-07-16.md](docs/daily/2026-07-16.md)). Canonical subset index lists: `src/modules/dataset/landmark/subsets.py`. Remaining ablations in `TODO.md` §3.1.
- **Architecture benchmarks (runs pending)** — `StreamingLSTM` (streaming-viable), `BiLSTM` (offline-only accuracy reference — prices the causality gap), `CausalConv1D` (dilated causal 1D-CNN, streaming-viable) in `src/gislr.1.model.{lstm,bilstm,cnn1d}.ipynb`. All four training notebooks are thin drivers over the shared stack in `src/modules/model/`. Still planned: the full 1st-place 1D-CNN + Transformer port, ST-GCN, TCN, Conformer. See `TODO.md` §4.

## Model registry

Every training run gets **one flat folder** at `src/data/models/<run_id>/`, where `run_id` is the **seconds since the Unix epoch** at training start. Dataset, architecture and subset are *fields in `meta.json`* (and columns of `index.csv`), not directory levels.

| file | content |
|---|---|
| `meta.json` | the single machine-readable run record (schema below) |
| `best.pt` | best-val-accuracy checkpoint (gitignored) |
| `last.pt` | latest checkpoint, saved every epoch — auto-resume state (gitignored) |
| `assets/` | every other artifact: learning curves, per-class CSVs/plots, landmark indices, eval summary — each linked from `meta.json["assets"]` |

All `meta.json` files are flattened into the queryable **[src/data/models/index.csv](src/data/models/index.csv)** by `src/modules/scripts/build_model_index.py` (runs from anywhere), which also answers filter queries directly — e.g. `--dataset gislr --architecture gru --top 3` or `--subset ME_126`. The training driver writes `meta.json` **every epoch** with `eval_status: "pending"`; `src/modules/scripts/eval_gru.py` fills in the canonical eval numbers and flips it to `"canonical"`.

> **Registry reset (2026-07-18).** The registry was restarted empty when the flat epoch-seconds layout was adopted. The 8 pre-reset runs (GRU full-543 baseline 70.59%, ME-126 73.73%, the xy ablations, …) survive only in git history (`3668dae` and earlier, under the old `src/models/` tree) and in the daily reports — their weights are gone, so their canonical evals cannot be completed; the numbers remain as historical references.

### meta.json schema

**This section is the source of truth for the schema** (`schema_version: 2`; the machine-side key check lives in `modules/model/registry.py::REQUIRED_KEYS`). All keys are required; unknown extra keys are not written.

| key | type | content |
|---|---|---|
| `schema_version` | int | `2` |
| `run_id` | int | seconds since Unix epoch at training start = run folder name |
| `created` | str | ISO-8601 local timestamp derived from `run_id` |
| `dataset` | str | e.g. `"gislr"` |
| `architecture` | str | key into `modules.model.ARCHS`: `gru` / `lstm` / `bilstm` / `cnn1d` |
| `model_name` | str | class name, e.g. `"StreamingGRU"` |
| `streaming` | bool | streaming-viable? (`false` = offline-only reference, never deployable) |
| `subset` | str | landmark-subset name from `modules/dataset/landmark/subsets.py` |
| `coords` | str | `"xyz"` or `"xy"` (z-drop ablation) |
| `n_landmarks` / `feature_dim` / `n_classes` / `n_params` | int | model/input dimensions |
| `split` | object | `{strategy, random_state, n_val}` — the canonical split (`stratified 90/10`, seed 42, 9,448 val) |
| `training` | object | `{regime, source, epoch_cap, epochs_trained, best_epoch, early_stopped, finished, wall_time_min}` |
| `hyperparameters` | object | full `HYP` dict + `seed`, `max_seq_len`, `num_workers`, `loss`, `precision` |
| `metrics` | object | `{train_val_acc, eval_status ("pending"\|"canonical"), overall_accuracy, macro_accuracy, median_class_accuracy, n_classes_below_50pct}` — canonical fields are `null` until `eval_gru.py` runs and then survive training-loop rewrites |
| `checkpoints` | object | `{best: "best.pt", last: "last.pt"}` — run-dir-relative |
| `assets` | object | `{name: run-dir-relative path}` for every asset file, e.g. `{"landmarks": "assets/landmarks.npy", "learning_curves": "assets/learning_curves.png"}` |
| `notes` | str | free-text run notes |

## Reports & docs

See [docs/README.md](docs/README.md): day-by-day logs in `docs/daily/<YYYY-MM-DD>.md`, weekly summaries in `docs/weekly/<YYYY>-<WW>.md` (ISO week, e.g. `2026-29.md`), standalone topic reports in `docs/reports/<topic>.md` — figures always under the matching `assets/` subfolder.

| date | report | contents |
|---|---|---|
| 2026-07-15 | [docs/daily/2026-07-15.md](docs/daily/2026-07-15.md) | GISLR landmark motion-energy analysis (z-noise finding, keep/discard recommendation) · 1st-place solution landmark cross-check · GRU training in detail: full-543 baseline vs ME-126 subset (+3.1 pts at half the parameters) |
| 2026-07-16 | [docs/daily/2026-07-16.md](docs/daily/2026-07-16.md) | Landmark-subset discriminability comparison (F-ratio / MI / probe classifier, 3 scopes): **ME-126 wins**; discriminability ≠ motion energy (rho −0.12) · POPSIGN extraction module + driver notebook (resource-capped, resumable) built & validated |
| 2026-07-17 | [docs/daily/2026-07-17.md](docs/daily/2026-07-17.md) | Registry tooling v1: per-run metadata + queryable index · fresh-run-folder policy · training regime **v2-plateau-300** · LSTM / BiLSTM / CausalConv1D benchmark notebooks built · eval script generalized (arch dispatch + xy mode) |
| 2026-07-18 | [docs/daily/2026-07-18.md](docs/daily/2026-07-18.md) | **Repo restructure**: unified `modules/model` training stack · flat epoch-seconds model registry + meta.json schema v2 (registry reset) · `data/{raw,cache,temp,models}` tree + temp-cleanup policy · docs daily/weekly/reports split · single-progress-bar training |

## Project structure

```
sign2speech/
├── README.md
├── TODO.md                   # living task list, organized by workstream
├── pyproject.toml            # uv-managed dependencies (Python >= 3.12, incl. torch cu130 via [tool.uv.sources])
├── uv.lock
├── .env                      # machine-specific config (POPSIGN_LANDMARKS_DRIVE) — not committed
├── docs/
│   ├── daily/<YYYY-MM-DD>.md          # day-by-day logs + dated analyses (assets in daily/assets/<date>/)
│   ├── weekly/<YYYY>-<WW>.md          # weekly summaries (ISO week number)
│   └── reports/<topic>.md             # standalone topic reports (assets in reports/assets/<topic>/)
├── scripts/                  # housekeeping / misc only (NO project Python scripts — those live in src/modules/scripts/)
│   └── sys_disk_usage.ps1    # disk-usage helper (POPSIGN raw video is ~870GB)
└── src/                      # all code; notebooks assume the kernel CWD is src/
    ├── gislr.0.dataset.motion-energy.ipynb      # diagnostic: landmark motion-over-time analysis (TODO §1)
    ├── gislr.0.dataset.subset-comparison.ipynb  # diagnostic: landmark-subset discriminability comparison (TODO §3)
    ├── gislr.1.model.gru.ipynb                  # GRU training driver (top-3 subsets) + deferred TFLite export
    ├── gislr.1.model.lstm.ipynb                 # StreamingLSTM benchmark (streaming-viable, TODO §4)
    ├── gislr.1.model.bilstm.ipynb               # BiLSTM offline-only accuracy reference (TODO §4)
    ├── gislr.1.model.cnn1d.ipynb                # CausalConv1D benchmark (streaming-viable, TODO §4)
    ├── popsign.0.dataset.extraction.ipynb       # POPSIGN landmark extraction driver (TODO §2)
    ├── popsign.1.mediapipe.ipynb                # stale — slated for retirement (TODO §0.1)
    ├── popsign.2.model.ipynb                    # label distribution, split, earlier TF experiments
    ├── popsign.3.pipeline.ipynb                 # end-to-end pipeline (stub)
    ├── modules/
    │   ├── paths.py                  # canonical tree constants (absolute, CWD-independent) + lazy dataset resolution + cleanup_temp()
    │   ├── model/                    # unified training stack shared by all gislr.1.model.* notebooks
    │   │   ├── architectures.py      # StreamingGRU / StreamingLSTM / BiLSTM / CausalConv1D + ARCHS registry (single definition, shared with eval)
    │   │   ├── data.py               # canonical split, per-subset feature caches, in-RAM dataset
    │   │   ├── registry.py           # run folders (epoch seconds), meta.json writing, asset registration
    │   │   ├── train.py              # training driver: auto-resume, early stopping, ONE progress bar per run
    │   │   └── report.py             # learning-curve plots into a run's assets/
    │   ├── scripts/                  # project Python CLIs (run from anywhere — they bootstrap their own imports)
    │   │   ├── eval_gru.py           # canonical per-class eval of any run (all archs, xy/xyz); promotes meta.json to "canonical"
    │   │   └── build_model_index.py  # flattens all meta.json files into data/models/index.csv + answers filter queries
    │   └── dataset/landmark/
    │       ├── subsets.py            # canonical landmark-subset registry (FULL_543, ME_126, FP_118, …)
    │       └── extraction.py         # MediaPipe Holistic video→landmark extraction (multiprocess, resource-capped, resumable)
    └── data/                         # gitignored EXCEPT the model registry (data/models/)
        ├── raw/                      # extracted-from-source data (POPSIGN landmark npz: raw/popsign/{train,test}/)
        ├── cache/                    # reusable derived artifacts, one subtree per dataset:
        │   ├── gislr/features/       #   per-subset feature caches (GBs, shared across architecture notebooks)
        │   ├── gislr/motion_analysis/, gislr/subset_comparison/   # diagnostic-notebook caches
        │   ├── popsign/dataframes/, popsign/extraction/           # POPSIGN manifests + extraction measurements
        │   └── runs/                 #   run-dir pointer files (auto-resume)
        ├── temp/                     # throwaway scratch (e.g. POPSIGN pilot npz) — deleted after use (cleanup_temp())
        ├── external/                 # third-party assets (MediaPipe holistic_landmarker.task)
        └── models/                   # COMMITTED model registry (weights gitignored)
            ├── index.csv             # queryable table of all runs
            └── <run_id>/             # one flat folder per run (epoch seconds): meta.json + best.pt/last.pt + assets/
```

### Conventions

- **Notebooks are flat in `src/`, named `<dataset>.<stage>.<topic>.ipynb`** — dataset first, then a stage number ordering the pipeline, then what the stage does. No nested notebook folders.
- **Shared logic lives in `src/modules/`, not in notebooks.** The four training notebooks differ only in their architecture-identity block (`ARCH`, `TRAIN_SUBSETS`, `HYP`) — everything else (model classes, data layer, training driver, registry, reporting) is `modules/model/`. Project Python CLIs live in **`src/modules/scripts/`**; the root `scripts/` folder holds housekeeping/misc scripts only.
- **One flat registry folder per training run** at `src/data/models/<epoch-seconds>/` holding `meta.json` + `best.pt`/`last.pt` + `assets/`. A run's artifacts are never split across parallel trees; `index.csv` is the queryable view.
- **Docs**: daily logs in `docs/daily/`, weekly summaries in `docs/weekly/` (`<YYYY>-<WW>.md`), standalone topic reports in `docs/reports/` — see `docs/README.md`.
- **Data placement policy**: extracted-from-source data → `data/raw/<dataset>/…`; reusable derived artifacts → `data/cache/<dataset>/…`; throwaway output → `data/temp/`, **deleted after use** (`modules.paths.cleanup_temp()`); third-party assets → `data/external/`. Raw kagglehub downloads never enter the repo — `modules/paths.py` resolves them lazily at call time (`gislr_dir()`, `resolve_datasets()`), never at import time.
- **POPSIGN's extracted landmarks go to `data/raw/popsign/{train,test}`**, rooted at the drive configured via `POPSIGN_LANDMARKS_DRIVE` in `.env` when set (falling back to `src/data/raw/`, gitignored). The extraction module (`modules/dataset/landmark/extraction.py`) resolves this in one place.
- **Paths are CWD-independent**: `modules/paths.py` anchors every constant to the `src/` directory, so notebooks (kernel CWD = `src/`) and the `modules/scripts/` CLIs (any CWD) agree on the same tree.
- **Canonical evaluation** (all GISLR runs must match to be comparable): stratified 90/10 split, `random_state=42`, 9,448-video val set, per-class accuracy from raw parquet — exactly what `modules/scripts/eval_gru.py` reproduces. A new run displaces a leaderboard entry only on this same split/metric.

## Environment setup

```bash
# Install dependencies (uv-managed, Python >= 3.12)
uv sync
```

Create a `.env` file at the project root (not committed) with:

```
POPSIGN_LANDMARKS_DRIVE=D:/    # or wherever the extraction-output drive is mounted
```

The Jupyter kernel must use this project's `uv`-managed virtual environment (`.venv`) and run with `src/` as its working directory — `import modules...` depends on it.

## Running the pipeline

**GISLR** (landmarks already extracted by Kaggle):

1. `src/gislr.0.dataset.motion-energy.ipynb` — *optional diagnostic*: per-video / per-category / global landmark motion-energy analysis. Executed end-to-end; findings in `docs/daily/2026-07-15.md`. (Caches live at `data/cache/gislr/motion_analysis/`; the pre-restructure caches were cleared, so a re-run recomputes them.)
2. `src/gislr.0.dataset.subset-comparison.ipynb` — *diagnostic*: landmark-subset discriminability comparison across three scopes, scoring the subsets registered in `modules/dataset/landmark/subsets.py`. Findings in `docs/daily/2026-07-16.md`.
3. `src/gislr.1.model.gru.ipynb` — the training stage: builds per-subset feature caches (`data/cache/gislr/features/`, skip-if-exists), trains one `StreamingGRU` run per subset in `TRAIN_SUBSETS` (regime **v2-plateau-300**: batch 512, lr 1e-3, ReduceLROnPlateau, epoch cap 300 with early stopping) via `modules.model.train_run` — one epoch-seconds registry folder each, `meta.json` updated every epoch, one progress bar per run. Per-class evaluation is handed off to `modules/scripts/eval_gru.py` (commands printed by the notebook); the ONNX → TF → TFLite export chain is a deferred, run-id-parameterized final section.
4. `src/gislr.1.model.{lstm,bilstm,cnn1d}.ipynb` — architecture benchmarks (TODO §4): identical drivers differing only in the architecture-identity block. `bilstm` is an **offline-only accuracy reference** (never a deployment candidate); `lstm` and `cnn1d` are streaming-viable. Runs pending.

**POPSIGN** (raw video, requires extraction first — in progress):

1. `src/popsign.0.dataset.extraction.ipynb` — the extraction driver: manifest verification (`data/cache/popsign/dataframes/` — needs regeneration after the restructure), a **pilot batch (≤100 videos)** writing throwaway npz to `data/temp/popsign_pilot/` (auto-cleaned), then the resumable bulk run to `data/raw/popsign/{train,test}` via `modules/dataset/landmark/extraction.py`. See `TODO.md` §2.
2. `src/popsign.2.model.ipynb` — label-distribution analysis, stratified split, earlier TensorFlow experiments.
3. `src/popsign.3.pipeline.ipynb` — end-to-end pipeline (stub).

(`src/popsign.1.mediapipe.ipynb` is stale — slated for retirement per `TODO.md` §0.1.)

## Constraints & known limitations

- **Streaming-viability drives architecture choice** — the deployment path is the unidirectional GRU; bidirectional models (BiLSTM) can only ever be offline accuracy benchmarks.
- **TensorFlow GPU is not supported on native Windows** — training uses PyTorch (CUDA); TFLite conversion happens post-hoc via ONNX → TensorFlow SavedModel → TFLite.
- **MediaPipe GPU delegate is Ubuntu-only** — landmark extraction runs on CPU on this Windows machine, parallelized across worker processes.
- Hardware: Windows 11, i7-14700K, RTX 4080 Super, 64GB DDR5 RAM.
