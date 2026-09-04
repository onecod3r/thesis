# sign2speech

A sign language recognition system focused on **streaming, real-time inference** rather than offline-only accuracy. The end goal is a deployable pipeline that classifies signs frame-by-frame with low latency, trained on hand/pose/face landmark sequences extracted via MediaPipe Holistic.

## Datasets

| Dataset | Role | Source (via `kagglehub`) | Status |
|---|---|---|---|
| **GISLR** | Fast-iteration dataset — landmarks are pre-extracted | Kaggle competition `asl-signs` | Ready to preprocess/train immediately |
| **POPSIGN** | Primary dataset (~870GB raw video) | `mrgeislinger/popsign-asl-v1-0-game-train-{a-e,f-m,n-s,t-z}-signs` + `...-game-test` | All 4 train parts + test now downloaded (2026-07-21); test-split landmarks extracted (33,599/33,600), train extraction pending manifest regeneration |

Raw data is **not stored in this repository**. It's downloaded on demand via `kagglehub` into its default cache (`~/.cache/kagglehub/`), resolved lazily by `src/modules/paths.py` (`gislr_dir()` / `resolve_datasets()` — importing the module never downloads anything). GISLR is a Kaggle *competition* download, so the Kaggle account in use must have accepted the competition rules and `kagglehub` must be authenticated.

POPSIGN's extracted landmarks (the large intermediate artifact, pre-feature-caching) are written to a separate drive configured via `.env` — see [Environment setup](#environment-setup).

## Models

- **GRU** (unidirectional `StreamingGRU`) — the deployment baseline. Chosen because it supports true causal/streaming inference.
- **Landmark-subset ablations** — motivated by the motion-energy analysis and the Kaggle 1st-place cross-check ([docs/logs/daily/2026-07-15.md](docs/logs/daily/2026-07-15.md)): the **ME-126** subset (hands + upper-body pose + lips + eyes/nose) beat the full-543 baseline **73.73% vs 70.59% val accuracy with half the parameters** (v1-regime runs, pre-reset — see registry note below), independently confirmed by the discriminability probe comparison ([docs/logs/daily/2026-07-16.md](docs/logs/daily/2026-07-16.md)). Canonical subset index lists: `src/modules/dataset/landmark/subsets.py`. Remaining ablations in `TODO.md` §3.1.
- **Architecture benchmarks** — `StreamingLSTM` (streaming-viable), `BiLSTM` (offline-only accuracy reference — prices the causality gap), `CausalConv1D` (dilated causal 1D-CNN, streaming-viable) — all trained from `src/gislr.1.models.training.ipynb`, a thin driver over the shared stack in `src/modules/model/` with hyperparameters in `src/config/gislr.training.json`. Still planned: ST-GCN, TCN, Conformer. See `TODO.md` §4.
- **1st-place solution port** (`Conv1DTransformer`, `TODO.md` §4.2) — the Kaggle GISLR winner recreated end to end: `FP_118` landmarks, reference-point normalization, lag-1/lag-2 motion features, six augmentations, 2 stages of (3× causal Conv1DBlock + Transformer), RAdam + Lookahead + AWP on a cosine one-cycle. Driven by `src/gislr.1.models.firstplace.ipynb` with its own config (`src/config/gislr.firstplace.json`), because its feature pipeline is incompatible with the shared training notebook's — but the same canonical split, registry and `meta.json` schema, so it stays comparable. **Offline-only** (`streaming: false`): global-average readout, unmasked self-attention and whole-sequence normalization. Its reported ~89% is a 4-seed ensemble trained on all 94,477 videos and scored on the Kaggle LB; a single run on the canonical 90/10 split is not the same measurement. **Both runs so far (`1787483814`, `1787492560`) diverged at epoch 15** — the step AWP and LateDropout switch on — so the recipe has not been measured yet; the best surviving checkpoint scores 0.7459. The driver now has collapse/plateau stopping conditions (the second run cost 9 min instead of 2 h), and notebook §5b is a 30-minute three-arm ablation that separates the two switches ([docs/logs/daily/2026-08-23.md](docs/logs/daily/2026-08-23.md)).

## Model registry

Every training run gets **one flat folder** at `src/data/models/<run_id>/`, where `run_id` is the **seconds since the Unix epoch** at training start. Dataset, architecture and subset are *fields in `meta.json`* (and columns of `index.csv`), not directory levels.

| file | content |
|---|---|
| `meta.json` | the single machine-readable run record (schema below) |
| `best.pt` | best-val-accuracy checkpoint (gitignored) |
| `last.pt` | latest checkpoint, saved every epoch — auto-resume state (gitignored) |
| `assets/` | every other artifact: learning curves, per-class CSVs/plots, landmark indices, eval summary — each linked from `meta.json["assets"]` |

All `meta.json` files are flattened into the queryable **[src/data/models/index.csv](src/data/models/index.csv)** by `src/modules/scripts/build_model_index.py` (runs from anywhere), which also answers filter queries directly — e.g. `--dataset gislr --architecture gru --top 3` or `--subset ME_126`. The training driver writes `meta.json` **every epoch** with `eval_status: "pending"`; `src/modules/scripts/evaluate.py` fills in the canonical eval numbers and flips it to `"canonical"`.

The index is derived, and it used to lag the run folders silently. It is now regenerated by **`src/modules/scripts/gen_docs.py`** along with everything else on this page that comes from code or the registry; `gen_docs.py --check` fails if any of it is stale.

<!-- generated:registry-summary -->
**42 runs** · 37 canonically evaluated · 0 scored on a held-out test set · 0 carrying provenance (schema v4).

Top 5 by canonical val accuracy (training-loop best where the canonical eval has not run):

| run | dataset | architecture | subset | coords | val acc | eval | params |
|---|---|---|---|---|---|---|---|
| `1784447175` | gislr | bilstm | ME_126 | xy | 0.7569 | canonical | 2,751,218 |
| `1784453891` | gislr | gru | ME_126 | xy | 0.7565 | canonical | 851,698 |
| `1784447187` | gislr | gru | ME_126 | xy | 0.7565 | canonical | 851,698 |
| `1787494351` | gislr | conv1d_transformer | FP_118 | xy | 0.7548 | pending | 1,830,040 |
| `1784397301` | gislr | bilstm | FP_118 | xy | 0.7525 | canonical | 2,718,418 |
<!-- /generated:registry-summary -->

### Checkpoints off this machine

`best.pt`/`last.pt` are gitignored, so every trained weight exists on one Windows machine — and the 2026-07-18 reset below already destroyed 8 runs' checkpoints. `src/modules/scripts/sync_models.py` keeps a copy in Cloudflare R2 (**42 checkpoints, ~707 MB** — inside the free tier):

```bash
.venv/Scripts/python.exe src/modules/scripts/sync_models.py status          # local vs manifest
.venv/Scripts/python.exe src/modules/scripts/sync_models.py push            # dry run
.venv/Scripts/python.exe src/modules/scripts/sync_models.py push --apply    # upload
.venv/Scripts/python.exe src/modules/scripts/sync_models.py pull 1784447175 # restore one run
.venv/Scripts/python.exe src/modules/scripts/sync_models.py pull --all      # restore everything
```

Needs `uv sync --group ops` (boto3, an optional group) and the `R2_*` keys in `.env` — see [Environment setup](#environment-setup). **Weights only**: not the ~30 GB of feature caches (derivable, and their content address makes that checkable) and not POPSIGN's ~870 GB of raw video (an immutable upstream release — a run records the reference, never the bytes).

[`src/data/models/checkpoints.manifest.json`](src/data/models/checkpoints.manifest.json) is committed and records each object's size, sha256 and upload time, so "is this run backed up, and is the copy still the file I trained?" is answerable with no credentials. `pull` verifies every download against that hash and refuses a mismatch rather than installing a checkpoint that is not the one that was trained.

> **Registry reset (2026-07-18).** The registry was restarted empty when the flat epoch-seconds layout was adopted. The 8 pre-reset runs (GRU full-543 baseline 70.59%, ME-126 73.73%, the xy ablations, …) survive only in git history (`3668dae` and earlier, under the old `src/models/` tree) and in the daily reports — their weights are gone, so their canonical evals cannot be completed; the numbers remain as historical references.

### meta.json schema

**The schema is defined in `modules/model/registry.py::FIELDS`** — that dict is what `write_meta` enforces, what [`schemas/meta.v4.json`](schemas/meta.v4.json) is generated from, and what the table below is rendered from by `modules/scripts/gen_docs.py`. It used to be defined here *and* in the code, which is two sources of truth and therefore one wrong one. All keys are required; unknown extra keys are not written.

<!-- generated:meta-schema -->
| key | type | content |
|---|---|---|
| `schema_version` | integer | `4` |
| `run_id` | integer | seconds since Unix epoch at training start = run folder name |
| `created` | string | ISO-8601 local timestamp derived from `run_id` |
| `dataset` | string | e.g. `"gislr"` — resolved through `modules/model/sources.py` |
| `architecture` | string | key into `modules.model.ARCHS`: `gru` / `lstm` / `bilstm` / `cnn1d` / `conv1d_transformer` |
| `model_name` | string | class name, e.g. `"StreamingGRU"` |
| `streaming` | boolean | streaming-viable? (`false` = offline-only reference, never deployable) |
| `subset` | string | landmark-subset name from `modules/dataset/landmark/subsets.py` |
| `coords` | string | `"xyz"` or `"xy"` (z-drop ablation) |
| `n_landmarks` | integer | landmarks fed to the model |
| `feature_dim` | integer | input width per frame |
| `n_classes` | integer | label-space size |
| `n_params` | integer | trainable parameters |
| `split` | object | `{strategy, random_state, n_val}` — the canonical split (`stratified 90/10`, seed 42, 9,448 val) |
| `training` | object | `{regime, source, epoch_cap, epochs_trained, best_epoch, early_stopped, finished, wall_time_min}`, plus `stop_reason` (`"completed"`/`"plateau"`/`"collapse"`/`"nan"`) on `fp-onecycle-300` runs. `source` is the DRIVER NOTEBOOK, not the dataset |
| `hyperparameters` | object | full `HYP` dict + `seed`, `max_seq_len`, `num_workers`, `loss`, `precision` |
| `provenance` | object \| null | what state of the world produced the run — see below. `null` for pre-v4 runs, and that is permanent |
| `metrics` | object | `{train_val_acc, eval_status ("pending"\|"canonical"), overall_accuracy, macro_accuracy, median_class_accuracy, n_classes_below_50pct}` — canonical fields are `null` until the eval script runs, and then survive training-loop rewrites |
| `checkpoints` | object | `{best: "best.pt", last: "last.pt"}` — run-dir-relative |
| `assets` | object | `{name: run-dir-relative path}` for every asset file, e.g. `{"landmarks": "assets/landmarks.npy", "history": "assets/history.json"}` |
| `submission` | object | `{tested, platform, submitted_at, public_score, private_score, reference, notes}` — see below |
| `notes` | string | free-text run notes |
<!-- /generated:meta-schema -->

#### The `submission` block (schema v3)

`tested` answers one question: **has this run been scored on the official/held-out test set?**
It is deliberately *dataset-agnostic* — for GISLR that means a Kaggle submission landed
(`platform: "kaggle"`), for a dataset with no leaderboard it means a local held-out evaluation
ran (`platform: "local"`). No Kaggle vocabulary appears in the required keys, so POPSIGN runs
reuse the block unchanged.

It exists because Kaggle allows **100 submissions/day** and every trained model should
eventually be scored, so "which models still need submitting" has to be a query rather than
something remembered by hand:

```sql
dataset = 'gislr' AND submission.tested = false   ORDER BY accuracy DESC   LIMIT 100
```

`modules/model/submission.py::untested_runs` is exactly that query; `registry.mark_tested()`
flips the flag after a successful submission, and `registry.write_meta` protects the block from
the training loop's per-epoch rewrites (the same protection canonical eval metrics get).
Pre-v3 records are backfilled with the default (never submitted) by `registry.migrate_all`,
which `build_model_index.py` runs automatically.

#### The `provenance` block (schema v4)

Hyperparameters say what a run was *configured* with; `provenance` says what actually
**ran** — so a number in the registry can be defended and, if need be, rebuilt. Written by
both training drivers via `modules/model/provenance.py`, flattened into `prov_*` columns of
`index.csv`.

| field | content |
|---|---|
| `git_commit` / `git_branch` | HEAD at the time the driver was invoked |
| `git_dirty` | anything in the working tree modified — **information, not an alarm** |
| `code_dirty` / `dirty_code_paths` | whether `src/modules/` + `src/config/` were dirty — *this* is the alarm |
| `config_path` / `config_sha256` | the config file, and a hash of the values that actually ran (a config edited in a cell hashes differently from the file on disk) |
| `feature_pipeline` | `base_v1` (`modules/model/data.py`, NaN→0 at cache build) or `firstplace_v1` (`modules/model/features.py`, NaN-preserving) |
| `feature_cache_key` | content hash of the feature cache the run read (TODO §9.2) |
| `source` | `{name, kaggle_ref, version, n_videos, manifest, manifest_sha256, resolved_dir}` — the dataset *reference*, never its bytes |
| `env` | `python`, `platform`, `torch`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `mediapipe`, `gpu`, `cuda` |

Two deliberate choices:

- **`git_dirty` is not what warns you.** Training here starts by editing and re-running a
  notebook, so the tree is dirty for essentially every run; a blanket warning would be
  ignored within a day. Only `code_dirty` — the code and config that execute — prints the
  loud "this run will not be reproducible" banner.
- **`scikit-learn` is in `env` on purpose**: `train_test_split` defines the canonical split,
  so a version change there moves the val set itself.
- **The 42 pre-v4 runs carry `provenance: null` forever.** Their commit and environment were
  never recorded and cannot be recovered; a reconstructed block would make them look
  reproducible when they are not.

## Reports & docs

See [docs/README.md](docs/README.md): day-by-day logs in `docs/logs/daily/<YYYY-MM-DD>.md`, weekly summaries in `docs/logs/weekly/<YYYY>-<WW>.md` (**weeks run Sunday → Saturday**, e.g. `2026-30.md` = 2026-07-19 → 07-25), standalone topic reports in `docs/reports/<topic>.md` — figures always under the matching `assets/` subfolder.

**Weekly summaries**

| week | dates | summary |
|---|---|---|
| 2026-29 | Jul 12 – Jul 18 | [docs/logs/weekly/2026-29.md](docs/logs/weekly/2026-29.md) — motion energy → discriminability (ME-126 wins on both, rho −0.12 between them) · xy beats xyz · registry v1 → the restructure (registry reset, pre-reset weights gone) · 4 architecture notebooks |
| 2026-30 | Jul 19 – Jul 25 | [docs/logs/weekly/2026-30.md](docs/logs/weekly/2026-30.md) — *in progress*: plateau diagnosed as **overfitting** · TFLite export working for all 4 archs (Keras rebuild) · training consolidated to one notebook + config · POPSIGN **test-split extraction finished** (33,599/33,600) and all 4 train dataset parts now downloaded (train manifest regeneration + bulk run still pending) |

**Standalone reports**

| report | contents |
|---|---|
| [docs/reports/motion-energy.md](docs/reports/motion-energy.md) | GISLR per-landmark motion analysis (three scopes, 94,477 videos): **~92% of pose "motion" is z-axis noise** · seeded 50-video samples reproduce the global ranking (rho 0.95+) · ME-126 keep/discard recommendation, cross-checked against the Kaggle 1st-place subset |
| [docs/reports/subset-comparison.md](docs/reports/subset-comparison.md) | Landmark-subset discriminability (F-ratio / MI / probe classifier, 3 scopes): **ME-126 wins** the 6-subset leaderboard (49.9% global probe) · discriminability ≈ uncorrelated with motion energy (rho −0.12) · probe difficulty profile tracks the trained GRU's (rho 0.640) |
| [docs/reports/confidence-tuning.md](docs/reports/confidence-tuning.md) | POPSIGN extraction-quality threshold sweep (**partial — 2 of 7 arms**): `min_hand_landmarks_confidence` is inert and the *pose* thresholds gate the hands · thresholds move hand detection by only ~0.02 · **the quality proxies are dominated by clip padding** — ~half of every clip is non-signing lead-in/lead-out, and detection is 0.85–0.94 within the signing span |

**Daily logs**

| date | report | contents |
|---|---|---|
| 2026-07-15 | [docs/logs/daily/2026-07-15.md](docs/logs/daily/2026-07-15.md) | GISLR landmark motion-energy analysis (z-noise finding, keep/discard recommendation) · 1st-place solution landmark cross-check · GRU training in detail: full-543 baseline vs ME-126 subset (+3.1 pts at half the parameters) |
| 2026-07-16 | [docs/logs/daily/2026-07-16.md](docs/logs/daily/2026-07-16.md) | Landmark-subset discriminability comparison (F-ratio / MI / probe classifier, 3 scopes): **ME-126 wins**; discriminability ≠ motion energy (rho −0.12) · POPSIGN extraction module + driver notebook (resource-capped, resumable) built & validated |
| 2026-07-17 | [docs/logs/daily/2026-07-17.md](docs/logs/daily/2026-07-17.md) | Registry tooling v1: per-run metadata + queryable index · fresh-run-folder policy · training regime **v2-plateau-300** · LSTM / BiLSTM / CausalConv1D benchmark notebooks built · eval script generalized (arch dispatch + xy mode) |
| 2026-07-18 | [docs/logs/daily/2026-07-18.md](docs/logs/daily/2026-07-18.md) | **Repo restructure**: unified `modules/model` training stack · flat epoch-seconds model registry + meta.json schema v2 (registry reset) · `data/{raw,cache,temp,models}` tree + temp-cleanup policy · docs daily/weekly/reports split · single-progress-bar training |
| 2026-07-19 | [docs/logs/daily/2026-07-19.md](docs/logs/daily/2026-07-19.md) | **Plateau diagnosed as overfitting** (train 90–99% vs val ~75%, gap 0.16–0.24) · most-confused pairs are semantic near-synonyms · GISLR **stage-2 evaluation/submission notebook** + arch-generic TFLite export + meta.json **schema v3** (`submission.tested`, submission queue as a query) · `docs/` split into logs/ vs reports/ · POPSIGN **confidence-tuning** first results ([report](docs/reports/confidence-tuning.md)) — thresholds barely matter, the quality proxies are measuring clip padding · POPSIGN extraction driver given a **CLI handoff** (`extract_popsign.py`) · **bulk test-split extraction running** (15,549/33,600, 1 failed, 1.46 videos/s) + the on-disk npz format recorded |
| 2026-08-23 | [docs/logs/daily/2026-08-23.md](docs/logs/daily/2026-08-23.md) | **The 1st-place port diverges at epoch 15** — the step AWP + LateDropout switch on: loss pinned at ln(250), stem weight norm 16→224, `stem_bn.running_var` 4.8→8467, 285 of 300 epochs wasted; the re-run with clipping + frozen BN stats died the same way but cost 9 min, so §5b now ablates the two switches apart · canonical **0.7459** from the surviving epoch-15 checkpoint (below the 0.7565 GRU, and not a measurement of the recipe) · its confused pairs are the **same semantic near-synonyms** the 18-run aggregate found · fixes: `grad_clip: 1.0`, BatchNorm stats frozen during AWP's adversarial pass, and collapse/plateau **stopping conditions** for `fp-onecycle-300` |

## Project structure

```
sign2speech/
├── README.md
├── TODO.md                   # living task list, organized by workstream
├── pyproject.toml            # uv-managed dependencies (Python >= 3.12, incl. torch cu130 via [tool.uv.sources])
├── uv.lock
├── .env                      # machine-specific config (POPSIGN_LANDMARKS_DRIVE) — not committed
├── docs/
│   ├── logs/                          # time-ordered: what happened when
│   │   ├── daily/<YYYY-MM-DD>.md      #   day-by-day logs (assets in logs/daily/assets/<date>/)
│   │   └── weekly/<YYYY>-<WW>.md      #   weekly summaries (weeks run Sunday → Saturday)
│   └── reports/<topic>.md             # standalone test/analysis reports (assets in reports/assets/<topic>/)
├── schemas/                  # GENERATED machine contracts — never hand-edit
│   └── meta.v4.json          # JSON Schema for a run record, rendered from modules/model/registry.py::FIELDS
├── scripts/                  # housekeeping / misc only (NO project Python scripts — those live in src/modules/scripts/)
│   └── sys_disk_usage.ps1    # disk-usage helper (POPSIGN raw video is ~870GB)
└── src/                      # all code; notebooks assume the kernel CWD is src/
    ├── config/
    │   ├── gislr.training.json   # SHARED TRAINING HYPERPARAMETERS (source of truth, not in any cell)
    │   └── gislr.firstplace.json # 1st-place port hyperparameters (separate regime, same principle)
    ├── gislr.0.dataset.motion-energy.ipynb      # diagnostic: landmark motion-over-time analysis (TODO §1)
    ├── gislr.0.dataset.subset-comparison.ipynb  # diagnostic: landmark-subset discriminability comparison (TODO §3)
    ├── gislr.1.models.training.ipynb            # ALL GISLR training: one section per architecture (TODO §4)
    ├── gislr.1.models.firstplace.ipynb          # 1st-place solution port: own feature pipeline + config (TODO §4.2)
    ├── gislr.2.models.evaluation.ipynb          # ALL GISLR evaluation + export + Kaggle submission (TODO §6)
    ├── popsign.0.dataset.extraction.ipynb       # POPSIGN landmark extraction driver (TODO §2)
    ├── popsign.0.dataset.confidence-tuning.ipynb # diagnostic: extraction-quality threshold sweep (TODO §2.3)
    ├── popsign.0.dataset.output-inspection.ipynb # diagnostic: what an extracted npz contains (read-only, safe during a live run)
    ├── popsign.1.mediapipe.ipynb                # stale — slated for retirement (TODO §0.1)
    ├── popsign.2.model.ipynb                    # label distribution, split, earlier TF experiments
    ├── popsign.3.pipeline.ipynb                 # end-to-end pipeline (stub)
    ├── modules/
    │   ├── paths.py                  # canonical tree constants (absolute, CWD-independent) + lazy dataset resolution + cleanup_temp()
    │   ├── model/                    # unified training stack behind gislr.1.models.training.ipynb
    │   │   ├── architectures.py      # StreamingGRU / StreamingLSTM / BiLSTM / CausalConv1D / Conv1DTransformer + ARCHS registry (single definition, shared with eval)
    │   │   ├── data.py               # canonical split, content-addressed feature caches, in-RAM dataset
│   │   ├── provenance.py         # what actually ran: commit, config hash, feature-cache key, dataset ref, env
│   │   ├── sources.py            # dataset seam: dir resolver + label map + canonical split + sample reader per dataset
    │   │   ├── features.py           # 1st-place input pipeline: NaN-preserving cache, normalization, lag features, augmentation
    │   │   ├── optim.py              # Lookahead, AWP, cosine one-cycle (what the 1st-place recipe needs and torch lacks)
    │   │   ├── train_fp.py           # 1st-place training driver (cosine, AWP, collapse/plateau stops) — same registry/split as train.py
    │   │   ├── registry.py           # run folders (epoch seconds), meta.json writing, asset registration
    │   │   ├── train.py              # training driver: auto-resume, early stopping, ONE progress bar per run
    │   │   ├── report.py             # learning curves, per-epoch history, confusion matrices
    │   │   ├── export.py             # arch-generic TFLite export → submission.zip (parity-gated)
    │   │   ├── keras_export.py       # PyTorch → native Keras rebuild + weight transfer (the export route)
    │   │   └── submission.py         # DuckDB submission queue (untested runs, daily cap) + the Kaggle call
    │   ├── scripts/                  # project Python CLIs (run from anywhere — they bootstrap their own imports)
    │   │   ├── evaluate.py           # canonical per-class eval of any run (all archs, xy/xyz); promotes meta.json to "canonical"
    │   │   ├── build_model_index.py  # flattens all meta.json files into data/models/index.csv + answers filter queries
│   │   ├── migrate_feature_caches.py # one-shot: flat name-keyed caches → content-addressed dirs (rename, never rebuild)
│   │   ├── sync_models.py        # off-machine copy of the run weights (R2): status / push / pull, manifest-verified
│   │   ├── gen_docs.py           # regenerates schemas/meta.v4.json + the README's generated blocks + index.csv (--check fails on drift)
    │   │   ├── extract_popsign.py    # POPSIGN extraction: pilot benchmark + resumable bulk run (worker pool — cannot run in a notebook)
    │   │   └── tune_confidence.py    # POPSIGN confidence-threshold sweep (same reason)
    │   └── dataset/landmark/
    │       ├── spec.py               # LANDMARK_TENSOR v1: the stage-1 → stage-2 contract (row layout, dtype, NaN policy, npz keys) + validators
    │       ├── subsets.py            # canonical landmark-subset registry (FULL_543, ME_126, FP_118, …)
    │       ├── extraction.py         # MediaPipe Holistic video→landmark extraction (multiprocess, resource-capped, resumable)
    │       ├── quality.py            # extraction-quality proxies + composite score (confidence tuning)
    │       └── overlay.py            # landmark-on-video rendering + contact sheets (the visual quality test)
    └── data/                         # gitignored EXCEPT the model registry (data/models/)
        ├── raw/                      # extracted-from-source data (POPSIGN landmark npz: raw/popsign/{train,test}/)
        ├── cache/                    # reusable derived artifacts, one subtree per dataset:
        │   ├── gislr/features/<pipeline>/<key>/   #   feature caches, CONTENT-ADDRESSED (GBs, shared across
        │   │                         #   architectures): <key> hashes pipeline+version, the subset's index
        │   │                         #   array, coords, NaN policy and the dataset manifest, so a redefined
        │   │                         #   subset can't silently reuse the old array. cache_key.json per dir
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
- **Shared logic lives in `src/modules/`, not in notebooks; shared *parameters* live in a config file, not in cells.** All GISLR training is one notebook (`gislr.1.models.training.ipynb`) with a section per architecture, and every hyperparameter comes from **[`src/config/gislr.training.json`](src/config/gislr.training.json)** via `modules/model/config.py`. Architectures inherit the `shared` block; a deviation must be declared as an explicit `overrides` entry, which the notebook prints — so "all else identical" is enforced rather than maintained by hand. Project Python CLIs live in **`src/modules/scripts/`**; the root `scripts/` folder holds housekeeping/misc scripts only.
- **One flat registry folder per training run** at `src/data/models/<epoch-seconds>/` holding `meta.json` + `best.pt`/`last.pt` + `assets/`. A run's artifacts are never split across parallel trees; `index.csv` is the queryable view.
- **Docs**: daily logs in `docs/logs/daily/`, weekly summaries in `docs/logs/weekly/` (`<YYYY>-<WW>.md`, weeks Sunday → Saturday), standalone topic reports in `docs/reports/` — see `docs/README.md`.
- **Data placement policy**: extracted-from-source data → `data/raw/<dataset>/…`; reusable derived artifacts → `data/cache/<dataset>/…`; throwaway output → `data/temp/`, **deleted after use** (`modules.paths.cleanup_temp()`); third-party assets → `data/external/`. Raw kagglehub downloads never enter the repo — `modules/paths.py` resolves them lazily at call time (`gislr_dir()`, `resolve_datasets()`), never at import time.
- **POPSIGN's extracted landmarks go to `data/raw/popsign/{train,test}`**, rooted at the drive configured via `POPSIGN_LANDMARKS_DRIVE` in `.env` when set (falling back to `src/data/raw/`, gitignored). The extraction module (`modules/dataset/landmark/extraction.py`) resolves this in one place.
- **Paths are CWD-independent**: `modules/paths.py` anchors every constant to the `src/` directory, so notebooks (kernel CWD = `src/`) and the `modules/scripts/` CLIs (any CWD) agree on the same tree.
- **Canonical evaluation** (all GISLR runs must match to be comparable): stratified 90/10 split, `random_state=42`, 9,448-video val set, per-class accuracy from raw parquet — exactly what `modules/scripts/evaluate.py` reproduces. A new run displaces a leaderboard entry only on this same split/metric.

## Environment setup

```bash
# Install dependencies (uv-managed, Python >= 3.12)
uv sync
```

Create a `.env` file at the project root (not committed) with:

```
POPSIGN_LANDMARKS_DRIVE=D:/    # or wherever the extraction-output drive is mounted

# Checkpoint remote (sync_models.py) — Cloudflare R2 speaks the S3 API
R2_ACCOUNT_ID=...
R2_BUCKET=sign2speech-models
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_PREFIX=models               # optional, defaults to "models"
```

`.env` is read by `modules/paths.py::env_value` (process environment first, then the file), so any of these can also come from the shell. Checkpoint sync additionally needs its optional dependency group:

```bash
uv sync --group ops            # boto3, for src/modules/scripts/sync_models.py
```

The Jupyter kernel must use this project's `uv`-managed virtual environment (`.venv`) and run with `src/` as its working directory — `import modules...` depends on it.

## Running the pipeline

**GISLR** (landmarks already extracted by Kaggle):

1. `src/gislr.0.dataset.motion-energy.ipynb` — *optional diagnostic*: per-video / per-category / global landmark motion-energy analysis. Executed end-to-end; findings in `docs/logs/daily/2026-07-15.md`. (Caches live at `data/cache/gislr/motion_analysis/`; the pre-restructure caches were cleared, so a re-run recomputes them.)
2. `src/gislr.0.dataset.subset-comparison.ipynb` — *diagnostic*: landmark-subset discriminability comparison across three scopes, scoring the subsets registered in `modules/dataset/landmark/subsets.py`. Findings in `docs/logs/daily/2026-07-16.md`.
3. `src/gislr.1.models.training.ipynb` — the training stage, **all four architectures in one notebook**: shared feature caches (`data/cache/gislr/features/<pipeline>/<key>/`, content-addressed so skip-if-exists is safe, built once and reused by every architecture), then one section per architecture (`StreamingGRU`, `StreamingLSTM`, `BiLSTM`, `CausalConv1D`) calling `modules.model.train_from_config`. Hyperparameters come from `src/config/gislr.training.json` (regime **v2-plateau-300**), not from cells. One epoch-seconds registry folder per (architecture, subset), `meta.json` + `assets/history.json` updated every epoch, one progress bar per run. `bilstm` is an **offline-only accuracy reference** (never a deployment candidate); the other three are streaming-viable. Per-class evaluation is handed off to `modules/scripts/evaluate.py`. **Training only** — no export section.
4. `src/gislr.1.models.firstplace.ipynb` — the **1st-place solution recreation** (TODO §4.2), a parallel stage-1 track rather than a step in the main sequence. Builds its own NaN-preserving feature cache, then trains `Conv1DTransformer` under regime `fp-onecycle-300`. Read its §0 before running: it explains why the model is offline-only, why a single run on the canonical split should not be expected to reproduce the ~89% leaderboard figure, and which cell to check for the time estimate. Hands off to the same `evaluate.py` canonical evaluation.
5. `src/gislr.2.models.evaluation.ipynb` — **everything after training** (TODO §6): DuckDB leaderboard over all `meta.json` files, canonical-eval backfill, top-5 learning-curve overlay, per-run and aggregate confusion matrices + most-confused pairs, arch-generic TFLite export, and the Kaggle submission queue (untested runs only, 100/day cap). All GISLR evaluation and submission lives here and nowhere else.

**POPSIGN** (raw video, requires extraction first — in progress):

1. `src/popsign.0.dataset.confidence-tuning.ipynb` — *diagnostic, run before bulk extraction* (TODO §2.3): sweeps `HolisticLandmarker` thresholds over a seeded 50-video sample (5 classes × 10), scores each config with quality proxies (detection rates, jitter, gaps, rigid-bone variance) **and** renders 100 landmark-overlay frames weighted toward the worst detections. Establishes which thresholds actually matter — measured: `min_hand_landmarks_confidence` is inert; the *pose* thresholds gate the hands.
2. `src/popsign.0.dataset.extraction.ipynb` — the extraction driver, and where extraction actually runs: manifest generation + verification (`data/cache/popsign/dataframes/{train,test}.csv` — regenerated from the raw video tree, 30,867 train / 33,600 test), a **pilot batch (≤100 videos)** writing throwaway npz to `data/temp/popsign_pilot/` (auto-cleaned), then the resumable bulk run to `data/raw/popsign/{train,test}`. The worker pool runs in the kernel; MediaPipe's C++ logs are redirected per worker to `<out_dir>/<split>/_worker_stderr.log` so they never reach cell output. `modules/scripts/extract_popsign.py` (`pilot` / `run <split>`) is an optional CLI for unattended runs — same module, same manifests, resumable either way. See `TODO.md` §2.
3. `src/popsign.0.dataset.output-inspection.ipynb` — *diagnostic, safe to run **during** an extraction*: opens one completed npz and shows the saved format — keys/shapes/dtypes, the 543-row holistic group layout, per-group detection rates, a frame plot, and the reference npz→model-input loader. Read-only by construction: no writes to the landmarks tree, no worker pool, no MediaPipe import, `.tmp.npz` staging files excluded so a half-written video is never opened.

   **Saved format** — the stage-1 → stage-2 contract. Its definition is **[`src/modules/dataset/landmark/spec.py`](src/modules/dataset/landmark/spec.py)** (`LANDMARK_TENSOR` v1), not this paragraph: `spec.GROUPS` is the single row layout that `extraction.py`, `quality.py` and `subsets.py` all derive from, and `validate_tensor` / `validate_npz` enforce it. One `np.savez_compressed` per video at `<root>/data/raw/popsign/<split>/<label>/<video_id>.npz`, the label carried by the path rather than stored inside: `landmarks` `(T, 543, 3)` float16 (NaN where undetected), `fps` float32, `num_frames` int32. Row order is GISLR holistic order (face 0–467, left hand 468–488, pose 489–521, right hand 522–542) — that is what makes the `subsets.py` indices apply to POPSIGN unchanged, so extraction validates it before every write. ~165 KB/video → **~5.4 GB** for the full test split.
4. `src/popsign.2.model.ipynb` — label-distribution analysis, stratified split, earlier TensorFlow experiments.
5. `src/popsign.3.pipeline.ipynb` — end-to-end pipeline (stub).

(`src/popsign.1.mediapipe.ipynb` is stale — slated for retirement per `TODO.md` §0.1.)

## Constraints & known limitations

- **Streaming-viability drives architecture choice** — the deployment path is the unidirectional GRU; bidirectional models (BiLSTM) can only ever be offline accuracy benchmarks.
- **TensorFlow GPU is not supported on native Windows** — training uses PyTorch (CUDA); TFLite conversion happens post-hoc by **rebuilding the trained model in native Keras and transferring the weights** (`modules/model/keras_export.py`), gated on numerical parity with the PyTorch model. The PyTorch → ONNX → `onnx2tf` route was tried and abandoned: onnx2tf failed on 3 of the 4 architectures (see `TODO.md` §6.2).
- **MediaPipe GPU delegate is Ubuntu-only** — landmark extraction runs on CPU on this Windows machine, parallelized across worker processes.
- **Runs are not yet reproducible from their records** — `meta.json` captures hyperparameters but no commit SHA, environment versions, dataset version or feature-cache identity, and the feature caches are keyed by subset *name* rather than by content. Both are being fixed under `TODO.md` §9 (schema v4 provenance block, content-addressed caches); until then a number in the registry is comparable to another only by convention, not by check.
- **Trained weights exist on one machine** (`best.pt`/`last.pt` are gitignored, 675 MB over 42 runs). The 2026-07-18 registry reset already destroyed 8 runs' weights — off-machine sync is `TODO.md` §9.3.
- Hardware: Windows 11, i7-14700K, RTX 4080 Super, 64GB DDR5 RAM.
