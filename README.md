# signbridge

> Renamed from `sign2speech` on 2026-09-04, alongside the workspace restructure. The old name described one direction; the repo is now laid out so the other one (speech → sign, `packages/sb-synthesize/`) can be built against the same landmark contract. The working directory and the git remote are unchanged.

A sign language recognition system focused on **streaming, real-time inference** rather than offline-only accuracy. The end goal is a deployable pipeline that classifies signs frame-by-frame with low latency, trained on hand/pose/face landmark sequences extracted via MediaPipe Holistic.

## Datasets

| Dataset | Role | Source (via `kagglehub`) | Status |
|---|---|---|---|
| **GISLR** | Fast-iteration dataset — landmarks are pre-extracted | Kaggle competition `asl-signs` | Ready to preprocess/train immediately |
| **POPSIGN** | Primary dataset (~870GB raw video) | `mrgeislinger/popsign-asl-v1-0-game-train-{a-e,f-m,n-s,t-z}-signs` + `...-game-test` | All 4 train parts + test now downloaded (2026-07-21); test-split landmarks extracted (33,599/33,600), train extraction pending manifest regeneration |

Raw data is **not stored in this repository**. It's downloaded on demand via `kagglehub` into its default cache (`~/.cache/kagglehub/`), resolved lazily by `sb.core.paths` (`gislr_dir()` / `resolve_datasets()` — importing the module never downloads anything). GISLR is a Kaggle *competition* download, so the Kaggle account in use must have accepted the competition rules and `kagglehub` must be authenticated.

POPSIGN's extracted landmarks (the large intermediate artifact, pre-feature-caching) are written to a separate drive configured via `.env` — see [Environment setup](#environment-setup).

## Models

- **GRU** (unidirectional `StreamingGRU`) — the deployment baseline. Chosen because it supports true causal/streaming inference.
- **Landmark-subset ablations** — motivated by the motion-energy analysis and the Kaggle 1st-place cross-check ([docs/logs/daily/2026-07-15.md](docs/logs/daily/2026-07-15.md)): the **ME-126** subset (hands + upper-body pose + lips + eyes/nose) beat the full-543 baseline **73.73% vs 70.59% val accuracy with half the parameters** (v1-regime runs, pre-reset — see registry note below), independently confirmed by the discriminability probe comparison ([docs/logs/daily/2026-07-16.md](docs/logs/daily/2026-07-16.md)). Canonical subset index lists: `packages/sb-core/src/sb/core/subsets.py`. Remaining ablations in `TODO.md` §3.1.
- **Architecture benchmarks** — `StreamingLSTM` (streaming-viable), `BiLSTM` (offline-only accuracy reference — prices the causality gap), `CausalConv1D` (dilated causal 1D-CNN, streaming-viable) — all trained from `experiments/recognition/gislr.1.models.training.ipynb`, a thin driver over the shared stack in `packages/sb-recognize/` with hyperparameters in `experiments/recognition/configs/gislr.training.json`. Still planned: ST-GCN, TCN, Conformer. See `TODO.md` §4.
- **1st-place solution port** (`Conv1DTransformer`, `TODO.md` §4.2) — the Kaggle GISLR winner recreated end to end: `FP_118` landmarks, reference-point normalization, lag-1/lag-2 motion features, six augmentations, 2 stages of (3× causal Conv1DBlock + Transformer), RAdam + Lookahead + AWP on a cosine one-cycle. Driven by `experiments/recognition/gislr.1.models.firstplace.ipynb` with its own config (`experiments/recognition/configs/gislr.firstplace.json`), because its feature pipeline is incompatible with the shared training notebook's — but the same canonical split, registry and `meta.json` schema, so it stays comparable. **Offline-only** (`streaming: false`): global-average readout, unmasked self-attention and whole-sequence normalization. Its reported ~89% is a 4-seed ensemble trained on all 94,477 videos and scored on the Kaggle LB; a single run on the canonical 90/10 split is not the same measurement. **Both runs so far (`1787483814`, `1787492560`) diverged at epoch 15** — the step AWP and LateDropout switch on — so the recipe has not been measured yet; the best surviving checkpoint scores 0.7459. The driver now has collapse/plateau stopping conditions (the second run cost 9 min instead of 2 h), and notebook §5b is a 30-minute three-arm ablation that separates the two switches ([docs/logs/daily/2026-08-23.md](docs/logs/daily/2026-08-23.md)).

## Model registry

Every training run gets **one flat folder** at `registry/runs/<run_id>/`, where `run_id` is the **seconds since the Unix epoch** at training start. Dataset, architecture and subset are *fields in `meta.json`* (and columns of `index.csv`), not directory levels.

| file | content |
|---|---|
| `meta.json` | the single machine-readable run record (schema below) |
| `best.pt` | best-val-accuracy checkpoint (gitignored) |
| `last.pt` | latest checkpoint, saved every epoch — auto-resume state (gitignored) |
| `assets/` | every other artifact: learning curves, per-class CSVs/plots, landmark indices, eval summary — each linked from `meta.json["assets"]` |

All `meta.json` files are flattened into the queryable **[registry/index.csv](registry/index.csv)** by `.venv/Scripts/sb-index.exe` (runs from anywhere), which also answers filter queries directly — e.g. `--dataset gislr --architecture gru --top 3` or `--subset ME_126`. The training driver writes `meta.json` **every epoch** with `eval_status: "pending"`; `.venv/Scripts/sb-evaluate.exe` fills in the canonical eval numbers and flips it to `"canonical"`.

The index is derived, and it used to lag the run folders silently. It is now regenerated by **`.venv/Scripts/sb-docs.exe`** along with everything else on this page that comes from code or the registry; `gen_docs.py --check` fails if any of it is stale.

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

### Checkpoints live on Kaggle

**A run folder normally contains no weights.** `best.pt` is uploaded to a Kaggle Model and then deleted locally — this is what keeps 700 MB+ of `.pt` off a single disk after the 2026-07-18 reset destroyed 8 runs' weights.

**Nothing needs fetching by hand.** `sb-evaluate` and the TFLite export call `sb.mlops.artifacts.ensure_local`, which downloads the run's checkpoint through `kagglehub` and verifies its sha256 against the manifest before using it — so testing a run is one command whether or not the weights happen to be on disk. `--no-fetch` reports where the file is instead of downloading. Learning curves never need a checkpoint at all: they read the committed `assets/history.json`.

The backend is chosen by `SB_ARTIFACT_BACKEND` in `.env`:

| backend | what it is | setup |
|---|---|---|
| **`kaggle`** (default, in use) | a **Kaggle Model**: `bracu23101281/signbridge-gislr/pyTorch/<architecture>/<version>`, with subset/coords/score in the version note. No new account — this repo already authenticates to Kaggle for the GISLR data | nothing; `KAGGLE_ARTIFACT_{OWNER,MODEL,FRAMEWORK}` override the defaults |
| `local` | any filesystem path: external drive, NAS share, or a OneDrive/Drive/Dropbox-synced folder. Zero dependencies | `SB_ARTIFACT_DIR=D:/backup/signbridge` |
| `s3` | any S3-compatible endpoint — Backblaze B2, Wasabi, MinIO, Storj (or Cloudflare R2, if it is ever enabled) | `S3_ENDPOINT_URL` + `S3_BUCKET` + keys, and `uv sync --group ops` |

A second folder on the same physical disk is not a backup — point `local` at something that survives this machine.

#### Naming

```
bracu23101281/signbridge-gislr/pyTorch/gru-me126-xy/2
└─ owner ──┘ └─── model ────┘ └─fw─┘ └ variation ┘ └ver┘
```

Derived from each run's `meta.json`, never typed by hand:

| segment | rule |
|---|---|
| **model** | the family. GISLR recognizers are `signbridge-gislr`; a POPSIGN model becomes `signbridge-popsign` rather than a variation, because a different label space is a different model |
| **framework** | `pyTorch` — these are `.pt` state dicts. A TFLite export (§6.2) goes under the *same* model as `tfLite`, which is what the segment is for |
| **variation** | the **architecture** alone — `gru`, `bilstm`, `conv1d-transformer` |
| **version** | any run of that architecture, in chronological run-id order |

Everything the slug does not say — subset, coords, score and whether it is canonical, params, landmark count, feature_dim, regime, epochs, an UNFINISHED marker — goes in the **version note**, with the full record in the `meta.json` uploaded beside the weights.

**Versions are chronological, not ranked**, and since a variation now spans subsets they are **not all-else-equal** either: a version list cannot be read as a learning curve. Restores are unaffected — the manifest pins an exact `<variation>/<version>` handle per run — but a human comparing two versions has to read the notes. "Which run should be deployed" remains `sb-promote`'s question.

`sb-sync rescheme` migrates existing uploads after a naming change, since Kaggle has no rename: it pulls each affected run back, re-uploads under the new handle, and re-prunes. It never deletes remotely, so superseded variations must be removed from the model page by hand.

**Why Models and not Datasets.** A Kaggle *Dataset* versions as one directory, so every push would re-upload all 707 MB and every restore would download the lot. A *Model*'s variations version independently, so a push sends **only the runs that are new** and a restore fetches **one file** (`model_download(handle, path="best.pt")`). And a Kaggle inference kernel can attach a model directly, so the backup and the artifact a submission run loads (§6.3) are the same object.

#### Deleting local copies

Never `rm` a checkpoint. `sb-sync prune` downloads the remote copy and deletes the local file only when **three hashes agree** — the manifest's, the remote's, and the local file's. The third check matters as much as the second: a local file that has diverged from what was uploaded is something the remote does *not* have.

`sb-sync drop-resume` removes `last.pt` for **finished** runs only — the registry never resumes a finished run, so that file is dead weight — and leaves an unfinished run's resume state alone.

```bash
sb-sync status                 # local vs manifest
sb-sync push                   # dry run
sb-sync push --apply           # upload (--limit N to trial a few first)
sb-sync pull 1784447175        # restore one run
sb-sync pull --all             # restore everything
sb-sync prune --apply          # verify against the remote, then delete local copies
sb-sync drop-resume --apply    # drop last.pt for finished runs
```

Only the `s3` backend needs `uv sync --group ops` (boto3); `kaggle` and `local` work out of the box — see [Environment setup](#environment-setup). **Weights only**: not the ~30 GB of feature caches (derivable, and their content address makes that checkable) and not POPSIGN's ~870 GB of raw video (an immutable upstream release — a run records the reference, never the bytes).

[`registry/checkpoints.manifest.json`](registry/checkpoints.manifest.json) is committed and records each object's size, sha256, backend and upload time, so "is this run backed up, and is the copy still the file I trained?" is answerable with no credentials. `pull` verifies every download against that hash and refuses a mismatch rather than installing a checkpoint that is not the one that was trained.

> **Registry reset (2026-07-18).** The registry was restarted empty when the flat epoch-seconds layout was adopted. The 8 pre-reset runs (GRU full-543 baseline 70.59%, ME-126 73.73%, the xy ablations, …) survive only in git history (`3668dae` and earlier, under the old `src/models/` tree) and in the daily reports — their weights are gone, so their canonical evals cannot be completed; the numbers remain as historical references.

### meta.json schema

**The schema is defined in `sb.mlops.registry::FIELDS`** — that dict is what `write_meta` enforces, what [`schemas/meta.v4.json`](schemas/meta.v4.json) is generated from, and what the table below is rendered from by `sb.mlops.docs`. It used to be defined here *and* in the code, which is two sources of truth and therefore one wrong one. All keys are required; unknown extra keys are not written.

<!-- generated:meta-schema -->
| key | type | content |
|---|---|---|
| `schema_version` | integer | `4` |
| `run_id` | integer | seconds since Unix epoch at training start = run folder name |
| `created` | string | ISO-8601 local timestamp derived from `run_id` |
| `dataset` | string | e.g. `"gislr"` — resolved through `modules/model/sources.py` |
| `architecture` | string | key into `sb.recognize.ARCHS`: `gru` / `lstm` / `bilstm` / `cnn1d` / `conv1d_transformer` |
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

`sb.mlops.submission::untested_runs` is exactly that query; `registry.mark_tested()`
flips the flag after a successful submission, and `registry.write_meta` protects the block from
the training loop's per-epoch rewrites (the same protection canonical eval metrics get).
Pre-v3 records are backfilled with the default (never submitted) by `registry.migrate_all`,
which `build_model_index.py` runs automatically.

#### The `provenance` block (schema v4)

Hyperparameters say what a run was *configured* with; `provenance` says what actually
**ran** — so a number in the registry can be defended and, if need be, rebuilt. Written by
both training drivers via `sb.mlops.run`, flattened into `prov_*` columns of
`index.csv`.

| field | content |
|---|---|
| `git_commit` / `git_branch` | HEAD at the time the driver was invoked |
| `git_dirty` | anything in the working tree modified — **information, not an alarm** |
| `code_dirty` / `dirty_code_paths` | whether `packages/` + the `experiments/*/configs/` were dirty — *this* is the alarm |
| `config_path` / `config_sha256` | the config file, and a hash of the values that actually ran (a config edited in a cell hashes differently from the file on disk) |
| `feature_pipeline` | `base_v1` (`sb.recognize.data`, NaN→0 at cache build) or `firstplace_v1` (`sb.recognize.features.firstplace_v1`, NaN-preserving) |
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
| [docs/reports/plateau-diagnosis.md](docs/reports/plateau-diagnosis.md) | **Why accuracy stops at ~75%** (31 canonical runs): semantically similar signs are **not** the cause — the top-20 confusable pairs absorb only **10.5% of errors**, and solving them perfectly gives **0.7433 → 0.7704** against +0.007 for a random-pair control · the plateau is a **generalization gap**: symmetric confusion is **0.012 on train vs 0.273 on val**, 19 of 20 pairs at exactly zero on training data · pooled probes sit near chance on the worst pairs (`awake`/`wake` 0.463) and velocity does not rescue them · levers are **§7.2 normalization / §7.4 augmentation**; a rescoring layer is bounded at ~2.7 points |
| [docs/reports/extractor-parity.md](docs/reports/extractor-parity.md) | **The Deno/TypeScript extractor cannot run** (12-clip test run, 0/12): `@mediapipe/tasks-vision` creates a **WebGL** context during graph construction regardless of `delegate: "CPU"`, and Deno has `ImageData`/`OffscreenCanvas`/WebGPU but **no WebGL** · Node is no better placed — the requirement is the *web* MediaPipe build, not the runtime · found two defects on the way that would have made any parity number meaningless (fixed `scale=640:480 -r 30` vs POPSIGN's 1944×2592 portrait at 30/120 fps; two different model assets) · POPSIGN extraction is **not** blocked — `sb-extract` is native C++ with no GL requirement |
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
| 2026-09-04 | [docs/logs/daily/2026-09-04.md](docs/logs/daily/2026-09-04.md) | **Reproducibility §9.1–§9.7**: meta.json **schema v4** with a provenance block (42 runs backfilled) · **content-addressed** feature caches (30.3 GB moved by rename, not rebuild) · `LANDMARK_TENSOR` v1 spec + validators · the `DatasetSource` seam (no `gislr_dir` left in the training stack) · schema/index/README tables generated from `registry.FIELDS` · `eval_gru.py` → `evaluate.py` · **§9.8 the workspace restructure**: six `packages/sb-*`, `experiments/`, `registry/` and `data/` at the top level, renamed **signbridge** · checkpoint sync given **kaggle/local/s3 backends** after R2 turned out to be unavailable · TODO audit (11 done-but-unticked, 4 obsolete) |
| 2026-09-05 | [docs/logs/daily/2026-09-05.md](docs/logs/daily/2026-09-05.md) | **The plateau is a generalization gap, not a label ceiling** — the diagnosis's GPU arms: symmetric confusion **0.012 on train vs 0.273 on val**, 19 of 20 confusable pairs at exactly zero on training data · per-pair probes near chance (`awake`/`wake` 0.463) and pooled velocity does not rescue them (§8b), which bounds *pooled summaries* rather than the landmarks · §8 rescoring bounded at ~2.7 pts; §7.2/§7.4 are the evidence-backed levers |

## Project structure

A **uv workspace**: six installable packages under `packages/`, notebooks that only drive them
under `experiments/`, and the two data trees — one committed, one never — at the top level.

```
signbridge/
├── pyproject.toml            # workspace root (virtual: owns no code), uv members + the cu130 torch index
├── uv.lock
├── .env                      # machine-specific config (POPSIGN drive, artifact backend) — not committed
├── packages/                 # ALL library code. PEP 420 namespace: no `sb/__init__.py` anywhere,
│   │                         # so each distribution ships part of the same `sb` namespace
│   ├── sb-core/              # THE SEAM — imports no torch/mediapipe/tensorflow, so anything may depend on it
│   │   └── src/sb/core/
│   │       ├── schema.py     #   LANDMARK_TENSOR v1: row layout, dtype, NaN policy, npz keys, validators
│   │       ├── subsets.py    #   landmark-subset registry (FULL_543, ME_126, FP_118, …)
│   │       ├── vocab.py      #   sign name ↔ class index
│   │       ├── io.py         #   atomic landmark-npz read/write, schema-checked both ways
│   │       └── paths.py      #   the repo tree (found by walking up) + lazy dataset resolution
│   ├── sb-extract-ts/        # STAGE 1 in Deno/TypeScript — MediaPipe WASM. DOES NOT RUN: the web
│   │                         # build needs WebGL (any delegate) and Deno has none. See
│   │                         # docs/reports/extractor-parity.md. Use sb-extract for extraction.
│   ├── sb-extract/           # STAGE 1 — video → landmarks (Python; produced the 33,599 test clips)
│   │   └── src/sb/extract/
│   │       ├── holistic.py   #   MediaPipe worker pool, manifest-resumable, resource-capped
│   │       ├── sources/      #   per-dataset adapters (popsign.py) — adapters, not branches
│   │       ├── quality.py    #   extraction-quality proxies + composite score
│   │       ├── overlay.py    #   landmark-on-video rendering (the visual quality test)
│   │       ├── cli.py        #   `sb-extract`: pilot benchmark + resumable bulk run
│   │       ├── popsign_cycle.py # download one part → extract → VERIFY → delete (~870 GB won't fit)
│   │       ├── parity.py     #   do the Python and TypeScript extractors agree? gate the switch on this
│   │       ├── parity_run.py #   `python -m sb.extract.parity_run` — runs both extractors over the
│   │       │                 #   same seeded clips and hands the two trees to parity.py
│   │       └── tune.py       #   detector-threshold sweep
│   ├── sb-recognize/         # STAGE 2 — landmarks → gloss
│   │   └── src/sb/recognize/
│   │       ├── architectures.py  # the SINGLE definition of every model class + ARCHS
│   │       ├── data.py          # the canonical split and the constants every run agrees on
│   │       ├── features/        # named, versioned pipelines over one content-addressed cache:
│   │       │                    #   cache.py (addressing) · base_v1.py · firstplace_v1.py
│   │       ├── sources.py       # the dataset seam: dir, label map, split, sample reader
│   │       ├── config.py        # the shared training config loader
│   │       ├── train.py         # training driver: auto-resume, early stopping, ONE progress bar
│   │       ├── train_firstplace.py # the 1st-place recipe (cosine one-cycle, AWP, collapse stops)
│   │       ├── optim.py         # Lookahead, AWP, cosine one-cycle (what torch lacks)
│   │       ├── evaluate.py      # `sb-evaluate`: the canonical per-class eval, every architecture
│   │       ├── report.py        # learning curves, confusion matrices
│   │       ├── export/          # keras.py (rebuild + weight transfer) → tflite.py (parity-gated)
│   │       └── migrate_caches.py # one-shot: flat name-keyed caches → content-addressed dirs
│   ├── sb-mlops/             # run records and what happens to a model after training.
│   │   └── src/sb/mlops/     # Depends on sb-core ONLY — the registry must be readable without torch
│   │       ├── registry.py   #   run folders, meta.json schema (FIELDS) + writing, asset registration
│   │       ├── run.py        #   provenance: commit, config hash, feature-cache key, dataset ref, env
│   │       ├── index.py      #   `sb-index`: every meta.json → registry/index.csv + queries
│   │       ├── submission.py #   the DuckDB submission queue (untested runs, daily cap)
│   │       ├── artifacts.py  #   `sb-sync`: second copy of the weights (kaggle/local/s3) + the manifest
│   │       ├── promote.py    #   `sb-promote`: aliases — which run is champion
│   │       └── docs.py       #   `sb-docs`: regenerate schemas/, index.csv, README generated blocks
│   ├── sb-rescore/           # top-k → sentence (TODO §8 — scoped skeleton, see its __init__)
│   │   └── src/sb/rescore/   #   prompts/v1/ (versioned, hashed into the run record) · evalset/ (frozen)
│   └── sb-synthesize/        # speech → sign (future) — emits the SAME tensor sb.core.schema defines
├── experiments/              # notebooks are thin drivers; CWD does not matter any more
│   ├── extraction/           #   POPSIGN extraction + its diagnostics
│   ├── recognition/          #   GISLR analysis, training, evaluation
│   │   └── configs/          #   gislr.training.json · gislr.firstplace.json (hyperparameters, never in a cell)
│   └── synthesis/            #   (empty)
├── registry/                 # COMMITTED run records — top level, NOT inside the ignored data tree
│   ├── runs/<run_id>/        #   meta.json + best.pt/last.pt (gitignored) + assets/
│   ├── index.csv             #   generated: the queryable table of every run
│   ├── aliases.json          #   generated: champion/candidate → run id
│   └── checkpoints.manifest.json  # what is backed up, with sha256s (written by sb-sync)
├── data/                     # GITIGNORED ABSOLUTELY — no negation rules, nothing committed
│   ├── raw/                  #   extracted-from-source (POPSIGN landmark npz)
│   ├── cache/<dataset>/      #   derived artifacts; features/<pipeline>/<key>/ is content-addressed
│   ├── temp/                 #   throwaway scratch — deleted after use (cleanup_temp())
│   └── external/             #   third-party assets (MediaPipe .task model)
├── apps/                     # deployment surfaces (scaffolded)
│   ├── web/ · edge/ · shared-ts/
├── schemas/                  # GENERATED machine contracts — never hand-edit
│   └── meta.v4.json          #   JSON Schema for a run record, rendered from sb.mlops.registry::FIELDS
├── ops/                      # housekeeping (PowerShell etc.), no project Python
│   ├── envs.ps1              #   per-stage venvs: uv sync --package <member> into .venvs/<stage>
│   └── sys_disk_usage.ps1    #   disk-usage helper (POPSIGN raw video is ~870GB)
└── docs/
    ├── logs/{daily,weekly}/  # time-ordered: what happened when
    └── reports/<topic>.md    # standalone test/analysis findings
```

### Conventions

- **Notebooks live in `experiments/<domain>/`, named `<dataset>.<stage>.<topic>.ipynb`** — domain (extraction / recognition / synthesis), then dataset, a stage number ordering the pipeline, and what the stage does. Their configs sit beside them in `experiments/<domain>/configs/`.
- **Library code lives in `packages/sb-*/`, not in notebooks; shared *parameters* live in a config file, not in cells.** All GISLR training is one notebook (`gislr.1.models.training.ipynb`) with a section per architecture, and every hyperparameter comes from **[`experiments/recognition/configs/gislr.training.json`](experiments/recognition/configs/gislr.training.json)** via `sb.recognize.config`. Architectures inherit the `shared` block; a deviation must be declared as an explicit `overrides` entry, which the notebook prints — so "all else identical" is enforced rather than maintained by hand. Project Python CLIs live in **`packages/scripts/`**; the root `scripts/` folder holds housekeeping/misc scripts only.
- **One flat registry folder per training run** at `registry/runs/<epoch-seconds>/` holding `meta.json` + `best.pt`/`last.pt` + `assets/`. A run's artifacts are never split across parallel trees; `index.csv` is the queryable view.
- **Docs**: daily logs in `docs/logs/daily/`, weekly summaries in `docs/logs/weekly/` (`<YYYY>-<WW>.md`, weeks Sunday → Saturday), standalone topic reports in `docs/reports/` — see `docs/README.md`.
- **Data placement policy**: extracted-from-source data → `data/raw/<dataset>/…`; reusable derived artifacts → `data/cache/<dataset>/…`; throwaway output → `data/temp/`, **deleted after use** (`modules.paths.cleanup_temp()`); third-party assets → `data/external/`. Raw kagglehub downloads never enter the repo — `sb.core.paths` resolves them lazily at call time (`gislr_dir()`, `resolve_datasets()`), never at import time.
- **POPSIGN's extracted landmarks go to `data/raw/popsign/{train,test}`**, rooted at the drive configured via `POPSIGN_LANDMARKS_DRIVE` in `.env` when set (falling back to `data/raw/`, gitignored). The extraction module (`sb.extract.holistic`) resolves this in one place.
- **Paths are CWD-independent, and so are imports.** The packages are installed as editable workspace members, so `import sb...` works from anywhere; `sb.core.paths` finds the repo root by walking up for the workspace marker, so a notebook and an installed console script resolve the same tree. `SIGNBRIDGE_ROOT` overrides it.
- **Canonical evaluation** (all GISLR runs must match to be comparable): stratified 90/10 split, `random_state=42`, 9,448-video val set, per-class accuracy from raw parquet — exactly what `sb.recognize.evaluate` reproduces. A new run displaces a leaderboard entry only on this same split/metric.

## Environment setup

```bash
# Install the workspace: every package under packages/ is installed into .venv
# as an editable member, so `import sb...` works from any directory.
uv sync
```

**Per-stage environments (optional).** The default `.venv` holds every workspace member — torch *and* tensorflow *and* mediapipe *and* opencv — which is convenient and is also where dependency conflicts come from. `./ops/envs.ps1 -Stage train|mlops|extract` builds `.venvs/<stage>` from a single member's dependency closure:

```powershell
./ops/envs.ps1 -Stage mlops
.venvs/mlops/Scripts/python.exe -c "import sb.mlops.registry; import torch"  # must fail on torch
```

Measured: an `sb-mlops`-only environment drops torch, tensorflow, mediapipe and opencv — so the "`sb-mlops` must not import `sb-recognize`" invariant is executable rather than merely documented. Notebooks keep using the default `.venv`.

The six console scripts it puts on `.venv/Scripts/`:

| command | what it does |
|---|---|
| `sb-extract` | POPSIGN landmark extraction: pilot benchmark + resumable bulk run |
| `sb-evaluate <run_dir>` | the canonical per-class evaluation of one run |
| `sb-index` | rebuild `registry/index.csv` + answer filter queries |
| `sb-sync` | off-machine checkpoint copy (status / push / pull) |
| `sb-promote` | aliases: which run is champion |
| `sb-docs` | regenerate `schemas/`, the index and the README's generated blocks (`--check` fails on drift) |

Create a `.env` file at the project root (not committed) with:

```
POPSIGN_LANDMARKS_DRIVE=D:/    # or wherever the extraction-output drive is mounted

# Checkpoint backup (sb-sync): kaggle | local | s3
SB_ARTIFACT_BACKEND=kaggle
# KAGGLE_ARTIFACT_OWNER=<user>                            # kaggle, all optional
# KAGGLE_ARTIFACT_MODEL=signbridge-gislr
# KAGGLE_ARTIFACT_FRAMEWORK=pyTorch                       # tfLite for an export
# SB_ARTIFACT_DIR=D:/backup/signbridge                    # local
# S3_ENDPOINT_URL=... S3_BUCKET=... S3_ACCESS_KEY_ID=... S3_SECRET_ACCESS_KEY=...
```

`.env` is read by `sb.core.paths::env_value` (process environment first, then the file), so any of these can also come from the shell. Checkpoint sync additionally needs its optional dependency group:

```bash
uv sync --group ops            # boto3 — only needed for the `s3` backend
```

The Jupyter kernel must use this project's `uv`-managed virtual environment (`.venv`). **Its working directory no longer matters** — the six packages are installed into that venv, so `import sb...` and every path constant resolve identically wherever the kernel starts.

## Running the pipeline

**GISLR** (landmarks already extracted by Kaggle):

1. `experiments/recognition/gislr.0.dataset.motion-energy.ipynb` — *optional diagnostic*: per-video / per-category / global landmark motion-energy analysis. Executed end-to-end; findings in `docs/logs/daily/2026-07-15.md`. (Caches live at `data/cache/gislr/motion_analysis/`; the pre-restructure caches were cleared, so a re-run recomputes them.)
2. `experiments/recognition/gislr.0.dataset.subset-comparison.ipynb` — *diagnostic*: landmark-subset discriminability comparison across three scopes, scoring the subsets registered in `sb.core.subsets`. Findings in `docs/logs/daily/2026-07-16.md`.
3. `experiments/recognition/gislr.1.models.training.ipynb` — the training stage, **all four architectures in one notebook**: shared feature caches (`data/cache/gislr/features/<pipeline>/<key>/`, content-addressed so skip-if-exists is safe, built once and reused by every architecture), then one section per architecture (`StreamingGRU`, `StreamingLSTM`, `BiLSTM`, `CausalConv1D`) calling `modules.model.train_from_config`. Hyperparameters come from `experiments/recognition/configs/gislr.training.json` (regime **v2-plateau-300**), not from cells. One epoch-seconds registry folder per (architecture, subset), `meta.json` + `assets/history.json` updated every epoch, one progress bar per run. `bilstm` is an **offline-only accuracy reference** (never a deployment candidate); the other three are streaming-viable. Per-class evaluation is handed off to `sb.recognize.evaluate`. **Training only** — no export section.
4. `experiments/recognition/gislr.1.models.firstplace.ipynb` — the **1st-place solution recreation** (TODO §4.2), a parallel stage-1 track rather than a step in the main sequence. Builds its own NaN-preserving feature cache, then trains `Conv1DTransformer` under regime `fp-onecycle-300`. Read its §0 before running: it explains why the model is offline-only, why a single run on the canonical split should not be expected to reproduce the ~89% leaderboard figure, and which cell to check for the time estimate. Hands off to the same `evaluate.py` canonical evaluation.
5. `experiments/recognition/gislr.2.models.evaluation.ipynb` — **everything after training** (TODO §6): DuckDB leaderboard over all `meta.json` files, canonical-eval backfill, top-5 learning-curve overlay, per-run and aggregate confusion matrices + most-confused pairs, arch-generic TFLite export, and the Kaggle submission queue (untested runs only, 100/day cap). All GISLR evaluation and submission lives here and nowhere else.

**POPSIGN** (raw video, requires extraction first — in progress):

1. `experiments/extraction/popsign.0.dataset.confidence-tuning.ipynb` — *diagnostic, run before bulk extraction* (TODO §2.3): sweeps `HolisticLandmarker` thresholds over a seeded 50-video sample (5 classes × 10), scores each config with quality proxies (detection rates, jitter, gaps, rigid-bone variance) **and** renders 100 landmark-overlay frames weighted toward the worst detections. Establishes which thresholds actually matter — measured: `min_hand_landmarks_confidence` is inert; the *pose* thresholds gate the hands.
2. `experiments/extraction/popsign.0.dataset.extraction.ipynb` — the extraction driver, and where extraction actually runs: manifest generation + verification (`data/cache/popsign/dataframes/{train,test}.csv` — regenerated from the raw video tree, 30,867 train / 33,600 test), a **pilot batch (≤100 videos)** writing throwaway npz to `data/temp/popsign_pilot/` (auto-cleaned), then the resumable bulk run to `data/raw/popsign/{train,test}`. The worker pool runs in the kernel; MediaPipe's C++ logs are redirected per worker to `<out_dir>/<split>/_worker_stderr.log` so they never reach cell output. `modules/scripts/extract_popsign.py` (`pilot` / `run <split>`) is an optional CLI for unattended runs — same module, same manifests, resumable either way. See `TODO.md` §2.
3. `experiments/extraction/popsign.0.dataset.output-inspection.ipynb` — *diagnostic, safe to run **during** an extraction*: opens one completed npz and shows the saved format — keys/shapes/dtypes, the 543-row holistic group layout, per-group detection rates, a frame plot, and the reference npz→model-input loader. Read-only by construction: no writes to the landmarks tree, no worker pool, no MediaPipe import, `.tmp.npz` staging files excluded so a half-written video is never opened.

   **Saved format** — the stage-1 → stage-2 contract. Its definition is **[`packages/sb-core/src/sb/core/schema.py`](packages/sb-core/src/sb/core/schema.py)** (`LANDMARK_TENSOR` v1), not this paragraph: `schema.GROUPS` is the single row layout that `holistic.py`, `quality.py` and `subsets.py` all derive from, and `validate_tensor` / `validate_npz` enforce it. One `np.savez_compressed` per video at `<root>/data/raw/popsign/<split>/<label>/<video_id>.npz`, the label carried by the path rather than stored inside: `landmarks` `(T, 543, 3)` float16 (NaN where undetected), `fps` float32, `num_frames` int32. Row order is GISLR holistic order (face 0–467, left hand 468–488, pose 489–521, right hand 522–542) — that is what makes the `subsets.py` indices apply to POPSIGN unchanged, so extraction validates it before every write. ~165 KB/video → **~5.4 GB** for the full test split.
4. `experiments/recognition/popsign.2.model.ipynb` — label-distribution analysis, stratified split, earlier TensorFlow experiments.
5. `experiments/recognition/popsign.3.pipeline.ipynb` — end-to-end pipeline (stub).

(`experiments/extraction/popsign.1.mediapipe.ipynb` is stale — slated for retirement per `TODO.md` §0.1.)

## Constraints & known limitations

- **Streaming-viability drives architecture choice** — the deployment path is the unidirectional GRU; bidirectional models (BiLSTM) can only ever be offline accuracy benchmarks.
- **TensorFlow GPU is not supported on native Windows** — training uses PyTorch (CUDA); TFLite conversion happens post-hoc by **rebuilding the trained model in native Keras and transferring the weights** (`sb.recognize.export.keras`), gated on numerical parity with the PyTorch model. The PyTorch → ONNX → `onnx2tf` route was tried and abandoned: onnx2tf failed on 3 of the 4 architectures (see `TODO.md` §6.2).
- **MediaPipe GPU delegate is Ubuntu-only** — landmark extraction runs on CPU on this Windows machine, parallelized across worker processes.
- **Runs are not yet reproducible from their records** — `meta.json` captures hyperparameters but no commit SHA, environment versions, dataset version or feature-cache identity, and the feature caches are keyed by subset *name* rather than by content. Both are being fixed under `TODO.md` §9 (schema v4 provenance block, content-addressed caches); until then a number in the registry is comparable to another only by convention, not by check.
- **Trained weights exist on one machine** (`best.pt`/`last.pt` are gitignored, 675 MB over 42 runs). The 2026-07-18 registry reset already destroyed 8 runs' weights — off-machine sync is `TODO.md` §9.3.
- Hardware: Windows 11, i7-14700K, RTX 4080 Super, 64GB DDR5 RAM.
