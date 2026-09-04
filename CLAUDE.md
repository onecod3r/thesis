# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Notebook-driven ML research: streaming (frame-by-frame, causal) sign language recognition on MediaPipe landmark sequences. Two datasets — **GISLR** (Kaggle `asl-signs`, landmarks pre-extracted, fast iteration) and **POPSIGN** (~870GB raw video, landmark extraction still in progress). There is no app, no test suite, and no CI; notebooks in `experiments/` are the primary dev surface, driving library code in `packages/sb-*/`, with `README.md` / `TODO.md` / `docs/` as the committed record of results.

**Never run model training yourself.** When a task calls for training (or any long GPU run), build the appropriate notebook (following the conventions below) and hand it to the user to execute — they run it, you analyze the results afterwards.

**Bookend every task with `README.md` and `TODO.md`.** Before starting, read both to orient — TODO.md is the source of truth for workstreams and open items. After finishing, update both: mark TODO items done / in-progress, file follow-ups under the matching numbered workstream section (add a new `## N.` section only if none fits), and reflect any change to results, structure, or plans in the README.

## Environment & commands

```bash
uv sync                     # install deps (Python >= 3.12; torch cu130 via [tool.uv.sources])
```

- **Never `uv pip install` ad-hoc** — `uv sync` removes anything not declared in `pyproject.toml` (torch was once lost this way). Declare new deps in `pyproject.toml` instead.
- **Per-stage environments**: `./ops/envs.ps1 -Stage train|mlops|extract` builds `.venvs/<stage>` from one workspace member's dependency closure (`uv sync --package <member>` with `UV_PROJECT_ENVIRONMENT`). The default `.venv` holds everything and is what notebooks use; the split exists because torch and tensorflow are resolved together in one env and because it makes the "`sb-mlops` must not import `sb-recognize`" invariant executable — an mlops-only env verifiably has no torch, tensorflow, mediapipe or opencv.
- **`packages/sb-extract-ts` is a Deno package**, excluded from the uv workspace (`[tool.uv.workspace].exclude`) because it has no `pyproject.toml` and `packages/*` would otherwise break every `uv sync`.
- Run project Python via `.venv/Scripts/python.exe` (Windows). **CWD no longer matters**: the six packages are installed into the venv as editable workspace members, so `import sb...` resolves from anywhere, and `sb.core.paths` finds the repo root by walking up for the workspace marker (`ROOT_DIR`, `DATA_DIR`, `RAW_DIR`, `CACHE_DIR`, `TEMP_DIR`, `EXTERNAL_DIR`, `REGISTRY_DIR`, `MODELS_DIR`, `MODEL_INDEX`, `EXPERIMENTS_DIR`). `SIGNBRIDGE_ROOT` overrides the walk.
- The project CLIs are **console scripts** installed into the venv by `uv sync`; they run from any CWD and need no interpreter prefix:
  ```bash
  .venv/Scripts/sb-evaluate.exe <run_dir>   # canonical per-class eval (all archs); fetches the checkpoint if absent
  .venv/Scripts/sb-index.exe [...]          # rebuild registry/index.csv + query it
  .venv/Scripts/sb-sync.exe status          # checkpoint backup: status / push / pull / prune / rescheme
  .venv/Scripts/sb-promote.exe list         # aliases: which run is champion
  .venv/Scripts/sb-docs.exe                 # regenerate index.csv + schemas/ + README generated blocks
  .venv/Scripts/sb-extract.exe --help       # POPSIGN extraction (Python path)
  ```
  `ops/` holds housekeeping only (PowerShell etc., no project Python).
- Dataset resolution is **lazy**: importing `sb.core.paths` downloads nothing; call `sb.core.paths.gislr_dir()` for GISLR only, `resolve_datasets()` for everything (POPSIGN included — huge). Requires an authenticated Kaggle account that has accepted the `asl-signs` competition rules.
- `.env` at repo root (gitignored) holds secrets — currently `KAGGLE_MCP_TOKEN` (Kaggle MCP auth, TODO §6.3). `POPSIGN_LANDMARKS_DRIVE` (meant to send extracted POPSIGN landmarks to a separate drive, never into the repo) is **not currently set**, so extraction falls back to `data/raw/popsign` — set it before a bulk POPSIGN run if that's not where you want ~hundreds of GB to land.
- Type checking: `ty` is canonical (`[tool.ty.environment]` points at `./.venv`) — run `.venv/Scripts/ty.exe check` from the repo root. `pyrefly` was dropped 2026-07-22 (TODO §0.2).
- No `jq` on this machine — parse notebook JSON with `.venv/Scripts/python.exe -c "import json; ..."`.

## Git workflow

- **Create a Git commit after completing each logical unit of work** — don't batch unrelated changes into one commit.
- **Automatically stage modified files** with `git add` as part of committing.
- **Always format messages as [conventional commits](https://www.conventionalcommits.org/)** (e.g. `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).

## Windows constraints (shape architecture decisions)

- **Training is PyTorch + CUDA.** TensorFlow GPU doesn't work on native Windows; TFLite export happens post-hoc by rebuilding the model in native Keras and transferring weights (`sb.recognize.export.keras`, driven by `gislr.2.models.evaluation.ipynb`) — the ONNX/onnx2tf route was tried and abandoned (TODO §6.2).
- **MediaPipe extraction is CPU-only** here (GPU delegate is Ubuntu-only), parallelized across worker processes. There are **two extractors**: `packages/sb-extract` (Python, the one that produced the 33,599 test clips) and `packages/sb-extract-ts` (Deno/TypeScript, chosen because Deno gives `ImageData` and Web Workers natively so MediaPipe's WASM build needs no `canvas` native module). **They are not yet known to agree** — run `python -m sb.extract.parity` before letting the TS path extract anything that will be trained on, because a systematic difference between extractors used on different splits is invisible to every accuracy metric.
- **POPSIGN is downloaded one part at a time.** ~870 GB does not fit; `python -m sb.extract.popsign_cycle run --part <name>` does download → extract → **verify** → delete, and refuses to delete a part whose clips are not all extracted and spec-valid.
- DataLoader multiprocessing (spawn) is fragile from ad-hoc scripts — in-RAM arrays with `num_workers=0` train GISLR at ~0.3 min/epoch, which is plenty (`sb.recognize.features.base_v1::SubsetArrayDataset`).

## Architecture & conventions

- **Streaming viability drives everything.** The deployment path is the unidirectional `StreamingGRU`; bidirectional/offline models (BiLSTM etc.) are only ever accuracy references, never deployment candidates. New features (e.g. lag differences) must be causal.
- **Shared logic lives in `packages/`, notebooks stay thin.** The recognition stack is `packages/sb-recognize/` (`architectures.py` = the *single* definition of every model class, imported by both training and `evaluate.py` — never redefine a model class elsewhere; `data.py`, `train.py`, `report.py`, `features/`), with run records in `packages/sb-mlops/` and the shared contracts in `packages/sb-core/`. **`sb-mlops` must never import `sb-recognize`** — the registry has to be readable without importing torch. **All GISLR training is one notebook**, `gislr.1.models.training.ipynb`, with a markdown section per architecture. **Hyperparameters live in `experiments/recognition/configs/gislr.training.json`, never in a cell** — architectures inherit the `shared` block and may deviate only via an explicit `overrides` entry (`sb.recognize.config` validates this and rejects an override naming a key that isn't in `shared`). This is what keeps the architecture comparison and subset ablations all-else-identical; four separate notebooks made that a manual chore and it silently broke (cnn1d's `num_layers=5` got flattened to 2, cutting its receptive field from 125 frames to 13).
- **Notebooks live in `experiments/<domain>/`, named `<dataset>.<stage>.<topic>.ipynb`** — domain (`extraction` / `recognition` / `synthesis`), then dataset, a stage number ordering the pipeline, and the topic. Configs sit beside them in `experiments/<domain>/configs/`. Pipeline order per dataset is described in `README.md` §"Running the pipeline".
- **Model registry** (reset empty 2026-07-18, repopulated since — 42 indexed runs as of 2026-09-04; pre-reset runs live in git history ≤ `3668dae`, their weights are gone): one **flat** folder per run at `registry/runs/<run_id>/` with `run_id` = **epoch seconds** at training start, containing only `meta.json` + `best.pt`/`last.pt` (gitignored) + `assets/` (everything else, each linked from `meta.json["assets"]`). Dataset/architecture/subset are meta.json fields, not directory levels. **The meta.json schema is defined in `sb.mlops.registry::FIELDS`** (v4 — adds the `provenance` block, `sb.mlops.run`); `README.md` § "meta.json schema" and `schemas/meta.v4.json` are both *rendered* from it, so edit the code and re-run `gen_docs.py`. The training driver rewrites `meta.json` every epoch. **`index.csv`, `schemas/meta.v4.json` and the README's `<!-- generated:… -->` blocks are all derived** — regenerate them with `gen_docs.py` (and `gen_docs.py --check` fails on drift) rather than editing them by hand.
- **Model artifacts live on Kaggle, not on this disk.** `best.pt` is pushed to a
  Kaggle **Model** and then deleted locally, so *a run folder normally has no
  weights in it*. Anything that loads a checkpoint (`sb-evaluate`, TFLite export)
  fetches it automatically via `sb.mlops.artifacts.ensure_local` — a `kagglehub`
  download, sha256-verified against the manifest — so **never add a "pull it
  first" step to a workflow**; call the thing and let it fetch. Learning curves
  read the committed `assets/history.json` and need no checkpoint at all. Naming follows Kaggle's own convention and
  is derived, never typed by hand:
  `bracu23101281/signbridge-gislr/pyTorch/<architecture>/<version>` — the
  **model** is the family (a POPSIGN model becomes `signbridge-popsign`, not a
  variation, because a different label space is a different model), the
  **variation** is just the **architecture** (`gru`, `bilstm`,
  `conv1d-transformer`), and a **version** is any run of it. Everything the slug
  does not say — subset, coords, score, params, regime — goes in the **version
  note**, and in full in the `meta.json` uploaded beside the weights. Consequence
  to keep in mind: versions of one variation are *not* all-else-equal, so a
  version list mixes subsets and cannot be read as a learning curve; restores are
  unaffected because the manifest pins an exact `<variation>/<version>` handle.
  `sb-sync rescheme` migrates existing uploads after a naming change (Kaggle has
  no rename, so it re-uploads; old variations must be deleted by hand).
- **Never delete a checkpoint with `rm`.** `sb-sync prune` is the only safe path:
  it downloads the remote copy, and deletes only when the manifest hash, the
  remote hash and the local hash all agree. `sb-sync drop-resume` removes
  `last.pt` for *finished* runs only (the registry never resumes a finished run);
  an unfinished run keeps its resume state.
- **Data placement policy** — `data/` is gitignored **absolutely**, and the committed registry lives at the top level in `registry/` rather than inside it: extracted-from-source → `data/raw/<dataset>/`; reusable derived artifacts → `data/cache/<dataset>/` (content-addressed feature caches, analysis caches, manifests); throwaway output → `data/temp/`, **deleted after use** via `sb.core.paths.cleanup_temp()` (give producing notebooks a cleanup cell); third-party assets → `data/external/`. Raw kagglehub data never enters the repo.
- **Canonical evaluation** (all GISLR runs must match to be comparable): stratified 90/10 split, `random_state=42`, 9,448-video val set, per-class accuracy from raw parquet — exactly what `sb.recognize.evaluate` reproduces (it also promotes the run's `meta.json` to `eval_status: "canonical"`). A new run displaces a leaderboard entry only on this same split/metric.
- **Docs** split two ways — **logs** (time-ordered) and **reports** (standalone findings): daily logs `docs/logs/daily/<YYYY-MM-DD>.md`; weekly summaries `docs/logs/weekly/<YYYY>-<WW>.md` (**weeks run Sunday → Saturday**, week 1 = the week containing Jan 1 — *not* ISO; `2026-30.md` = 2026-07-19 → 07-25, and the title always states the date range); standalone topic reports from a test/analysis (motion-energy, subset-comparison, confidence-tuning, …) `docs/reports/<topic>.md`. Figures under the matching `assets/` subfolder. Convention: `docs/README.md`.

### How to build a notebook (and any big task)

Three core rules:

1. **Every cell is independently re-runnable.** Tweaking a parameter and re-running *one* cell must be enough to redo that subtask — never a series of cells. Put a subtask's tunables at the top of its own cell; have cells load their inputs from disk/cache rather than from live memory produced by other subtask cells. The only allowed dependencies are the setup cell (imports, shared constants, paths) and `sb.*` imports.
2. **Well documented with markdown cells.** Title cell first: what the notebook does, pipeline stage vs standalone diagnostic, a table of the artifacts it produces (path per output), how resumability works, and design decisions vs the TODO spec it implements. Then a numbered `## N.` markdown cell per section, cross-referencing TODO items (e.g. `## 3. Scope 1 — per-video (TODO §1.3)`).
3. **Long tasks (extraction, training, …) save state as they go** — an error/interrupt must never force a complete rerun. Use the existing manifest-driven resumable pattern (per-unit artifact written *before* marking `done` in a `data/cache/.../<scope>_manifest.json`; atomic saves via temp file + `os.replace`; `done` skipped, `failed` retried) rather than inventing a second one; training uses the driver's auto-resume checkpointing. Record seeded samples to JSON in the cache so re-runs are stable.

Supporting conventions:

- **Code cells open with a banner comment** (`# ===== / # <what this cell does> / # =====`).
- **One setup cell** right after the title: all imports together, then every shared tunable as an UPPERCASE constant with a short inline comment; end by printing the resolved data/cache paths.
- **Download only what's needed**: `sb.core.paths.gislr_dir()` for GISLR work — never `resolve_datasets()` unless POPSIGN raw video is genuinely required.
- **Progress reporting uses ONE bar per long task** (`tqdm.auto`), with everything (sub-progress, metrics) in that bar's description/postfix — no nested bars, no per-iteration prints (`sb.recognize.train` is the pattern).
- **Heavy outputs go to `data/cache/` (or a run's `assets/`), not into cell outputs**: write per-unit parquets/PNGs to disk and display only a couple of representative figures inline (a notebook once hit 17MB from animation outputs and had to be stripped).

## Key domain facts

- GISLR frames have **543 landmarks** (`ROWS_PER_FRAME`), xyz each; sequences uniformly subsampled to `MAX_SEQ_LEN=128`; NaN → 0 (constants: `sb.recognize.data`).
- **ME-126** landmark subset (hands + upper-body pose {11–16, 23, 24} + lips + eyes/nose) is the current leaderboard winner: **~75.7% canonical val acc** (gru and bilstm essentially tied, xy coords, `v2-plateau-300` regime, current 36-run registry as of 2026-07-22 — `registry/index.csv` can lag the actual run folders, so regenerate it with `build_model_index.py` first, or glob `registry/runs/*/meta.json` directly, or run `gislr.2.models.evaluation.ipynb` for the live leaderboard number). Originally discovered pre-registry-reset at 73.73% vs the full-543 baseline's 70.59% (v1 regime, xyz, weights since gone) — that comparison is what motivated the subset in the first place, on the finding that the z channel is mostly noise for pose landmarks (~92% of pose "motion"). Evidence: `docs/logs/daily/2026-07-15.md` and `docs/logs/daily/2026-07-16.md`.

## Known broken / stale (verify before relying on)

This list rots within days (things here have gone stale mid-week before) — treat
`TODO.md` as the source of truth, not this file. Read `TODO.md` §0 (repo-restructure
follow-ups) and §2 (POPSIGN extraction) before relying on any specific broken/stale
claim; don't snapshot specifics here.
