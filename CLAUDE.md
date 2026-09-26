# CLAUDE.md

## What this repo is

Notebook-driven ML research: streaming (causal, frame-by-frame) sign language recognition on MediaPipe landmarks. **GISLR** (Kaggle dataset `bracu23101281/gislr-stratified`, pre-converted npz landmarks derived from `asl-signs`) is the sole active dataset. **POPSIGN is deprecated (2026-09-22)** — its extracted data (`data/raw/popsign/`, `data/cache/popsign/`) was deleted; `sb-extract`/`sb-extract-ts` and `popsign.*.ipynb` notebooks are paused, not removed (`TODO.md` §2). No app, no test suite, no CI. `experiments/` notebooks are the dev surface, driving `packages/sb-*/`; `README.md` / `TODO.md` / `docs/` are the committed record of results.

**Never run model training yourself.** Build the notebook, hand it to the user to execute, analyze results after.

**Bookend every task with `README.md` and `TODO.md`.** Read both before starting — TODO.md is the source of truth for workstreams. After finishing: mark items done/in-progress, file follow-ups under the matching `## N.` section (new section only if none fits), reflect any result/structure/plan change in the README.

**Every response that learns or changes something updates `TODO.md` before it ends — not just at task end.** This includes status checks and Q&A, not only code changes. Record in the matching section, dated (`YYYY-MM-DD`):
- state you observed (a notebook half-run, an artifact present or missing, a run finished or failed);
- results and numbers read from outputs;
- decisions and answers the user gave, and questions you asked them that are still open;
- new blockers, bugs, or follow-ups;
- the "next action" as you would now state it.

Keep the "Current focus" table at the top current — re-derive it whenever what's next changes, and never leave a stale date on it. A response that only chats and learns nothing new needs no edit. If a session ends mid-task, TODO.md must say exactly where it stopped, so the next session can resume from the file alone.

## Environment & commands

- `uv sync` installs deps. **Never `uv pip install` ad-hoc** — it removes anything undeclared in `pyproject.toml` (torch was lost this way once). Declare new deps there.
- `./ops/envs.ps1 -Stage train|mlops|extract` builds `.venvs/<stage>` from one workspace member's dependency closure — exists to make "`sb-mlops` must not import `sb-recognize`" checkable (an mlops-only env has no torch/tensorflow/mediapipe/opencv). Default `.venv` has everything and is what notebooks use.
- `packages/sb-extract-ts` is Deno, excluded from the uv workspace (no `pyproject.toml`). **Paused** along with POPSIGN, its only consumer (`TODO.md` §2).
- Run Python via `.venv/Scripts/python.exe`. **CWD doesn't matter** — packages are editable-installed workspace members; `sb.core.paths` finds repo root by walking up for a workspace marker (`SIGNBRIDGE_ROOT` overrides).
- Console scripts (`.venv/Scripts/`, any CWD, no interpreter prefix):
  ```bash
  sb-evaluate.exe <run_dir>   # canonical per-class eval; fetches checkpoint if absent
  sb-index.exe [...]          # rebuild registry/index.csv + query it
  sb-sync.exe status          # checkpoint backup: status / push / pull / prune / rescheme
  sb-promote.exe list         # aliases: which run is champion
  sb-docs.exe                 # regenerate index.csv + schemas/ + README generated blocks
  sb-extract.exe --help       # POPSIGN extraction — PAUSED, POPSIGN deprecated
  ```
  `ops/` is housekeeping only (PowerShell etc.), no project Python.
- Dataset resolution is **lazy** — importing `sb.core.paths` downloads nothing. `gislr_dir()` for GISLR, the only active dataset. `resolve_datasets()` also resolves POPSIGN (deprecated, huge) — don't call it for new work.
- `.env` (gitignored) holds `KAGGLE_MCP_TOKEN`. `POPSIGN_LANDMARKS_DRIVE` is vestigial now that POPSIGN is deprecated.
- Type checking: `.venv/Scripts/ty.exe check` from repo root (canonical; `pyrefly` dropped 2026-07-22).
- No `jq` on this machine — parse notebook JSON with `python -c "import json; ..."`.

## Git workflow

- Commit after each logical unit of work — don't batch unrelated changes. Stage explicitly with `git add`. Conventional-commit format (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).
- **At the end of every response with a significant change, commit and push (`git push origin main`) — standing authorization, no need to ask.** Significant = code, config, notebook source, or docs/TODO/README content that records a result, decision, status, or plan. Not significant (don't commit on their own): typo/whitespace-only fixes, or leaving a notebook mid-run. Those ride along with the next significant commit. Never commit a notebook whose run is partial — wait until it finishes. If a push fails (auth, non-fast-forward), report it; never force-push.

## Windows constraints

- Training is PyTorch + CUDA. TensorFlow GPU doesn't work on native Windows — TFLite export rebuilds the model in native Keras and transfers weights (`sb.recognize.export.keras`, via `gislr.2.models.evaluation.ipynb`); the ONNX/onnx2tf route was tried and abandoned.
- MediaPipe extraction is CPU-only (GPU delegate is Ubuntu-only), parallelized across workers. Two extractors exist — `packages/sb-extract` (Python, produced the 33,599 POPSIGN test clips) and `packages/sb-extract-ts` (Deno/TS). **Both paused (2026-09-22)** — POPSIGN, their only workload, is deprecated; kept for a future raw-video dataset or live-camera deployment, not maintained until one exists.
- POPSIGN's `python -m sb.extract.popsign_cycle run --part <name>` (download → extract → verify → delete, one part at a time) is **deprecated** along with the rest of POPSIGN (`TODO.md` §2/§10.3).
- DataLoader multiprocessing (spawn) is fragile from ad-hoc scripts — in-RAM arrays with `num_workers=0` train GISLR at ~0.3 min/epoch, plenty (`SubsetArrayDataset` in `sb.recognize.features.base_v1`).

## Architecture & conventions

- **Streaming viability drives everything.** Deployment target is the unidirectional `StreamingGRU`; bidirectional/offline models (BiLSTM etc.) are accuracy references only. New features must be causal.
- **Shared logic in `packages/`, notebooks stay thin.** `sb-recognize` (`architectures.py` = the *single* definition of every model class — never redefine elsewhere; also `data.py`, `train.py`, `report.py`, `features/`), run records in `sb-mlops`, shared contracts in `sb-core`. `sb-mlops` must never import `sb-recognize` (registry must be readable without torch).
- **All GISLR training is one notebook**, `gislr.1.models.training.ipynb`, one markdown section per architecture. **Hyperparameters live only in `experiments/recognition/configs/gislr.training.json`, never in a cell** — architectures inherit `shared` and may deviate only via `overrides` naming a key that exists in `shared` (`sb.recognize.config` enforces this). Keeps architecture/subset comparisons all-else-equal — a past bug with per-notebook configs silently flattened cnn1d's `num_layers` 5→2.
- **Notebooks**: `experiments/<domain>/<dataset>.<stage>.<topic>.ipynb` (domain = extraction/recognition/synthesis). Configs beside them in `experiments/<domain>/configs/`. Pipeline order in README.md § "Running the pipeline".
- **Model registry**: flat `registry/runs/<run_id>/` (`run_id` = epoch seconds at training start), only `meta.json` + `best.pt`/`last.pt` (gitignored) + `assets/`. Dataset/architecture/subset are meta.json fields, not directories. Schema defined in `sb.mlops.registry::FIELDS` (v4) — README schema section, `schemas/meta.v4.json`, `index.csv` are all *generated*; edit code + run `gen_docs.py` (`--check` catches drift), never hand-edit.
- **Checkpoints live on Kaggle, not on disk** — `best.pt` pushed then deleted locally; a run folder normally has no weights. `sb-evaluate`/TFLite export auto-fetch via `sb.mlops.artifacts.ensure_local` (sha256-verified) — **never add a manual "pull it first" step**. Slug `bracu23101281/signbridge-<dataset>/pyTorch/<architecture>/<version>` (model=dataset family, variation=architecture, version=any run); everything else goes in the version note + meta.json, so a version list is *not* a learning curve. `sb-sync rescheme` migrates uploads after a naming change (Kaggle has no rename).
- **Never delete a checkpoint with `rm`.** `sb-sync prune` only deletes when manifest/remote/local hashes all agree. `sb-sync drop-resume` clears `last.pt` for finished runs only.
- **Data placement**: `data/` is gitignored absolutely; registry lives outside it at top level. raw→`data/raw/<dataset>/`, reusable derived→`data/cache/<dataset>/`, throwaway→`data/temp/` (delete after use via `sb.core.paths.cleanup_temp()`), third-party→`data/external/`. Raw kagglehub data never enters the repo.
- **Canonical GISLR eval**: GISLR_Stratified's fixed 80/20 split (18,896-video val set, stratified on `sign` only, upstream seed 42), per-class accuracy from the raw npz files — `sb.recognize.evaluate` reproduces this and promotes `meta.json` to `eval_status: "canonical"`. A new run displaces the leaderboard only on this exact split/metric. **Reset 2026-09-16** when GISLR moved off the live `asl-signs` competition parquet download to the self-produced `bracu23101281/gislr-stratified` Kaggle dataset (pre-converted npz) — the 37 runs evaluated on the old self-computed 90/10 split (9,448 val) are historical references only.
- **Docs**: logs (time-ordered) vs reports (standalone findings). `docs/logs/daily/<YYYY-MM-DD>.md`; weekly `docs/logs/weekly/<YYYY>-<WW>.md` (Sun→Sat, week 1 = week containing Jan 1, not ISO); topic reports `docs/reports/<topic>.md`. Figures under matching `assets/`.

### Building a notebook

1. **Every cell independently re-runnable** — a subtask's tunables live at the top of its own cell; cells load inputs from disk/cache, not live memory from other cells. Only allowed cross-cell dependency: the setup cell + `sb.*` imports.
2. **Title cell**: what it does, pipeline stage vs diagnostic, artifact table (path per output), resumability, design decisions vs the TODO spec. Then one `## N.` markdown cell per section, cross-referencing TODO items.
3. **Long tasks save state as they go** — manifest-driven resumable pattern (write artifact before marking `done` in `data/cache/.../<scope>_manifest.json`, atomic via temp file + `os.replace`, `done` skipped/`failed` retried); training uses the driver's auto-resume. Record seeded samples to JSON.
- Code cells open with a `# ===== / <what this cell does> / =====` banner.
- One setup cell after title: imports, then UPPERCASE tunables with inline comments, then print resolved paths.
- `gislr_dir()` for all dataset work — the only active dataset. Never `resolve_datasets()` (POPSIGN-inclusive, deprecated).
- One `tqdm.auto` bar per long task, everything in its description/postfix — no nested bars, no per-iteration prints.
- Heavy outputs (parquets, PNGs) go to `data/cache/`/`assets/`, not cell outputs — display only a couple of representative figures inline (a notebook once hit 17MB from animation outputs).

## Key domain facts

- GISLR frames: 543 landmarks (`ROWS_PER_FRAME`), xyz, subsampled to `MAX_SEQ_LEN=128`, NaN→0 (`sb.recognize.data`).
- **Current-split leader (2026-09-26): `gru_phono_raw` on ME_134, 0.7632 canonical, streaming** — the `PhonologyFrontend` (84 handshape/orientation/location features) plus ME_134 raw xy. Best raw-landmark model: `gru` ME_132/xy 0.7517. `registry/index.csv` can lag; regenerate with `sb-docs.exe` for the live number (`docs/reports/phonology-models.md`).
- **ME-126** subset (hands + upper-body pose {11-16,23,24} + lips + eyes/nose) was chosen for the raw-landmark models. Motivated by a pre-registry-reset comparison (73.73% vs full-543's 70.59%) showing the z channel is mostly noise for pose landmarks (~92% of pose "motion"). Evidence: `docs/logs/daily/2026-07-15.md`, `2026-07-16.md`.

## Known broken / stale

This list rots fast — `TODO.md` §0 (repo-restructure follow-ups) is the source of truth, not this file. §2 (POPSIGN extraction) is deprecated, not stale-and-pending.
