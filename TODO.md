# TODO — sign2speech

Living project TODO, organized by workstream so new tasks can be filed under an
existing section or a new one added without restructuring.

**Status legend:** `[ ]` open · `[~]` in progress · `[x]` done · `[?]` open question / decision needed

**How to add a task:** file it under the matching workstream section below. If it
doesn't fit an existing one, add a new `## N. <Workstream Name>` section at the end
(before "Backlog / Someday") rather than bolting it onto an unrelated section.

---

## 0. Repo Restructure Follow-ups & Tooling

Cleanup left over from the move to the flat `src/` layout (notebooks renamed to
`<dataset>.<stage>.<topic>.ipynb`), plus the **2026-07-18 restructure** (§0.4:
unified `modules/model` training stack, flat epoch-seconds registry with
meta.json schema v2, `data/{raw,cache,temp,models}` placement policy,
docs daily/weekly/reports split).

### 0.1 Stale imports / references broken by the restructure

- [x] **`src/modules/data/` no longer exists on disk** (2026-07-16): `dataset.py`
  (`GISLRRawDataset`) and `landmark_worker.py` are gone. `landmark_worker.py` is
  superseded by `modules/dataset/landmark/extraction.py` (§2). **Resolved
  2026-07-16:** `GISLRRawDataset` no longer needs restoring — the rebuilt
  `gislr.1.model.gru.ipynb` defines its dataset in-notebook (in-RAM arrays,
  `num_workers=0`, the pattern proven by the ME-126 training script).
- [x] `src/gislr.1.model.gru.ipynb` imports `from modules.dataset import ...` —
  **resolved 2026-07-16** by the notebook overhaul (§3.1): it now imports only
  the subset registry (`modules.dataset.landmark.subsets`) and defines the
  dataset class itself.
- [ ] `src/popsign.1.mediapipe.ipynb` imports `from modules.datasets import DATASETS`
  and uses `DATASETS["ISLR"]` — `datasets/` was deleted; it's now
  `modules.paths.DATASETS` with key `"GISLR"`.
- [x] `src/modules/` has no `__init__.py` files — **resolved 2026-07-18**: every
  package level ships one (`modules/`, `modules/model/`, `modules/scripts/`,
  `modules/dataset/`, `modules/dataset/landmark/`).
- [ ] `src/popsign.1.mediapipe.ipynb` currently contains early **GISLR** motion-energy
  exploration code, not POPSIGN extraction — retire that content (superseded by
  `gislr.0.dataset.motion-energy.ipynb`) and rebuild the notebook as the extraction
  driver (§2).

### 0.2 Packaging / config

- [x] `pyproject.toml`: torch/torchvision/torchaudio sat under an invalid top-level
  `[dependencies]` table. **Fixed 2026-07-15:** `torch>=2.13.0` declared in
  `[project].dependencies` with the cu130 index pinned via `[tool.uv.sources]`;
  torchvision/torchaudio dropped (nothing imports them, and torchaudio has no
  cu130 build that resolves for the full `requires-python` range). `uv sync`
  verified: torch 2.13.0+cu130, CUDA available.
- [x] `pyproject.toml`: placeholder description replaced (2026-07-15).
- [x] **Type checker: `ty` chosen, `pyrefly` dropped (2026-07-22).** `pyrefly`
  had no config left (`pyrefly.toml` was deleted) while `ty` already had a
  working `[tool.ty.environment]` block; `pyrefly>=1.1.1` removed from
  `[project].dependencies` and `uv sync` re-run (clean single-package
  uninstall, torch untouched). `ty` stays in `[dependency-groups].dev`. Run via
  `.venv/Scripts/ty.exe check` from `src/` (35 pre-existing diagnostics as of
  2026-07-22, mostly in the stale `popsign.2`/`popsign.3` notebooks — not
  triaged, just confirming the tool runs).
- [ ] `.gitignore`: the bare `data/` pattern also ignores `src/data/external/`
  (the MediaPipe `holistic_landmarker.task`) and `src/data/cache/dataframes/`
  (POPSIGN manifests) — decide whether to narrow the ignore and commit those, or
  document them as download/generate-on-setup. (Still open after the 2026-07-15
  rewrite — the pattern was kept as-is pending this decision.)
- [x] `.gitignore`: rewritten 2026-07-15 for the `src/` layout — stale root-level
  `cache/*.npy` lines and the self-ignoring `.gitignore` line removed; now covers
  `src/cache/`, model weights (`src/models/**/*.pt`, bare `gru_best.pt` /
  `gru_latest.pt` from notebook runs), export artifacts (`*.onnx`, `*.tflite`,
  `src/saved_model_dir/`, `src/final_saved_model/`) and `.ipynb_checkpoints/`.
- [x] Repo size: `src/gislr.0.competition.entry.1st.ipynb` was 17 MB — 15 MB of
  landmark-animation cell outputs (cells 10–13) stripped 2026-07-15 → 247 KB.
  Code, markdown and the training-log output are intact. Full copy with outputs:
  `src/cache/gislr.0.competition.entry.1st.with-outputs.ipynb` (gitignored) or
  Kaggle discussion 406978.

### 0.3 Model-run metadata & queryable index (2026-07-17, superseded by §0.4's schema v2)

Structured run records so "best 3 gru runs on gislr" / "all runs on subset X"
is a query, not a folder crawl:

- [x] Per-run **`metadata.json`** schema (dataset, architecture, subset, coords,
  n_params, hyperparameters, train-loop vs canonical accuracies, `eval_status`
  pending/canonical) — backfilled for all 5 existing gislr/gru runs 2026-07-17.
- [x] **`scripts/build_model_index.py`** — flattens every run's `metadata.json`
  into `src/models/index.csv` (committed) and answers filter queries
  (`--dataset/--architecture/--subset/--top`); warns on runs missing metadata.
- [x] `gislr.1.model.gru.ipynb` §7 (run-docs cell) now also writes
  `metadata.json` (preserving canonical-eval fields on re-runs);
  `scripts/eval_gru.py` promotes `eval_status` to `canonical` after the
  per-class eval.
- [x] Run folders are always fresh (2026-07-17): `gislr.1.model.gru.ipynb`'s
  `resolve_run_dir` no longer reuses/skips a **completed** run — every new
  training gets a new `<timestamp>` folder (timestamp = training start), even
  under identical conditions. Auto-resume still continues an *interrupted*
  run in its own folder.
- [~] Future training notebooks must write the same `metadata.json` schema —
  **done for GISLR** (the lstm/bilstm/cnn1d siblings inherit the GRU
  notebook's §7 emission, 2026-07-17); still applies to future POPSIGN
  training notebooks.
- [x] ~~Rebuild `index.csv` after the canonical evals of the six pending runs~~ —
  **voided 2026-07-18**: those runs' weights were deleted with the old
  `src/models/` tree during the restructure, so their canonical evals can
  never run. Their train-loop numbers stand as historical references
  (git history ≤ `3668dae` + docs/logs/daily/); the registry restarted empty (§0.4).

### 0.4 Restructure 2026-07-18 — unified stack, flat registry, data-tree policy

Executed 2026-07-18 (full write-up: `docs/logs/daily/2026-07-18.md`):

- [x] **`src/modules/model/`** — unified training stack (`architectures.py`
  single model-class definition shared with eval, `data.py`, `registry.py`,
  `train.py`, `report.py`); the four `gislr.1.model.*.ipynb` notebooks
  regenerated as thin drivers (identity block + `modules.model` calls).
- [x] **Registry v2**: flat `src/data/models/<epoch-seconds>/` folders holding
  only `meta.json` + `best.pt`/`last.pt` (gitignored) + `assets/`;
  schema v2 documented in README.md § "meta.json schema" (machine check:
  `modules/model/registry.py::REQUIRED_KEYS`); `meta.json` rewritten every
  epoch by the driver; **registry reset to empty** (header-only `index.csv`).
- [x] **Single progress bar per training run** (batch progress, metrics, LR,
  plateau counter in one bar) — replaces the nested-bar + per-epoch-print spam.
- [x] **`modules/paths.py`**: absolute CWD-independent tree constants
  (`RAW/CACHE/TEMP/EXTERNAL/MODELS`), lazy dataset resolution (import no
  longer downloads), `cleanup_temp()`.
- [x] **Data placement policy** applied: POPSIGN pilot npz → `data/temp/popsign_pilot/`
  with cleanup cell (stale `data/raw/popsign/_pilot` deleted); diagnostic
  caches → `data/cache/gislr/{motion_analysis,subset_comparison}`;
  manifests → `data/cache/popsign/dataframes/`; `.gitignore` reworked
  (all of `src/data/` ignored except `data/models/` minus weights/exports).
- [x] **Scripts split**: project CLIs in `src/modules/scripts/` (`eval_gru.py`
  takes a run folder, `build_model_index.py` flat-layout; both run from any
  CWD); root `scripts/` = housekeeping only.
- [x] **Docs split**: `docs/logs/daily/` + `docs/logs/weekly/<YYYY>-<WW>.md` +
  `docs/reports/<topic>.md` (convention in `docs/README.md`).
- [x] Regenerate the POPSIGN video manifests at
  `data/cache/popsign/dataframes/{train,test}.csv` — **done 2026-07-19**:
  §1 of `popsign.0.dataset.extraction.ipynb` now generates them from the raw
  video tree (30,867 train / 33,600 test, ids unique, labels cross-checked
  against the filename). Unblocks the §2 pilot/bulk runs. Still only 1 of 4
  train parts (§2.2).
- [ ] First v2-regime training runs (user) to seed the fresh registry —
  re-establishes the FULL_543 baseline and ME_126 leader under the new
  schema before any new ablation conclusions.
- [ ] `popsign.2.model.ipynb` / `popsign.3.pipeline.ipynb` still predate the
  restructure (old paths, TF-era code) — modernize or retire alongside
  `popsign.1` (§0.1).

### 0.5 Docs tree: `logs/` vs `reports/` (2026-07-19)

The docs tree grew a third top-level sibling (`daily/`, `weekly/`, `reports/`)
where only two *kinds* of document exist: time-ordered logs and standalone
test/analysis reports. Collapse the first two under one `logs/` parent:

```
docs/
├── logs/
│   ├── daily/<YYYY-MM-DD>.md
│   └── weekly/<YYYY>-<WW>.md
└── reports/<topic>.md        # motion-energy, subset-comparison, confidence-tuning, …
```

- [x] Move `docs/logs/daily/` → `docs/logs/daily/`, `docs/logs/weekly/` → `docs/logs/weekly/`
  (assets subfolders move with them).
- [x] Update every reference: `docs/README.md`, `README.md` (report table +
  project-structure block + conventions), `CLAUDE.md`, this file.
- [x] Backfill the standalone reports that currently live only as daily entries
  (motion-energy, subset-comparison) into `docs/reports/<topic>.md`, leaving the
  daily logs as the dated narrative that links to them. **Done 2026-07-22**:
  `docs/reports/motion-energy.md` and `docs/reports/subset-comparison.md`.
- [x] **Weekly logs started 2026-07-19**: `docs/logs/weekly/2026-29.md` (Jul 12–18)
  and `2026-30.md` (Jul 19–25, running). **Week numbering corrected**: weeks run
  **Sunday → Saturday**, week 1 = the week containing Jan 1 — *not* ISO, which
  `docs/README.md`, `README.md` and `CLAUDE.md` all previously said and which would
  number 2026-07-19 as the tail of week 29 rather than the start of week 30. All
  three updated; weekly titles now state the date range explicitly.
- [ ] Close out `2026-30.md` on Sat 2026-07-25 (drop the "in progress" marker) and
  open `2026-31.md`.

### 0.6 Stray notebook/module state to clean up (found 2026-07-19)

- [x] `src/popsign.0.dataset.extraction.ipynb` had a bare
  `DATASETS = resolve_datasets()` scratch cell in §1 (an unguarded call that
  would hit kagglehub on every top-to-bottom run). **Resolved 2026-07-19**: it
  became the proper manifest-generation cell — guarded by `FORCE_REGENERATE`
  and skipped entirely when both CSVs exist, so a plain re-run never reaches
  `resolve_datasets()` at all. (`force_download=True` is not in the current
  `modules/paths.py`; the ~870GB re-download hazard the original note described
  no longer exists.)
- [ ] `modules/paths.py::resolve_datasets` (uncommitted working-tree change)
  hardcodes `D:/`/`E:/` `PATH_PARTS` and calls `p.unlink()` on directories
  (raises on a real directory) — reconcile with the `.env`-driven policy.
  **Status 2026-07-22**: this is the actual change that downloaded the
  remaining 3 of 4 POPSIGN train dataset parts (§2.2) — `train-n-s-signs` to
  `D:/datasets/…` (complete 07-20) and `train-t-z-signs` to `E:/datasets/…`
  (complete 07-21) — so it did its job, but it's still uncommitted and still
  hardcoded rather than `.env`-driven. `src/temp.py` (untracked) is a scratch
  copy of an earlier version of this file, used to manually trigger the
  a-e/f-m downloads by hand — clean up both once the real fix is committed.
- [ ] `.env` is empty/missing at the repo root, so `POPSIGN_LANDMARKS_DRIVE`
  is unset and extraction output falls back into `src/data/raw/popsign`.

---

## 1. Landmark Motion-Over-Time Analysis (GISLR)

**Goal:** One robust, resumable notebook measuring how much each landmark moves
over time, at three scopes: per-video, per-category, global. Builds on existing
motion-energy pipeline findings (RMS speed, `["type","landmark_index"]` grouping,
Savitzky-Golay filtering) rather than re-deriving them.

**Location:** `src/gislr.0.dataset.motion-energy.ipynb`

**Status: ✅ executed end-to-end 2026-07-15 — all three scopes complete, 0 failed
units. Findings, stats, figures and the landmark keep/discard recommendation are
written up in `docs/2026-07-15.md`.** Remaining work moved to §1.8.

### 1.0 Decision to lock in

- [x] **Confirm DuckDB as the loading layer.** Adopted — `gislr.0` builds on
  `get_duckdb_conn()` querying parquet via `CREATE VIEW ... glob`, so aggregation
  happens before pulling into pandas and peak memory stays bounded by one query's
  result, not dataset size. Revisit only if a blocker turns up.

### 1.1 Reusable core (build once, use in all three scopes)

- [x] `get_duckdb_conn()` — validated end-to-end (94,477 videos · 250 signs · 21
  participants resolved from the meta view).
- [x] `load_landmarks_for_paths(paths: list[str]) -> pd.DataFrame` — validated; the
  one function all three scopes call (explicit parquet file list, not glob filter).
- [x] `compute_motion_energy(df: pd.DataFrame) -> pd.DataFrame` — validated:
  Savitzky-Golay (window=7, polyorder=2) → RMS speed over raw-valid transitions
  only (NaN policy) → tidy long format incl. `n_valid_transitions`.
- [x] `plot_motion_gridspec(df, title)` — validated across all three scopes
  (auto-detects per-video vs aggregate frames, std as error bars).
- [x] Run the reusable-core cells against real data — done, all scopes ran clean.

### 1.2 Resumable caching / state management

- [x] Manifest per scope: `cache/motion_analysis/<scope>_manifest.json` — with
  atomic saves (temp file + `os.replace`).
- [x] Idempotent write pattern: per-unit parquet written **before** marking `done`
  in the manifest — validated over 50 + 10 + 189 units, 0 failures.
- [x] Resume check (skip `done`, retry `failed`) + per-unit try/except — validated
  by the executed runs (skip path exercised on re-runs; a deliberate
  mid-run-interrupt drill wasn't needed given the invariants held over 249 units).
- [x] Final aggregation reads only cached per-unit files (`load_cached_units` /
  in-SQL over chunk parquets), never raw parquet — validated for all three scopes.

### 1.3 Scope 1 — Per-video (50 random samples)

- [x] Sample 50 video paths (seed 42, recorded to `per_video_sample.json`).
- [x] Per video: load → compute → cache → plot (50 PNGs in `per_video/plots/`).
- [x] Output: `cache/motion_analysis/per_video/summary.parquet` (27,150 rows =
  50 videos × 543 landmarks).

### 1.4 Scope 2 — Per-category (10 sampled sign categories)

- [x] Sample 10 sign labels (seed 42, recorded to `per_category_sample.json`).
- [x] Per category: batched load (100 videos/read) → aggregate RMS speed per
  landmark (mean + std — feeds the future within-class consistency analysis).
- [x] Output: `cache/motion_analysis/per_category/summary.parquet`
  (`sign, type, landmark_index, rms_speed_mean, rms_speed_std, n_videos`).

### 1.5 Scope 3 — Global (entire dataset)

- [x] Decision: full in-SQL aggregation rejected (Savitzky-Golay needs ordered
  per-frame series) → chunked Python compute, then **in-SQL aggregation over the
  cached chunk parquets** (the memory-bounded part still happens in DuckDB).
- [x] 189 chunks of ≤500 videos through load → compute → cache, manifest tracking
  chunk completion — full run ≈25 min at ~65 videos/s, 0 failures.
- [x] Output: `cache/motion_analysis/global/summary.parquet` + `global_overview.png`.

### 1.6 Cross-scope comparison

- [x] Overlay plot + rank correlation: per-video sample rho **0.954**, per-category
  sample rho **0.996** vs global (n=543) — the seeded samples reproduce the global
  per-landmark pattern, so sample-based landmark analyses can be trusted.
- [x] Cross-check against the competition 1st-place landmark subset — done in
  `docs/2026-07-15.md` §5: agrees on hands / face-mass-discard / legs / z-drop;
  diverges on upper-body pose (ME-126 keeps, 1st place drops) and lips (1st
  place keeps, on linguistic grounds motion energy can't see).

### 1.8 Follow-ups from the 2026-07-15 findings (report §7)

- [x] xy-vs-xyz decomposition on the 50-video sample — **~92% of pose "motion" is
  z-axis noise** (pose-head 99%, legs 95%; hands only 24%, face 14%). Cached at
  `cache/motion_analysis/xy_vs_xyz_sample50.parquet`, chart in the report.
- [ ] Re-run the **global** scope with xy-only RMS (store `rms_speed_xy` alongside
  `rms_speed` in the chunk schema) so landmark-importance numbers at full-dataset
  scale aren't z-contaminated.
- [ ] Consider adding the xy/xyz split to `compute_motion_energy` itself (cheap —
  same smoothed array, second reduction) before any re-run.

### 1.7 Explicitly out of scope here

- Within-class / cross-class ANOVA-style discriminability analysis (separate
  future task — this notebook only produces its motion-energy inputs)
- Gradient saliency / SHAP (needs a trained model; this is pre-training analysis)
- Spectrogram-format conversion

---

## 2. Bulk Landmark Extraction (POPSIGN)

**Decision (resolved):** extracted landmarks go to **`data/raw/popsign/{train,test}`**,
rooted at the separate drive configured via `POPSIGN_LANDMARKS_DRIVE` in `.env`
when set (fallback: `src/data/`, gitignored) — too large to live next to the code.

### 2.1 Extraction module — `modules/dataset/landmark/extraction.py` (2026-07-16)

Replaces the deleted `modules/data/landmark_worker.py` (whose known bugs —
landmarks never written to the npz, stale hardcoded model path and output dir —
must not be reproduced):

- [x] Per-video MediaPipe Holistic extraction saving **all** landmark groups
  (face + pose + both hands → one `(T, 543, 3)` float16 npz in GISLR holistic
  row order + fps/num_frames metadata), atomic writes (temp file + `os.replace`).
  Smoke-validated 2026-07-16: 211-frame video → npz with 7% NaN, hand/pose/face
  blocks populated.
- [x] Multiprocess pool **capped at ≤70% of CPU / RAM** (worker count from
  `cpu_count × 0.70`, workers pinned to 1 math thread, RAM backpressure via
  `psutil`); MediaPipe is CPU-only on Windows so GPU is not touched.
  Pool + Windows-spawn path and resume-skip validated on a 4-video run.
- [x] Output layout `data/raw/popsign/{train,test}/<label>/<id>.npz`, root
  resolved via `POPSIGN_LANDMARKS_DRIVE` (fallback `src/data/`, gitignored).
- [x] **BUG (2026-07-19, found by the first pilot): leaked MediaPipe graphs
  deadlocked the pool.** POPSIGN mixes resolutions (1944×2592 and 1080×1920);
  the persistent per-worker landmarker raises `RET_CHECK ... current_mat->rows
  == previous_mat->rows` on a resolution change, and the retry handler rebuilt
  it by **rebinding `_LANDMARKER` without `.close()`** — leaking a native graph
  and ~70 threads each time. The wedged worker reached **219 threads / 1.3 GB**
  (vs 76 / 614 MB for its siblings), then stopped at **0% CPU**; the run stalled
  at 18/20 with no error, because `imap_unordered` cannot tell a live-but-hung
  worker from a slow one. An overnight 30K-video run would have hung silently.
  Fixed three ways: `_reset_landmarker()` closes before rebuilding; the
  landmarker is rebuilt **proactively on a resolution change** (checked once per
  video, so the exception path isn't used at all); `maxtasksperchild=64`
  recycles workers as a safety net. Verified on the exact failure sequence
  (2592 → 1920 → 2592 ending on the two hung videos): 4/4 in 21.7 s, 0 failed.
- [x] `extract_popsign.py pilot` **clears the temp tree before benchmarking** —
  `extract_dataset` skips done videos, so leftover npz from an interrupted pilot
  would time an empty trial and report a meaningless throughput.
- [ ] Consider a per-video **watchdog timeout** in the driver. The three fixes
  above address the known cause, but nothing yet bounds an unknown one: a worker
  that stops returning still hangs the whole run indefinitely.

### 2.2 Extraction driver — `src/popsign.0.dataset.extraction.ipynb` (2026-07-16)

Replaces the deleted `popsign.0.dataset.ipynb` stub as the extraction driver
(`popsign.1.mediapipe.ipynb` stays stale pending retirement, §0.1):

- [x] Manifest verification (`data/cache/dataframes/{train,test}.csv`) — both
  verified 2026-07-16 (30,867 train / 33,600 test rows, unique ids, spot-checked
  paths exist; train covers 72 labels = the 1 enabled dataset).
- [x] **Pilot batch (≤100 videos) cells built**: seeded sample, worker-count
  sweep (6/10/14/19) on disjoint 20-video slices under the 70% cap, measured
  videos/s + frames/s + CPU%, ETA + disk projection (`cache/popsign_extraction/`).
- [x] Resumable bulk-extraction cells with progress bars + QC section
  (interruption-safe over ~30K videos) — §1.2 manifest pattern (per-unit artifact
  before `done`, atomic saves, `failed` retried, batched manifest rewrites).
- [x] Pilot output moved to the temp tree (2026-07-18): npz →
  `data/temp/popsign_pilot/w<N>/`, deleted by the notebook's cleanup cell once
  `eta.json` is recorded in `data/cache/popsign/extraction/`.
- [x] **Extraction runs in the notebook (2026-07-19).** §2/§3/§4 call
  `extract_dataset` directly — the "pool can't run in Jupyter" guard was
  disproved and removed (§2.3). Two supporting fixes made it safe:
  the leaked-graph deadlock (§2.1), and **`_init_worker` redirecting each
  worker's fd 2** to `<out_dir>/<split>/_worker_stderr.log`. That second one is
  not cosmetic: MediaPipe logs from C++ straight to fd 2 and ipykernel captures
  fd-level output into cell output by default (`IPKernelApp.capture_fd_output`),
  so a 30K-video run would have written hundreds of lines *per worker* into the
  `.ipynb` — the 17 MB-notebook failure mode of §0.2.
  Verified by executing the notebook's own cells in a real ipykernel
  (setup → manifests → verify → pilot → ETA → cleanup): 0 failed,
  **0 noise lines, ~2 KB of cell output total**.
  `modules/scripts/extract_popsign.py` stays as an **optional** CLI for
  unattended runs; both share the manifests and are resumable, so they can be
  used interchangeably.
- [x] **Manifests regenerated 2026-07-19** by the notebook's new §1 cell —
  30,867 train / 33,600 test, verified against the raw tree (ids unique, every
  label agreeing with the sign encoded in its filename, spot-checked paths all
  present). The pilot is **no longer blocked**.
- [x] All 4 POPSIGN train dataset parts are now downloaded (2026-07-21) — see
  §0.6: `train-n-s-signs` → `D:/datasets/…` (complete 07-20 17:19),
  `train-t-z-signs` → `E:/datasets/…` (complete 07-21 16:29), `a-e`/`f-m`
  already in the default kagglehub cache. ~650GB that was outstanding all last
  week is now on disk.
- [ ] **Regenerate the train manifest** — `data/cache/popsign/dataframes/train.csv`
  was last written 2026-07-19 17:51, *before* the 3 new parts finished, so it
  still reflects only the old 72-label / 30,867-video single-part state. Consequence
  today: **train still covers only 72 labels while test covers all 250**, so 178
  test labels have no train videos yet — purely because the manifest hasn't been
  regenerated, not because the video is missing. Re-run §1 of
  `popsign.0.dataset.extraction.ipynb` with `FORCE_REGENERATE = True` before
  starting the train bulk run.
- [x] Run the pilot (user), review videos/s + resource headroom, then run the
  bulk extraction for train + test:
  ```
  .venv/Scripts/python.exe src/modules/scripts/extract_popsign.py pilot
  .venv/Scripts/python.exe src/modules/scripts/extract_popsign.py run train --confidence default
  ```
  **Test split: done 2026-07-20 02:06** — 33,599/33,600, **1 failed** (unchanged
  from initial report — investigate below), 1.461 videos/s wall, 8 workers,
  6.4 h total wall time. Output in `src/data/raw/popsign/test/`
  (`POPSIGN_LANDMARKS_DRIVE` still unset, §0.6); confirmed the resolution-change
  fix (§2.1) holds at scale across the full 33.6K videos, zero deadlocks.
  **Train: not started** — 0/30,867 done, blocked on the manifest regeneration
  above. A second pilot run on 2026-07-20 (post test-split, pre-manifest-regen)
  measured only 0.246–0.365 videos/s at 4–8 workers — **roughly a quarter of**
  what the test bulk run sustained at 8 workers. Likely explanation: the pilot's
  window overlaps the `train-n-s-signs` download finishing on the same drive
  that day, but this hasn't been confirmed — **re-run the pilot with no
  concurrent downloads before picking a worker count for the train run.**
  Full write-up: `docs/logs/weekly/2026-30.md` §3.
- [ ] Investigate the one failed test video —
  `gtsignstudy4a.8035-into-2023_01_30_12_00_12.563-0`, `cv2` cannot open the source
  mp4. Likely a truncated/corrupt download rather than an extraction bug; the
  manifest retries `failed` on the next run, so confirm the source file first.

### 2.4 Output inspection — `src/popsign.0.dataset.output-inspection.ipynb` (2026-07-19)

- [x] Standalone read-only diagnostic showing how an extracted sign is stored:
  archive keys/shapes/dtypes, the 543-row holistic group layout, per-group
  detection rates, one frame + a presence timeline, and the reference
  npz→model-input loader (NaN→0, subset gather, uniform subsample to
  `MAX_SEQ_LEN`) mirroring `modules/model/data.py`.
  **Safe to run against a live extraction by construction**: no writes anywhere in
  the landmarks tree, no worker pool, no MediaPipe import, and `.tmp.npz` staging
  files are excluded from sampling so a half-written video can never be opened.
  Format recorded in `docs/logs/daily/2026-07-19.md` §4c.

### 2.3 Extraction-quality / confidence tuning — `src/popsign.0.dataset.confidence-tuning.ipynb` (2026-07-19)

**Goal:** before committing ~30K videos of CPU time to bulk extraction, find out
which `HolisticLandmarker` confidence thresholds actually produce good landmarks
on POPSIGN video — with both a numeric score and a visual check, because no
ground-truth landmarks exist for this dataset.

**Sample:** 50 videos = 5 classes × 10 videos, seeded, recorded to the cache so
every re-run scores the same clips. Videos are globbed straight off the raw
video drives — this notebook deliberately does **not** depend on the missing
`data/cache/popsign/dataframes/` manifests (§2.2), so it is unblocked today.

- [x] **Finding (2026-07-19, measured): `min_hand_landmarks_confidence` is inert
  and the *pose* thresholds are what gate the hands.** Driving each field to
  0.01 vs 0.99 on one clip: the hand threshold produces **bit-identical**
  landmarks, while `min_pose_{detection,landmarks}_confidence` swing hand
  detection rate from **0.52 → 0.09**, and the face thresholds at 0.99 drop the
  face entirely (91% NaN). Cause: holistic derives hand ROIs from the pose
  landmarks. The naive grid (sweep the hand threshold) would have produced a
  table of identical rows and read as "tuning doesn't matter".
  Note also: the task API has **no** `min_tracking_confidence` and no separate
  hand *detection* threshold — the real field list is
  `extraction.CONFIDENCE_FIELDS`.
- [x] Threshold grid over the fields that actually move the output — pose
  detection/landmarks confidence (jointly and one at a time) plus a face-off
  arm — each config extracted over the 50-video sample, resumable per
  (config, video) via the §1.2 manifest pattern.
- [x] **Numeric quality proxies** (no ground truth, so these are proxies and are
  documented as such): per-group detection rate (fraction of frames with a
  non-NaN block), hand-presence rate, temporal jitter (median frame-to-frame
  landmark displacement — high = flicker), longest detection gap, and
  bone-length coefficient of variation (a rigid bone such as shoulder-elbow
  should keep constant length; variance = detection instability).
- [x] Composite `quality_score` per (config, video) combining those proxies, with
  the weighting exposed as a tunable so the ranking can be re-derived without
  re-extracting.
- [x] **Visual test**: 100 rendered frames with landmarks overlaid — 10 from the
  best-scoring frames and 80 from the worst-scoring (plus a 10-frame
  median-quality strip for reference), written to
  `data/cache/popsign/confidence_tuning/overlays/`, contact sheets shown inline.
- [x] Supporting module work: `extraction.py` gained `CONFIDENCE_FIELDS` /
  `DEFAULT_CONFIDENCE` and a `confidence=` parameter threaded through
  `extract_dataset` → pool initializer → landmarker (unknown fields assert);
  new `modules/dataset/landmark/quality.py` (proxies + composite score) and
  `overlay.py` (landmark drawing, frame rendering, contact sheets).
  Smoke-validated end to end 2026-07-19 on 2 configs × 2 videos.
- [x] **RESOLVED 2026-07-19: "`multiprocessing.Pool` cannot run in a Jupyter
  kernel" was wrong, and the guard has been removed.** Two measurements
  retired it:
  1. `multiprocessing.spawn.get_preparation_data` only sets `main_path` when
     `__main__` has a `__file__`. A kernel has none, so the child **never
     re-imports `__main__`**; `sys_path` *is* propagated and the worker
     functions live in an importable module. (The classic Jupyter+spawn failure
     is about workers defined *in the notebook* — ours are not.)
  2. A real ipykernel driven over ZMQ ran the pool end to end: 4 videos,
     2 workers, 0 failed, **21.6 s** — the same wall time as the CLI, through
     the resolution sequence that used to deadlock.

  The hang that motivated the guard was real but is far better explained by the
  **leaked-graph deadlock in §2.1**, which wedges a worker at 0% CPU with no
  error and which `imap_unordered` cannot detect. That is now fixed.
  The other objection — MediaPipe flooding cell output — is handled by
  `_init_worker`'s `stderr_log` fd-2 redirect (see §2.2). Extraction therefore
  runs **in the notebook**; `extract_popsign.py` remains as an optional CLI for
  unattended runs that should outlive the kernel.
  Superseded sub-items (kept for history):
  - `extraction._assert_pool_usable` refuses the case up front with
    instructions, so it fails in a second instead of hanging indefinitely.
  - `n_workers=1` now runs genuinely **in-process** (no Pool at all) — the only
    mode usable in-notebook, fine for a smoke test.
  - `extract_dataset(bar=...)` accepts a caller-owned tqdm so a multi-config
    sweep still shows ONE progress bar (the old per-config bar also meant the
    display could not move until an entire 50-video config finished).
  - **`modules/scripts/tune_confidence.py`** is the supported way to run the
    sweep; notebook §4 is now a handoff cell that prints the command and
    reports per-config progress from the npz on disk. Same work: ~100 videos in
    ~10 min as a script vs **0 files in 30 min** in-notebook.
- [x] **`popsign.0.dataset.extraction.ipynb` had the same defect** — its pilot
  (§2) and bulk (§3/§4) cells called `extract_dataset` in-notebook and would now
  raise instead of hanging. **Resolved 2026-07-19**: new
  `modules/scripts/extract_popsign.py` (`pilot` / `run <split>` subcommands,
  resumable, `--confidence` naming a tuning arm, `--limit` for staged runs);
  the three notebook cells are handoff cells that print the exact command and
  read progress back off the split manifest. `CONFIDENCE_CONFIG` in the setup
  cell is threaded through, so the bulk output records which thresholds
  produced it.
- [x] **Report written: `docs/reports/confidence-tuning.md`** (2026-07-19) —
  covers the inert-hand-threshold finding, the default-vs-`pose_strict` paired
  comparison, and the padding finding below.

**Sweep results so far (2026-07-19): 2 of 7 arms measured** (`default` 50/50
videos, `pose_strict` 49/50). Full write-up in the report; the operating config
is **not** yet chosen, and the sweep should not simply be finished as-is:

- [x] **Thresholds barely move the output, and `default` leads.** Paired over the
  49 videos both arms extracted, `pose_strict` costs 0.022 of any-hand detection
  rate (worse on 31 videos, better on 5) and buys 0.002 less hand jitter.
  Composite score 0.160 vs −0.163 — but with two arms the z-scored score is ±1
  by construction, so it is directional only.
- [ ] **BLOCKER — the proxies are measuring clip padding, not extraction quality.**
  Hand presence peaks at **0.86** mid-clip and sits at 0.12–0.19 across the first
  and last fifths; the median clip's first hand detection is at 27% of its
  duration and its last at 72%. Restricted to that span the same extraction
  scores **0.85 mean / 0.94 median** rather than 0.427. `longest_gap_frames`
  correlates with `n_frames` at **rho 0.84** — that proxy is very largely a
  measurement of the lead-in/lead-out. The effect being ranked (~0.02) is an
  order of magnitude below the artifact (~0.4). **Restrict the proxies to the
  signing span (or add `*_span` variants) before scoring anything else**, then
  re-derive the comparison above.
- [ ] Follow-up: `pose_rate` was 1.0 in every arm tested, including
  `min_pose_*_confidence = 0.99` — the pose block appears to be emitted
  whenever *any* pose is found, so the proxy can't discriminate pose quality.
  Either find a per-landmark visibility signal or drop `pose_rate` from the
  composite score's weighting (it currently contributes a constant offset).
- [ ] Then run the remaining five arms (`pose_permissive`, `pose_very_permissive`,
  `pose_det_only`, `pose_lm_only`, `face_off`) — ~250 extractions, ~25 min at 19
  workers — and record the chosen config as `CONFIDENCE_CONFIG` in
  `popsign.0.dataset.extraction.ipynb` (and as the default in `extraction.py`).
- [ ] **Separate deficiency, bigger than any threshold: only 1.1% of frames carry
  both hands** (left 9.5%, right 34.3%), and `hand_rate` tops out at exactly 0.50
  across the sample — the signature of "exactly one hand, always". Several
  sampled signs (`car`, `bath`) are two-handed in ASL. This is about *which*
  landmarks holistic returns and no confidence threshold addresses it. Inspect
  the `default` overlay frames before accepting any config.
- [ ] **Downstream consequence** (not a §2.3 item, filed here so it isn't lost):
  ~50% of every POPSIGN clip is non-signing lead-in/lead-out. Trimming, or a
  learned attention over the signing span, belongs in the POPSIGN
  feature-building stage.

---

## 3. Data-Driven Landmark Importance

- [x] Motion energy (feeds from §1) — delivered; keep/discard recommendation in
  `docs/2026-07-15.md` §4 (keep: hands 42 + upper-body pose 8 + lips 40 + eyes/
  nose 36 = **ME-126**; discard: 392 face, pose head/legs, z channel).
- [x] Within-class consistency + cross-class discriminability — **delivered
  2026-07-16** (`docs/2026-07-16.md`): per-landmark **ANOVA F-ratio** +
  **mutual information** on per-video descriptors, **probe classifier** as the
  subset score. Verdict: **ME-126 wins** (49.9% global probe; FP-118 48.6%,
  FULL-543 last at 40.6%); pose divergence adjudicated in ME's favor;
  discriminability is ~uncorrelated with motion energy (rho −0.12). Caveat
  discovered: marginal F cannot rank face landmarks either (rigid-head
  redundancy) — the subset-level probe is the instrument that prices it.
- [ ] Position as complementary to gradient saliency and SHAP from trained models
  (not a replacement)

### 3.0.1 Landmark-reduction findings write-up (2026-07-22 remark)

**Blocked on input from the user:** the remark says the rationale, alternatives
and next steps are "already locked in inside the draft paper," but no paper
file exists in this repo and the link wasn't provided when asked (2026-07-22).
Placeholder filed so it isn't lost — fill in the URL/doc and re-derive this
item once available, since the actual next steps may differ once the paper's
existing content is known:

- [ ] Get the draft paper location (Google Doc / Overleaf / other) from the
  user, then reconcile its landmark-reduction section against what's already
  written here (`docs/reports/motion-energy.md`, `docs/reports/subset-comparison.md`)
  before drafting anything new, so the write-up doesn't contradict or duplicate
  decisions already locked in.
- [ ] Findings write-up should cover, at minimum: why reduce at all (streaming
  latency/model-size budget — `CLAUDE.md` "streaming viability drives
  everything"), the evidence trail (motion-energy z-noise finding →
  discriminability probe → ME-126 selection, §1/§3), the subset-comparison
  cross-check against the Kaggle 1st-place subset (agreement + divergence
  points, §1.6), and honest caveats (motion-energy computed pre-normalization,
  §7.7 flags this may shift once §7.2 lands).
  Possible solutions/next steps to include: the pending global xy-only re-run
  (§1.8), the face-anchor-reduction candidate (§3.0), and the normalization-first
  re-validation of ME-126 vs FP-118 (§7.7).
- [ ] Decide the destination: a new `docs/reports/landmark-reduction.md`
  (this repo's existing convention) vs content destined for the external
  paper — depends on what the paper already contains.

### 3.0 Landmark-subset registry + comparison notebook (2026-07-16)

- [x] `src/modules/dataset/landmark/subsets.py` — canonical registry of every
  landmark subset in play (FULL_543, FP_118 = 1st-place, ME_126, ME_132,
  HANDS_42, HANDS_POSE_50, plus component groups) with holistic row indices —
  `ME_126.array` verified equal to the trained run's `landmarks.npy`.
- [x] `src/gislr.0.dataset.subset-comparison.ipynb` — **executed end-to-end
  2026-07-16** (scope A 10 videos / scope B 10 classes / scope C global 189
  chunks, 0 failures; global descriptors ≈50 min, probes ≈7 min). All 6
  registered subsets scored; `probe_acc_global` written back into
  `subsets.py`; report `docs/2026-07-16.md`.
- [~] Feed the winning subset + per-landmark rankings into the §3.1 training
  ablations (probe predicts: pose helps, pose-wrist points {17-22} don't) —
  top-3 probe subsets queued in the rebuilt `gislr.1.model.gru.ipynb`
  (2026-07-16), awaiting user run.
- [ ] Candidate new subset: **face-anchor reduction** (eyes/nose 36 → ~8 rigid
  anchors) — the face's discriminative signal is one rigid transform; needs a
  trained ablation before admission to the registry (report §4).
- [ ] Note for feature work: `x_std` is the most discriminative descriptor
  (median F 25.1 vs 17.3 for speed) — causal running-std features are
  streamable and worth a §3.1-style ablation.

### 3.1 Landmark-subset training ablations (GRU, all-else-identical)

Controlled runs that change ONLY the input subset vs the full-543 baseline
(historical run `20260713-213000`, val acc 70.59%).

**NOTE (2026-07-18):** all runs below predate the registry reset (§0.4) — their
weights are gone, so "canonical evals pending" can no longer be satisfied for
them. Their train-loop numbers stand as historical references; the ablation grid
should be re-run under regime v2 into the fresh registry before drawing final
subset conclusions (the run-to-run-variance caveat below makes this doubly true):

**2026-07-16:** `src/gislr.1.model.gru.ipynb` overhauled into the subset-ablation
driver: trains the top-3 probe subsets (`ME_126`, `ME_132`, `FP_118`) as
all-else-identical runs (per-subset caches + auto-resume + auto-generated run
docs), with `TRAIN_SUBSETS`/`COORDS` as the only knobs — the xy ablation below
is now a one-line config change. Awaiting user run.

- [x] **ME-126, xyz** — done 2026-07-15 (`src/models/gislr/gru/20260715-190729`):
  **73.73% val vs 70.59% baseline (+3.14) with 50.4% fewer params** (0.95M),
  failing classes 22→9. Leaderboard updated in `src/models/README.md`.
  (A reproducibility re-run is included in the rebuilt notebook's default
  `TRAIN_SUBSETS`; drop it there to save ~25 GPU-min.)
- [~] Exact 1st-place 118 (ME-126 minus the 8 pose landmarks) — isolates whether
  upper-body pose helps a *streaming* model (hand-dropout fallback hypothesis).
  Queued as `FP_118` in the rebuilt `gislr.1` notebook — awaiting user run.
- [~] **ME-132** (`ME_126` + pose wrist points {17-22}) — #2 by probe score;
  tests the probe's prediction that the extra 6 landmarks add nothing. Queued
  in the rebuilt `gislr.1` notebook — awaiting user run.
- [~] **xy only** (drop z) — tests the z-noise finding in-model. Trained
  2026-07-17 for all three subsets (v1 regime, `COORDS="xy"`): train-loop val
  acc **ME_126-xy 74.92 / ME_132-xy 74.95 / FP_118-xy 74.54 — each beats its
  xyz counterpart** (73.73 / 72.47–74.95 / 74.60), consistent with the z-noise
  finding. Canonical evals pending. `scripts/eval_gru.py` xy mode added
  2026-07-17 (reads the checkpoint's `coords` key). **Caveat**: ME_132's two
  same-config xyz runs differ by ~2.5 pts (72.47 vs 74.95) — run-to-run
  variance is on the order of the subset deltas, so ablation conclusions need
  the canonical evals (and ideally repeat runs).
- [ ] ME-126 + lag-1/lag-2 difference features (the 1st-place motion features) —
  note these are causal, so streaming-safe.

### 3.2 GRU training-regime update (batch ↑, epochs 300, early stopping) — implemented 2026-07-17

`src/gislr.1.model.gru.ipynb` now trains under regime **`v2-plateau-300`**
(no run executed yet — awaiting user):

- [x] **Batch size 192 → 512**, lr scaled 3e-4 → 1e-3 (16 GB GPU has ample
  headroom for the ~1M-param models; per-epoch time at 512 untested).
- [x] **Epochs 60 → 300 (cap) with early stopping** on val-accuracy plateau:
  no gain > `es_min_delta=1e-3` for `es_patience=15` epochs (both HYP tunables).
- [x] **Scheduler decision: ReduceLROnPlateau** (factor 0.5, patience 5, on
  val acc) — watches the same plateau signal as the stopper with shorter
  patience, so the LR gets a chance to rescue a plateau before the run ends.
  OneCycleLR dropped (fixed-epoch anneal would be truncated by early stops).
- [x] Comparability: `metadata.json`/`hyp.json`/checkpoints carry
  `training_regime` (`v1-onecycle-60` backfilled for all 8 prior runs;
  `index.csv` has the column). v1 vs v2 runs are not
  hyperparameter-comparable; canonical eval split/metric unchanged.
- [x] Resume-safety: plateau counter + scheduler state persist in the
  checkpoint (`epochs_since_gain`, `finished`); resolve_run_dir resumes only
  unfinished runs (finished = early-stopped or cap reached).
- [ ] Run the v2 regime (user) — start with the best subset — and compare
  against its v1 counterpart on the canonical eval before adopting v2 as the
  default family.

---

## 4. Architecture Benchmarking

GISLR architecture-benchmark notebooks (filed 2026-07-17): one flat notebook
per architecture (`gislr.1.model.<arch>.ipynb`), reusing the `gislr.1.model.gru.ipynb`
pattern — per-subset feature caches, canonical split, auto-resume, one
timestamped run folder per training start, auto-generated
`data.md`/`README.md`/`metadata.json` so every run lands in `src/models/index.csv`.
Train on the best-known subset for comparability with the GRU runs.

- [x] **`gislr.1.model.lstm.ipynb`** — unidirectional `StreamingLSTM`
  (streaming-viable, direct cell-vs-cell GRU comparison; 1.24M params at
  hidden 256×2). **Built 2026-07-17, trained 2026-07-19** — best 0.7453 (lstm
  ME_126/xy). **Status was stale here** (said "awaiting user run" though the
  18-run sweep in `docs/logs/daily/2026-07-19.md` already covers all 3
  subsets) — corrected 2026-07-22.
- [x] **`gislr.1.model.bilstm.ipynb`** — `BiLSTM`, offline-only **accuracy
  reference, never a deployment candidate** (prices the causality gap;
  fwd-last + bwd-first readout, 3.0M params). **Built 2026-07-17, trained
  2026-07-19** — best 0.7526 (bilstm FP_118/xy), also the **largest train/val
  gap** of the four archs (0.22–0.24 vs gru's 0.155) — read in the 07-19 log
  as "most capacity + can see the future ⇒ memorizes hardest," not as a
  feature/architecture win. **Status was stale here** — corrected 2026-07-22.
- [x] **`gislr.1.model.cnn1d.ipynb`** — `CausalConv1D`: 5 dilated causal
  Conv1d blocks (kernel 5, dilations 1..16, per-frame LayerNorm, 125-frame
  receptive field ≈ MAX_SEQ_LEN; 1.86M params). Streaming-viable; first step
  toward the 1st-place port. **Built 2026-07-17, trained 2026-07-19** — but at
  the crippled `num_layers=2` config (§5.5.1 bug), so its 0.5414 best is not a
  valid architecture result; **re-run still pending** (§5.5.1) with the fixed
  5-layer config before it's comparable to the other three.

### 4.1 BiLSTM investigation (2026-07-22 remark — scoped as diagnostic only)

- [?] **Conflict to flag, not silently resolve:** the remark asks to "figure out
  why BiLSTM is performing better" and "try increasing the depth of the
  model." Two problems with taking that at face value:
  1. **The premise is already stale.** As of the 07-19 canonical evals, GRU
     (ME_126/xy, 0.7566) leads the leaderboard — BiLSTM's best is 0.7526. BiLSTM
     was ahead only in the interim 07-17→07-19 window before GRU's best run.
  2. **Deepening BiLSTM for accuracy conflicts with `CLAUDE.md`'s "streaming
     viability drives everything"** — BiLSTM is explicitly "only ever an
     accuracy reference, never a deployment candidate" because it's
     bidirectional (needs the whole sequence, can't run causally frame-by-frame).
     Chasing its accuracy by adding depth doesn't move the deployable model
     forward and risks quietly re-centering the project on a model that can
     never ship.
  - [ ] **If the goal is understanding the causality gap** (legitimate, already
    partly answered by the memorization read above): quantify it properly —
    train/val gap by architecture, params-controlled comparison (BiLSTM at
    GRU-equivalent param count), and whether the gap is genuinely "sees the
    future" signal or just "has more capacity." This is diagnostic, feeds §7.1,
    and doesn't require adopting BiLSTM for anything.
  - [ ] **If the goal is a higher-accuracy *deployable* model**, depth should go
    into GRU/LSTM/CausalConv1D (the streaming-viable three), not BiLSTM —
    consistent with the existing plan (§7 plateau-breaking phases, §4's ST-GCN/
    TCN/Transformer/Conformer evaluation).
  - Needs a decision from the user on which of these two this remark meant
    before either sub-item is actioned.

All three verified 2026-07-17 by CPU smoke test: forward shapes correct,
future-frame corruption provably doesn't change logits for the two causal
models, and notebook state_dicts load into `scripts/eval_gru.py`'s classes
with identical logits (the script now dispatches on the checkpoint's `arch`
key and handles xy/xyz via its `coords` key).
- [ ] ST-GCN, TCN, Transformer, Conformer — evaluate against the recurrent baselines
- [ ] Caution: 1st-place GISLR Kaggle solution found hand-crafted angle/distance
  features didn't help and GCNs underperformed simpler sequence models — keep this
  in mind when scoping the ST-GCN evaluation

### 4.2 Full 1st-place-solution recreation (2026-07-22 remark)

**Not a new item** — this is already tracked, split across three places: the
1D-CNN+Transformer port (README "Still planned" line, this section), the
normalization scheme cross-check (§7.7), and the landmark subset it uses
(FP_118, already in the registry, §3.0). The remark's "figure out how they
did it and recreate it, they hit 89%" bumps it in priority; consolidating
so it isn't chased as three separate untracked efforts:

- [x] **The 1st-place notebook itself is no longer in the working tree** —
  `src/gislr.0.competition.entry.1st.ipynb` (referenced in §0.2) was removed
  in commit `f7be9a1`. **Recovered 2026-08-23** with
  `git show fd1c7aa:src/gislr.0.competition.entry.1st.ipynb`.
- [x] **Read in full 2026-08-23** (all 28 cells), not re-derived from memory.
  What it actually is, and the correction it forces to the plan below: the
  architecture is **one of five changes**, and not the biggest one.
  - **Landmarks**: `POINT_LANDMARKS` = LIP 40 + LHAND 21 + RHAND 21 + NOSE 4 +
    REYE 16 + LEYE 16 = **118 — already registered as `FP_118`**. Its `POSE`
    list is defined but *commented out* of the gather.
  - **Normalization**: translation by the clip-mean position of **raw holistic
    landmark 17** (a lip point) — **not** the shoulder-centre §7.2 assumes —
    then divide by the per-channel std over all frames and landmarks. **Both
    statistics are whole-sequence, i.e. NOT causal.**
  - **Features**: xy (z dropped *after* normalizing) + lag-1 + lag-2
    differences = **6 channels/landmark = 708 input dims**. The differences are
    **forward** (`dx[t] = x[t+1] - x[t]`) — a 2-frame lookahead, trivially
    flipped to causal.
  - **Augmentation** (6, on raw coordinates): temporal resample 0.5-1.5x,
    left/right mirror with full landmark-pair swap, random affine
    (scale/shear/rotate±30°/shift), temporal crop to 384, temporal mask and
    spatial mask — both of which write **NaN**, so NaN must survive to
    training time.
  - **Architecture**: 2 stages of (3x Conv1DBlock + TransformerBlock), dim 192.
    Conv1DBlock = expand x2 -> **causal** depthwise conv k=17 -> BatchNorm ->
    ECA channel attention -> project, residual dropped per-sample
    (stochastic depth). Readout = **global average pool** over the sequence.
  - **Regime**: RAdam + Lookahead(5), cosine one-cycle 300-400 epochs **no
    early stopping**, decoupled wd 0.1, label smoothing 0.1, LateDropout 0.8
    from epoch 15, and **AWP** (adversarial weight perturbation, λ=0.2) from
    epoch 15.
  - **Streaming verdict**: OFFLINE-ONLY. The convolutions are causal, but the
    global-average readout and the unmasked self-attention are not, and neither
    is the normalization. Registered `streaming: false`.
  - **The ~89% is a 4-seed ensemble trained on ALL 94,477 videos** and scored on
    the Kaggle LB — not a single model on a held-out split. A single run on our
    canonical 90/10 split should be expected around 0.84-0.88.
- [x] **Ported 2026-08-23** — code written, CPU-smoke-validated, **not yet run**:
  - `modules/model/features.py` — NaN-preserving cache, the normalization, the
    lag features, all 6 augmentations, mirror-permutation builder (asserts the
    subset is closed under left/right swap; FP_118, ME_126, ME_132 all are),
    dataset + collate padding to the **batch max** rather than a fixed 384.
  - `modules/model/architectures.py` — `Conv1DTransformer` + `ECA`,
    `CausalDWConv1D`, `Conv1DBlock`, `TransformerBlock`, `LateDropout`,
    `MaskedBatchNorm1d`; registered in `ARCHS` as `conv1d_transformer`
    (`streaming=False`). `build_model` now forwards arch-specific HYP keys that
    a model class declares (existing four architectures verified unchanged).
  - `modules/model/optim.py` — Lookahead, AWP, cosine one-cycle.
  - `modules/model/train_fp.py` + `src/config/gislr.firstplace.json` — the
    driver and its config; same canonical split, registry and meta.json schema.
  - `modules/scripts/eval_gru.py` — dispatches on the checkpoint's `features`
    key so these runs are scored through the 1st-place preprocessing.
  - **`src/gislr.1.models.firstplace.ipynb`** — the driver notebook.
- [x] **Run it (user)** — done 2026-08-23, run `1787483814`, 300 epochs in
  2.0 h. **It diverged**; see the next item. Canonical eval of the surviving
  checkpoint: **0.7459** overall / 0.7433 macro / 0.7632 median / 13 classes
  below 50%. That is *below* the 0.7565 GRU and is not a measurement of the
  recipe — it is a model 15 epochs into a 300-epoch cosine.
- [~] **Re-run it with the fixed driver (user).** Full write-up:
  `docs/logs/daily/2026-08-23.md`. What happened and what changed:
  - **The run died at epoch 15** — the exact step `awp_start_epoch` and
    `late_dropout_start_epoch` both fire. Train loss 2.01 -> 5.79 and then
    pinned at ln(250) = 5.52 (a uniform predictor) for ~45 epochs; val acc
    0.746 -> 0.017; recovered only to 0.179 by epoch 300. **285 of 300 epochs
    trained a dead model.** Not a NaN (the reference's documented failure) and
    not a plateau: `best.pt` vs `last.pt` shows `stem.weight` norm 16 -> 224
    and `stem_bn.running_var` 4.8 -> 8467, with every downstream BatchNorm
    scale collapsed toward zero to suppress it.
  - **Cause 1 — `grad_clip: 0.0`** (faithful to the reference, fine without
    AWP). AWP steps from a gradient measured where every weight tensor was
    pushed 20% of its own norm *up* the loss surface. Now `1.0`.
  - **Cause 2 — AWP's adversarial forward was updating the BatchNorm running
    statistics.** That pass exists only for its gradient, but a `train()`
    forward also writes batch stats into the running buffers, so eval-time
    statistics were fed perturbed-weight activations twice per step. Fixed by
    `modules/model/optim.py::frozen_bn_stats`, used by `train_fp.py`. The
    reference's Keras AWP has the same flaw.
  - **Stopping conditions added to `fp-onecycle-300`** (it deliberately had
    none): `collapse_ratio`/`collapse_patience` (val acc below 50% of the run's
    own best for 3 consecutive epochs — replayed on this history it stops at
    epoch 18), `es_patience`/`es_min_delta` (30 / +0.001, the ordinary plateau
    stop, generous because a one-cycle cosine gains late), and a non-finite-loss
    stop. `meta.json` gains `training.stop_reason` ∈ {completed, plateau,
    collapse, nan}; README's schema table updated.
  - **Notebook**: §0.4 rewritten around this failure, §5 documents the stops,
    §6a marks the AWP/LateDropout switch epoch on every curve and warns when the
    best epoch lands at or before it, and a **new §7b** prints per-class
    best/worst, the confused-pair table and the confusion matrix from the cached
    predictions.
  - **The re-run collapsed too.** `1787492560`, same config plus all three
    changes: val acc 0.7415 at epoch 15, 0.039 at 16, stopped by the guard at
    epoch 18 after **9.1 min** (against 300 epochs / 2.0 h for the same failure).
    The guard works; the fixes only softened the blow-up (epoch-16 loss 5.08 vs
    5.79). From that epoch average the model survives ~20 steps into epoch 16
    before dying — a systematic runaway, so clipping was treating the wrong
    shape of problem.
  - [ ] **Run notebook §5b (user, ~30 min) — the next action on this item.**
    Epoch 15 switches on AWP (λ=0.2) *and* LateDropout (p=0.8) and neither has
    ever been run alone. Three arms, same seed/data/LR schedule, each capped at
    epoch 25: `awp_delta: 0` · `late_dropout: 0` · control. Whichever survives
    past epoch 16 is the innocent one. Readings and the fix implied by each are
    tabulated in §5b itself.
  - New config key **`stop_after_epoch`** (0 = off) ends a run after N epochs
    *without* changing the cosine schedule — lowering `epochs` to 25 instead
    would put epoch 15 at ~15% of peak LR and answer a different question.
  - **Until §5b returns, no number from this architecture means anything**, and
    a second full 2 h §5 run is a coin flip. The GRU feature ablation (below) is
    independent of all of it and is the item that can still move the deployable
    models.
- [ ] **Registry housekeeping**: `src/data/models/1787473998/` contains only
  `assets/landmarks.npy` (an aborted start, no `meta.json`) and makes
  `build_model_index.py` warn on every rebuild. Delete it or give it a meta.
- [x] ~~Port the architecture as a new `gislr.1.models.training.ipynb` section,
  not a standalone notebook~~ — **superseded 2026-08-23.** That instruction
  assumed only the architecture differs. It does not: the features, the
  sequence handling (no uniform subsample to 128), the NaN policy and the
  augmentation are all incompatible with the shared notebook's data path, and
  the shared notebook exists specifically to enforce *all-else-identical*
  across architectures. Injecting this run would destroy that property. It
  therefore has its own notebook + config, but keeps the canonical split, the
  registry and the meta.json schema, so the leaderboard still compares like
  with like.
- [ ] Feed findings into §7.7's cross-check once §7.2 normalization is
  implemented — don't re-normalize twice. **Note §7.2 needs correcting**: the
  reference point is a lip landmark (17), not shoulder-centre, and the scale
  reference is the clip's own std, not inter-shoulder distance.
- [ ] **Highest-value follow-up: ablate the feature pipeline on the GRU.**
  Normalization alone, then normalization + lag features, all else identical
  (§7.2/§7.3). The features are architecture-independent; the architecture is
  not deployable. If most of the gap comes from features, the *streaming*
  models get most of it — that is the result that matters for this project.
- [ ] Causal variant of `Conv1DTransformer` (masked attention + last-frame
  readout + `diff_mode: "backward"` + a running-statistics normalizer) — turns
  an offline reference into a deployment candidate and prices the causality gap
  for a modern architecture.
- [ ] **TFLite export does not cover this architecture.**
  `modules/model/keras_export.py` rebuilds GRU/LSTM/BiLSTM/CausalConv1D in
  native Keras; exporting the port needs Keras equivalents of Conv1DBlock /
  ECA / TransformerBlock. Not needed to measure accuracy, required before any
  Kaggle submission of this model (§6.3).

---

## 5. Spectrogram-Format Checkpoint (CNN/ViT arm)

- [ ] xyz-as-RGB channels, landmarks on y-axis, frames on x-axis
- [ ] Linear interpolation to fixed frame count
- [ ] Scoped only to this benchmarking arm — image quantization deferred to
  spectrogram-build time, not baked into the shared checkpoint format

---

## 5.5 Training consolidation — one notebook, one config (2026-07-19)

Four `gislr.1.model.<arch>.ipynb` notebooks each carried their own `HYP` dict, so
"all else identical" — the premise of both the architecture comparison (§4) and
the subset ablations (§3.1) — was a manual chore across four files.

- [x] **`src/gislr.1.models.training.ipynb`** replaces all four: shared setup /
  config / split / feature-cache sections, then one markdown+code section per
  architecture, then cross-architecture comparison and the eval handoff. Each
  architecture section re-reads the config from disk, so it is independently
  re-runnable. The four old notebooks are removed (git history ≤ `2d7f668`).
- [x] **`src/config/gislr.training.json`** is the source of truth for every
  hyperparameter, read at run time by **`modules/model/config.py`**.
  Architectures inherit `shared`; a deviation must be an explicit `overrides`
  entry, surfaced by §2 of the notebook and by `TrainingConfig.overrides_for`.
  Validation rejects: unknown architectures, unknown per-arch keys, an override
  naming a key absent from `shared` (a typo can't become a silent no-op),
  missing required HYP keys, bad `coords`, wrong `schema_version`. All six
  failure modes tested.
- [x] `modules/model/train.py::train_from_config(arch)` — what each section
  calls; trains every subset for that architecture and prints the resolved
  hyperparameters plus any overrides.
- [x] Feature caches are built **once** for every (subset, coords) pair the
  config needs, instead of once per notebook.
- [x] `report.comparison_row` now includes `finished`, and the comparison
  section separates interrupted runs from finished ones — an interrupted run
  was otherwise indistinguishable from a bad architecture.

### 5.5.1 Bug this consolidation exposed — cnn1d's receptive field

- [x] **The hand-sync that made all four `HYP` dicts identical also flattened
  `cnn1d`'s `num_layers` from 5 to 2.** For `CausalConv1D` that parameter is the
  number of dilated conv *blocks*, i.e. the receptive field:
  `1 + (kernel-1) * sum(2^i)` = **125 frames at 5 blocks** (dilations 1,2,4,8,16,
  matching `MAX_SEQ_LEN=128`) but only **13 frames at 2**. A 13-frame window
  cannot see a whole sign.
  **This is almost certainly the whole explanation for cnn1d's ~0.54 vs ~0.75**,
  and it means the "1D-CNN is 20 points behind" reading in the 2026-07-19 log is
  an artifact, not an architecture result. Restored as an explicit
  `overrides: {"num_layers": 5}` in the config, with the reasoning in its
  `notes` field.
- [ ] **Re-run cnn1d** with the corrected receptive field before drawing any
  conclusion about the architecture (all three subsets; ~1.7M params now).
- [ ] Once re-run, revisit `docs/logs/daily/2026-07-19.md` §1.3, which currently
  reports the crippled numbers.
- [ ] Consider making the receptive field an explicit, asserted quantity in
  `CausalConv1D.__init__` (e.g. warn when it is much shorter than
  `MAX_SEQ_LEN`), so a future misconfiguration fails loudly rather than
  training quietly at a fraction of the intended context.

---

## 6. Evaluation, Export & Kaggle Submission (GISLR)

**Location:** `src/gislr.2.models.evaluation.ipynb` — the single place where
**all** GISLR model evaluation and submission happens. The `gislr.1.model.*`
notebooks are training drivers only; they no longer carry export code.

### 6.1 Evaluation notebook (2026-07-19)

- [x] **Export code removed from `gislr.1.model.gru.ipynb`** (§7, cells 13–16 —
  the only training notebook that had it) and rehomed, arch-generic, in
  `modules/model/export.py` + the evaluation notebook.
- [x] **DuckDB leaderboard**: glob every `data/models/*/meta.json`, filter
  `dataset = 'gislr'`, ordered by accuracy (canonical `overall_accuracy` first,
  falling back to the training-loop `train_val_acc` for un-evaluated runs).
  DuckDB reads the meta.json files directly — `index.csv` stays the committed
  snapshot, not the query path.
- [x] **Learning curves**: the best 5 models overlaid on one loss+accuracy figure
  (train and val), sourced from each run's `assets/history.json` so the figure
  never needs the gitignored checkpoints.
- [x] **Confusion matrices**: the top 5 runs individually (250×250), then one
  row-normalized confusion matrix aggregated over **all** evaluated gislr runs
  (where the models agree on a mistake, the confusion is a property of the
  data/labels, not the architecture — this is the Phase-1 §7.4 instrument).
- [x] **Most-confused pairs** table extracted from the aggregate matrix (feeds
  §7 Phase 1.4).
- [ ] Run the notebook (user) once the fresh-registry runs have canonical evals.

### 6.2 Supporting module work (2026-07-19)

- [x] `modules/model/architectures.py`: every arch gains `forward_full(x)` —
  a batch-1, unpacked, ONNX-friendly forward used only by export. Parity against
  the packed training forward is asserted at export time.
- [x] `modules/model/export.py`: arch-generic ONNX → TF SavedModel → TFLite chain
  + `submission.zip` packaging + a validation pass using the grader's exact
  calling convention (raw `(T, 543, 3)` with NaNs in, `(250,)` out). NaN→0 and
  the landmark-subset gather stay **inside** the exported graph.
- [x] **The ONNX route was abandoned; export now goes through a native Keras
  rebuild** (`modules/model/keras_export.py`, 2026-07-19). **All 4
  architectures export**, all under the 40 MB cap:

  | arch | tflite | keras parity | tflite parity |
  |---|---|---|---|
  | gru | 3.44 MB | 4.1e-06 | 3.8e-06 |
  | lstm | 4.48 MB | 3.5e-06 | 3.8e-06 |
  | bilstm | 11.06 MB | 3.2e-06 | 2.9e-06 |
  | cnn1d | 2.89 MB | 6.1e-06 | 5.4e-06 |

  Why the ONNX route was dropped — five distinct failures, in order:
  1. `onnx2tf` declares **no dependencies of its own** (needs `tf-keras`,
     `onnx-graphsurgeon`, `sng4onnx`, `ai-edge-litert`).
  2. `torch.nan_to_num`'s infinity handling emits an ONNX **`IsInf`** op it
     cannot convert.
  3. It permutes 3D input layouts (emitted a model wanting `(T, 3, 543)`);
     needs `-kat inputs`.
  4. **Op coverage**: only **lstm** converted at all — **gru** died inside
     onnx2tf's own GRU handler (`tf.split(tR, 3)` on a 256-dim, expects 768),
     **bilstm** on "mixed types to Tensor", **cnn1d** on a squeeze.
  5. The one arch that did convert then failed **TFLite** conversion on a
     malformed `Squeeze` onnx2tf generated (`squeeze_dims` size > 8).

  `onnx`, `onnx2tf`, `onnxruntime`, `onnx-graphsurgeon`, `sng4onnx`,
  `ai-edge-litert` and `tf-keras` are all **removed from `pyproject.toml`**.
- [x] **Two parity gates** guard the weight transfer (a gate-order or bias
  mistake would convert cleanly and predict garbage): `keras_parity` (rebuild vs
  `forward_full`) and `tflite_parity` (the **final .tflite** vs PyTorch through
  the whole deployed path, raw `(T, 543, 3)` with NaNs). Conventions handled:
  GRU gate reorder `[r,z,n]`→`[z,r,h]` with `reset_after=True`; LSTM's two
  biases summed into Keras' one; LayerNorm epsilon forced to PyTorch's 1e-5
  (Keras defaults to 1e-3); Conv1d `(out,in,k)`→`(k,in,out)`.
- [x] Two bugs the gates caught, both fixed:
  - **int8 quantization**: `tf.lite.Optimize.DEFAULT` shifted logits by ~1e-1.
    Now off by default — these models are 3–11 MB against a 40 MB cap, so it
    bought nothing (`export_tflite(quantize=...)` if that ever changes).
  - **Uninitialized resource variables**: saving the Keras-backed module
    directly produced a TFLite model that died at invoke with
    `READ_VARIABLE ... variable != nullptr` inside the RNN's WHILE loop.
    Weights are now frozen to constants — and since freezing via
    `from_concrete_functions` loses the output *name* the grader indexes by
    (`output["outputs"]`), the frozen function is re-wrapped in a module that
    re-declares the exact signature.
- [x] `modules/scripts/eval_gru.py` refactored so `evaluate_run(run_dir)` is
  importable (the CLI is a thin wrapper), and it now also writes
  `assets/val_predictions.npz` (labels + preds) — that file is what makes the
  confusion matrices cheap and reproducible.
- [x] `modules/model/train.py` writes `assets/history.json` every epoch.

### 6.3 Submission tracking — meta.json schema v3 (2026-07-19)

Kaggle allows **100 submissions/day**, and every trained model should be
evaluated on the real test set, so submission state has to be part of the run
record rather than something remembered by hand.

- [x] **Schema v3** adds a `submission` object, deliberately dataset-agnostic
  (`tested` means "scored on the held-out/official test set", whatever that
  means for the dataset — no Kaggle vocabulary in the required keys):

  ```json
  "submission": {
    "tested": false,        // scored on the official/held-out test set yet?
    "platform": null,       // "kaggle" | "local" | … (null until tested)
    "submitted_at": null,   // ISO-8601
    "public_score": null,   // official metric (Kaggle public LB accuracy)
    "private_score": null,
    "reference": null,      // kernel slug + version, run id, … — free-form
    "notes": ""
  }
  ```

- [x] `registry.py`: `SCHEMA_VERSION = 3`, `submission` in `REQUIRED_KEYS`,
  `SUBMISSION_DEFAULT`, and `write_meta` preserves an existing submission block
  across the training loop's per-epoch rewrites (same protection the canonical
  metrics already had). `mark_tested()` records a result.
- [x] `build_model_index.py` exposes `submission_tested` / `submission_platform`
  / `public_score` columns and a `--untested` filter.
- [x] Backfill: every pre-v3 `meta.json` in the registry gets the default
  submission block (idempotent migration in `registry.migrate_meta`).
- [x] **Submission queue**: DuckDB globs all meta.json, filters
  `dataset = 'gislr' AND submission.tested = false`, `LIMIT 100` — so each run of
  the cell submits only untested models and respects the daily cap by
  construction. Exports each to `submission.zip`, submits, marks `tested`.
- [x] Declare `kaggle` in `pyproject.toml` and `uv sync` — done 2026-07-19.
- [~] Submission mechanics. The `kaggle` **CLI** path submits through a Kaggle
  kernel (`-k <owner>/<notebook> -v <version>`), so each zip must be attached to
  a kernel version first — and there are currently **no credentials on this
  machine** (`~/.kaggle/kaggle.json` absent, `KAGGLE_USERNAME` unset), so a
  non-dry-run submit can only fail or hang.
- [~] **Kaggle MCP server** (offered 2026-07-19) — likely the better path: it
  exposes `mcp_kaggle_start_competition_submission_upload` +
  `kaggle_mcp_submit_to_competition`, i.e. **upload a file and submit it
  directly**, with no kernel-version dance. `.mcp.json` added at the repo root
  using the **OAuth variant** (`npx mcp-remote https://www.kaggle.com/mcp`, then
  call the server's `authorize` tool) so that **no API token is stored at rest**
  — `.mcp.json` is committed and is not gitignored. Token auth is the fallback:
  add `--header "Authorization: Bearer ${KAGGLE_MCP_TOKEN}"` to the args and set
  that variable in the environment, never inline.
  - [ ] Authorize the server from an **interactive** session (OAuth cannot run
    in a non-interactive one), then submit one model by hand end to end.
  - [ ] Once proven, decide whether `modules/model/submission.py::submit_run`
    keeps shelling out to the CLI or the notebook drives the MCP tools instead;
    the queue query and `mark_tested` bookkeeping are unaffected either way.
- [ ] **Security**: an API token was pasted in plaintext into a chat transcript
  on 2026-07-19 and must be treated as compromised — rotate it (Kaggle
  Settings → Generate New Token) and never commit one.

---

## 7. Breaking the ~73% Accuracy Plateau

**Context:** GRU, 1D-CNN, LSTM and BiLSTM all converge to ~70–74% on the FP_118
subset. Architecture-independent ⇒ the ceiling is upstream, in
features/normalization/data, not the model. Ordered by priority: diagnose first,
then fix the highest-leverage causes, then ablate to confirm what actually
helped.

### 7.1 Phase 1 — Diagnose before changing anything

Figure out whether this is overfitting, underfitting or a data/label ceiling
*before* spending compute on new features.

- [ ] Run the canonical eval on the current best checkpoint and fill in the
  pending metrics: overall / macro / median class accuracy, `n_classes_below_50pct`
  (§6.1 leaderboard surfaces all four).
- [ ] Train/val accuracy gap at the best epoch. **train ≫ val ⇒ overfitting**,
  prioritize §7.4 augmentation + regularization; **train ≈ val, both ~73% ⇒
  underfitting the real signal**, prioritize §7.2 normalization and §7.3 motion.
- [ ] Full 250×250 confusion matrix on the val set (§6.1 produces it).
- [ ] Top 20–30 most-confused class pairs by off-diagonal mass (§6.1).
- [ ] **[Elevated to top priority per 2026-07-22 remarks]** Manually inspect a few
  sequences per confused pair (landmark-trajectory visualization) and classify
  each pair as distinguished by: **handshape only** (hand landmark
  resolution/features insufficient) · **motion trajectory only** (velocity
  features should help most) · **location on/near the body** (absolute
  position must be preserved, not normalized away — note the tension with
  §7.2). Confirmed by the 07-19 aggregate confusion matrix (`docs/logs/daily/2026-07-19.md`
  §1.2): top pairs (`awake↔wake` 0.42/0.37, `mouth↔lips` 0.31/0.23, `give→gift`
  0.30, `cut→scissors` 0.26, `goose→duck` 0.24, `listen↔hear` 0.20/0.19,
  several others) are near-synonyms/morphologically related, several confused
  **symmetrically** — a label-ceiling candidate, not just a feature deficiency.
  This is not yet formally verified as **manual** (human eyeball on raw
  sequences), only inferred from the confusion matrix — do that check.
- [x] **New (2026-07-22 remark):** cross-reference the confused-pair list above
  against the **per-class accuracy** list (the other §7.1 bullet below) to
  confirm the semantically-similar pairs are the same classes the models
  actually miss, rather than two findings that happen to coexist.
  **Done 2026-08-23 on run `1787483814`** (`conv1d_transformer`/FP_118, the
  1st-place port): **9 of the 15 worst classes** are the "true" side of a
  top-25 confused pair (`give`, `mouth`, `hear`, `sleep`, `pencil`, `look`,
  `that`, `close`, `zipper`), 14 of the worst 30. A quarter of a weak class's
  errors (median over the worst 30) land on its *single* most-confused
  neighbour. The overlap is high — added evidence for the label-ceiling read.
  **And it replicates across a completely different model**: this run uses a
  different architecture, a different feature pipeline (reference-point
  normalization + lag features) and a different regime from the 18 runs behind
  the 07-19 aggregate, yet reproduces the same pairs (`awake↔wake` 0.45/0.33,
  `pencil↔pen` 0.41/0.20, `give→gift` 0.31, `mouth→lips` 0.31, `hear→listen`
  0.29, `cat↔kitty`, `goose→duck`). Confusion that survives that much variation
  is a property of the labels/representation, not of any model.
  Re-check on the fixed re-run (§4.2), since this checkpoint is only 15 epochs
  into its schedule.
- [~] Per-class sample count vs per-class accuracy. If low accuracy correlates
  with low sample count this is **class imbalance**, and the fix is
  oversampling/class weighting, *not* feature engineering — record this
  separately. **First data point 2026-08-23** (run `1787483814`):
  `corr(n_val, accuracy) = 0.35` — positive but weak, and the canonical val set
  spans only 30–41 videos per class, so imbalance is a minor effect here rather
  than the plateau's cause. Worth re-running across several canonical runs
  before treating 0.35 as the number.
- [ ] Write the verdict up (overfitting / underfitting / imbalance / specific
  confusable pairs) — it decides which phase below runs next.

### 7.2 Phase 2 — Fix normalization (remove signer-appearance bias)

**Goal:** make all relative geometry invariant to a signer's physical proportions.

- [ ] Pick the reference landmark by **semantic identity** (shoulder-center =
  midpoint of left/right shoulder), never by "whatever index sits at position 0"
  in a reordered subset like FP_118.
- [ ] Check that reference's non-NaN rate across the dataset — needs to be ~100%;
  otherwise pick a more reliable landmark or define a fallback.
- [ ] Per-frame **translation** normalization: subtract the reference position
  from every landmark in that frame.
- [ ] Choose a **scale** reference that is also reliably detected (inter-shoulder
  distance, or a fixed bone such as shoulder→elbow).
- [ ] Per-frame **scale** normalization: divide by that distance, with an epsilon
  floor for frames where the scale landmarks are missing or coincident.
- [ ] [?] Optional log-compression on top of the scale-normalized values (not on
  raw distances) — empirical, not assumed to help.
- [ ] Re-verify the NaN policy after these transforms: missing landmarks must stay
  NaN-flagged, not silently become 0 through subtraction/division.
- [ ] Confirm the whole pipeline is **causal** — frame *t* uses only data from
  frame *t*. This feeds a streaming model; no lookahead, no whole-sequence stats.
- [ ] Re-run the motion-energy analysis (§1) on normalized coordinates and compare
  with the existing ME-126 findings — normalization may change which landmarks
  look important.

### 7.3 Phase 3 — Motion features

**Goal:** give the model velocity, which the 1st-place solution indicates was its
primary edge.

- [ ] Stacking order: normalize (§7.2) **first**, then compute motion features on
  the normalized coordinates — never on raw ones.
- [ ] Frame-to-frame velocity (first-order delta) as extra channels, causal
  (frame *t* uses *t* and *t−1* only).
- [ ] Decide and **document** the first-frame policy (zero-velocity vs repeating
  the first delta) — it changes the model's first observation.
- [ ] Concatenate position + velocity; record the new `feature_dim` (≈2×).
- [ ] Retrain **GRU** (fastest, ~15 min wall) on position+velocity vs the
  position-only baseline under identical hyperparameters.
- [ ] Only if position+velocity clearly wins: evaluate acceleration (delta-delta)
  as a third channel. Smooth positions first (reuse the Savitzky-Golay pipeline
  from §1) — raw double-differencing amplifies MediaPipe jitter. Track
  `feature_dim` growth and its inference-latency cost (TFLite, 100 ms/video budget).
- [ ] If acceleration doesn't measurably help, **drop it** rather than keeping it
  "just in case" — it costs training time and on-device latency.

### 7.4 Phase 4 — Augmentation (especially if Phase 1 showed overfitting)

- [ ] Jitter: random rotation / scale / translation **per sequence**, not per
  frame (per-frame destroys temporal coherence).
- [ ] Temporal resampling: randomly speed up / slow down a sequence to simulate
  different signing speeds.
- [ ] Random frame dropout + interpolation — simulates detection gaps, builds
  robustness to missing landmarks.
- [ ] Left/right mirroring for handedness (see §7.5 — decide augmentation vs
  canonicalization vs both).
- [ ] Retrain with augmentation and compare the **train/val gap** before vs after,
  to confirm it reduces overfitting rather than just adding noise.

### 7.5 Phase 5 — Handedness canonicalization

- [ ] Determine whether GISLR labels signer dominant hand, or whether it must be
  inferred (which hand has higher motion energy / detection rate per sequence).
- [ ] Decide the canonical handedness (e.g. always right-handed).
- [ ] Mirroring transform: flip x-coordinates **and** swap left/right landmark
  indices.
- [ ] [?] Deterministic canonicalization of all data, or random mirroring at train
  time? Different design choices with different generalization effects.
- [ ] Re-run the §7.1 confusion-matrix analysis afterwards to see whether
  handedness confusion specifically improved.

### 7.6 Phase 6 — Controlled ablations

**Goal:** isolate which change mattered instead of stacking everything and getting
an unattributable result.

- [ ] Fixed protocol for every arm: GRU, same split, same hyperparameters, same
  seed.
- [ ] **Arm A** — position-only, normalized (§7.2 alone vs the current baseline).
- [ ] **Arm B** — position + velocity, normalized (motion on top of fixed
  normalization).
- [ ] **Arm C** — velocity-only, normalized. Tests whether static pose/location is
  necessary at all. Prediction: underperforms B and possibly A (loss of
  static-hold and sign-location information) — run it to confirm, not to assume.
- [ ] **Arm D** (only if B > A) — position + velocity + acceleration, normalized.
- [ ] **Arm E** (only if §7.4 is implemented) — best arm from A–D + augmentation.
- [ ] Tabulate overall accuracy, macro accuracy and train/val gap for all arms
  side by side.
- [ ] Re-validate the winning feature combination on **one other architecture**
  (1D-CNN) to confirm the gain is feature-driven, not GRU-specific.
- [ ] Record the normalization + feature configuration **per run** in the meta.json
  schema, so future comparisons stay attributable and nobody has to re-litigate
  "was it the normalization or the velocity?" later. (Schema change — coordinate
  with §6.3; likely a `features` object alongside `subset`/`coords`.)

### 7.7 Phase 7 — Validate against prior work

- [ ] Cross-check the §7.2 scheme against the 1st-place solution's specific
  single-reference-point normalization — match the validated approach rather than
  a variant of it.
- [ ] Revisit ME-126 / motion-energy using normalized coordinates: motion energy
  computed on un-normalized data may have been biased by signer scale.
- [ ] Resolve the long-open ME-126 vs Kaggle-suggested-subset cross-validation
  (§1.6, §3) — landmark importance rankings may shift once normalization is fixed.
- [ ] Plain-language write-up for supervisor progress reporting in
  `docs/reports/plateau-breakout.md`: what Phase 1 diagnosed, what changed, what
  moved accuracy and by how much.

---

## 8. Post-Processing: Context-Aware Correction Layer (2026-07-22 remark, new)

**The idea:** instead of (or alongside) improving raw model accuracy on
semantically-confused pairs (§7.1/§3.0.1), feed predictions through a
correction LLM that uses sentence-level context to pick the right word among
near-synonyms (`awake`/`wake`, `mouth`/`lips`, etc. — the exact pairs §7.1
already identified as semantic, not geometric).

- [?] **Scope conflict to resolve before filing real sub-tasks:** GISLR and
  POPSIGN as used in this repo are **isolated single-sign classification**
  (one video → one of 250 labels), not continuous sentence recognition —
  there is currently no stage that assembles a sequence of predicted signs
  into a sentence for an LLM to have "context of a sentence" over. This
  remark presupposes that downstream stage exists or is in scope. Before
  doing anything else: is `sign2speech`'s roadmap meant to extend to
  continuous/sentence-level signing (which would need a whole new
  segmentation + sequence-assembly pipeline, well beyond the current
  per-video classifier), or is this meant as a smaller-scope idea (e.g.
  n-best/beam re-ranking within a single prediction using label
  co-occurrence stats, no real "sentence")? The two read very differently in
  scope.
  - Note also: this repo is notebook-driven ML research with **no app**
    (`CLAUDE.md`) — an LLM-correction *pipeline component* is a reasonable
    research notebook (train/eval a re-ranker), but an actual inference
    service wiring model → LLM → output would be new territory for this repo.
- [ ] If continuous/sentence-level is in scope: this is a substantial new
  workstream (data: does either dataset have sentence-level
  labels/transcripts to train or even evaluate this against? POPSIGN and
  GISLR are both isolated-sign as extracted here) — needs its own numbered
  section once scoped, not folded into §7.
- [ ] If the smaller-scope reading is intended: prototype using the existing
  aggregate confusion matrix (`docs/logs/daily/2026-07-19.md` §1.2) as a
  confusability prior — e.g. an LLM or even a simple bigram/co-occurrence
  re-ranker over the confused pairs — as a notebook-based offline experiment,
  measuring accuracy lift on exactly the pairs §7.1 identified, before
  deciding whether it's worth the added complexity/latency over just fixing
  the underlying signal (§7.2/§7.3).
- [ ] Either way, note this doesn't replace §7's normalization/motion-feature
  work — the remark itself frames it as "instead of," but a correction layer
  papering over confusable classes without first knowing whether they're a
  genuine label ceiling (§7.1) risks masking a data problem rather than
  fixing or correctly diagnosing it.

---

## 9. Reproducibility, Artifact Storage & Repo Layout (2026-09-04, new)

Filed from an external architecture review of the repo (three-tier
Git/DVC/MLflow advice, plus a proposed `signbridge/` uv-workspace layout).
The review's *diagnosis* is largely correct and its *tooling prescription* is
mostly not — see the verdicts below. This section holds only the changes that
survived being checked against the repo; the rejected/deferred ones are kept in
§9.8 with the reason, so they don't get re-proposed from scratch.

Ordering is by (value ÷ cost), cheapest first. §9.1–§9.3 are independently
useful and none of them requires the layout change.

Related: §0.4 (the 2026-07-18 restructure this builds on), §6.3 (schema v3),
§8 (the scope question that blocks the rename and the rescore/synthesis packages).

### 9.1 Provenance block — meta.json schema v4

**The fault (confirmed).** `meta.json` records what was configured but not what
*ran*: no commit SHA, no environment versions, no dataset version, no link to
the feature cache that produced the inputs. `hyperparameters` + `training.source`
+ the committed `src/config/*.json` cover part of it, but a run cannot be
rebuilt from its record. 43 run folders are in this state.

- [x] Add a `provenance` block to `meta.json` (`schema_version: 4`):
  `git_commit`, `git_dirty`, `dirty_code_paths`, `config_path`,
  `config_sha256`, `feature_pipeline` (`base_v1` | `firstplace_v1`),
  `feature_cache_key` (§9.2), `source` (`{name, kaggle_ref, version, n_videos}`),
  `env` (`{python, torch, numpy, mediapipe, platform, gpu}`).
- [x] **Don't gate on `git_dirty` alone.** Every training run in this repo starts
  from a dirty tree — the driver notebook is edited and re-run as part of
  starting the run (the working tree had `M src/gislr.1.models.firstplace.ipynb`
  when this section was filed). A blanket dirty warning would fire on 100% of
  runs and be ignored within a day. Hash **the code that actually executes** —
  `src/modules/` + the resolved config file — and warn only when *that* is
  dirty; record notebook dirtiness separately as information, not as an alarm.
- [x] `REQUIRED_KEYS` + `SCHEMA_VERSION = 4` in `modules/model/registry.py`;
  `migrate_all` backfills `provenance: null` for the 43 existing runs — unknown
  provenance must read as unknown, never be reconstructed after the fact.
- [x] Both drivers write it (`train.py` and `train_fp.py`), and
  `eval_gru.py`/`evaluate.py` (§9.7) records its own eval-time env, since the
  canonical metric is produced there, not in training.
- [x] `build_model_index.py`: new `prov_*` columns; README schema table updated
  (or generated — §9.6).

**Done 2026-09-04.** `modules/model/provenance.py` (12 fields: git commit/branch +
the two dirtiness flags, config path + hash of the values that ran, feature
pipeline, feature-cache key, dataset ref with a `train.csv` fingerprint,
environment incl. scikit-learn because it defines the split, captured_at). Wired
into both drivers, into `eval_gru.py`'s `eval_summary.json` (the canonical number
is produced there, not in training), and into `index.csv` as 16 `prov_*` columns.
`migrate_all` backfilled **42 runs to v4 with `provenance: null`**, and
`index.csv` was rebuilt — 42 rows, up from the 38 it had drifted to. One run
folder (`1787473998`) has no `meta.json` at all — an aborted 1st-place start that
only ever wrote `assets/landmarks.npy`; it is skipped with a warning, not
indexed.

### 9.2 Content-addressed feature caches

**The fault (confirmed, but narrower than the review claimed).** The cache key
is `subset_tag(subset.name, coords)` — subset *name* plus `"xy"`/`"xyz"`, and
nothing else (`modules/model/data.py::subset_tag`, and the same tag with a
`_nan_` infix in `modules/model/features.py::build_nan_cache`). Both builders
are skip-if-exists. So editing the `ME_126` index list in
`modules/dataset/landmark/subsets.py` leaves the tag unchanged and every
subsequent run silently trains on the **old** 3.2 GB array, with no record that
the definition moved.

**Correction to the review's framing:** the two feature pipelines do *not*
collide (distinct `_nan_` suffix, deliberate per §4.2), and the leaderboard
metric is *not* at risk — `eval_gru.py` reproduces the split and preprocessing
straight from raw parquet and never touches a feature cache. The hazard is to
**training inputs**, which is bad enough on its own.

- [x] Key caches by a hash of everything that determines their bytes: pipeline
  name + pipeline version, `sha256(subset.array.tobytes())`, coords, NaN policy,
  and the dataset ref — not the subset's human name.
- [x] **Migrate by rename + sidecar, not rebuild.** `data/cache/gislr/features/`
  is **29 GB**; recomputing it means re-decoding 94,477 parquets per subset.
  Compute the key for each existing file from the current subset definitions,
  rename in place, and drop a `<key>.json` sidecar recording the inputs. If a
  definition has already drifted, the rename produces a key that no config asks
  for — which is exactly the detection this task is for.
- [x] Record `feature_cache_key` in the §9.1 provenance block, so "are these two
  runs comparable on inputs?" becomes a field equality check instead of a
  promise.

**Done 2026-09-04.** `data.feature_cache_key()` hashes pipeline + version,
dataset, subset name **and its index array**, coords, NaN policy, rows-per-frame,
split strategy/seed, and the `train.csv` fingerprint; caches moved to
`features/<pipeline>/<key>/{train,val}_{data,offsets}.npy` with a
`cache_key.json` sidecar. `modules/scripts/migrate_feature_caches.py` relocated
**14 groups / 30.3 GB by rename** (dry-run by default, `--apply` to move), all 14
passing a shape check (offsets ↔ split size, bytes ↔ frames × landmarks ×
channels). Migrated sidecars are stamped `assigned_by: "migration"`,
`verified: "shape"` — the key is *asserted* from today's subset definitions,
because the old layout recorded nothing about the definitions that built it;
that is exactly the hole this closes going forward. Verified after the move:
every (subset, coords) resolves to its migrated directory with no rebuild, and a
subset redefined under the same name — including a same-count index swap, which
the shape check cannot catch — produces a different key. `subset_tag` survives as
the *human* handle (registry pointer keys, progress bars) and is documented as no
longer being the cache identity.

### 9.3 Off-machine artifact store (Cloudflare R2)

**The fault (confirmed).** `best.pt`/`last.pt` are gitignored and exist on one
Windows machine. The 2026-07-18 reset (§0.4) already destroyed 8 runs' weights
including the ME-126 result still cited in the README, and those evals can never
be completed.

- [x] `ops/sync_models.ps1` (or `.py`) — `rclone` / `aws s3 sync` of
  `data/models/*/best.pt` to R2. Current total: **707 MB across 42 runs**
  (~17 MB/run), so this is inside R2's free tier and takes minutes.
  **Built as `src/modules/scripts/sync_models.py`, not `ops/`** — a project
  Python CLI belongs in `modules/scripts/` under the existing convention, and
  creating `ops/` would pre-empt the layout change §9.8 defers. Uses boto3
  (R2 speaks S3) from a new optional `ops` dependency group, so the default
  environment stays lean: `uv sync --group ops`.
- [x] Run it at the end of every training session; document the restore path in
  the README registry section.
- [x] **Scope it to weights only.** Not the 29 GB feature caches (derivable —
  §9.2 makes that checkable) and emphatically not POPSIGN's ~870 GB of raw
  video (immutable upstream Kaggle releases; record the ref, never the bytes).
- [x] Skip DVC. Its one real advantage over this — `dvc.yaml` stage DAGs
  catching stale derived artifacts — is what §9.2 buys directly, and DVC fights
  the notebook-driven workflow for the rest.
- [ ] **Not yet run: the actual upload.** This machine has no R2 credentials
  (`.env` holds only `KAGGLE_MCP_TOKEN`) and no bucket exists yet, so `push
  --apply` has never executed. Create the bucket, put the five `R2_*` keys in
  `.env`, `uv sync --group ops`, then `push --apply` — 42 objects, 707 MB. Until
  that runs, the weights are still single-copy.

**Done 2026-09-04 (tooling).** `sync_models.py status | push | pull`, dry-run by
default, never deletes remotely. `data/models/checkpoints.manifest.json`
(committed) records size + sha256 + upload time per object, so "is this backed
up, and is it still the file I trained?" is answerable with no credentials;
`pull` verifies every download against that hash and refuses a mismatch rather
than installing a checkpoint that is not the one that was trained. Saved after
each object, so an interrupted push loses nothing. Verified end to end except
the network calls: `status` (42 local, 0 in manifest), `push` dry run (42
objects / 707 MB), and the missing-credentials path (names the exact missing
keys before touching boto3). The `.env` reader moved from
`extraction.py::_read_env_file` to `paths.py::env_value` now that two consumers
need it.

### 9.4 Landmark tensor spec + validator

**The fault (confirmed).** The npz contract between extraction and training is a
paragraph of README prose: `landmarks (T, 543, 3)` float16 NaN-where-undetected,
`fps`, `num_frames`, GISLR holistic row order (face 0–467, left hand 468–488,
pose 489–521, right hand 522–542). That row order is what makes the
`subsets.py` indices valid for POPSIGN, i.e. it is load-bearing for a
cross-dataset claim, and nothing checks it.

- [x] `modules/dataset/landmark/spec.py`: versioned `LANDMARK_TENSOR_V1`
  (row count, group offsets, dtype, NaN policy, required npz keys) +
  `validate_tensor(arr)` / `validate_npz(path)`.
- [x] Call it in `extraction.py` before the atomic write, and in every loader
  that reads an npz.
- [x] README's prose format block becomes a pointer to the spec (§9.6's rule:
  one source of truth, and it is the code).

**Done 2026-09-04.** `spec.py` holds `GROUPS` / `GROUP_SLICES` / `POSE_OFFSET` /
`NPZ_KEYS` and `spec()` (the contract as data). The row layout had been restated
in **four** places — `extraction.GROUP_LAYOUT`, `quality.GROUPS`, `subsets`'
constants, README prose — and all four now derive from the one definition;
`overlay.py`'s `from quality import GROUPS, POSE_OFFSET` still works because
those are re-exports. `validate_tensor` sits on the extraction write path
(structural only, measured **0.3 µs/call**, so 33.6k videos cost ~10 ms total)
and `quality.load_landmarks` validates on read with `dtype=None` (widening
float16 → float32 is that loader's job).

Two judgement calls worth recording:
- **T = 0 is valid.** A clip whose frames all failed to decode is a recorded
  outcome, not a malformed file.
- **An all-NaN tensor is valid too** — first written as a rejection, then
  removed: "nothing was detected in this clip" is a *quality* signal that
  `quality.py`'s detection-rate proxies exist to score, not a structural
  violation. Infinities stay rejected: they cannot come out of the pipeline and
  they silently destroy normalization downstream.

Verified: six violation classes rejected (row count, dtype, `num_frames`
mismatch, missing key, infinities, non-array), valid/T=0/all-NaN files accepted,
and — the thing that mattered after refactoring `subsets.py` — every §9.2 cache
key is byte-identical, so no migrated cache was orphaned.

### 9.5 Dataset seam in the training stack (do before POPSIGN training)

**The fault (real, but not where the review put it).** The review blamed
`dataset` appearing in filenames (`gislr.1.models.training.ipynb`,
`data/cache/gislr/`). Those are deliberate, documented conventions and the cache
subtree-per-dataset *is* the data-placement policy. The actual coupling is in
code: `modules/model/data.py` hardcodes `FEATURES_DIR = CACHE_DIR/"gislr"/"features"`,
`load_label_map` reads GISLR's `sign_to_prediction_index_map.json`,
`get_canonical_split` reads GISLR's `train.csv`, and both drivers default
`data_dir` to `gislr_dir()`. That is what doubles when POPSIGN arrives.

- [x] Introduce a dataset adapter (split builder, label map, per-sample loader,
  feature-cache root) and make `train.py` / `train_fp.py` / the eval script take
  one, with GISLR as the first implementation.
- [x] Keep the `<dataset>.<stage>.<topic>.ipynb` notebook convention as-is —
  renaming notebooks is churn that fixes nothing.
- [ ] POPSIGN's canonical split needs the same treatment GISLR's got (fixed
  seed, asserted val size) before any POPSIGN number is comparable to anything.
  This is now a `DatasetSource` entry in `modules/model/sources.py`, not a
  second training driver — `get_source("popsign")` already fails with the list
  of what that entry must provide.

**Done 2026-09-04.** `modules/model/sources.py`: `DatasetSource` bundles dir
resolver, label map, canonical split, per-sample reader, sample-path builder,
plus the identity used for cache addressing and provenance (name, kaggle ref,
manifest). `grep gislr_dir` over `train.py`, `train_fp.py` and `eval_gru.py`
now returns **nothing** — all three go through the source, and the eval script
takes its dataset from the run's own `meta.json` instead of assuming GISLR.
`data.FEATURES_DIR` became `data.features_root(dataset)` (the const stays as
the GISLR default that notebooks import), and both cache builders take a
`dataset` argument. `provenance.build` takes the source's `kaggle_ref` /
`manifest` rather than looking GISLR up itself.

Caught in review: the first wiring named the local `source`, which **shadowed
`train_run`'s existing `source` parameter** — the driver-notebook name recorded
as `training.source`. Renamed to `ds`; a regression check now asserts
`training.source` still comes out as `gislr.1.models.training.ipynb`. Verified
too: all §9.2 cache keys unchanged, both NaN policies reachable through
`read_sample`, and an unregistered dataset raises with instructions rather than
a bare `KeyError`.

### 9.6 Kill the doc/schema drift

**The fault (confirmed, with live examples).** README says its meta.json table
"is the source of truth" while `registry.py::REQUIRED_KEYS` is what actually
enforces it. Observed drift as of 2026-09-04: `data/models/index.csv` holds
**38 runs against 43 run folders**; `CLAUDE.md` says "36 runs as of 2026-07-22";
the README daily-log table listed 07-19 before 07-18.

- [x] Fix the README daily-log table ordering (2026-09-04).
- [x] `schemas/meta.v4.json` **generated from** `registry.py`, and the README
  schema table generated from the JSON Schema — so the code stays the single
  definition and the doc is a rendering of it.
- [x] `build_model_index.py --markdown`: emit the leaderboard / run-count table
  the README embeds, and regenerate the index as part of it (the index lagging
  the run folders is the recurring failure).
- [x] Leave the narrative tables (weekly, daily, reports) hand-written —
  generating prose summaries is not a drift fix, it's a worse changelog.
- [x] Refresh the stale run count in `CLAUDE.md` (done in §9.1's pass), and
  correct its claim that the README is the schema's source of truth — the
  README section is now a rendering of `registry.py::FIELDS`.

**Done 2026-09-04.** `registry.FIELDS` (key → JSON type + one-line description)
is now THE schema; `REQUIRED_KEYS = tuple(FIELDS)` so the machine check cannot
fall out of step with the documentation. `modules/scripts/gen_docs.py` renders
`schemas/meta.v4.json` (JSON Schema draft 2020-12), the README's
`<!-- generated:meta-schema -->` and `<!-- generated:registry-summary -->`
blocks, and `index.csv` — with `--check` failing on drift (verified: a hand-edit
to a generated table is caught and reported by file). `build_model_index.py
--markdown` emits the counts + leaderboard the README embeds.

All **42 run records validate against the generated schema** — required keys,
declared types, no extra keys. The three drift instances that motivated this are
gone: `index.csv` is regenerated by the same command that renders the docs, the
`CLAUDE.md` count is current, and the daily-log ordering was fixed earlier.

### 9.7 Rename `eval_gru.py` → `evaluate.py`

- [ ] The name predates everything it now does: it dispatches all five
  architectures in `ARCHS` and both feature pipelines (it branches on
  `ckpt["features"] == "firstplace"`). Rename the file and its `--help` text;
  update README, `CLAUDE.md`, `docs/`, and the notebook import
  (`from modules.scripts.eval_gru import evaluate_run`).

### 9.8 Deferred / rejected — with the reason, so they aren't re-proposed

- [?] **Repo rename `sign2speech` → `signbridge`.** Rejected *for now*, not on
  taste: the justification is bidirectionality (speech → sign), and there is no
  synthesis direction anywhere in the repo, the README, or this TODO. §8 has not
  even settled whether *sentence-level* recognition is in scope. Blocked on §8.
- [?] **The `packages/sb-*` uv workspace split.** All of `src/modules/` is
  **5,532 lines** across 20 files, single developer, no test suite, no CI. Six
  workspace members rooted at `packages/*/src/sb/<pkg>/` would add six
  `pyproject.toml`s, editable installs, and an import-root change to every
  notebook and CLI — and would break the `CWD = src/` kernel convention that
  `modules/paths.py` and the `sys.path` bootstrap in `modules/scripts/` are both
  built around. The seam the split is meant to create (shared landmark schema +
  subset indices) already exists as `modules/dataset/landmark/`, and §9.4 makes
  it enforceable without moving a single file. Revisit when POPSIGN training
  starts (§9.5 is the real preparation for it).
- [ ] **Move the registry out of `src/data/` to a top-level `registry/`.** The
  review's stated reason — that `.gitignore` negation is fragile and can be
  "silently defeated" — does not apply as written: `src/data/*` globs the
  *contents* (not the directory), so `!src/data/models/` works, and verifiably
  does today (264 files tracked; `git check-ignore` does not match
  `src/data/models/<id>/meta.json`). What remains is a naming/legibility
  argument, worth ~1 line in `paths.py` plus a `git mv` of 264 files and every
  path reference in docs and notebooks. Low value alone — bundle it with the
  layout change if that ever happens.
- [ ] **`aliases.json` + a `promote` command.** The right idea, but a promotion
  pointer needs something to promote *to*. There is no app, no deployment
  target, and TFLite export already exists in
  `gislr.2.models.evaluation.ipynb`. File it properly when a deployment target
  is real; `submission.tested` (§6.3) already covers the query that exists today.
- [ ] **MLflow as a mirror.** Recommend against. It is a second write path for
  data `meta.json` already holds, needs a server process, and its stated payoff
  (parallel-coordinates / run-comparison views for the ablation write-up) is a
  plotting cell over `index.csv`, which is already a flat, DuckDB-queryable
  table with one row per run. If the ablation chapter needs those views, add
  the plot to `gislr.2.models.evaluation.ipynb` — hours cheaper, and it cannot
  drift from the registry.
- [ ] **`apps/` (web/edge/shared-ts), `sb-synthesize/`, `sb-rescore/`
  skeletons.** No code, and in §8's case no scope decision. The review's own
  step 6 says "empty scaffolding rots" while its structure diagram creates four
  such directories; the advice is right and the diagram is wrong. Prompt
  versioning + a frozen eval set is a genuinely good idea and belongs under §8
  the moment §8's scope question is answered — not before.

---

## Backlog / Someday

- [ ] (add unscoped ideas here as they come up, promote to a numbered section once
  they have a concrete plan)

---

*Last updated: September 4, 2026 (**§9 filed**: an external architecture review
of the repo was checked against the code and split into what survives and what
does not. Confirmed and actionable: runs record no commit/env/dataset version
(§9.1), feature caches are keyed by subset *name* so a subset-definition edit is
silently reused (§9.2), 675 MB of weights exist on one machine after already
losing 8 runs (§9.3), the npz contract is prose (§9.4), the training stack
hardcodes GISLR (§9.5), README/index/CLAUDE.md have drifted (§9.6). Rejected or
deferred with reasons in §9.8: the `signbridge` rename (blocked on §8), the
six-package uv workspace (5,532 LOC total), the `registry/` move (the gitignore
negation demonstrably works), MLflow, DVC, and empty `apps/`/synthesis/rescore
scaffolding.)*

*Previously: August 23, 2026, later still (§4.2: the fixed re-run
`1787492560` **collapsed at epoch 15 as well** — the collapse guard caught it in
9.1 min instead of 2 h, so the guards work but the cause is still open. Neither
of the two switches that fire at epoch 15 has been run alone; notebook **§5b**
is a ~30 min three-arm ablation that settles it, and it is the next action.)*

*Previously: August 23, 2026 (§4.2: the 1st-place port's first run
`1787483814` **diverged at epoch 15** — the step AWP + LateDropout switch on —
and burned 285 of 300 epochs on a dead model; canonical 0.7459 from the
surviving checkpoint. Causes found (`grad_clip: 0.0`; AWP's adversarial forward
updating BatchNorm running stats) and fixed, and `fp-onecycle-300` gained
collapse/plateau **stopping conditions** + `training.stop_reason`. §7.1: two
diagnostic bullets closed — the confused-pair and per-class lists overlap 9/15,
and the semantic pairs replicate on a completely different architecture and
feature pipeline. Full write-up: `docs/logs/daily/2026-08-23.md`.)*

*Previously: August 23, 2026 (1st-place solution recreated in code — reference
notebook recovered from git history and read in full, ported as
`src/gislr.1.models.firstplace.ipynb` + `modules/model/{features,optim,train_fp}.py`
+ `Conv1DTransformer` in `architectures.py`; awaiting the user's run. Correction
filed to §7.2: the reference point is lip landmark 17, not shoulder-centre.)*

*Previously: July 22, 2026 (POPSIGN test-split extraction finished · all 4 train dataset parts downloaded, train manifest regeneration still pending · motion-energy and subset-comparison reports backfilled · filed 5 new remarks: elevated §7.1 semantic-confusion diagnostic, corrected stale §4 architecture-run status + flagged BiLSTM-depth conflict (§4.1), consolidated 1st-place-solution recreation (§4.2), filed landmark-reduction write-up pending draft-paper link (§3.0.1), filed new LLM correction-layer idea pending scope decision (§8))*
