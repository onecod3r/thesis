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
`<dataset>.<stage>.<topic>.ipynb`, `modules/` slimmed to `paths.py` + `data/`,
checkpoints+metrics merged into one per-run folder).

### 0.1 Stale imports / references broken by the restructure

- [ ] `src/gislr.1.model.gru.ipynb` imports `from modules.dataset import ...` —
  actual module is now `modules.data.dataset`.
- [ ] `src/popsign.1.mediapipe.ipynb` imports `from modules.datasets import DATASETS`
  and uses `DATASETS["ISLR"]` — `datasets/` was deleted; it's now
  `modules.paths.DATASETS` with key `"GISLR"`.
- [ ] `src/modules/` has no `__init__.py` files (the old ones were deleted) — either
  restore them or confirm namespace-package imports work from the `src/` kernel CWD.
- [ ] `src/popsign.1.mediapipe.ipynb` currently contains early **GISLR** motion-energy
  exploration code, not POPSIGN extraction — retire that content (superseded by
  `gislr.0.dataset.motion-energy.ipynb`) and rebuild the notebook as the extraction
  driver (§2).

### 0.2 Packaging / config

- [ ] `pyproject.toml`: torch/torchvision/torchaudio sit under an invalid top-level
  `[dependencies]` table — uv expects the index pinning under `[tool.uv.sources]`
  and the packages listed in `[project].dependencies`. As written, **torch is not a
  declared dependency at all** despite being the training framework.
- [ ] `pyproject.toml`: placeholder `description = "Add your description here"`.
- [ ] Type checker: `pyrefly.toml` was deleted but `pyrefly` is still a runtime
  dependency, and the dev group also pulls in `ty` — pick one, drop the other, and
  move it to the dev group.
- [ ] `.gitignore`: the bare `data/` pattern also ignores `src/data/external/`
  (the MediaPipe `holistic_landmarker.task`) and `src/data/cache/dataframes/`
  (POPSIGN manifests) — decide whether to narrow the ignore and commit those, or
  document them as download/generate-on-setup.
- [ ] `.gitignore`: still ignores old root-level paths (`cache/train_data.npy`, bare
  `gru_best.pt`) — update patterns to the `src/` layout (e.g. `src/cache/`,
  `src/checkpoints/**/*.pt`).

---

## 1. Landmark Motion-Over-Time Analysis (GISLR)

**Goal:** One robust, resumable notebook measuring how much each landmark moves
over time, at three scopes: per-video, per-category, global. Builds on existing
motion-energy pipeline findings (RMS speed, `["type","landmark_index"]` grouping,
Savitzky-Golay filtering) rather than re-deriving them.

**Location:** `src/gislr.0.dataset.motion-energy.ipynb` (exists; core drafted but
not yet executed end-to-end — statuses below reflect the drafted-vs-validated split)

### 1.0 Decision to lock in

- [x] **Confirm DuckDB as the loading layer.** Adopted — `gislr.0` builds on
  `get_duckdb_conn()` querying parquet via `CREATE VIEW ... glob`, so aggregation
  happens before pulling into pandas and peak memory stays bounded by one query's
  result, not dataset size. Revisit only if a blocker turns up.

### 1.1 Reusable core (build once, use in all three scopes)

- [~] `get_duckdb_conn()` — drafted in notebook with smoke test; not yet validated
  end-to-end. Path normalization: strip `islr_str`, normalize `\\` → `/`.
- [~] `load_landmarks_for_paths(paths: list[str]) -> pd.DataFrame` — drafted with
  single-video smoke test; the one function all three scopes call.
- [~] `compute_motion_energy(df: pd.DataFrame) -> pd.DataFrame` — drafted: group by
  `["type", "landmark_index"]` → Savitzky-Golay (window=7, polyorder=2) → RMS speed
  (not mean-squared) → tidy long format: `type, landmark_index, rms_speed, video_id[, sign]`.
- [~] `plot_motion_gridspec(df, title)` — drafted: gridspec layout (combined overview
  ordered pose → left_hand → right_hand → face, plus per-type panels), parameterized
  for single-video / category-aggregate / global use.
- [ ] Run the reusable-core cells against real data and fix whatever falls out.

### 1.2 Resumable caching / state management

- [~] Manifest per scope: `cache/motion_analysis/<scope>_manifest.json` — helpers
  drafted (`load_manifest` / `save_manifest` / `_mark`).
- [~] Idempotent write pattern: write per-unit result file
  (`cache/motion_analysis/<scope>/<id>.parquet`) **before** marking `done` in the
  manifest — drafted in `process_units()`.
- [~] Resume check at notebook start (skip `done`, retry `failed`) + per-unit
  try/except with failures logged to manifest — drafted in `process_units()`;
  verify behavior with a forced mid-run interrupt.
- [ ] Final aggregation reads only cached per-unit files (`load_cached_units`),
  never recomputes from raw parquet — validate once a scope has real cached output.

### 1.3 Scope 1 — Per-video (50 random samples)

- [~] Sample 50 video paths (fixed seed, recorded in output for reproducibility) —
  sampling cell drafted.
- [ ] Per video: load → compute → cache → plot (save PNG, don't just display inline).
- [ ] Output: `cache/motion_analysis/per_video/summary.parquet`
  (`video_id, sign, type, landmark_index, rms_speed`).

### 1.4 Scope 2 — Per-category (10 sampled sign categories)

- [ ] Sample 10 sign labels (fixed seed).
- [ ] Per category: batched load for all videos in that sign → aggregate RMS speed
  per landmark (mean + std — feeds the future within-class consistency analysis).
- [ ] Output: `cache/motion_analysis/per_category/summary.parquet`
  (`sign, type, landmark_index, rms_speed_mean, rms_speed_std, n_videos`).

### 1.5 Scope 3 — Global (entire dataset)

- [ ] Prefer in-SQL aggregation via DuckDB where possible (memory-bounded — this is
  where DuckDB's advantage over pandas matters most).
- [ ] If full in-SQL aggregation isn't feasible (Savitzky-Golay needs per-frame
  ordering, which SQL can't do cleanly), fall back to chunked batches (e.g. 500
  videos/chunk) through the same load → compute → cache path, manifest tracking
  chunk completion instead of per-video.
- [ ] Output: `cache/motion_analysis/global/summary.parquet` (same schema as
  per-category, one row set for the whole dataset).

### 1.6 Cross-scope comparison

- [ ] Overlay/side-by-side plot: per-video sample vs per-category vs global RMS
  speed per landmark — sanity check whether the samples represent the global
  pattern before trusting them for downstream landmark-importance decisions.
- [ ] Note in notebook: findings are **not conclusive** until cross-checked against
  the competition-suggested landmark subset (per report §3.1).

### 1.7 Explicitly out of scope here

- Within-class / cross-class ANOVA-style discriminability analysis (separate
  future task — this notebook only produces its motion-energy inputs)
- Gradient saliency / SHAP (needs a trained model; this is pre-training analysis)
- Spectrogram-format conversion

---

## 2. Bulk Landmark Extraction (POPSIGN)

**Decision (resolved):** extracted landmarks go to a **separate drive** configured
via `POPSIGN_LANDMARKS_DRIVE` in `.env` — too large to live next to the code.

### 2.1 Fix `modules/data/landmark_worker.py` before any bulk run

- [ ] **Bug: landmarks are never saved.** `process_video()` collects
  `pose_frames` / `lh_frames` / `rh_frames` but `np.savez_compressed()` writes only
  `fps` and `num_frames` — a full extraction run would produce empty archives.
- [ ] `MODEL_PATH_STRING` points at `tools/mediapipe/tasks/holistic_landmarker.task`,
  but the task file lives at `src/data/external/mediapipe/tasks/` — resolve via
  `modules.paths` instead of a hardcoded stale string.
- [ ] `OUTPUT_DIR` is hardcoded to `./data/landmarks/` — wire it to
  `POPSIGN_LANDMARKS_DRIVE` from `.env` (and actually populate `.env`, currently empty).

### 2.2 Bulk run

- [ ] Rebuild `src/popsign.1.mediapipe.ipynb` as the extraction driver (its current
  contents are stale GISLR exploration — see §0.1). The old `run_extraction.py`
  driver was dropped in the restructure.
- [ ] Resumable bulk extraction with QC manifests (interruption-safe over ~30K
  videos) — reuse the manifest pattern from §1.2 rather than inventing a second one.
- [ ] Regenerate/verify the video manifests (`src/data/cache/dataframes/train.csv`,
  `test.csv`) from `popsign.0.dataset.ipynb` — currently only 1 of 4 POPSIGN train
  datasets is enabled in `modules/paths.py` (the other three downloads are
  commented out).

---

## 3. Data-Driven Landmark Importance

- [ ] Motion energy (feeds from §1) + within-class consistency + cross-class
  discriminability (ANOVA-style between/within variance ratio)
- [ ] Position as complementary to gradient saliency and SHAP from trained models
  (not a replacement)

---

## 4. Architecture Benchmarking

- [ ] LSTM / BiLSTM baselines vs the existing GRU (BiLSTM offline-only — accuracy
  reference, never a deployment candidate)
- [ ] ST-GCN, TCN, Transformer, Conformer — evaluate against the recurrent baselines
- [ ] Caution: 1st-place GISLR Kaggle solution found hand-crafted angle/distance
  features didn't help and GCNs underperformed simpler sequence models — keep this
  in mind when scoping the ST-GCN evaluation

---

## 5. Spectrogram-Format Checkpoint (CNN/ViT arm)

- [ ] xyz-as-RGB channels, landmarks on y-axis, frames on x-axis
- [ ] Linear interpolation to fixed frame count
- [ ] Scoped only to this benchmarking arm — image quantization deferred to
  spectrogram-build time, not baked into the shared checkpoint format

---

## Backlog / Someday

- [ ] (add unscoped ideas here as they come up, promote to a numbered section once
  they have a concrete plan)

---

*Last updated: July 15, 2026*
