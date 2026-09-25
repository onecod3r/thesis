# TODO — signbridge

Living project TODO, organized by workstream so new tasks can be filed under an
existing section or a new one added without restructuring.

**Status legend:** `[ ]` open · `[~]` in progress · `[x]` done · `[?]` open question / decision needed · `[-]` closed without doing (obsolete, superseded or rejected — reason inline)

**How to add a task:** file it under the matching workstream section below. If it
doesn't fit an existing one, add a new `## N. <Workstream Name>` section at the end
(before "Backlog / Someday") rather than bolting it onto an unrelated section.

**POPSIGN is deprecated (2026-09-22).** Every open item under §2 (Bulk Landmark
Extraction), §10.1 (Deno/TS extractor parity), and §10.3 (`popsign_cycle`) is
closed, not pursued — see §2's banner for the decision. `data/raw/popsign/`
and `data/cache/popsign/` (~6.2 GB) have been deleted from disk; the extraction
packages (`sb-extract`, `sb-extract-ts`) and `popsign.*.ipynb` notebooks are
kept but marked paused, not deleted. GISLR is the only active dataset.

---

## Current focus (2026-09-25)

The workstream sections below are the source of truth; this is just the short
list of what is actually next, in order. Re-derived at each audit — if it looks
stale, trust the sections. Rows 1–12 are carried over from the 2026-09-04 audit
and were re-checked against the repo on 2026-09-24 (the stale-TODO audit).

| # | next action | where | why now |
|---|---|---|---|
| 0a | **§12.8 live camera fails on sentences (user, 2026-09-25).** Probes: low fps (repeated frames) → missed signs, jitter → extra signs, landscape framing → both; app-side interp + EMA + reframe measured (`live-streaming-gap.md`). **Next: user answers fps/mirror/distance, and decides Fix 1 (app) / Fix 2 (record real sentences) / Fix 3 (retrain C1 v2)** | §12.8 | the deployed model is unusable live until this is fixed |
| 0a2 | **§3.8 sign patterns**: the first variables gave no per-sign pattern; **with handshape/orientation/location/movement, every ASL-LEX parameter is recovered on unseen signs and templates reach 38.8% top-1 (was 4.9%)**; **B3 done: DTW over per-frame phonology 41.6% top-1, 15.2% from one example**, no training. **Follow-up built as §3.9 (see row 0a3)** | §3.8 | the user's current priority (2026-09-25: "start on the sign pattern first") |
| 0a3 | **§3.9 phonology front-end models, built 2026-09-25; user runs:** `gislr.1.models.training.ipynb` §4 → §8b `gru_phono` → §8c `gru_phono_raw` (PH_55, ME_134) → §8d importance → (config check) → §8e `bilstm_phono` → §10 evals. Streaming-ready (RecurrentSession parity 2e-5) with a continuous port (`gru_continuous_phono`) | §3.9 | the user's request: a streamable model from the pattern findings, subset combos + explainability, and the best offline model |
| ~~0~~ | ~~Re-run `gislr.3.streaming.continuous-eval.ipynb`~~ — **done 2026-09-24**: C1 D3 c **GER 0.293** (eval signers) vs baseline 0.507, oracle 0.221 → `docs/reports/continuous-models.md`. Optional follow-up: D5 = D3 ∪ D1 decoder for hard-cut | §12.3 | — |
| 0b | Plan §12.4 add-a-sign (enroll C-open's 20 held-out glosses from 1/5/10 examples) | §12.4 | C-open (`1790146838`) is trained and waiting; decides how custom signs (§12.7) work |
| 0c | §12.5 **web app built 2026-09-24** (`apps/web`, `apps/edge`): sign → speech runs in the browser (Holistic → LiteRT.js step model → lag-2 lattice + trigram → rule English → browser voice). Parity tests pass and a headless Chrome replay of 24 held-out streams is identical to Python. **Next (user): first camera test**: `cd apps/web && npm run dev`; check mirroring, fps, a few known sentences | §12.5 | first time real webcam landmarks reach the model |
| 0d | §12.6: next-gloss sweep **done 2026-09-24**: the trigram prior gives −6% GER (0.293 → 0.276); **noise is the real problem** (0.982 unfiltered; the confidence floor → 0.580 but 39% of noise still spoken). **Decide: retrain C1 with noise as null?** **Floor-recall experiment done (2026-09-24): on the evaluation signers the lag-2 lattice beats the floor on missed (−1.4/100 clean, −0.7 noisy), wrong and extra signs, and noise (−2.1 pts); GER 0.347→0.320 clean; about 27 frames delay. Now the web app's default rule** Clean floor deletes 511 correct signs (10.4%); `peak` scoring recovers +122 at equal errors on clean but not noisy Also: **add Cloudflare creds to `.env`** for the LLM/TTS arms; **review the 132 draft references** | §12.6 | stage 2 (prediction + fusion + noise rejection) and stage 3 (gloss → English) have no numbers on real decoding yet |
| 0e | §12.7 custom-sign feature | §12.7 | product layer over 12.4 + 12.5 |
| 0f | §13 **speech → gloss, with the real T5 checkpoint, live and tuned for speed** — **https://signbridge.onecoder1.workers.dev**. Built + deployed 2026-09-25; then, on the user's request to reduce inference time, re-tuned same day: T5 ships **mixed precision** (int8 encoder + fp32 decoder — same guard-acceptance as full fp32, ~30% smaller) with `num_beams` 4→2 and `max_length` 64→56 for the browser's `guarded` preset, all benchmarked on the real checkpoint before shipping. Checkpoint files moved from `C:\Users\Public\Downloads\...` into the repo. Headless-Chrome-verified end to end (camera stream, Holistic, T5 manifest live) | §13 | the full speech → gloss pipeline runs client-side with the real T5 refiner, tuned for latency, and is live |
| 0g | §12.5 **sign → speech camera overlay improved 2026-09-25** (user: "should show the video along with mediapipe overlay live"). Audit found the video + overlay were already both showing (verified live with headless Chrome + a fake camera device) — the overlay was just barely visible (2px pale dots, no pose skeleton). Rewrote `drawFrame` with a bright pose skeleton over the model's own ME-126 upper-body landmarks and clearer hand markers; `npm test` (11/11) and the 24/24 browser replay check still pass | §12.5 | the live camera view now reads as an obviously "live" overlay, not just technically-present dots |
| ~~1~~ | ~~Restart the Jupyter kernels, then run one short training~~ — **effectively done**: the four §12.3 continuous runs trained end to end through the restructured stack on 2026-09-23 | §9.8 | notebooks have been parsed, never executed since the move. `import modules...` is gone. This is the only unverified thing about the restructure |
| 2 | ~~Run the first checkpoint backup~~ — **done 2026-09-04**: 42 on Kaggle, local copies pruned after hash verification. Model confirmed **private** 2026-09-24 | §9.3 | was the last single-copy risk |
| 3 | **Notebook §5b: the three-arm AWP/LateDropout ablation** (~30 min) | §4.2 | the 1st-place port has collapsed at epoch 15 twice and neither switch has been run alone, so the recipe is still unmeasured |
| ~~4~~ | ~~Install deno + ffmpeg, then run the TS extractor once~~ — **deprecated 2026-09-22**: POPSIGN (the only workload this parity check gates) is deprecated; see §2/§10.1 | §10.1 | — |
| ~~4b~~ | ~~Regenerate the POPSIGN train manifest, then `popsign_cycle run --part test`~~ — **deprecated 2026-09-22**: POPSIGN deprecated, extracted data deleted from disk; see §2 | §2.2, §10.3 | — |
| 5 | **§7.2 normalization or §7.4 augmentation**, under §7.6's ablation protocol | §7.1 → §7.2/§7.4 | the diagnosis is complete: the plateau is a generalization gap (train confusion 0.012 vs val 0.273), and these are the two levers that attack one |
| ~~6~~ | ~~Re-run the evaluation notebook~~ — **done 2026-09-19** (55 runs, 37 canonical; 5 skipped for missing `best.pt`) | §6.1 | it last ran against 18 runs; only 1 of 42 run folders has a confusion matrix |
| ~~7~~ | ~~Run `gislr.1.models.landmark-importance.ipynb`~~ — **done 2026-09-18** | §3.3 | custom DNN/LSTM/GRU + full-543 engineered features + rotating k-fold — the model-derived complement to the motion-energy/probe landmark rankings (§1/§3.0) |
| ~~8~~ | ~~Run `gislr.0.dataset.motion-energy.ipynb`~~ — **done 2026-09-21**: all three scopes, 0 failed units, results in `docs/reports/motion-energy.md` §5 | §1 | — |
| 9 | **Run `gislr.1.models.training.ipynb` §§5b/6/7/8** (`gru_deep`/`lstm`/`bilstm`/`cnn1d` × 3 subsets, 12 runs) | §4.3 | closes the current-split benchmark gap — 52 of 55 registry runs are on the retired split, only `gru` has been re-run since the reset; no porting needed, the cells are already correct, just never executed |
| ~~10~~ | ~~Run `gislr.1.models.five-arch-benchmark.ipynb`~~ — **done 2026-09-21**: `bilstm` 0.7392 (offline) > `gru` 0.7380 > `lstm` 0.7286 > `cnn` 0.6696 > `dnn` 0.6485; `dnn`'s mean true-class confidence (0.24) a quarter of the rest — results in `docs/reports/five-arch-benchmark.md` | §3.7 | — |
| 12 | **Run `gislr.0.dataset.sentences-kaggle.ipynb` on Kaggle**, then Output → New Dataset (private) | §12.1 | first of the continuous-signing experiments; 12.2–12.4 all read this dataset |
| ~~11~~ | ~~Run `gislr.3.streaming.confidence-eval.ipynb`~~ — **done 2026-09-22**: fresh-start confidence is already well-calibrated (`gru` late-third 0.58); the blocker is un-reset state (bleed-through cut 96-98% by resetting), not the training objective — §11.2's retrain downgraded to optional. `docs/reports/streaming-confidence.md` | §11.1 | — |

Decisions still owed by the user, blocking real work:

- ~~**§8 scope**~~ — **answered 2026-09-24**: continuous/sentence-level, LLM +
  TTS downstream (§12.6). `sb-rescore/` is the LLM's home.
- **§4.1**: BiLSTM is the accuracy leader (0.7569) but can never ship. Is the
  goal understanding the causality gap, or a deployable model? The section
  flags this conflict and it is still unresolved.
- ~~Stale questions from the 2026-09-24 audit~~ — **all answered by the user 2026-09-24**: Kaggle token
  rotated; model private; paper = https://docs.google.com/document/d/12TNIMaL1yvOGZKazzSWrYWmiOuMg8gsneShv1_e228o; §5 re-scoped as a GISLR_Stratified
  spectrogram experiment; §7 kept; xy wins (no xyz re-benchmark); `minemy` kept.

---

## 0. Repo Restructure Follow-ups & Tooling

Cleanup left over from the move to the flat `src/` layout (notebooks renamed to
`<dataset>.<stage>.<topic>.ipynb`), plus the **2026-07-18 restructure** (§0.4:
unified `modules/model` training stack, flat epoch-seconds registry with
meta.json schema v2, `data/{raw,cache,temp,models}` placement policy,
docs daily/weekly/reports split).

### 0.1 Stale imports / references broken by the restructure

- [x] **`packages/data/` no longer exists on disk** (2026-07-16): `dataset.py`
  (`GISLRRawDataset`) and `landmark_worker.py` are gone. `landmark_worker.py` is
  superseded by `sb.extract.holistic` (§2). **Resolved
  2026-07-16:** `GISLRRawDataset` no longer needs restoring — the rebuilt
  `gislr.1.model.gru.ipynb` defines its dataset in-notebook (in-RAM arrays,
  `num_workers=0`, the pattern proven by the ME-126 training script).
- [x] `src/gislr.1.model.gru.ipynb` imports `from modules.dataset import ...` —
  **resolved 2026-07-16** by the notebook overhaul (§3.1): it now imports only
  the subset registry (`modules.dataset.landmark.subsets`) and defines the
  dataset class itself.
- [x] `experiments/extraction/popsign.1.mediapipe.ipynb` imported `DATASETS`,
  which no longer existed in any form. **Resolved by deletion** — the file no
  longer exists on disk (retired at some point before 2026-09-22, this entry
  just hadn't been closed). `popsign.2.model.ipynb`'s `tensorflow.keras`
  import is now moot too: that notebook is **deprecated along with POPSIGN**
  (2026-09-22, §2) and marked paused rather than fixed.
- [x] `src/modules/` had no `__init__.py` files — **resolved 2026-07-18**, and
  superseded 2026-09-04: the tree is now six installed packages under a `sb`
  PEP 420 namespace, so there is deliberately **no** `sb/__init__.py` — that
  absence is what lets separate distributions share the namespace.
- [x] `experiments/extraction/popsign.1.mediapipe.ipynb` had contained early
  **GISLR** motion-energy exploration code, not POPSIGN extraction. **Moot**:
  the file has since been deleted, GISLR's motion-energy work moved to
  `gislr.0.dataset.motion-energy.ipynb` long ago, and rebuilding it as a
  POPSIGN extraction driver is no longer wanted — POPSIGN is deprecated (§2).
- [x] `gislr.0.dataset.motion-energy.ipynb` — **deleted 2026-09-19** (it broke on
  2026-09-16 when GISLR moved to npz), **rebuilt 2026-09-21** against GISLR_Stratified
  npz (TODO §1.8/§7.7): xy-native per-landmark RMS speed (z dropped before the speed
  computation, not decomposed after), Savitzky-Golay jitter removal before the
  derivative, and new per-joint-angle RMS angular speed ("change of angles", reusing
  `kinematics.compute_angles`'s 28 joints). Core math factored into
  `sb.recognize.interp.motion_energy` (`landmark_motion_energy`, `joint_angle_motion`)
  + `reindex_interpolate`/`smooth_savgol` promoted to shared `geometry.py`. **Run
  2026-09-21** (all three scopes, 0 failed units) — the xy-native global run
  reproduces the old pre-npz 50-video sample almost exactly (detection rates match
  to the decimal); results in `docs/reports/motion-energy.md` §5, which now covers
  both runs.
- [ ] **BROKEN (2026-09-16): `gislr.0.dataset.subset-comparison.ipynb` no longer
  runs** — same cause (reads raw parquet via DuckDB). Its findings
  (`docs/reports/subset-comparison.md`) stand as historical results. Retire or
  rebuild on the npz format.

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
- [x] `.gitignore`: whether to narrow the `data/` ignore — **decided 2026-09-04
  by the restructure (§9.8): do not narrow it.** `data/` is ignored absolutely,
  with no negation rules, and the one thing that genuinely needed committing
  (the registry) moved out to `registry/` instead.
- [ ] Follow-on from that decision: the two ignored artifacts a fresh clone
  needs have **no one-command fetch/regenerate path** —
  `data/external/mediapipe/holistic_landmarker.task` (third-party download) and
  `data/cache/popsign/dataframes/{train,test}.csv` (regenerated from the raw
  video tree). Document or script both in the README setup section.
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
  `scripts/evaluate.py` promotes `eval_status` to `canonical` after the
  per-class eval.
- [x] Run folders are always fresh (2026-07-17): `gislr.1.model.gru.ipynb`'s
  `resolve_run_dir` no longer reuses/skips a **completed** run — every new
  training gets a new `<timestamp>` folder (timestamp = training start), even
  under identical conditions. Auto-resume still continues an *interrupted*
  run in its own folder.
- [x] Future training notebooks must write the same schema — **closed
  2026-09-04**: it is no longer a convention notebooks must remember. Both
  drivers build the record through `sb.mlops.registry`, `write_meta` asserts
  every key in `FIELDS`, and `schemas/meta.v4.json` is generated from that same
  dict (§9.6). A POPSIGN driver gets the schema by construction, because it goes
  through the `DatasetSource` seam (§9.5) rather than being a new notebook.
- [x] ~~Rebuild `index.csv` after the canonical evals of the six pending runs~~ —
  **voided 2026-07-18**: those runs' weights were deleted with the old
  `src/models/` tree during the restructure, so their canonical evals can
  never run. Their train-loop numbers stand as historical references
  (git history ≤ `3668dae` + docs/logs/daily/); the registry restarted empty (§0.4).

### 0.4 Restructure 2026-07-18 — unified stack, flat registry, data-tree policy

Executed 2026-07-18 (full write-up: `docs/logs/daily/2026-07-18.md`):

- [x] **`packages/model/`** — unified training stack (`architectures.py`
  single model-class definition shared with eval, `data.py`, `registry.py`,
  `train.py`, `report.py`); the four `gislr.1.model.*.ipynb` notebooks
  regenerated as thin drivers (identity block + `modules.model` calls).
- [x] **Registry v2**: flat `registry/runs/<epoch-seconds>/` folders holding
  only `meta.json` + `best.pt`/`last.pt` (gitignored) + `assets/`;
  schema v2 documented in README.md § "meta.json schema" (machine check:
  `sb.mlops.registry::REQUIRED_KEYS`); `meta.json` rewritten every
  epoch by the driver; **registry reset to empty** (header-only `index.csv`).
- [x] **Single progress bar per training run** (batch progress, metrics, LR,
  plateau counter in one bar) — replaces the nested-bar + per-epoch-print spam.
- [x] **`sb.core.paths`**: absolute CWD-independent tree constants
  (`RAW/CACHE/TEMP/EXTERNAL/MODELS`), lazy dataset resolution (import no
  longer downloads), `cleanup_temp()`.
- [x] **Data placement policy** applied: POPSIGN pilot npz → `data/temp/popsign_pilot/`
  with cleanup cell (stale `data/raw/popsign/_pilot` deleted); diagnostic
  caches → `data/cache/gislr/{motion_analysis,subset_comparison}`;
  manifests → `data/cache/popsign/dataframes/`; `.gitignore` reworked
  (all of `src/data/` ignored except `data/models/` minus weights/exports).
- [x] **Scripts split**: project CLIs in `packages/scripts/` (`evaluate.py`
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
- [x] First v2-regime training runs (user) to seed the fresh registry —
  **done**: the registry holds 42 runs, all `v2-plateau-300` except the six
  `fp-onecycle-300` 1st-place runs, 37 of them canonically evaluated. Note the
  FULL_543 baseline was **not** re-established under v2 — every run uses
  ME_126 / ME_132 / FP_118, so the "+3.1 pts over full-543" claim still rests
  on the pre-reset v1 numbers whose weights are gone. Filed as its own item in
  §3.1.
- [x] `popsign.2.model.ipynb` / `popsign.3.pipeline.ipynb` still predated the
  restructure (old paths, TF-era code). **Resolved 2026-09-22: retired, not
  modernized** — both marked paused/deprecated along with POPSIGN (§2), not
  worth updating for a workstream that isn't being pursued.

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
- [ ] **The log trail has a six-week hole (found 2026-09-04).** `2026-30.md`
  still carries its "in progress" marker six weeks after the week ended, and
  weeks **31–35 (2026-07-26 → 2026-08-29) have no weekly file at all** — including
  the week containing the 1st-place port (daily log `2026-08-23.md` exists with
  no weekly around it). Today's work (§9 execution + the workspace restructure)
  has no daily log yet either. Concretely:
  - [x] **Done (verified 2026-09-24: no marker left).** Close `2026-30.md` (drop the marker, final summary).
  - [ ] (**Still open 2026-09-24**: weeks 31–34 have no file; 35–39 exist.) Decide whether to backfill 31–35 or record them as "no dev work" weeks —
    do not invent narrative for weeks that had none; the honest version is a
    one-line stub per empty week and a real file for the 08-23 week.
  - [x] **Done (verified 2026-09-24: both files exist).** Write `daily/2026-09-04.md` and `weekly/2026-36.md` for the current week.
  - [ ] The deeper problem is that this is hand-maintained and drifts. §9.6
    generated the registry-derived tables for exactly this reason; the narrative
    logs were deliberately left manual, so the fix here is discipline (or a
    reminder), not another generator.
---

## 1. Landmark Motion-Over-Time Analysis (GISLR)

**Goal:** One robust, resumable notebook measuring how much each landmark moves
over time, at three scopes: per-video, per-category, global. Builds on existing
motion-energy pipeline findings (RMS speed, `["type","landmark_index"]` grouping,
Savitzky-Golay filtering) rather than re-deriving them.

**Location:** `experiments/recognition/gislr.0.dataset.motion-energy.ipynb`

**Status: ✅ executed end-to-end 2026-07-15 (pre-npz, raw `asl-signs` parquet) — all
three scopes complete, 0 failed units. Findings, stats, figures and the landmark
keep/discard recommendation are written up in `docs/2026-07-15.md`.** That version
broke 2026-09-16 (GISLR moved to npz) and was deleted 2026-09-19. **Rebuilt and
re-run 2026-09-21 against GISLR_Stratified npz** (§1.8/§7.7: xy-native RMS speed,
jitter smoothed before the derivative, + new per-joint-angle "change of angles"
scope) — all three scopes, 0 failed units, results in `docs/reports/motion-energy.md`
§5. §1.0–§1.7 below describe the original pre-npz build and are historical, kept for
the design rationale (loading-layer decision aside — see §1.8).

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
- [x] Re-run the **global** scope with xy-only RMS — **superseded rather than
  literally implemented as `rms_speed_xy` alongside `rms_speed`**: the rebuilt
  notebook computes `rms_speed` directly on xy (z dropped before the computation,
  no second xyz column to reconcile against). **Run 2026-09-21, all 94,477
  videos, 0 failed** — global xy pose-subgroup magnitudes land within ~15% of the
  old 50-video sample estimate (exact on head/hips/arms); results in
  `docs/reports/motion-energy.md` §5.1.
- [x] Added the xy/xyz split at the source rather than as a second reduction —
  `sb.recognize.interp.motion_energy.landmark_motion_energy` takes whatever
  coordinate subspace its caller passes, so xy-only *is* the primary computation
  now, not a follow-up pass over `compute_motion_energy`'s output.

### 1.9 Follow-ups from the 2026-09-21 rerun (report §5.5)

- [ ] Legs (0.0173 xy RMS) sit almost as high as arms (0.0185) at global scale —
  the old "out of frame, apparent motion is jitter" discard rationale for
  pose-legs (§3) was reasoned from xyz z-noise, not xy magnitude, and this
  doesn't fully support it. Needs the discriminability instrument (does leg
  motion correlate with sign identity?), not another motion-energy pass —
  explicitly not a revision of ME-126 on this evidence alone.
- [ ] Cross-check joint-angle *rate of change* against
  `feature-discriminability.md`'s finding that static angle *value* is the
  most information-dense feature type — is angular speed also discriminative?
- [ ] Explain the elbow L/R reversal — every other paired joint favors the
  right (dominant) hand, but `L_elbow_angle` is the single highest RMS value
  in the 28-angle table while `R_elbow_angle` sits mid-pack. Bracing/
  counterbalance motion of the non-dominant arm, or an elbow-angle-definition
  artifact (shoulder-elbow-wrist more sensitive to pose noise on one side)?
- [ ] A single-video sample is a much noisier estimate of the global *angle*
  ranking (rho 0.743) than the *landmark* ranking (rho 0.951) — any future
  angle-motion sampling should go category-level or larger, not per-video.

### 1.7 Explicitly out of scope here

- Within-class / cross-class ANOVA-style discriminability analysis (separate
  future task — this notebook only produces its motion-energy inputs)
- Gradient saliency / SHAP (needs a trained model; this is pre-training analysis)
- Spectrogram-format conversion

---

## 2. Bulk Landmark Extraction (POPSIGN) — DEPRECATED 2026-09-22

**POPSIGN is deprecated as a workstream.** Every open (`[ ]`) item below is
closed as "not pursued," not completed — kept as history, not a queue. What
changed on disk: `data/raw/popsign/` (5.6 GB, 19,899 npz across 72 labels —
one part's worth of train, extracted under the old single-part manifest) and
`data/cache/popsign/` (582 MB — manifests, confidence-tuning and pilot
outputs) were **deleted**. The ~870 GB of raw video this section originally
sized for was not actually on this machine when the decision was made — the
two train parts §2.2 records as downloaded to `D:`/`E:` and the kagglehub
dataset cache were already clear, most likely cleaned up by the ordinary
download→extract→delete cycle (§10.3) at some point this wasn't logged, or
the drives were repurposed for other work since. `sb-extract`/`sb-extract-ts`
(the extraction packages) and every `popsign.*.ipynb` notebook are marked
**paused**, not deleted — see their own docstrings/banners — in case a future
raw-video dataset or a live-camera deployment path (§10.2) reuses them.

**Original section, kept for history below.**

**Decision (resolved, historical):** extracted landmarks go to
**`data/raw/popsign/{train,test}`**, rooted at the separate drive configured
via `POPSIGN_LANDMARKS_DRIVE` in `.env` when set (fallback: `src/data/`,
gitignored) — too large to live next to the code.

### 2.1 Extraction module — `sb.extract.holistic` (2026-07-16)

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
- [-] **POPSIGN deprecated 2026-09-22.** Consider a per-video **watchdog timeout** in the driver. The three fixes
  above address the known cause, but nothing yet bounds an unknown one: a worker
  that stops returning still hangs the whole run indefinitely.

### 2.2 Extraction driver — `experiments/extraction/popsign.0.dataset.extraction.ipynb` (2026-07-16)

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
- [-] **POPSIGN deprecated 2026-09-22.** **Regenerate the train manifest** — `data/cache/popsign/dataframes/train.csv`
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
  .venv/Scripts/python.exe .venv/Scripts/sb-extract.exe pilot
  .venv/Scripts/python.exe .venv/Scripts/sb-extract.exe run train --confidence default
  ```
  **Test split: done 2026-07-20 02:06** — 33,599/33,600, **1 failed** (unchanged
  from initial report — investigate below), 1.461 videos/s wall, 8 workers,
  6.4 h total wall time. Output in `data/raw/popsign/test/`
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
- [-] **POPSIGN deprecated 2026-09-22.** Investigate the one failed test video —
  `gtsignstudy4a.8035-into-2023_01_30_12_00_12.563-0`, `cv2` cannot open the source
  mp4. Likely a truncated/corrupt download rather than an extraction bug; the
  manifest retries `failed` on the next run, so confirm the source file first.
- [x] **Pilot split into its own notebook + main extraction rewritten as a
  staged, one-dataset-at-a-time pipeline (2026-09-13).** Two changes, same day:
  - `popsign.0.dataset.pilot.ipynb` (new file) replaces the single-video pilot
    section briefly added to the extraction notebook earlier the same day.
    Instead of extracting an already-downloaded video, it lists a few candidate
    files straight from Kaggle (`KaggleApi.dataset_list_files` — metadata only,
    no download) and pulls just `N_PILOT_VIDEOS` of them individually via
    `kagglehub.dataset_download(handle, path=<file>)` — confirmed 2026-09-13
    that this fetches only that one file (3.4MB cache footprint after one
    call, not the 174GB `train-a-e` archive; a directory-prefix `path` 404s,
    so it really is one-file-at-a-time). Extracts in-process (`n_workers=1`),
    replays with `sb.extract.overlay.render_gif` (draws the `(543,3)` skeleton
    over every frame, vs. `render_frames`'s seek-and-grab PNG sampling — see
    the codec bug entry right below for why it's a GIF, not the MP4 it started
    as), then deletes only the files it downloaded.
    File-listing + per-file download + manifest construction were smoke-tested
    live against `train-a-e` (real Kaggle calls); the extraction step itself
    could not be — this machine has neither the POPSIGN manifests nor the
    MediaPipe holistic model downloaded.
  - `popsign.0.dataset.extraction.ipynb` rewritten: no more
    `resolve_datasets()` call pulling every enabled dataset onto disk at
    once. Now one stage per train part (download the whole part — still a
    bulk archive fetch, the efficient path — walk its tree into a manifest
    slice, extract via the existing CLI handoff, verify, **delete its
    kagglehub cache**, next part) followed by one test stage (downloaded
    **once**, then extracted in 4 `--limit`-based checkpoint "quarters" —
    per-file quartering was measured at ~5s/file, i.e. ~12h just in download
    overhead for one quarter, so quartering is an extraction cadence, not 4
    separate downloads; cache is deleted only after all 4 quarters finish).
    `sb.core.paths` gained `train_dir(index)` (download one part instead of
    all of `train_dirs()`), `dataset_cache_dir(handle)` / `clear_dataset_cache(handle)`
    (kagglehub's own cache layout, resolved without triggering a download —
    never touches GISLR, which lives under a separate `competitions/` cache
    subtree). `train.csv` is now the concat of per-part manifest slices
    (`dataframes/train_parts/*.csv`), rebuilt every time a part is added, so
    `sb-extract run train`/`test` needed **no CLI changes** — already-`done`
    rows for a part whose cache was since cleared are just skipped (`pending_jobs`
    only checks the npz artifact, not the source video). The manifest-walk,
    cumulative-CSV-rebuild, and per-part/per-split progress-tracking helpers
    were unit-tested against a synthetic 2-part fixture (not real Kaggle data)
    and pass; the actual multi-hundred-GB download+extract+delete stages are
    unverified end-to-end (would take hours-to-days per part and this machine
    lacks the holistic model, same limitation as above).
- [x] **BUG (found 2026-09-13, same day): the pilot's overlay video didn't
  play — blank 0:00 placeholder in the notebook.** Root cause: this machine's
  OpenCV/FFmpeg build has no working H.264 encoder (`libopenh264` DLL not
  installed; `avc1`/`H264`/`X264` fourccs all silently fall back to a
  non-decodable stream — confirmed by testing all four), and its one working
  fallback, `mp4v` (MPEG-4 Part 2), isn't a codec browsers decode inline, so
  `IPython.display.Video(embed=True)` renders nothing. Fixed by replacing
  `overlay.render_video` (MP4) with `overlay.render_gif` (animated GIF — no
  codec dependency, displays inline anywhere `IPython.display.Image` does).
  Full-resolution GIF measured 30-40MB on a real 211-frame/120fps clip (GIF
  compresses photographic content poorly), so `render_gif` downscales
  (`max_width=360`), subsamples (`max_frames=30`), and palette-quantizes
  (`palette_colors=64`) — same real clip: 2.3MB, verified inline-playable.
  Verified live end-to-end against the real `train-a-e` file downloaded for
  the earlier smoke test.
- [x] **Two follow-up bugs found running the pilot notebook for real
  (2026-09-13):**
  - The `## 4.` markdown cell's source was written with literal `\n`
    two-character sequences instead of real newlines (a `NotebookEdit` call
    that didn't escape correctly), so it rendered as one run-on line instead
    of paragraphs. Fixed; a repo-wide check confirmed no other markdown cell
    in either extraction notebook has the same corruption (code cells
    legitimately contain literal `\n` inside string literals, so the check
    is markdown-only).
  - `cleanup_temp()` in the last cell raised `PermissionError: WinError 32`
    on the extracted npz — `np.load()` returns a lazy `NpzFile` that keeps its
    zip file handle open for as long as the object is referenced, and
    `pilot_npz` kept holding it across cells; Windows refuses to delete an
    open file. Fixed by loading inside a `with np.load(...) as _zf:` block in
    the extract cell and storing a plain `{"landmarks":..., "fps":...}` dict
    instead of the `NpzFile` object (the overlay cell's `_z["landmarks"]`/
    `_z["fps"]` indexing needed no change). Verified end-to-end with the real
    MediaPipe model now present on this machine: real extraction (211 frames,
    7.0% NaN), a 2.1MB GIF, and `cleanup_temp()` completing with no error.
- [x] **BUG (found 2026-09-13): the staged rewrite above only enabled 3 of the
  4 train parts.** POPSIGN is 5 Kaggle datasets total — `train-a-e`,
  `train-f-m`, `train-n-s`, `train-t-z` (~175-250GB each) plus test (~17GB) —
  but `ENABLED_TRAIN_INDICES` shipped as `(0, 1, 2)`, silently dropping
  `train-t-z`. Fixed: `ENABLED_TRAIN_INDICES` is now `(0, 1, 2, 3)`, and the
  notebook gained a 4th train stage (download `train-t-z`, manifest, extract,
  verify, delete cache — same pattern as parts 1-3) between part 3 and the
  test stage, renumbering test/QC from §6-§8 to §7-§9. Title cell, caveats,
  and per-stage "part N/3" labels updated to "N/4" throughout. — `experiments/extraction/popsign.0.dataset.output-inspection.ipynb` (2026-07-19)

- [x] Standalone read-only diagnostic showing how an extracted sign is stored:
  archive keys/shapes/dtypes, the 543-row holistic group layout, per-group
  detection rates, one frame + a presence timeline, and the reference
  npz→model-input loader (NaN→0, subset gather, uniform subsample to
  `MAX_SEQ_LEN`) mirroring `sb.recognize.data`.
  **Safe to run against a live extraction by construction**: no writes anywhere in
  the landmarks tree, no worker pool, no MediaPipe import, and `.tmp.npz` staging
  files are excluded from sampling so a half-written video can never be opened.
  Format recorded in `docs/logs/daily/2026-07-19.md` §4c.

### 2.3 Extraction-quality / confidence tuning — `experiments/extraction/popsign.0.dataset.confidence-tuning.ipynb` (2026-07-19)

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
  new `sb.extract.quality` (proxies + composite score) and
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
- [-] **POPSIGN deprecated 2026-09-22.** **BLOCKER — the proxies are measuring clip padding, not extraction quality.**
  Hand presence peaks at **0.86** mid-clip and sits at 0.12–0.19 across the first
  and last fifths; the median clip's first hand detection is at 27% of its
  duration and its last at 72%. Restricted to that span the same extraction
  scores **0.85 mean / 0.94 median** rather than 0.427. `longest_gap_frames`
  correlates with `n_frames` at **rho 0.84** — that proxy is very largely a
  measurement of the lead-in/lead-out. The effect being ranked (~0.02) is an
  order of magnitude below the artifact (~0.4). **Restrict the proxies to the
  signing span (or add `*_span` variants) before scoring anything else**, then
  re-derive the comparison above.
- [-] **POPSIGN deprecated 2026-09-22.** Follow-up: `pose_rate` was 1.0 in every arm tested, including
  `min_pose_*_confidence = 0.99` — the pose block appears to be emitted
  whenever *any* pose is found, so the proxy can't discriminate pose quality.
  Either find a per-landmark visibility signal or drop `pose_rate` from the
  composite score's weighting (it currently contributes a constant offset).
- [-] **POPSIGN deprecated 2026-09-22.** Then run the remaining five arms (`pose_permissive`, `pose_very_permissive`,
  `pose_det_only`, `pose_lm_only`, `face_off`) — ~250 extractions, ~25 min at 19
  workers — and record the chosen config as `CONFIDENCE_CONFIG` in
  `popsign.0.dataset.extraction.ipynb` (and as the default in `extraction.py`).
- [-] **POPSIGN deprecated 2026-09-22.** **Separate deficiency, bigger than any threshold: only 1.1% of frames carry
  both hands** (left 9.5%, right 34.3%), and `hand_rate` tops out at exactly 0.50
  across the sample — the signature of "exactly one hand, always". Several
  sampled signs (`car`, `bath`) are two-handed in ASL. This is about *which*
  landmarks holistic returns and no confidence threshold addresses it. Inspect
  the `default` overlay frames before accepting any config.
- [-] **POPSIGN deprecated 2026-09-22.** **Downstream consequence** (not a §2.3 item, filed here so it isn't lost):
  ~50% of every POPSIGN clip is non-signing lead-in/lead-out. Trimming, or a
  learned attention over the signing span, belongs in the POPSIGN
  feature-building stage.

---

## 3. Data-Driven Landmark Importance

### 3.0.2 GISLR moved to a self-produced npz dataset — canonical-split reset (2026-09-16)


- [x] **`sb.core.paths.gislr_dir()` now downloads `bracu23101281/gislr-stratified`**
  (Kaggle dataset, not the `asl-signs` competition): pre-converted `(T,543,3)`
  npz per sequence + `train.csv`/`test.csv` (fixed 80/20 split, stratified on
  `sign` only, upstream seed 42). Training/eval code updated:
  `sb.recognize.data.get_canonical_split` reads the fixed split as-given
  (N_VAL 9,448 → 18,896) instead of computing its own 90/10;
  `base_v1`/`firstplace_v1`/`evaluate.py` read npz instead of parquet.
  `sign_to_prediction_index_map.json` (not shipped by the new dataset) is
  derived and written into the resolved dir on first use.
- [x] **Fixed a real landmark-order mismatch** in the producing notebook's npz:
  it wrote `pose[0:33], face[33:501], left_hand[501:522], right_hand[522:543]`,
  not this repo's canonical gislr-holistic order (`schema.GROUPS`) that every
  `subsets.py` index array (ME_126, FP_118, …) assumes. Fixed with a permutation
  (`sb.recognize.features.gislr_stratified.CANONICAL_TO_STORED`) applied at
  read time, verified against a synthetic npz with per-group markers.
- [x] **This is a canonical-split reset**, same shape as the 2026-07-18 registry
  reset: the 37 runs canonically evaluated on the old self-computed 90/10 split
  are historical references only, not comparable to anything evaluated on the
  new one. `sb-evaluate`'s docstring, README's "Canonical evaluation" section
  and the `meta.json` schema's `split` field description are updated; old
  `meta.json` records are left as-is (their `split` block describes the split
  they actually used).
- [ ] Old `asl-signs` competition kagglehub cache deleted from disk (user request,
  2026-09-16) — confirm nothing still expects it (only `gislr_dir()` referenced
  it, and that function no longer does).
- [ ] `sb.mlops.run.source_ref`'s `version` field is still hardcoded `None` even
  though `GISLR_Stratified` is a versioned Kaggle dataset (croissant metadata
  reports version 1) — not wired up in this pass, since it wasn't blocking
  anything. Low-effort follow-up: thread the dataset's own version through
  `DatasetSource`/`P.build()`.
- [ ] (2026-09-24: motion-energy was rebuilt on npz 2026-09-21 — only subset-comparison is still broken, §0.1.) The two GISLR diagnostic notebooks that read raw parquet directly
  (motion-energy, subset-comparison) are now broken by this switch — filed
  under §0.1.
- [x] **Replaced the Kaggle submission step with local testing (2026-09-16).**
  GISLR_Stratified's `test.csv` split IS the dataset's held-out test set now,
  and the canonical eval already scores on it — so `sb.recognize.evaluate.
  evaluate_run` marks `meta.json["submission"] = {tested: true, platform:
  "local"}` itself the moment it scores a run, rather than requiring a
  separate Kaggle-competition submission (which is no longer reachable —
  there's no live `asl-signs` download to submit against). Removed §8
  "Kaggle submission queue" from `gislr.2.models.evaluation.ipynb` entirely;
  reframed §7 (TFLite export) as a plain deployment export, no longer gated
  on or framed around a submission queue. `sb.mlops.submission`'s
  Kaggle-submit helpers were **deleted 2026-09-19**; the module was renamed
  `sb.mlops.query` and keeps only the DuckDB leaderboard queries.
  README's "submission" block section and TODO/CLAUDE.md updated to match.

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

- [~] **Location given 2026-09-24: https://docs.google.com/document/d/12TNIMaL1yvOGZKazzSWrYWmiOuMg8gsneShv1_e228o** (Google Doc). Next: read it and
  reconcile its landmark-reduction section against what's already
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

- [x] `packages/sb-core/src/sb/core/subsets.py` — canonical registry of every
  landmark subset in play (FULL_543, FP_118 = 1st-place, ME_126, ME_132,
  HANDS_42, HANDS_POSE_50, plus component groups) with holistic row indices —
  `ME_126.array` verified equal to the trained run's `landmarks.npy`.
- [x] `experiments/recognition/gislr.0.dataset.subset-comparison.ipynb` — **executed end-to-end
  2026-07-16** (scope A 10 videos / scope B 10 classes / scope C global 189
  chunks, 0 failures; global descriptors ≈50 min, probes ≈7 min). All 6
  registered subsets scored; `probe_acc_global` written back into
  `subsets.py`; report `docs/2026-07-16.md`.
- [x] **Done (closed 2026-09-24): the subset ablations ran and ME-126 won; `gislr.1.model.gru.ipynb` no longer exists.** Feed the winning subset + per-landmark rankings into the §3.1 training
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
- [x] Exact 1st-place 118 (ME-126 minus the 8 pose landmarks) — isolates whether
  upper-body pose helps a *streaming* model (hand-dropout fallback hypothesis).
  **Run and canonically evaluated**: `FP_118` has runs on all four v2
  architectures. Best `FP_118` is bilstm 0.7525 vs ME_126's 0.7569 — i.e. the 8
  upper-body pose landmarks are worth roughly +0.4 pts, small but consistent
  across architectures.
- [x] **ME-132** (`ME_126` + pose wrist points {17-22}) — #2 by probe score;
  tests the probe's prediction that the extra 6 landmarks add nothing. **Run and
  canonically evaluated** on all four architectures; the probe was right — best
  ME_132 0.7476 sits below ME_126's 0.7569, so the 6 extra wrist points cost
  rather than add. ME_126 remains the leader.
- [x] **Settled (closed 2026-09-24; the item's own text says so):** **xy only** (drop z) — tests the z-noise finding in-model. Trained
  2026-07-17 for all three subsets (v1 regime, `COORDS="xy"`): train-loop val
  acc **ME_126-xy 74.92 / ME_132-xy 74.95 / FP_118-xy 74.54 — each beats its
  xyz counterpart** (73.73 / 72.47–74.95 / 74.60), consistent with the z-noise
  finding. Canonical evals pending. `scripts/evaluate.py` xy mode added
  2026-07-17 (reads the checkpoint's `coords` key). **Caveat**: ME_132's two
  same-config xyz runs differ by ~2.5 pts (72.47 vs 74.95) — run-to-run
  variance is on the order of the subset deltas, so ablation conclusions need
  the canonical evals (and ideally repeat runs). **Canonical evals now exist**
  (37 of 42 runs, 2026-09-04) and every v2 run is `xy` — the z-drop is settled
  and is the default. The repeat-run variance caveat stands and is why §3.1's
  remaining conclusions should quote a spread, not a single number.
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
- [x] Run the v2 regime (user) — **done and adopted**: `v2-plateau-300` is the
  regime for all 36 non-1st-place runs. The v1 comparison it asked for cannot be
  completed as written — the v1 weights were destroyed in the 2026-07-18 reset —
  so the comparison that exists is train-loop numbers in the daily logs, not a
  canonical one.

### 3.3 Model-derived landmark importance: custom DNN/LSTM/GRU, full-543 (2026-09-16, built; run 2026-09-18)

Directly answers the §3.0's "position as complementary to gradient saliency
and SHAP from trained models" follow-up: three trained-from-scratch
architectures whose only job is exposing what weight the model itself assigns
each of the 543 landmarks, rather than inferring it from motion energy or a
probe classifier (§1/§3.0). Deliberately **not** part of the canonical
leaderboard/registry — see `sb.recognize.interp`'s module docstring.

- [x] **Feature pipeline** `sb.recognize.interp.features` (`landmark_interp_v1`):
  full 543 landmarks (no subset), per-frame translation-invariant
  (mid-shoulder-centered) + scale-invariant (inter-shoulder-normalized)
  position, velocity, acceleration, speed (10 channels/landmark = 5430) plus a
  12-scalar relational block (inter-hand fingertip/wrist distances,
  hand-to-face-anchor distances). Content-addressed cache under
  `data/cache/gislr/features/landmark_interp_v1/`, same skip-if-exists pattern
  as `base_v1`.
- [x] **Models** `sb.recognize.interp.models`: `LandmarkDNN` (memory-free,
  classifies one frame at a time — every valid frame gets the video's label,
  video-level prediction is the probability-averaged vote over frames) and
  `LandmarkRNN` (causal GRU/LSTM, same last-valid-frame streaming contract as
  `StreamingGRU`). All three share one learned `LandmarkAttention` gate
  (per-landmark sigmoid weight) as their first layer — the direct,
  architecture-comparable landmark-importance signal, rather than reading it
  off each architecture's own internal weight shapes.
- [x] **Training driver** `sb.recognize.interp.train`: resumable, config-driven
  (`experiments/recognition/configs/gislr.landmark-importance.json`), stratified
  5-fold CV over `train.csv` (every training video validated exactly once,
  out-of-fold), then one final model per architecture fit on all of `train.csv`
  and evaluated once on the untouched canonical `test.csv`.
- [x] **Importance metrics** `sb.recognize.interp.importance`: attention gate +
  gradient×input saliency + permutation importance, combined into one
  rank-averaged `ranking_score` per landmark, grouped by region (Face / Pose /
  Left hand / Right hand, `sb.core.schema.GROUPS`).
- [x] **Notebook**: `experiments/recognition/gislr.1.models.landmark-importance.ipynb`
  — OOF per-class accuracy + confusion matrix, held-out test-set evaluation,
  landmark ranking + region heatmap, cross-architecture ranking-agreement
  (Spearman ρ). All package code (`sb.recognize.interp.*`) unit-smoke-tested
  end-to-end on synthetic data (train_fold resume, k-fold row alignment,
  saliency/permutation shapes) — the notebook itself has NOT been run: per
  CLAUDE.md, training is handed to the user to execute.
- [x] **Run it** (user, 2026-09-18) — test accuracy DNN 71.02% / LSTM 70.55% /
  GRU 67.96% (own feature pipeline + full-543, not leaderboard-comparable);
  region-level ranking hands > pose > face for all three architectures,
  cross-architecture Spearman ρ > 0.8. Full results, per-class accuracy, top
  confusions: `docs/reports/landmark-importance.md`.
- [x] Compare the three architectures' `ranking_score` against the existing
  motion-energy (§1) and discriminability-probe (§3.0) landmark rankings —
  **region-level agreement confirmed** (2026-09-18): this model-derived
  ranking independently reproduces the ME-126 hands > pose > face ordering
  from a third, unrelated method. Landmark-level correlation (Spearman ρ of
  `ranking_score` vs motion-energy RMS speed / probe F-ratio) is still open,
  filed below.
- [ ] Landmark-level correlation of `ranking_score` vs motion-energy RMS speed
  and vs the discriminability-probe's per-landmark F-ratio (region-level
  agreement is confirmed above; per-landmark is a finer-grained open question).
- [ ] **Per-axis (x/y/z/speed) saliency** — added 2026-09-18 as a report-only
  analysis script (not in the notebook or `sb.recognize.interp`), since the
  attention gate and `gradient_saliency` both collapse each landmark's 10
  channels into one number. Finding: z carries 15–24% of saliency mass vs
  36–49% each for x/y (all three architectures) — same direction as the
  motion-energy report's z-noise finding but a much smaller gap; not yet
  reconciled against the pending canonical xy-vs-xyz ablation evals (§3.1).
  If this diagnostic is worth keeping, promote it into
  `sb.recognize.interp.importance` (e.g. `channel_saliency()`) plus a
  notebook cell, so it's reproducible from the notebook itself. Detail:
  `docs/reports/landmark-importance.md` §3/§6.

### 3.4 Engineered-feature discriminability: joint angles + kinematics (2026-09-19, built + run)

A finer-grained complement to §3.0 (per-landmark position/speed descriptors)
and §3.3 (trained-model attention/saliency): a **richer, hand-crafted**
per-frame feature set — joint angles, frame-gap-aware velocity/speed,
Savitzky-Golay jitter, rolling variance, on top of the existing mid-shoulder-
centered/inter-shoulder-scaled positions and hand/face relational distances —
scored not just by ANOVA F-ratio/probe accuracy but by a new **tolerance-band
overlap** metric: does a feature's per-class `mean ± k·std` range collide with
another class's. No training involved.

- [x] **Refactor**: `sb.recognize.interp.geometry` — extracted
  `center_and_scale`/`relational_block` out of `features.py`
  (`landmark_interp_v1`, unchanged behavior) so the new pipeline reuses the
  same normalization instead of re-deriving it.
- [x] **New pipeline** `sb.recognize.interp.kinematics`
  (`landmark_kinematics_v1`): reindex + linearly interpolate every gap (one
  undetected landmark or a fully-undetected frame) over a contiguous frame
  range — frames are never dropped mid-sequence, so a derivative computed
  across a gap reads as steady motion rather than a spike, and `n_frames`/
  `valid_frame_frac` are kept as diagnostics. Savitzky-Golay smoothing
  (window 7/polyorder 2, same constants as motion-energy) on normalized
  position before deriving velocity/speed; jitter = raw − smoothed, RMS'd per
  region; rolling variance of speed, region-averaged. 28 joint angles (arm,
  wrist orientation, 5-finger ×2-joint flexion ×2 hands, palm-facing) —
  `ANGLE_SPECS`/`ANGLE_NAMES`. Reduces straight to a flat, named, per-video
  descriptor dict (mean+std per signal) — not a training tensor.
- [x] **New analysis module** `sb.recognize.interp.discriminability`:
  `f_ratio`/`probe_classifier` (same recipe §3.0 used, now reusable — the old
  `subset-comparison.ipynb` code is unreachable, broken per §0.1), plus
  `tolerance_bands`/`band_overlap` (the new metric) and
  `nearest_neighbor_margins` (a small Scope-A visual diagnostic).
- [x] **Notebook**: `experiments/recognition/gislr.0.dataset.feature-discriminability.ipynb`
  — npz-based (not the broken DuckDB/parquet path §0.1 left behind), Scope A
  (3 classes × 10 videos, eyeball check) + Scope B (15 sampled classes,
  resumable chunked manifest driver reused from motion-energy/
  subset-comparison) + feature-type/region breakdown + verdict cells. Global
  scope (all 250 classes) deliberately **not** run in this pass — filed
  below.
- [x] Smoke-tested against real `GISLR_Stratified` data (not just synthetic):
  angle geometry verified against known synthetic poses (180°/90° arm bends);
  Scope A (30 real videos) and a shrunk Scope B (~1,200 real videos, 4
  classes) both ran end-to-end with plausible output. This caught a real bug
  (see below).
- [x] **Run it** (user, 2026-09-19) — Scope A (3 classes × 10 videos) + Scope
  B (15 classes, 4,497 videos). **Angles are the most information-dense
  feature type**: 56 angle features alone reach 68.3% probe accuracy vs 79.4%
  for 3,258 position features (~50× more information per feature); adding
  angles to the full feature set gains +2.5pt (79.6% → 82.1%) for 1.1% more
  features. Top individual feature by F-ratio is `angle_R_palm_facing_mean`
  (F=163.7), beating every raw landmark coordinate. Tolerance-band overlap
  agrees on *which* features are good but is flat across types in aggregate
  (0.97–1.0); every top feature still has `worst_pair_overlap == 1.0` — no
  single feature separates all 15 classes, only the full probe does. Full
  results: `docs/reports/feature-discriminability.md`.
- [x] **Caught + fixed a real aggregation bug during analysis**: the
  type/region breakdown cell didn't apply the same `F > 0` gate the
  per-feature top-N tables did, so 33 constant (never-detected-landmark)
  `detection_rate` features + 1 constant `meta` feature were ranking as the
  *tightest-separated feature types* purely from their trivially zero-width
  bands, despite carrying zero class information. Patched the notebook cell,
  re-ran it against cached data (no new video processing), re-embedded the
  corrected output. Detail: `docs/reports/feature-discriminability.md` §2.1.
- [ ] **Region-ranking divergence, unresolved**: this pass's region ordering
  (right_hand > left_hand > face > pose, pose last) disagrees with every
  prior region ranking (§1/§3.0/§3.3 all found hands > pose > face). Plausibly
  a 15-class-sample artifact, or that this pipeline uses raw mean position
  for pose rather than the `x_std`/speed descriptors §3.0 found pose
  informative on, or that it uses all 33 pose rows rather than
  `UPPER_BODY_POSE_8`. Not distinguished yet — `docs/reports/feature-discriminability.md` §4.
- [ ] Global scope (all 94,477 videos / 250 classes) — not run in this pass,
  same staging `subset-comparison` used (A→B before committing to global);
  needed before §1's feature-type ranking or the region divergence above can
  be treated as settled rather than a 15-class snapshot.
- [ ] Promote the feature-type probe ablation (angle-only/position-only/etc.
  accuracy, `docs/reports/feature-discriminability.md` §1's table) into a
  notebook cell — currently only a supplementary analysis script, same status
  as landmark-importance's per-axis-saliency addendum (§3.3).
- [ ] `TOLERANCE_K` (currently a single global 1.5) sensitivity check before
  treating `mean_overlap` rankings as final.
- [ ] If a winning feature/angle subset emerges, feed it into a trained-model
  ablation (mirrors how §3.0's probe findings fed §3.1) — `angle_R_palm_facing_mean`
  and the finger PIP-flex angles are the strongest individual candidates.

### 3.5 Curated-feature DNN + LSTM: ME-126 + xy + joint angles, top-N eval (2026-09-19, built + run)

Acts on §3.4's own follow-up above: trains real, evaluated models on exactly
the feature recipe the prior three interpretability experiments converged
on — **ME-126** landmark subset (§1/§3.0/§3.3), **xy only** (§3.1's
established default), **+ 28 engineered joint angles** (§3.4's
highest-information-density feature type) — instead of re-deriving
importance. DNN + LSTM only (no GRU). Answered separately (no code change):
neither `LandmarkDNN` nor `LandmarkRNN`/`StreamingGRU`/`StreamingLSTM` are
"trained frame by frame" in a live-updating sense — see "Backlog / Someday"
below.

- [x] **Refactor**: `sb.recognize.interp.geometry` gained `derivatives()`
  (promoted from `features._derivatives`, bit-identical verified before/after
  on a fixed seed — `features.py`'s already-reported `landmark_interp_v1`
  output is unchanged).
- [x] **New pipeline** `sb.recognize.interp.features_curated`
  (`landmark_curated_v1`): normalizes + computes all 28 angles on the full
  543-landmark 3D geometry **first** (2 angles need a z cross-product
  component), *then* subsets to ME-126 and drops z from the raw per-landmark
  coordinate channel — z-dropping and angle-computation don't conflict
  because angles are a derived quantity, not a raw coordinate. `FEATURE_DIM`
  = 882 (126 landmarks x 7 channels: xy position/velocity/acceleration +
  speed) + 28 angles + 12 relational = **922**, an 83% reduction from
  `landmark_interp_v1`'s 5,442.
- [x] **`sb.recognize.interp.models` generalized** (additive, backward
  compatible): `LandmarkAttention`/`split_features`/`LandmarkDNN`/
  `LandmarkRNN` now take explicit dimension args (`n_landmarks`,
  `channels_per_landmark`, `angle_dim`, `relational_dim`) defaulting to
  `landmark_interp_v1`'s shape — verified the old landmark-importance call
  sites (`LandmarkDNN(hidden_sizes, num_classes)`, no dim args) produce
  identical output before/after, in eval mode (dropout makes train-mode
  comparison noisy, not a bug). New `LandmarkRNN.forward_all(x, lengths) ->
  (B, T, C)` applies the trained head to every timestep instead of only the
  last — verified its value at each sequence's own last valid frame equals
  `forward()`'s output exactly. `sb.recognize.interp.train`'s two loader
  functions gained a `feature_dim` parameter (same default-preserving
  pattern).
- [x] **Output contract**: both models produce a confidence score for every
  of the 250 signs at every frame (`LandmarkDNN.forward` already did this;
  `forward_all` gives the LSTM the same property at inference) — read
  honestly: neither was trained with per-frame supervision (see Backlog
  item below), so this is an inference-time read-out, not a new training
  claim.
- [x] **Top-N evaluation**: true label counted correct if it's anywhere in
  the top N confidences (not just rank 1) — e.g. `wake` 0.9 / `awake` 0.8
  with true label `awake` is a top-2 hit. Reuses `sb.recognize.evaluate.py`'s
  existing `topk_idx`/`topk_prob` vocabulary rather than inventing a new
  metric. A confusable-pair section (the 16 pairs from §7.1/
  `docs/reports/pair-similarity.md`) checks specifically whether the *wrong*
  member of a pair winning rank 1 gets recovered at rank 2.
- [x] **Notebook**: `experiments/recognition/gislr.1.models.curated-features.ipynb`
  — curated feature cache → model factories → **single final fit** per
  architecture (no k-fold — unlike §3.3, this doesn't need out-of-fold
  landmark-ranking coverage) → held-out `test.csv` eval + top-N table →
  confusable-pair rank-recovery table → per-frame confidence-trace plots
  (diagnostic) → summary. Config: `experiments/recognition/configs/gislr.curated-features.json`.
- [x] Smoke-tested against real `GISLR_Stratified` data end-to-end (not just
  synthetic): feature shapes + angle-block bit-match against
  `kinematics.compute_angles`; a full mini run (5-class shrink, 2 epochs,
  both architectures, real npz) through caching → training → eval → top-N →
  confusable-pair table → per-frame trace plotting, all on GPU. Caught and
  fixed one real bug: the per-frame-trace example cell crashed
  (`plt.subplots(0, ...)`) when its hardcoded example signs weren't present
  in a shrunk run's class set — added a graceful fallback.
- [x] **Run it** (user, 2026-09-19) — full 250-class final fit + held-out
  eval. **LSTM 69.09% top-1 / 79.65% top-2 / 87.53% top-5; DNN 57.76% /
  72.53% / 85.08%.** Confusable-pair classes get ~2× the top-1→top-2 lift
  of every other class (DNN +25.1pp vs +13.3pp; LSTM +21.7pp vs +9.0pp) —
  model-level confirmation of §7.1's near-miss finding. Full results:
  `docs/reports/curated-features.md`.
- [x] Trained-model confirmation of whether the curated recipe beats
  `landmark_interp_v1`/ME-126-xyz baselines on accuracy — **mixed,
  architecture-dependent**: LSTM lost only 1.46pp vs full-543
  `landmark_interp_v1` (70.55%→69.09%) despite an 83% smaller feature
  space; DNN lost 13.26pp (71.02%→57.76%) — a memory-free model needs a
  richer per-frame signal than ME-126+angles alone provides, but a causal
  model's temporal integration mostly absorbs the same cut. Single runs,
  not repeated — direction is a reasonable read, the exact ratio isn't.
- [ ] Neither model converged within its 60-epoch cap (`es_patience=8`
  never triggered) — DNN's train≈val (well-fit, likely still underfit);
  LSTM's train 86%/val 69% (~17pp generalization gap, same signature
  `plateau-diagnosis.md` found for the canonical registry runs). Re-run
  with a real stopping condition / more epochs before treating these
  numbers as the recipe's ceiling.
- [ ] `landmark_curated_v1` zero-fills detection gaps (`geometry.
  center_and_scale`'s policy, same as `landmark_interp_v1`) rather than
  interpolating them the way `kinematics.py` (§3.4) does — visible as a
  multi-frame confidence dead-zone in one of the per-frame trace examples.
  Worth an ablation: does gap interpolation change the training numbers.
- [ ] **Concrete motivation for the backlogged live-prediction idea**: one
  `scissors` test video's LSTM confidence trace peaks at ~0.95–1.0 for the
  true label mid-sequence, then a competitor (`cut`) overtakes it by the
  last frame — the frame the model is actually read out at. A non-last-frame
  readout (max-over-time / average of the last K frames) would have gotten
  this one right; untested at the aggregate-accuracy level.
- [ ] Feed the curated recipe into a canonical-comparable architecture
  (`StreamingGRU`/`StreamingLSTM`, not the interp track's `LandmarkRNN`) if
  the smaller feature space is worth pursuing as a deployment candidate.

### 3.6 BiLSTM on the current split: exact replica vs curated-feature variant (2026-09-19, run + ablation 2026-09-21)

Every registry `bilstm` run predates the 2026-09-16 canonical-split reset
(`registry/index.csv`: all 7 have `split.n_val=9448`) — there is no BiLSTM
number on the split the three current-split `gru` runs (74–75%) and §3.5's
curated-feature DNN/LSTM (69.09%/57.76%) are measured on. Two arms, same
notebook, same training regime:

- [x] **`bilstm_base`** — the production `sb.recognize.architectures.BiLSTM`
  class, **unmodified**, fed `sb.recognize.features.base_v1`'s ME-126/xy raw
  landmarks (252-dim, the exact historical feature pipeline). Needs zero new
  model code: `BiLSTM.forward(x, lengths) -> (B, C)` already matches
  `sb.recognize.interp.train`'s generic `train_fold`/`predict_probs_indexed`
  contract.
- [x] **`bilstm_curated`** — new `sb.recognize.interp.models.LandmarkBiLSTM`
  (additive; attention gate + generalized dims, same pattern
  `LandmarkDNN`/`LandmarkRNN` established, plus the production `BiLSTM`'s
  own fwd-last/bwd-first readout convention), fed `features_curated`'s
  ME-126+xy+28-angle+relational pipeline (922-dim, already cached from
  §3.5).
- [x] **Notebook**: `experiments/recognition/gislr.1.models.bilstm-curated.ipynb`
  — mirrors §3.5's structure (feature caches → model factories → single
  final fit per arm → held-out `test.csv` eval + top-N → confusable-pair
  check), plus a closing comparison cell that live-queries
  `registry/index.csv` (`sb.mlops.query.query_runs`) for the three
  current-split canonical `gru` runs and pulls in §3.5's numbers, so this
  notebook situates itself against everything else on the same held-out
  set. No per-frame confidence-trace section — bidirectional models need
  the whole sequence, so frame-by-frame isn't a meaningful read-out here.
  Config: `experiments/recognition/configs/gislr.bilstm-curated.json`,
  hyperparameters copied verbatim from `gislr.training.json`'s `shared`
  block (same regime as every historical registry run) for both arms.
- [x] **Both arms are OFFLINE-ONLY** (bidirectional = reads future frames,
  same constraint the production `BiLSTM` class's own docstring states) —
  this prices bidirectionality + engineered features at the accuracy
  ceiling, not a deployment candidate.
- [x] Smoke-tested against real `GISLR_Stratified` data (not just
  synthetic): `LandmarkBiLSTM` forward-pass sanity-checked directionally
  (perturbing a padded frame doesn't change the output; perturbing frame 0,
  which only the backward pass reads at t=0, does) before touching real
  data; a real mini training+eval pass for `bilstm_base` (5-class shrink,
  real `base_v1` cache, real GPU training, `predict_probs_indexed`) ran
  clean end to end.
- [x] **GPU-utilization fix (2026-09-19 remark)**: the interp track's
  training loop (`sb.recognize.interp.train.py`) was missing the three
  things `sb.recognize.train`'s production driver already has —
  `non_blocking=True` H2D transfers (`run_epoch_dnn`, `run_epoch_rnn`,
  `predict_probs_indexed`), `pin_memory=True` on both loader factories, and
  `torch.backends.cudnn.benchmark = True` (added to this notebook's setup
  cell only — a global torch setting, so it's not retrofit onto already-run
  notebooks). None of these change a single computed value (CUDA's
  stream-ordering guarantees correctness), so they're safe even for a
  notebook whose results are already reported. On top of that,
  `bilstm_curated` (no historical regime to stay faithful to) gets a
  `batch_size: 4096, lr: 0.005656` override in the config — `bilstm_base`
  keeps `shared` byte-for-byte, preserving the exact-replica requirement.
  Verified with real data: override plumbing picks the right hyp per arm,
  training still runs clean end to end.
- [x] **Run it** (user, 2026-09-19) — `bilstm_base` trained cleanly, 73.71%
  top-1 / 86.15% top-3 / 89.36% top-5 (comparable to the current-split
  canonical GRU runs at 74–75%). **`bilstm_curated` collapsed**: early-stopped
  at epoch 16, `train_loss`/`val_loss` pinned at `ln(250)=5.52` the entire
  run and `val_acc` stuck at 0.42% — exactly chance for 250 classes, a dead
  uniform predictor from epoch 1, not slow convergence.
- [x] **Root cause, found and fixed**: the `batch_size=4096, lr=0.005656`
  override 4x'd both together (linear batch-size scaling of `lr`) — verified
  on real data that this combination collapses the model from epoch 1
  (`train_loss` 5.89→5.59, flat, `val_acc` flat at 0.0042 for 4 epochs),
  while dropping the `lr` override (batch 4096, production's unscaled
  0.001414) breaks the collapse (`val_acc` 0.0042→0.0103 within 3 epochs,
  genuinely rising off chance). Adam-family optimizers don't tolerate the
  linear LR-scaling rule without a warmup schedule, which this notebook
  doesn't have. Fixed: `gislr.bilstm-curated.json`'s `bilstm_curated` arm
  now overrides only `batch_size` (still 4096, for GPU utilization),
  `bilstm_base` is unaffected (never had an `lr` override). A `CAUTION` note
  in the config's `notes` field warns against re-adding an `lr` override
  without a warmup + a repeated real-data check.
- [x] **Re-run `bilstm_curated`** (user) with the fixed config — the collapse
  is gone (train_loss/val_acc both move normally, no epoch-1 plateau), but
  `bilstm_curated` still loses to `bilstm_base` by **~16pp top-1** (0.578 vs
  0.737) and overfits *more* despite 3.6x the raw feature dimensionality:
  internal-val peaks epoch ~80/96 then drifts down, `val_loss` rises from
  epoch ~48 while `train_loss` keeps falling — `bilstm_base`'s late-training
  curve is much flatter by comparison. Two things differ between the arms at
  once (attention-gate + projection architecture, and a 4x batch size at the
  same unscaled lr), so this result alone can't say which one is responsible.
- [x] **Ablation added (2026-09-2x)** to isolate the two confounded variables
  from the finding above: `bilstm_curated_plainarch` (same plain `BiLSTM`
  class as `bilstm_base`, fed the 922-dim curated features directly, at
  `bilstm_base`'s batch_size=1024 — isolates the feature set alone) and
  `bilstm_curated_b1024` (same `LandmarkBiLSTM` attention+projection
  architecture as `bilstm_curated`, but batch_size=1024 — isolates batch
  size alone). New notebook §6b sums the three deltas (feature set,
  architecture, batch size) back to the original gap as a sanity check.
  Config (`gislr.bilstm-curated.json`) updated with both arms' entries;
  `make_model`/`ARM_FEATURE_DIM`/`ARM_TRAIN_CACHE`/`ARM_TEST_CACHE` in the
  notebook generalized from a 2-arm dict to loop over `ARMS`. Both new arms
  reuse the already-built curated feature cache (no new cache build).
  Smoke-tested (model construction + forward pass, all 4 arms, correct
  per-arm batch_size and output shape) — not full training, per convention.
- [x] **Run the ablation** (user, 2026-09-21) — all four arms, clean runs, 0
  errors. **Verdict: the curated feature set is the dominant cause, not the
  architecture.** Decomposition of the −15.90pp `bilstm_curated` −
  `bilstm_base` gap: feature set **−17.37pp**, architecture (attention gate +
  projection) **+10.55pp** (recovers >60% of the feature-set loss — not the
  problem), batch size **−9.08pp** (real, secondary, with a mechanistic hint
  it's under-training from ~4x fewer gradient steps per epoch at batch 4096,
  not proven). Even the best curated arm (`b1024`) still loses to
  `bilstm_base` by 6.8pp — not a tunable config bug. Also: `bilstm_base`
  (0.7371) sits *below* every current-split `gru` run — bidirectionality
  buys nothing here once measured on the same split. Full write-up:
  `docs/reports/bilstm-curated.md`.
- [ ] If `bilstm_base`'s single-final-fit number is wanted as a literal new
  registry entry (this notebook's regime matches `gislr.training.json`'s
  hyperparameters but not its k-fold-free driver/registry-writing path),
  the simpler and separate route is to (re-)run
  `gislr.1.models.training.ipynb`'s existing `bilstm`/`ME_126`/`xy` config —
  it already resolves to the current split automatically, no code changes
  needed there.

### 3.7 Five-architecture benchmark on one feature pipeline, plus a mean true-class-confidence metric (built + run 2026-09-21)

**Ask:** run GRU, LSTM, BiLSTM, CNN and DNN under identical conditions, and
add a "top-n (maximum)" metric alongside top-1/3/5. Clarified the metric with
the user first: not another ranked top-N tier, but **the probability mass
the model assigns to the correct class specifically, independent of its
rank** — e.g. true label `sleep`, model outputs `dog: 0.9, cat: 0.65,
sleep: 0.2` → that sample scores **0.2**, averaged per gloss and overall.
Called **mean true-class confidence** in code/docs to avoid confusion with
ranked top-N accuracy.

- [x] **New notebook**: `experiments/recognition/gislr.1.models.five-arch-benchmark.ipynb`
  + `experiments/recognition/configs/gislr.five-arch-benchmark.json` —
  mirrors `bilstm-curated.ipynb`'s proven structure (single final fit, small
  internal-val carve-out, `sb.recognize.interp.train`'s generic driver).
  **All five arms share one feature pipeline** (`base_v1`, ME-126/xy,
  252-dim) and **one hyperparameter regime** (copied verbatim from
  `gislr.training.json`'s `shared` block — same batch/lr/epochs/patience the
  current-split `gru`/`lstm`/`bilstm`/`cnn1d` registry runs use), so all five
  are directly comparable to each other.
  - `gru`/`lstm`/`bilstm`/`cnn` — the unmodified production classes
    (`sb.recognize.architectures`), fed straight through
    `sb.recognize.interp.train`'s generic `forward(x, lengths) -> (B, C)`
    path (`run_epoch_rnn`) — `cnn`(`CausalConv1D`) fits this contract too
    despite not being recurrent.
  - `cnn`'s `num_layers` overrides to **5**, the same TODO §5.5.1 fix
    `gislr.training.json`'s `cnn1d` already has — `num_layers` means dilated
    conv *blocks* for this architecture (receptive field), not recurrent
    layers; at the shared `num_layers=2` it would crush to a 13-frame
    receptive field.
  - `dnn` — `sb.recognize.interp.models.LandmarkDNN`, fed the same flat
    252-dim vector directly (`n_landmarks=126, channels_per_landmark=2,
    angle_dim=0, relational_dim=0` — no angle/relational blocks, `base_v1`
    doesn't compute them). **Never measured on this raw pipeline before**
    (only on full-543 `landmark_interp_v1` in `landmark-importance.ipynb`, or
    the 922-dim curated pipeline in `curated-features.ipynb`) — this closes
    that gap. Per-frame classifier, memory-free, video prediction =
    softmax-averaged over valid frames (`run_epoch_dnn`,
    `predict_probs_indexed`'s `arch=="dnn"` branch).
  - **Mean true-class confidence**: `probs[arange(N), true_labels]` — one
    line, computed alongside the existing top-1/3/5 ranked-accuracy block
    from the same prediction matrix. §4b breaks it out **per gloss**
    (`per_gloss_confidence.csv`) — which signs does each architecture
    "believe in" least, even when it still ranks them correctly at top-1?
  - Confusable-pair check + registry-leaderboard comparison sections carried
    over from `bilstm-curated.ipynb` unchanged (same 16 pairs, same
    live-query pattern).
- [x] **Smoke-tested against real data** before handoff: model-factory
  construction + forward pass for all five arms (correct output shapes —
  `(B,C)` for four, `(B,T,C)` for `dnn`; correct `cnn` `num_layers=5`), then
  a full 2-epoch end-to-end pass (real feature cache, real training loop,
  real held-out eval) confirming the whole pipeline — including the new
  confidence metric, overall and per-gloss — runs clean for every arm.
  Cache/artifacts from the smoke test deleted before handoff (throwaway,
  like every other notebook this session).
- [x] **Run it** (user) — **run 2026-09-21, all five arms, 0 failures, ~51 min
  wall clock.** Held-out `test.csv` top-1: `bilstm` 0.7392 (offline-only) >
  `gru` 0.7380 > `lstm` 0.7286 > `cnn` 0.6696 > `dnn` 0.6485 — among
  streaming-viable arms `gru` leads, and every arm sits at or below the
  canonical `gru`/ME_132/xy registry entry (0.7517), so nothing here beats
  the existing streaming baseline. `bilstm`'s number agrees with the
  same-pipeline `bilstm_base` run from `bilstm-curated.ipynb` (0.7371) to
  within 0.2pp. **Standout: `dnn`'s mean true-class confidence (0.2387) is a
  quarter of every other arm's (0.59–0.61)** despite top-3/5 accuracy in the
  same range as `cnn` — ranked accuracy and confidence decouple for the
  per-frame, softmax-averaged architecture. `give`/`gift` is the hardest
  confusable pair and `give` the lowest-confidence gloss, for every single
  arm — two independent metrics agreeing on one sign. `cnn`/`dnn` trained
  far longer (137-138 vs 66-79 epochs) with almost no train/val gap
  (underfitting, not better generalization — both still land well below the
  recurrent arms on top-1). Full write-up: `docs/reports/five-arch-benchmark.md`.
  Note: the notebook's own confusable-pair (§5) and registry-comparison (§6)
  cells only print in-kernel rather than caching to disk — the report
  recomputed both from each arm's saved `test_predictions.npz` rather than
  losing them; filed as a follow-up to fix the notebook itself.
- [ ] **Cross-check `gru`/`lstm`/`bilstm`/`cnn` here (single-final-fit)
  against their canonical registry counterparts** once §4.3's 12-run gap is
  filled — right now only `gru` has a comparison point (0.7380 here vs
  0.7450 canonical, −0.70pp, protocols not identical), not a clean
  same-protocol check for the other three.
- [ ] **`dnn`'s mean-true-class-confidence collapse** — is it specific to
  averaging per-frame softmaxes, or would a per-frame-max/last-frame readout
  close the gap? Also: `dnn` scores *worse* on the 922-dim curated pipeline
  (0.5776, `curated-features.md`) than on this raw 252-dim pipeline
  (0.6485), but *better* on the full-543 engineered pipeline (0.7102,
  `landmark-importance.md`) — non-monotonic in dimensionality, possibly the
  same "no projection/attention gate" pattern `bilstm-curated.md` found for
  curated features specifically.
- [ ] Fill in the notebook's §4b/§5/§6 cells to actually cache their output
  (per-gloss confidence already does; confusable-pair table and registry
  comparison don't) so a re-run doesn't require reconstructing them from
  `test_predictions.npz` the way this write-up did.

---

### 3.8 Sign "patterns" without a model: intra- vs inter-gloss similarity of hand-built variables (2026-09-25, built + run)

**User request (2026-09-25):**
- per clip: normalize to a reference point, scale by the shoulder width, compute wrist/elbow/shoulder
  angles, hand↔hand and hand↔face distances, drop null frames, take the variance of displacement,
  velocity, acceleration and jerk, and count touches;
- run it with only x, only y, only z, and xyz;
- test whether same-gloss clips are similar and different glosses dissimilar, so each sign generalizes to
  a pattern;
- no model training.

- [x] `sb.recognize.patterns` + `gislr.0.dataset.sign-patterns.ipynb`: run by Claude on all 94,477 clips,
  no training. Report: `docs/reports/sign-patterns.md`.
  - **Answer: no.**
    - Intra > inter on average, but every gloss has a nearest rival more similar than its own clips.
    - Silhouette is negative in every setting. For the user's combined variables in xyz, it is negative for
      all 250 glosses.
    - Nearest-template train → test: best top-1 **4.9%** / top-5 16.3% (xy, all variables, dominant hand);
      chance is 0.4%, the GRU gets ~74%.
  - **Axes:** x ≈ y (3.2% / 3.0%) > z (1.2%). xy > xyz in every family but touch; z adds noise.
  - **Strongest variables:** dominant hand ↔ face touch fraction (Fisher 0.91) and distance (0.50). Angles
    and velocity/acceleration/jerk variances are nearly constant across glosses.
  - **Handedness:** the labeled "left" hand is the more-seen hand in 42% of clips. Relabeling to the
    dominant hand raises top-1 from 3.8% to 4.9%.
- [x] **Parameter-level patterns: done (user said "start on the sign pattern first", 2026-09-25).** B1 + B2 from
  `improvements-research.md`, no training. Report `sign-patterns.md` §7.
  - New code: `sb.recognize.aslex` (ASL-LEX 2.0: **233/250 glosses mapped as of 2026-09-25**,
    209 exact + 24 synonyms — 4 more synonyms added 2026-09-25: `eye`=`eyes`/`shoe`=`shoes`
    (`_norm` doesn't collapse singular/plural), `wake`=`awake`, `police`=`policeman`; was
    229/250, 209+20) and
    `patterns.phonology_descriptors` (handshape / orientation / location / movement / signtype, left hand
    mirrored).
  - Notebook §7 of `gislr.0.dataset.sign-patterns.ipynb` (run by Claude).
  - **Every ASL-LEX parameter is recovered on unseen signs** (gloss-disjoint, p < 0.005). Best cases:
    - selected fingers 0.81 (chance 0.17, handshape);
    - repeated movement 0.80 (movement);
    - ulnar rotation 0.75 (orientation);
    - major location 0.65 (chance 0.25).
  - The family × parameter grid is near-diagonal.
  - **Whole-sign templates: 4.9% → 38.8% top-1** (61.9% top-5); separable glosses 16% → 70%. Handshape
    alone gives 26.1%.
  - Weak spots: sign type (non-dominant hand tracked too rarely) and movement shape (orderless).
- [x] **B3 DTW temporal templates: done (Claude, 2026-09-25, no training).** `sign-patterns.md` §8.
  - Method: 39 per-frame dominant-hand channels (`patterns.phonology_sequence`), resampled to 32 steps;
    GPU DTW (`patterns.dtw_distances`, checked against a naive DTW to 7e-6); 20 train exemplars per gloss
    → 18,183 test clips.
  - All channels, top-1: orderless 34.5% → lockstep 36.8% → **DTW 41.6%** (top-5 66.8%). Order +2.3,
    warping +4.8; a wider band is better up to ±16 of 32.
  - Order matters most for location (3.3% → 8.4%) and orientation (4.9% → 10.0%).
  - Few-shot: 1 exemplar 15.2%, 5 exemplars 27.8%.
  - Worst glosses are near-identical pairs (`give`→`gift`, `pen`→`pencil`, `mouth`→`lips`) and short
    pointing signs.
  - A display bug (a pivot averaged over K) briefly showed ±8 at 27.8%; the stored results were right.
- [ ] Optional next: a learned per-parameter embedding (PhonSSM-style) instead of hand-set channel weights;
  or DTW templates as a §12.4/§12.7 custom-sign enrollment baseline vs cosine-head imprinting.
  *Awaiting the user's call.*
- [ ] Optional: use parameter patterns for §12.4/§12.7 custom-sign enrollment (describe a new sign as a
  parameter combination) and to explain confusable pairs (`give`/`gift`).
- [x] ~~Follow-up (optional, no training): add handshape; keep time with DTW; speed normalization~~ —
  **done 2026-09-25**: handshape (§7), DTW over per-frame phonology with time resampling (§8).

### 3.9 Phonology front-end models: streaming GRU, subset combos, feature importance, BiLSTM (2026-09-25, built, awaiting the user's run)

**User request (2026-09-25), after §3.8:**
1. train a model that can stream later, with its training set up to match downstream, so an isolated win
   ports to continuous signing;
2. see whether the phonology features combine with the landmark subsets into a best combination, and use
   explainability to find which features contribute;
3. run the best type of model regardless of streaming.

**User decisions (2026-09-25):**
- **Claude prepares the experiment; the user runs the notebook** (CLAUDE.md's "never train" rule stands).
- Sweep = "focused": 3 arms + feature-group importance, then BiLSTM with the winner.

- [x] **Built (Claude, 2026-09-25):**
  - `PhonologyFrontend` (in `architectures.py`):
    - non-learned, per-frame, causal;
    - every frame shoulder-centred and shoulder-width scaled, which also removes §12.8's framing
      dependence;
    - 41 features per hand (handshape 25 as joint cosines, tip distances, spreads and the thumb gap;
      orientation 6; location 9 vs nose / chin / forehead / mouth / shoulder / chest / other hand;
      present 1) + 2 elbow angles = 84;
    - left hand mirrored; `phono+raw` mode appends the subset's shoulder-normalized xy;
    - optional `mirror_p` handedness augmentation, 0 in every arm for all-else-equal.
  - Classes and `ARCHS` keys:
    - `PhonoGRU` (`gru_phono`, `gru_phono_raw`);
    - `PhonoBiLSTM` (`bilstm_phono`, offline);
    - `ContinuousPhonoGRU` (`gru_continuous_phono`, the §12.3 port).
  - New subsets: `PH_55` (hands + upper pose + 5 face anchors) and `ME_134` (ME_132 + chin/forehead).
  - Drivers pass `landmark_subset` in `hyp`, so every rebuild site gets it: `sb-evaluate`, the
    continuous loader, the baselines.
  - `RecurrentSession` accepts the new recurrent archs.
  - `sb.recognize.phonology_eval`: feature-group permutation importance on the canonical val split.
  - Config `gislr.training.json`: shared `frontend`/`mirror_p` (ignored by old classes) + the 3 new
    archs.
  - Notebook `gislr.1.models.training.ipynb` §8b–§8e.
- [x] **Wiring checks (no training, no registry writes):**
  - every arch builds; front-end outputs are finite on real clips;
  - mirroring swaps the r/l hand blocks to 1e-5;
  - autocast train steps OK;
  - **RecurrentSession = batch forward to 2e-5** for `gru_phono`, `gru_phono_raw` and
    `gru_continuous_phono`;
  - importance runs end to end on the real val split (~14 s per repeat); the permutation hook changes
    the logits;
  - `ty` clean on new code; `sb-docs --check` up to date.
  - Built the `PH_55/xyz` val cache (0.47 GB).
- **Observed 2026-09-25 ~15:10:**
  - No phonology run exists yet.
  - A new plain `gru` ME_126/xy run `1790327033` is in progress (epoch 38, val 0.7297, unfinished). It
    looks like §5 is being re-run, which retrains the existing baselines. The phonology arms only need
    §4 then §8b–§8e.
  - The best streaming model is still `gru` ME_132/xy `1789559734`, **0.7517** canonical.
- [ ] **Next (user):** in `gislr.1.models.training.ipynb`:
  - run §4 (builds the `PH_55/xyz` + `ME_134/xyz` caches);
  - §8b `gru_phono` (PH_55);
  - §8c `gru_phono_raw` (PH_55, ME_134);
  - §8d importance, which names the best arm; if it isn't phono+raw on ME_134, edit `bilstm_phono`
    in the config;
  - §8e `bilstm_phono`;
  - then `sb-evaluate` each run (§10).

  Estimated ~25–45 min per GRU run (the `gru` baseline regime), longer for BiLSTM. Compare against
  `gru` ME_132/xy **0.7517** (canonical) and `bilstm`.
- [ ] After the runs: Claude analyzes (arm comparison, importance, BiLSTM gap). If a phonology arm
  beats `gru`, the continuous port needs:
  - a continuous config run with `subset`/`coords` = the winner's and `arch: gru_continuous_phono`
    (its global `subset`/`coords` are ME_132/xy today);
  - the export (`sb.recognize.export.step`) to support the front-end (it has cross products and
    divisions; it would need a Keras port) before the web app can use it.
- [ ] Optional ablation later: `mirror_p` 0.5 on the winning arm (handedness invariance).

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
  - [x] **Built 2026-09-12**: `gru_deep` — same `StreamingGRU` class as `gru`,
    config override to `hidden_size=384, num_layers=4` (vs shared 256×2),
    still `streaming=True`. Registered in `ARCHS` (`architectures.py`), config
    block in `gislr.training.json`, notebook §5b in
    `gislr.1.models.training.ipynb`. **Awaiting user training run** (this
    agent never trains).
  - Needs a decision from the user on which of these two this remark meant
    before either sub-item is actioned.

All three verified 2026-07-17 by CPU smoke test: forward shapes correct,
future-frame corruption provably doesn't change logits for the two causal
models, and notebook state_dicts load into `scripts/evaluate.py`'s classes
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
  - `sb.recognize.features.firstplace_v1` — NaN-preserving cache, the normalization, the
    lag features, all 6 augmentations, mirror-permutation builder (asserts the
    subset is closed under left/right swap; FP_118, ME_126, ME_132 all are),
    dataset + collate padding to the **batch max** rather than a fixed 384.
  - `sb.recognize.architectures` — `Conv1DTransformer` + `ECA`,
    `CausalDWConv1D`, `Conv1DBlock`, `TransformerBlock`, `LateDropout`,
    `MaskedBatchNorm1d`; registered in `ARCHS` as `conv1d_transformer`
    (`streaming=False`). `build_model` now forwards arch-specific HYP keys that
    a model class declares (existing four architectures verified unchanged).
  - `modules/model/optim.py` — Lookahead, AWP, cosine one-cycle.
  - `sb.recognize.train_firstplace` + `experiments/recognition/configs/gislr.firstplace.json` — the
    driver and its config; same canonical split, registry and meta.json schema.
  - `sb.recognize.evaluate` — dispatches on the checkpoint's `features`
    key so these runs are scored through the 1st-place preprocessing.
  - **`experiments/recognition/gislr.1.models.firstplace.ipynb`** — the driver notebook.
- [x] **Run it (user)** — done 2026-08-23, run `1787483814`, 300 epochs in
  2.0 h. **It diverged**; see the next item. Canonical eval of the surviving
  checkpoint: **0.7459** overall / 0.7433 macro / 0.7632 median / 13 classes
  below 50%. That is *below* the 0.7565 GRU and is not a measurement of the
  recipe — it is a model 15 epochs into a 300-epoch cosine.
- [x] **Ran (closed 2026-09-24): `1787492560` collapsed at epoch 15 again; the open work is §5b below.** **Re-run it with the fixed driver (user).** Full write-up:
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
- [ ] **Registry housekeeping**: `registry/runs/1787473998/` contains only
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
  `sb.recognize.export.keras` rebuilds GRU/LSTM/BiLSTM/CausalConv1D in
  native Keras; exporting the port needs Keras equivalents of Conv1DBlock /
  ECA / TransformerBlock. Not needed to measure accuracy, required before any
  Kaggle submission of this model (§6.3).

### 4.3 Full-registry benchmark on the current split + legacy-run deprecation (2026-09-2x)

**Audited 2026-09-2x** whether any training notebook/config still targets the
retired `asl-signs` parquet / self-computed 90/10 split, since the request was
to "port outdated training regimes" to GISLR_Stratified. **Finding: nothing
needs porting.** Every training-producing notebook and config already resolves
the current split automatically (`gislr_dir()` / `get_canonical_split` /
`get_source("gislr")`) — `gislr.1.models.training.ipynb`,
`gislr.1.models.firstplace.ipynb`, `gislr.1.models.landmark-importance.ipynb`,
`gislr.1.models.curated-features.ipynb`, `gislr.1.models.bilstm-curated.ipynb`
all read npz already. The 52-of-55 legacy-split registry entries are old
because they were trained **before** the 2026-09-16 reset, not because their
code is stale — re-running the exact same notebook cells now trains on the
current split with zero code changes. (Two non-training, diagnostic
notebooks — `gislr.0.dataset.subset-comparison.ipynb` and TF-era
`popsign.2.model.ipynb` — are still on the old path; tracked separately under
§0.1/§0.4, not part of this item since they don't produce registry runs.)

**Legacy-run deprecation — done (schema-light, no migration).** Rather than
add a persisted `deprecated` field to every one of 52 `meta.json` files
(schema v4 → v5, a real migration for zero new information — `dataset` +
`split.n_val` already determine this permanently), declared obsolete via a
**derived** `legacy_split` boolean (GISLR only, `split.n_val == 9448`):
- `sb.mlops.registry.is_legacy_gislr_split(meta)` — the one predicate
  everything below calls, so the rule can't drift.
- `index.csv` gained a `legacy_split` column (`build_model_index.py
  --legacy` / `--current` to filter); the README's generated leaderboard
  block now excludes legacy runs by default (`index.markdown_summary`) —
  **52 on the retired split, only 3 current** (all `gru`).
- `sb.mlops.query.leaderboard()` keeps its existing default (`include_legacy=True`
  — every caller, including `gislr.2.models.evaluation.ipynb`'s top-N
  selection for learning curves/confusion matrices/**TFLite export**, behaves
  identically to before); pass `include_legacy=False` for a current-split-only
  view. `query_runs` exposes `split_n_val`/`legacy_split` directly for any
  `WHERE` clause.
- `sb.mlops.promote.check()` now **refuses to promote a legacy-split GISLR
  run** to any alias — closes a real gap (nothing previously stopped a
  stale-split run from becoming `champion.recognize.streaming`).

**Benchmark gap — what's left to actually run** (all via existing,
unmodified code; no porting). `gislr.training.json` already configures
`gru`/`lstm`/`bilstm`/`gru_deep`/`cnn1d` × `{ME_126, ME_132, FP_118}` × `xy`
(15 combos) in one notebook; `gislr.1.models.training.ipynb`'s cells train
all 3 subsets for one architecture per cell (`train_from_config(ARCH,
subsets=None)`). Current status:
- [x] `gru` × 3 subsets — **already run 2026-09-16**, the only current-split
  entries so far (0.7517/0.7450/0.7425).
- [ ] `gru_deep`, `lstm`, `bilstm`, `cnn1d` × 3 subsets each = **12 runs** —
  cells built, **never executed even once** (`execution_count: null` in the
  notebook as of this audit) — run notebook §§5b/6/7/8 (user).
- [ ] `conv1d_transformer` (1st-place port) × `FP_118`/xy — **not a simple
  re-run**: §4.2 tracks its own unresolved collapse investigation
  independently of the split reset; its next run (once that's fixed) will
  land on the current split automatically like everything else, so no
  separate porting action is needed here, only the §4.2 fix.
- [-] **Decided by the user 2026-09-24: xy wins — no xyz re-benchmark.** `gru` × `xyz` (any subset) — existed pre-reset (52-run legacy set
  includes xyz variants) but `gislr.training.json` only configures `xy`.
  **Decision needed, not silently resolved**: the whole point of the ME-126
  subset work was "xy beats xyz" (`docs/reports/motion-energy.md` §2.3,
  `docs/reports/subset-comparison.md`) — re-benchmarking xyz on the current
  split would satisfy "replace every legacy combo" literally, but may just
  reproduce a settled conclusion at real GPU cost. Left open rather than
  added to the config unasked.

Once the 12 runs above land, re-run `gen_docs.py` to refresh the leaderboard
and cross-check every architecture's current-split number against its
legacy-split counterpart in `index.csv` (`--legacy` vs `--current`) as a
migration sanity check — same architecture/subset should land in the same
neighborhood; a large unexplained swing would flag a real regression, not
just a stale-split artifact.

---

## 5. Spectrogram experiment on GISLR_Stratified (CNN/ViT arm) — re-scoped 2026-09-24

**User decision 2026-09-24: keep it, as a real experiment on `gislr-stratified`**
(it had sat untouched since July as a format note). Each landmark sequence
becomes an image: landmarks on the y-axis, frames on the x-axis, coordinates
as channels. Train image models on those images and compare them with the
recurrent models on the canonical split. Needs a plan + user review before it
is built.

- [ ] **Plan** (for review). Decisions to put to the user:
  - channels: xy only (consistent with "xy wins", §4.3) vs xyz-as-RGB (the
    original idea) vs xy + speed;
  - landmark subset/ordering on the y-axis (ME-126, grouped by region so
    neighbouring rows are anatomically related);
  - fixed width: linear interpolation to T frames (T = 64/128?), and how
    NaN/undetected is encoded (0 plus a mask channel?);
  - models: a small CNN from scratch, a pretrained ResNet/EfficientNet
    (ImageNet stats don't match landmark images, so test both), and a
    small ViT;
  - augmentation that makes sense on these images (time-crop/stretch is
    fine; flips on the landmark axis are not).
- [ ] Build the images from `gislr_dir()` npz into a content-addressed cache
  under `data/cache/gislr/features/spectrogram_v1/` (the existing
  `features/cache.py` pattern). Quantization happens at build time and is
  not baked into the shared format (the original §5 note).
- [ ] Notebook `experiments/recognition/gislr.1.models.spectrogram.ipynb`, with
  hyperparameters in `configs/gislr.spectrogram.json`, never in cells. The
  model classes go in `architectures.py`, `streaming: false`.
- [ ] Canonical eval (18,896-video val split, `sb-evaluate`) and a
  comparison with the registry's recurrent models.
- [ ] Framing: fixed-width interpolation needs the whole clip, so this is an
  **offline accuracy reference** like BiLSTM, not a deployment candidate,
  unless a causal sliding-window variant is added later.

---

## 5.5 Training consolidation — one notebook, one config (2026-07-19)

Four `gislr.1.model.<arch>.ipynb` notebooks each carried their own `HYP` dict, so
"all else identical" — the premise of both the architecture comparison (§4) and
the subset ablations (§3.1) — was a manual chore across four files.

- [x] **`experiments/recognition/gislr.1.models.training.ipynb`** replaces all four: shared setup /
  config / split / feature-cache sections, then one markdown+code section per
  architecture, then cross-architecture comparison and the eval handoff. Each
  architecture section re-reads the config from disk, so it is independently
  re-runnable. The four old notebooks are removed (git history ≤ `2d7f668`).
- [x] **`experiments/recognition/configs/gislr.training.json`** is the source of truth for every
  hyperparameter, read at run time by **`sb.recognize.config`**.
  Architectures inherit `shared`; a deviation must be an explicit `overrides`
  entry, surfaced by §2 of the notebook and by `TrainingConfig.overrides_for`.
  Validation rejects: unknown architectures, unknown per-arch keys, an override
  naming a key absent from `shared` (a typo can't become a silent no-op),
  missing required HYP keys, bad `coords`, wrong `schema_version`. All six
  failure modes tested.
- [x] `sb.recognize.train::train_from_config(arch)` — what each section
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
- [-] **Folded into §4.3 (2026-09-24): its 12-run grid includes `cnn1d` × 3 subsets on the current split.** **Re-run cnn1d** with the corrected receptive field before drawing any
  conclusion about the architecture (all three subsets; ~1.7M params now).
- [-] **Obsolete (2026-09-24): that log reports the retired split; §4.3's runs replace it.** Once re-run, revisit `docs/logs/daily/2026-07-19.md` §1.3, which currently
  reports the crippled numbers.
- [ ] Consider making the receptive field an explicit, asserted quantity in
  `CausalConv1D.__init__` (e.g. warn when it is much shorter than
  `MAX_SEQ_LEN`), so a future misconfiguration fails loudly rather than
  training quietly at a fraction of the intended context.

---

## 6. Evaluation, Export & Kaggle Submission (GISLR)

**Location:** `experiments/recognition/gislr.2.models.evaluation.ipynb` — the single place where
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
- [x] Run the notebook (user) — run on 2026-07-19 (18-run aggregate) and again
  on 2026-08-23 for the 1st-place port; its findings are what §7.1 closed on.
- [x] **Done 2026-09-19 (verified 2026-09-24 from the notebook's outputs): 55 runs, 37 canonical.** Two leftovers filed below. Original text: **Re-run it on the current 42-run registry.** It last ran against 18 runs;
  since then the four-architecture × three-subset grid completed and six
  1st-place runs landed. Only 1 of 42 run folders has a `confusion.png`, so the
  per-run confusion artifacts are mostly missing.
- [ ] (2026-09-24) Only 1 run folder has a `confusion.png` — the per-run confusion artifacts are still missing.
- [ ] (2026-09-24) The 09-19 run **skipped 5 runs for "no best.pt on disk"** (`1787494351`, `1787492560`,
  `1787493805`, `1787494807`, `1787495142`) — the notebook does not go through
  `ensure_local`'s auto-fetch, or those weights were never pushed. Check `sb-sync status`.

### 6.2 Supporting module work (2026-07-19)

- [x] `sb.recognize.architectures`: every arch gains `forward_full(x)` —
  a batch-1, unpacked, ONNX-friendly forward used only by export. Parity against
  the packed training forward is asserted at export time.
- [x] `modules/model/export.py`: arch-generic ONNX → TF SavedModel → TFLite chain
  + `submission.zip` packaging + a validation pass using the grader's exact
  calling convention (raw `(T, 543, 3)` with NaNs in, `(250,)` out). NaN→0 and
  the landmark-subset gather stay **inside** the exported graph.
- [x] **The ONNX route was abandoned; export now goes through a native Keras
  rebuild** (`sb.recognize.export.keras`, 2026-07-19). **All 4
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
- [x] `sb.recognize.evaluate` refactored so `evaluate_run(run_dir)` is
  importable (the CLI is a thin wrapper), and it now also writes
  `assets/val_predictions.npz` (labels + preds) — that file is what makes the
  confusion matrices cheap and reproducible.
- [x] `sb.recognize.train` writes `assets/history.json` every epoch.

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
- [-] **Obsolete (2026-09-24): submission was replaced by local testing 2026-09-16 (§3.0.2), and `.mcp.json` is now empty.** Submission mechanics. The `kaggle` **CLI** path submits through a Kaggle
  kernel (`-k <owner>/<notebook> -v <version>`), so each zip must be attached to
  a kernel version first — and there are currently **no credentials on this
  machine** (`~/.kaggle/kaggle.json` absent, `KAGGLE_USERNAME` unset), so a
  non-dry-run submit can only fail or hang.
- [-] **Obsolete (2026-09-24): submission was replaced by local testing 2026-09-16 (§3.0.2), and `.mcp.json` is now empty.** **Kaggle MCP server** (offered 2026-07-19) — likely the better path: it
  exposes `mcp_kaggle_start_competition_submission_upload` +
  `kaggle_mcp_submit_to_competition`, i.e. **upload a file and submit it
  directly**, with no kernel-version dance. `.mcp.json` added at the repo root
  using the **OAuth variant** (`npx mcp-remote https://www.kaggle.com/mcp`, then
  call the server's `authorize` tool) so that **no API token is stored at rest**
  — `.mcp.json` is committed and is not gitignored. Token auth is the fallback:
  add `--header "Authorization: Bearer ${KAGGLE_MCP_TOKEN}"` to the args and set
  that variable in the environment, never inline.
  - [-] **Obsolete, see parent.** Authorize the server from an **interactive** session (OAuth cannot run
    in a non-interactive one), then submit one model by hand end to end.
  - [-] **Obsolete, see parent.** Once proven, decide whether the (since-deleted, 2026-09-19) `submit_run`
    keeps shelling out to the CLI or the notebook drives the MCP tools instead;
    the queue query and `mark_tested` bookkeeping are unaffected either way.
- [x] **Rotated by the user 2026-09-24.** **Security**: an API token was pasted in plaintext into a chat transcript
  on 2026-07-19 and must be treated as compromised — rotate it (Kaggle
  Settings → Generate New Token) and never commit one.

---

## 7. Breaking the ~73% Accuracy Plateau

**User decision 2026-09-24: keep §7.2–§7.7**, running alongside the §12 continuous work.

**Context:** GRU, 1D-CNN, LSTM and BiLSTM all converge to ~70–74% on the FP_118
subset. Architecture-independent ⇒ the ceiling is upstream, in
features/normalization/data, not the model. Ordered by priority: diagnose first,
then fix the highest-leverage causes, then ablate to confirm what actually
helped.

### 7.1 Phase 1 — Diagnose before changing anything

Figure out whether this is overfitting, underfitting or a data/label ceiling
*before* spending compute on new features.

- [x] Run the canonical eval on the current best checkpoint and fill in the
  pending metrics — **done**: 37 of 42 runs are `eval_status: canonical`, best
  is `1784447175` (bilstm/ME_126/xy) at 0.7569, best *streaming* is
  `1784453891` (gru/ME_126/xy) at 0.7565. The 5 pending are the 1st-place
  ablation arms from 2026-08-23.
- [x] Train/val accuracy gap at the best epoch — **answered 2026-07-19**: train
  90–99% vs val ~75%, gap 0.16–0.24. That is `train ≫ val`, so the verdict is
  **overfitting**, and §7.4 (augmentation/regularization) outranks §7.3 on this
  evidence. Full write-up: `docs/logs/daily/2026-07-19.md`.
- [x] Full 250×250 confusion matrix on the val set — done, aggregate over 18 runs
  (`docs/logs/daily/2026-07-19.md` §1.2) and per-run for the 1st-place port.
- [x] Top 20–30 most-confused class pairs by off-diagonal mass — done; the pairs
  are **semantic near-synonyms** (`awake`/`wake`, `mouth`/`lips`), and they
  replicate on a completely different architecture and feature pipeline
  (2026-08-23), which is what makes them a label/data property rather than an
  artefact of one model.
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
- [x] **Is semantic similarity the cause of the plateau? — measured 2026-09-04, and the
  answer is NO.** `experiments/recognition/gislr.2.models.plateau-diagnosis.ipynb`, over
  31 canonical runs at ≥0.70 (mean 0.7433). Write-up:
  [`docs/reports/plateau-diagnosis.md`](docs/reports/plateau-diagnosis.md).
  - The top-20 confusable pairs absorb **10.5% of all errors**. Solving them
    perfectly moves 0.7433 → **0.7704** (+2.7 pts), against **+0.007** for a
    random-pair control — the effect is unmistakably real *and* small. Even
    declaring 50 pairs (100 of 250 classes) solved leaves accuracy below 0.79.
  - Semantic errors are near-misses (**77%** recovered at rank 2, 95% at rank 5);
    the other 90% of errors are not (31% / 56%). For the bulk of the error the
    representation simply does not carry the answer.
  - Imbalance ruled out (support spans only 30–42 videos/class); the error is
    diffuse, not concentrated (worst 10 classes = 8.6% of error vs 4.0% uniform).
  - **Consequence for §8:** a rescoring layer is worth **at most ~2.7 points**,
    and only on the near-miss tenth. Quote it that way, not as "fixes the plateau".
  - **Consequence for §7.2–§7.4:** the plateau is ~22 points of diffuse error
    where the true label is usually outside the top 5. That points at the input
    representation, and **§7.2 normalization is the untested candidate** —
    nothing in the stack currently removes signer appearance or position.
- [x] **Both GPU arms run 2026-09-05 — the plateau is a GENERALIZATION GAP.**
  - **§7 train-split confusion (run `1784447187`, 2,000 stratified train clips):**
    mean symmetric confusion **0.273 on val vs 0.012 on train**, with **19 of 20
    pairs at exactly zero** on train (only `awake`→`wake` shows any, at 0.25, and
    one-directionally). The models separate these pairs on data they have seen
    and fail on data they have not. **Not a label ceiling.** This settles the
    §7.1 tension: the 07-19 overfitting verdict is the story, not the 08-23
    label-ceiling read.
  - **§8 separability probes:** median **0.628** on a *binary* task;
    `awake`/`wake` at **0.463** (chance), robust to standardisation and stronger
    regularisation (0.464 at C=0.01). `corr(confusion_rate, probe_accuracy) =
    −0.72`. **§8b added and run:** pooled *velocity* does not rescue it either
    (`velocity_gain` ≈ 0 for 6 of 7 pairs).
  - Read §8 precisely: it bounds **time-pooled summaries**, not the landmarks.
    Pooling destroys ordering and trajectory shape — exactly where `awake`
    (repeated) and `wake` (single) differ — and the sequence models do beat the
    probes (≈57% vs 46% recall on that pair). **It does not refute §7.3**, which
    proposes per-frame velocity channels in a sequence model, a different claim.
- [ ] Backfill top-k on more runs (count below is from 2026-09-04; re-count 2026-09-24) — only **1 of 31** has it (`sb-evaluate` began
  storing top-5 on 2026-09-04), so the near-miss split rests on a single run.
- [ ] Two of the top-20 pairs (`finger`/`wait`, `animal`/`have`) are **not**
  semantically related. Whatever drives those is not meaning — and `finger`/`wait`
  has the **highest** probe score of the whole set (0.790), i.e. it is the most
  separable pair yet among the most confused. Worth an eyeball on raw sequences;
  it may be the more informative case.
- [ ] **§7.2 and §7.4 are now the evidence-backed levers**, because they are the
  two that attack a generalization gap: normalization removes nuisance variance
  from the input (nothing in the stack currently removes signer appearance or
  position), augmentation expands the training distribution. Pick one and run
  §7.6's controlled ablation protocol against the current ME-126 baseline.
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
- [x] **Re-test §6's binary separability probes with the richer angle+kinematics
  feature set** (`experiments/recognition/gislr.0.dataset.pair-similarity.ipynb`,
  built + run 2026-09-19, TODO §3.4's `sb.recognize.interp.kinematics`/
  `discriminability`, new `probe_classifier_cv` for the same 5-fold-CV
  methodology §6 used) — all 16 documented pairs + 8 random-pair controls,
  per-pair probe broken down by feature type (angle/position/speed/...).
  **Full run: `corr(confusion_rate, probe_accuracy)` = Spearman ρ −0.71,
  matching §6's −0.72 almost exactly** with a completely different feature
  set. `awake`/`wake` stays at chance (0.512) under every single feature
  type, individually or combined; control pairs average 0.954. **Confirms
  §6's own reading that pooling itself, not the channel choice, is the
  ceiling** — angles don't rescue what pooling loses. Full write-up:
  `docs/reports/pair-similarity.md`.
- [x] **Turned the hand-picked 16-pair list into reusable code, re-derived from scratch on
  the current canonical split (2026-09-25, user: "analyze all the glosses... merge
  semantically similar ones... then check accuracy of the top performing models").**
  `sb.recognize.label_merge` (new): 9 merge groups (19 words -> 9 canonical labels:
  `awake`/`wake`, `give`/`gift`, `listen`/`hear`, `nap`/`sleep`/`sleepy`, `kitty`/`cat`,
  `pencil`/`pen`, `look`/`see`, `mouth`/`lips`, `puppy`/`dog`), each requiring **both**
  a semantic check (WordNet Wu-Palmer, dominant sense -- alone it's a poor filter, rating
  `cat`/`dog` or any two colors just as "similar" as true synonyms) **and** empirical
  confusion in a freshly-rebuilt current-split confusion matrix (not the possibly stale
  cached `confusion_all_normalized.npy`). Matches this section's own `finger`/`wait` and
  `animal`/`have` read (confused but not semantically related -- excluded) and extends it:
  also excluded several plausible-sounding semantic guesses the model doesn't actually
  confuse (`not`/`no`, `cute`/`pretty`, `shower`/`bath`, `talk`/`say`, `fall`/`drop`,
  `bad`/`yucky`, `loud`/`noisy` -- all near-zero). `experiments/recognition/
  gislr.2.models.label-merging.ipynb` (run) + `docs/reports/label-merging.md`.
  **Provisional top-model numbers only** (7 current-split canonical runs exist right
  now, one a broken outlier at 0.335 acc; merge lift so far +1.4 to +1.6 points,
  consistent across architectures) -- **blocked on `gislr.1.models.training.ipynb`'s
  §4.3 grid finishing and being evaluated**, per the user's explicit instruction to wait;
  re-run §2-§4 of the notebook once it has. Two borderline pairs surfaced by the noisier
  current matrix, below the merge bar for now: `mouth`/`tooth` (0.063), `lips`/`tooth`
  (0.057) -- watch after more runs land, don't merge yet.
- [x] **Direct answer to "how similar are the signs, really" (2026-09-25, user: "run an
  experiment... no model training, work with the landmark files").** `gislr.0.dataset.
  sign-patterns.ipynb` §9: intra- vs. inter-gloss DTW distance (GPU, unconstrained,
  reused §8's cached per-frame phonology sequences -- no recomputation, no training) for
  the 9 merge pairs, the 12 rejected-but-confused pairs, and 12 random pairs. **Merge
  pairs are kinematically near-indistinguishable (confusability 1.043 mean) and clearly
  separable from random pairs (1.271)** -- validates the merge decision as a real
  property of the sign. **Correction to this section's own earlier framing**: the
  rejected pairs turned out just as kinematically close (1.060, barely above the merge
  groups) -- `goose`/`duck` and `wait`/`finger` are *more* identical than several merge
  pairs. These are true near-homophones for unrelated concepts, not a separate,
  non-kinematic model weakness as speculated; the label-merge decision to keep them
  distinct is unaffected (different meanings regardless of how alike the signs look),
  but the "why" changes. Per-family breakdown names which parameter each merge pair
  shares (mostly all four; `look`/`see` shares only location, the weakest pair by both
  this test and raw confusion -- consistent across two methods). `docs/reports/
  sign-patterns.md` §9; data `data/cache/gislr/sign_patterns/sign_similarity.csv`.
- [x] **ASL-LEX coverage improved while checking the above (2026-09-25): 229/250 ->
  233/250** (209 exact + 24 synonym, was +20). Investigated the surprisingly-unmapped
  basic words (`eye`, `shoe`, `wake`, `police`) instead of assuming ASL-LEX genuinely
  lacked them: found real near-misses (`_norm` doesn't collapse ASL-LEX's plural
  `EntryID`s against GISLR's singular gloss) and added 4 well-justified `SYNONYMS`
  entries to `sb.recognize.aslex` (`eye`=`eyes`, `shoe`=`shoes`, `wake`=`awake`,
  `police`=`policeman`). **A real tension surfaced, not resolved**: ASL-LEX's own
  docstring explicitly treats `look`/`see` and `sleepy`/`tired` as *different* citation
  signs, while both this section's empirical confusion matrix and the new DTW check
  above find them (`look`/`see`, and `sleepy` via the `sleep` group) kinematically
  near-identical *in GISLR's actual clips*. Flagged in `sign-patterns.md` §7 rather than
  silently picking a side — a citation-form lexicon and a corpus of real signing can
  legitimately disagree.

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
  computed on un-normalized data may have been biased by signer scale. **Partial:**
  the rebuilt motion-energy notebook (2026-09-21) deliberately keeps per-landmark
  motion energy un-normalized (see its title cell's rationale — normalizing would
  conflate camera distance with actual motion, a legitimate question for
  discriminability but not for "how much does this landmark move"); its new joint
  angles are computed on `center_and_scale`-normalized positions, since angles are
  already scale-invariant and this keeps them comparable to
  `feature-discriminability.ipynb`'s numbers. A normalized-coordinates re-run of
  the *landmark* RMS speed itself, if wanted, is still open.
- [ ] Resolve the long-open ME-126 vs Kaggle-suggested-subset cross-validation
  (§1.6, §3) — landmark importance rankings may shift once normalization is fixed.
- [ ] Plain-language write-up for supervisor progress reporting in
  `docs/reports/plateau-breakout.md`: what Phase 1 diagnosed, what changed, what
  moved accuracy and by how much.

---

## 8. Post-Processing: Context-Aware Correction Layer (2026-07-22 remark, new)

**Cross-reference (2026-09-22):** §11 ("Streaming Confidence & Reset") now
also wants an LLM in this pipeline, in a different role than this section
originally scoped — reading per-frame confidence scores and deciding when to
reset a streaming model's state (matching a "next-word suggestion"), not
correcting/reranking a finished prediction among near-synonyms. Both readings
hit the same open gap below: neither GISLR nor POPSIGN has sentence-level
data to build any next-word/context model from. Resolve that gap once, not
twice — §11.3 is explicitly blocked on it.

**The idea:** instead of (or alongside) improving raw model accuracy on
semantically-confused pairs (§7.1/§3.0.1), feed predictions through a
correction LLM that uses sentence-level context to pick the right word among
near-synonyms (`awake`/`wake`, `mouth`/`lips`, etc. — the exact pairs §7.1
already identified as semantic, not geometric).

- [x] **Answered by the user 2026-09-24: continuous/sentence-level is in
  scope** — the LLM fuses next-gloss prediction with recognizer confidence,
  writes fluent English, and feeds TTS. Work moved to **§12.6**; the data gap
  below is covered by §12.1's (synthetic) corpus. Original question kept for
  context:
- [?] ~~**Scope conflict to resolve before filing real sub-tasks:**~~ GISLR and
  POPSIGN as used in this repo are **isolated single-sign classification**
  (one video → one of 250 labels), not continuous sentence recognition —
  there is currently no stage that assembles a sequence of predicted signs
  into a sentence for an LLM to have "context of a sentence" over. This
  remark presupposes that downstream stage exists or is in scope. Before
  doing anything else: is this repo's roadmap meant to extend to
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
- [x] **Done: this became §12 (2026-09-24).** If continuous/sentence-level is in scope: this is a substantial new
  workstream (data: does either dataset have sentence-level
  labels/transcripts to train or even evaluate this against? POPSIGN and
  GISLR are both isolated-sign as extracted here) — needs its own numbered
  section once scoped, not folded into §7.
- [-] **Obsolete (2026-09-24): the user chose the continuous reading.** If the smaller-scope reading is intended: prototype using the existing
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
  starting the run (the working tree had `M experiments/recognition/gislr.1.models.firstplace.ipynb`
  when this section was filed). A blanket dirty warning would fire on 100% of
  runs and be ignored within a day. Hash **the code that actually executes** —
  `packages/` + the resolved config file — and warn only when *that* is
  dirty; record notebook dirtiness separately as information, not as an alarm.
- [x] `REQUIRED_KEYS` + `SCHEMA_VERSION = 4` in `modules/model/registry.py`;
  `migrate_all` backfills `provenance: null` for the 43 existing runs — unknown
  provenance must read as unknown, never be reconstructed after the fact.
- [x] Both drivers write it (`train.py` and `train_fp.py`), and
  `evaluate.py` (§9.7) records its own eval-time env, since the
  canonical metric is produced there, not in training.
- [x] `build_model_index.py`: new `prov_*` columns; README schema table updated
  (or generated — §9.6).

**Done 2026-09-04.** `sb.mlops.run` (12 fields: git commit/branch +
the two dirtiness flags, config path + hash of the values that ran, feature
pipeline, feature-cache key, dataset ref with a `train.csv` fingerprint,
environment incl. scikit-learn because it defines the split, captured_at). Wired
into both drivers, into `evaluate.py`'s `eval_summary.json` (the canonical number
is produced there, not in training), and into `index.csv` as 16 `prov_*` columns.
`migrate_all` backfilled **42 runs to v4 with `provenance: null`**, and
`index.csv` was rebuilt — 42 rows, up from the 38 it had drifted to. One run
folder (`1787473998`) has no `meta.json` at all — an aborted 1st-place start that
only ever wrote `assets/landmarks.npy`; it is skipped with a warning, not
indexed.

### 9.2 Content-addressed feature caches

**The fault (confirmed, but narrower than the review claimed).** The cache key
is `subset_tag(subset.name, coords)` — subset *name* plus `"xy"`/`"xyz"`, and
nothing else (`sb.recognize.data::subset_tag`, and the same tag with a
`_nan_` infix in `sb.recognize.features.firstplace_v1::build_nan_cache`). Both builders
are skip-if-exists. So editing the `ME_126` index list in
`sb.core.subsets` leaves the tag unchanged and every
subsequent run silently trains on the **old** 3.2 GB array, with no record that
the definition moved.

**Correction to the review's framing:** the two feature pipelines do *not*
collide (distinct `_nan_` suffix, deliberate per §4.2), and the leaderboard
metric is *not* at risk — `evaluate.py` reproduces the split and preprocessing
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

### 9.3 Off-machine artifact store (backend-agnostic; R2 ruled out)

**The fault (confirmed).** `best.pt`/`last.pt` are gitignored and exist on one
Windows machine. The 2026-07-18 reset (§0.4) already destroyed 8 runs' weights
including the ME-126 result still cited in the README, and those evals can never
be completed.

- [x] `ops/sync_models.ps1` (or `.py`) — `rclone` / `aws s3 sync` of
  `data/models/*/best.pt` to R2. Current total: **707 MB across 42 runs**
  (~17 MB/run), so this is inside R2's free tier and takes minutes.
  **Built as `.venv/Scripts/sb-sync.exe`, not `ops/`** — a project
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
- [x] **R2 ruled out (2026-09-04): the account does not have R2 activated.**
  `sb-sync` now dispatches on `SB_ARTIFACT_BACKEND` instead of assuming one
  provider — the manifest, hashing and verification are shared, only the byte
  transport differs:
  - **`kaggle`** (new default) — a **Kaggle Model**, one variation per run:
    `bracu23101281/signbridge-gislr/pyTorch/<arch>-<subset><-coords>/<version>` via
    `kagglehub.model_upload`. This repo *already* authenticates to Kaggle
    (`whoami` → `bracu23101281`), so there is no new account, no card and no new
    secret, and a Kaggle **inference kernel can attach a model directly** —
    the backup and the artifact a submission run loads (§6.3) become the same
    object. **Models, not Datasets, on purpose**: a Dataset versions as one
    directory, so every push would re-send all 707 MB and every restore would
    pull the lot; a Model's variations version independently, so a push uploads
    only the new runs and `model_download(handle, path="best.pt")` restores
    exactly one. Each variation carries its run's `meta.json` beside the
    weights. `pyTorch` is the framework segment; a TFLite export (§6.2) would
    go under the same model as `tfLite`.
  - **`local`** — any path: external drive, NAS share, or a synced folder. Zero
    dependencies, works this minute. Refuses to run without `SB_ARTIFACT_DIR`,
    and the docs say plainly that a folder on the same disk is not a backup.
  - **`s3`** — any S3-compatible endpoint (Backblaze B2, Wasabi, MinIO, Storj,
    and R2 if it is ever enabled). The old code path, now generic.
- [x] Verified end to end on the `local` backend (2026-09-04): 42 objects
  pushed, one checkpoint deleted locally and restored **byte-identical**, and a
  deliberately corrupted backup copy was **refused** on sha256 with no
  half-written `.pt` left behind. The test manifest and staging directory were
  then removed, so the committed state still honestly says "nothing pushed yet".
- [x] **Done 2026-09-04: all 42 checkpoints are on Kaggle.**
  `bracu23101281/signbridge-gislr`, 16 variations, versions in chronological
  run-id order (`conv1d-transformer-fp118-xy` has 6, the gru/lstm configs 3
  each, bilstm/cnn1d 2 each). Naming is derived from each run's `meta.json`, so
  it cannot drift from what was trained.
- [x] **Local copies deleted after verification, 2026-09-04.** `sb-sync prune`
  downloaded all 42 remote copies and required three hashes to agree (manifest,
  remote, local) before unlinking: **42 verified, 0 refused, 707 MB freed**.
  `drop-resume` then removed `last.pt` for the 41 *finished* runs (675 MB); run
  `1784459817` keeps its resume state because it stopped at epoch 50 of 300.
  **The registry now holds 0 checkpoints locally.**
- [x] Round trip proven from a cold cache: pulled `1784447175` back from Kaggle,
  sha256 verified, and `torch.load` returned the expected bilstm/ME_126/xy at
  0.7569 with 22 state-dict tensors — then re-pruned.
- [x] **Confirmed private by the user 2026-09-24.** **Check the model's visibility on Kaggle.** `kagglehub.model_upload` does
  not expose a visibility argument and the API does not report one back, so
  whether `signbridge-gislr` was created public or private is unverified from
  here. These are unpublished thesis weights — confirm on the model page.
- [ ] Consequence to keep in mind: **a run folder no longer contains weights.**
  `sb-evaluate` and the TFLite export now start with `sb-sync pull <run_id>`;
  the eval script says so instead of raising `FileNotFoundError`.

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

- [x] `sb.core.schema`: versioned `LANDMARK_TENSOR_V1`
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
code: `sb.recognize.data` hardcodes `FEATURES_DIR = CACHE_DIR/"gislr"/"features"`,
`load_label_map` reads GISLR's `sign_to_prediction_index_map.json`,
`get_canonical_split` reads GISLR's `train.csv`, and both drivers default
`data_dir` to `gislr_dir()`. That is what doubles when POPSIGN arrives.

- [x] Introduce a dataset adapter (split builder, label map, per-sample loader,
  feature-cache root) and make `train.py` / `train_fp.py` / the eval script take
  one, with GISLR as the first implementation.
- [x] Keep the `<dataset>.<stage>.<topic>.ipynb` notebook convention as-is —
  renaming notebooks is churn that fixes nothing.
- [x] POPSIGN's canonical split would have needed the same treatment GISLR's
  got (fixed seed, asserted val size) before any POPSIGN number was
  comparable to anything. **Moot: POPSIGN is deprecated (2026-09-22, §2)** —
  `get_source("popsign")` still fails, and the seam is being kept general on
  principle (`sb.recognize.sources`), not because a second dataset is coming.

**Done 2026-09-04.** `sb.recognize.sources`: `DatasetSource` bundles dir
resolver, label map, canonical split, per-sample reader, sample-path builder,
plus the identity used for cache addressing and provenance (name, kaggle ref,
manifest). `grep gislr_dir` over `train.py`, `train_fp.py` and `evaluate.py`
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
fall out of step with the documentation. `sb.mlops.docs` renders
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

- [x] The name predates everything it now does: it dispatches all five
  architectures in `ARCHS` and both feature pipelines (it branches on
  `ckpt["features"] == "firstplace"`). Rename the file and its `--help` text;
  update README, `CLAUDE.md`, `docs/`, and the notebook import
  (`from modules.scripts.evaluate import evaluate_run`).

**Done 2026-09-04.** `git mv` + **42 references** updated across 15 files
(modules, all three GISLR notebooks, README, CLAUDE.md, this file).
`registry.eval_command()` and the notebooks' handoff cells now print the new
path.

**`docs/logs/**` was deliberately left alone.** Those are time-ordered records
of what existed on the day they were written; rewriting a 2026-07-17 log to name
a file that would not exist for another seven weeks would falsify the record.
The daily logs are the one place the old name legitimately survives.

### 9.8 The workspace restructure — done 2026-09-04

This section was filed as "deferred / rejected, with the reason". The user
overrode the deferral: the layout is judged necessary for where the repo is
going, so it was executed in full. **The original reasoning is kept verbatim in
§9.9** rather than deleted — it was the honest read of the evidence at the time,
and a decision record that quietly erases what it overruled teaches nothing.

What actually landed:

- [x] **uv workspace, six members** under `packages/`, PEP 420 namespace (no
  `sb/__init__.py` anywhere): `sb-core`, `sb-extract`, `sb-recognize`,
  `sb-mlops`, `sb-rescore`, `sb-synthesize`. Root `pyproject.toml` is a virtual
  project (`[tool.uv] package = false`) that owns no code.
- [x] **The dependency direction is enforced by the manifests**: `sb-mlops`
  depends on `sb-core` only and *not* on `sb-recognize`, so the registry, the
  index and the artifact sync all work without importing torch. `sb-core`
  imports no torch/mediapipe/tensorflow, which is what lets everything depend
  on it.
- [x] **`features/` became real** rather than a rename: `cache.py` (content
  addressing), `base_v1.py` (NaN→0, subsample) and `firstplace_v1.py`
  (NaN-preserving, crop) now expose the *same*
  `cache_inputs`/`cache_key`/`cache_dir`/`build_cache` surface, so the two
  pipelines are substitutable instead of merely adjacent. The awkward
  `pipeline=`/`nan_policy=` kwargs threading through every call site is gone.
- [x] **`experiments/{extraction,recognition,synthesis}/`** with configs beside
  the notebooks. The `CWD = src/` convention is retired: packages are installed
  editable, and `sb.core.paths` finds the repo root by walking up for the
  workspace marker (`SIGNBRIDGE_ROOT` overrides).
- [x] **`registry/` at the top level**, `data/` gitignored absolutely. Committed
  artifacts no longer live inside a tree whose whole policy is "never commit
  this", and there is no negation rule left to get wrong.
- [x] **`apps/{web,edge,shared-ts}`, `ops/`, `schemas/`** scaffolded; six console
  scripts (`sb-extract`, `sb-evaluate`, `sb-index`, `sb-sync`, `sb-promote`,
  `sb-docs`) replace the `sys.path`-bootstrapping scripts.
- [x] **Rename to `signbridge`** (project, README, schema `$id`). The working
  directory and git remote were left alone on purpose — renaming the folder
  would break `.venv`'s absolute paths, the kernel spec and the VSCode
  workspace for no gain.
- [x] **`aliases.json` + `sb-promote`** (was deferred for lack of a deployment
  target; `apps/` now scaffolds one). Promotion refuses a run that is
  offline-only, not canonically evaluated, or not backed up off-machine —
  the three ways it has gone wrong here before — unless `--force`, which
  records the waiver.
- [x] **Nothing was rebuilt.** 29 GB of feature caches and 6.4 GB of extracted
  POPSIGN landmarks moved by rename; all three §9.2 cache keys are
  byte-identical, so every migrated cache still resolves. 42 run records intact
  at schema v4, auto-resume pointers re-aimed.

Verified end to end: `uv lock` + `uv sync` clean (torch 2.13.0+cu130, CUDA
available; `dask` dropped because nothing imports it), every package imports,
all six console scripts run, every notebook parses, and `ty` is back at its
56-diagnostic baseline.

Still open, and deliberately so:

- [-] **Rejected (recorded 2026-09-24).** **MLflow.** Still not recommended, and still not installed: it is a second
  write path for data `meta.json` already holds, needs a server process, and its
  payoff (parallel-coordinates / run comparison for the ablation write-up) is a
  plotting cell over `index.csv`, which is one row per run and DuckDB-queryable.
  If the write-up needs those views, add the plot to
  `experiments/recognition/gislr.2.models.evaluation.ipynb`.
- [-] **Rejected (recorded 2026-09-24).** **DVC.** Rejected in §9.3 and unchanged: its one advantage over the current
  setup — stage DAGs catching stale derived artifacts — is what §9.2's content
  addressing buys directly.
- [~] **Partly answered (2026-09-24): §8 resolved as in scope, so `sb-rescore` is §12.6's home. `sb-synthesize` (speech → sign) is now scoped by §13 (2026-09-24). `apps/` was scaffolded 2026-09-24 (web/ · edge/ · shared-ts/, README contracts only, for §12.5's Cloudflare deployment).** **`sb-rescore` and `sb-synthesize` are skeletons with no implementation**,
  and `apps/*` is empty. That is the known cost of building the full tree before
  the code exists (§9.9's "empty scaffolding rots" argument). Each carries a
  docstring saying what is fixed regardless of the open scope question, so the
  directories are at least load-bearing as contracts: prompts are versioned and
  hashed, the eval set is frozen, and synthesis emits the same tensor
  `sb.core.schema` defines. **If §8 resolves toward "not in scope", delete
  `sb-rescore`/`sb-synthesize` rather than leaving them to rot.**
- [-] **Obsolete (2026-09-24): notebooks have run many times since (continuous training 2026-09-23).** **Notebooks have not been re-executed** under the new layout — only parsed.
  Their imports resolve and the CLIs run, but the first real training run is the
  proof. Restart the Jupyter kernels: `import modules...` is gone.
- [x] `experiments/extraction/popsign.1.mediapipe.ipynb` had still imported
  `DATASETS`, and `popsign.2.model.ipynb` still imports `tensorflow.keras` —
  both pre-existing breakage (§0.1). **Resolved 2026-09-22**: the first file
  no longer exists; the second is deprecated along with POPSIGN, not fixed.


### 9.9 The reasoning §9.8 overruled (kept 2026-09-04)

> **Historical record — nothing here is an open task.** These bullets were
> written as checkboxes on 2026-09-04 and are kept in that form so the text is
> unaltered, but every one of them was overruled or completed the same day by
> §9.8. Do not pick work out of this section.

Filed on 2026-09-04 as the case for deferring the restructure, and
overruled the same day. Kept verbatim, because the numbers in it are the
measurements the decision was actually made against, and because a
decision record that erases what it overruled teaches nothing. Where it
was wrong is now checkable: it argued the split would break the
`CWD = src/` convention — it did, and that convention turned out to be
the thing worth losing.

- [x] **Done anyway — overruled by §9.8 on 2026-09-04 (closed 2026-09-24).** **Repo rename `sign2speech` → `signbridge`.** Rejected *for now*, not on
  taste: the justification is bidirectionality (speech → sign), and there is no
  synthesis direction anywhere in the repo, the README, or this TODO. §8 has not
  even settled whether *sentence-level* recognition is in scope. Blocked on §8.
- [x] **Done anyway — overruled by §9.8 (closed 2026-09-24).** **The `packages/sb-*` uv workspace split.** All of `packages/` is
  **5,532 lines** across 20 files, single developer, no test suite, no CI. Six
  workspace members rooted at `packages/*/src/sb/<pkg>/` would add six
  `pyproject.toml`s, editable installs, and an import-root change to every
  notebook and CLI — and would break the `CWD = src/` kernel convention that
  `sb.core.paths` and the `sys.path` bootstrap in `modules/scripts/` are both
  built around. The seam the split is meant to create (shared landmark schema +
  subset indices) already exists as `modules/dataset/landmark/`, and §9.4 makes
  it enforceable without moving a single file. Revisit when POPSIGN training
  starts (§9.5 is the real preparation for it).
- [x] **Done by §9.8 (closed 2026-09-24).** **Move the registry out of `src/data/` to a top-level `registry/`.** The
  review's stated reason — that `.gitignore` negation is fragile and can be
  "silently defeated" — does not apply as written: `src/data/*` globs the
  *contents* (not the directory), so `!registry/runs/` works, and verifiably
  does today (264 files tracked; `git check-ignore` does not match
  `registry/runs/<id>/meta.json`). What remains is a naming/legibility
  argument, worth ~1 line in `paths.py` plus a `git mv` of 264 files and every
  path reference in docs and notebooks. Low value alone — bundle it with the
  layout change if that ever happens.
- [x] **Done by §9.8 — `sb-promote` exists (closed 2026-09-24).** **`aliases.json` + a `promote` command.** The right idea, but a promotion
  pointer needs something to promote *to*. There is no app, no deployment
  target, and TFLite export already exists in
  `gislr.2.models.evaluation.ipynb`. File it properly when a deployment target
  is real; `submission.tested` (§6.3) already covers the query that exists today.
- [-] **Duplicate of §9.8's MLflow item (2026-09-24).** **MLflow as a mirror.** Recommend against. It is a second write path for
  data `meta.json` already holds, needs a server process, and its stated payoff
  (parallel-coordinates / run-comparison views for the ablation write-up) is a
  plotting cell over `index.csv`, which is already a flat, DuckDB-queryable
  table with one row per run. If the ablation chapter needs those views, add
  the plot to `gislr.2.models.evaluation.ipynb` — hours cheaper, and it cannot
  drift from the registry.
- [-] **Duplicate of §9.8's skeleton item (2026-09-24).** **`apps/` (web/edge/shared-ts), `sb-synthesize/`, `sb-rescore/`
  skeletons.** No code, and in §8's case no scope decision. The review's own
  step 6 says "empty scaffolding rots" while its structure diagram creates four
  such directories; the advice is right and the diagram is wrong. Prompt
  versioning + a frozen eval set is a genuinely good idea and belongs under §8
  the moment §8's scope question is answered — not before.

## 10. Extraction in TypeScript, staged environments, artifact naming (2026-09-05)

Three changes requested together. §10.1 is the large one and is **not finished** —
the code exists and has never run. **§10.1 and §10.3 deprecated 2026-09-22**
along with POPSIGN (§2) — both existed to serve POPSIGN extraction and there
is no other workload to run them against. §10.2, §10.4, §10.5 are unaffected
(dataset-agnostic infrastructure / GISLR-specific).

### 10.1 Deno/TypeScript extractor — `packages/sb-extract-ts` — DEPRECATED 2026-09-22

**Kept for history; not pursued.** This section blocked on measuring parity
against POPSIGN video, which no longer matters — POPSIGN is deprecated (§2)
and its extracted data deleted. `sb-extract-ts` is marked paused, not
deleted (its own README explains why).

Deno rather than Node for one concrete reason: it implements `ImageData` and Web
Workers natively, so MediaPipe's WASM build needs no `canvas` native module in a
long-running frame loop.

- [x] Built as a package, not a script: `schema.ts` (LANDMARK_TENSOR v1, ported
  from `sb.core.schema`, which stays authoritative), `npz.ts` (float16 + NPY 1.0
  + stored-ZIP, atomic write), `frames.ts` (ffmpeg rawvideo), `worker.ts` (one
  landmarker, many videos), `cli.ts` (pool + resumable manifest).
- [x] Three defects in the original sketch fixed rather than carried forward,
  each a *silent data bug*:
  - **chunk-boundary frame shear** — a pipe chunk routinely ends mid-frame;
    taking whole frames per chunk and discarding the remainder desyncs every
    later frame into a shear of two, and MediaPipe returns plausible landmarks
    for them. `frames.ts` carries a buffer.
  - **wrong output contract** — JSON of four arrays instead of `(T,543,3)`
    float16 npz in GISLR holistic row order. That row order is what makes the
    `subsets.py` index lists valid for POPSIGN.
  - **NaN policy and channel count** — undetected must be NaN, not 0, and the
    spec is xyz, not xyz+visibility.
- [x] Worker reuse (a worker per video re-downloads and re-compiles the WASM
  graph) and manifest-driven resumability, matching the Python extractor.
- [x] **Executed 2026-09-05, and it does not work.** `deno` (2.9.6) and `ffmpeg`
  (9.0.1) are both installed now. 12 clips through
  `python -m sb.extract.parity_run`: **0/12 extracted**, every one failing with
  `ReferenceError: WebGLRenderingContext is not defined` inside
  `_emscripten_webgl_do_create_context`, during `Module._changeBinaryGraph` —
  i.e. graph *construction*, before a frame is submitted.
- [-] **POPSIGN deprecated 2026-09-22.** **BLOCKER, and it is not ours to fix in this package.**
  `@mediapipe/tasks-vision` is the *web* build: its graph creates a WebGL
  context whatever `delegate` is asked for (`"CPU"` selects the inference
  backend, not the image pipeline), and Deno has `ImageData`, `OffscreenCanvas`,
  `createImageBitmap` and WebGPU but **no WebGL** — `getContext("webgl2")`
  returns null. Node is no better placed; it would need `headless-gl`, the
  native module the Deno choice existed to avoid. Three options, in
  `docs/reports/extractor-parity.md` §5: drive the web build from headless
  Chrome (also the §10.2 livestream target, so the work carries over), keep
  extraction on `packages/sb-extract` (native C++, no GL — it has already done
  33,599 clips), or native GL in Deno (not recommended).
- [x] `deno check` now passes. It did **not** before: under Deno 2.9.6 / TS 6,
  `Uint8ClampedArray<ArrayBufferLike>` is not assignable to `ImageData`'s
  `ArrayBuffer`.
- [x] Two defects found by trying to run it, either of which would have produced
  a confident and meaningless parity number:
  - **geometry could never have matched** — the TS side hardcoded
    `scale=640:480 -r 30` while the Python side feeds cv2's native frames at the
    video's own rate. POPSIGN is 1944x2592 *portrait* at 30 / ~29.92 / 120 fps,
    so that inverted the aspect ratio *and* resampled time (frame counts off by
    up to 4x). Geometry now defaults to per-clip native via `probe()`, which
    existed for exactly this and had never been called; the flags are overrides.
  - **different models** — the TS side defaulted to the bucket's unpinned
    `latest` while Python loads the local `.task`. `--model` now points both at
    `data/external/mediapipe/tasks/holistic_landmarker.task`.
- [x] Ported the Python extractor's resolution-change lesson: the landmarker is
  rebuilt when frame size changes, or the reused graph fails INTERNAL
  `RET_CHECK ... current_mat->rows == previous_mat->rows` on POPSIGN's mixed
  1944x2592 / 1080x1920 video.
- [x] `src/dom_shim.ts` — the browser globals MediaPipe reaches for
  (`document`, a `window`, a `<script>` loader that fetches and evals, `process`
  hidden so Emscripten does not take its node branch). **Not a fix and labelled
  as such**: it exists so the failure names the real constraint instead of
  stopping at `document is not defined`, which reads like a missing polyfill.
- [x] **`npz.ts` format logic validated** — `tools/verify_npz_format.py`
  transcribes `encodeNpy`/`encodeNpz`/`toFloat16` into Python and numpy reads the
  result: header padding, ZIP offsets, float16 with NaN preserved and exact
  binary fractions intact. Passes.
- [-] **POPSIGN deprecated 2026-09-22.** Still unproven: that the **TypeScript itself** runs correctly. Everything
  up to the MediaPipe call now is — ffmpeg spawns and decodes, the pool
  dispatches, the manifest records all 12 units — but no npz has ever been
  written by it, so `--file <clip>.npz` still has nothing to check.
- [-] **POPSIGN deprecated 2026-09-22.** **BLOCKER before it extracts anything trainable: parity — still not
  measured.** `python -m sb.extract.parity_run --limit 12` now does the whole
  thing (seeded selection → hardlink staging → both extractors → npz format
  check → `sb.extract.parity`), and reports `BLOCKED` because the TS half
  produces nothing. 33,599 POPSIGN test clips already exist from the Python
  path; if the train split came from the TS path and the two disagree
  systematically, that difference sits **between the splits**, a model learns
  it, and no accuracy metric reveals it. Do not mix extractors across a split.
- [x] The Python **reference tree** for that comparison is reproducible on
  demand: 12 clips (one per label, seeded 42), 47.7 s, face and pose detected on
  1.00 of frames, hands 0.00–0.73. The tree itself lives in `data/temp/parity/`
  and is deleted per the temp policy, but the selection is cached to
  `data/cache/popsign/parity/selection.json`, so a re-run rebuilds exactly those
  clips. Parity needs only the TS side to appear.
- [-] **POPSIGN deprecated 2026-09-22.** Note for whoever picks this up: the parity sample comes from POPSIGN
  **train a–e**, not test. The test split's videos are gone — `popsign_cycle`
  deletes video once landmarks verify, which is its whole point — and parity is
  a comparison on identical inputs, so the split does not matter but the video
  existing does.
- [-] **POPSIGN deprecated 2026-09-22.** Model asset is pinned to the bucket's `latest`, the only published path.
  Mirror the `.task` file if extraction reproducibility matters.

### 10.2 Livestream mode — the end goal, not yet started

**2026-09-24: now part of §12.5** (Cloudflare Workers deployment research), which
decides where MediaPipe runs. The `schema.ts` dependency below points at the
paused `sb-extract-ts`; re-check it in 12.5.

**Unaffected by POPSIGN's deprecation** — this is about a live camera feed,
not a stored dataset, and the deployment target is GISLR-trained models
regardless. It depends on `sb-extract-ts`'s `schema.ts` (paused, not
deleted, §10.1) and the deployment-target decision §9.8 left open, not on
POPSIGN itself.

- [ ] `runningMode: "LIVE_STREAM"` is a genuinely different contract from
  `VIDEO`: a result callback rather than a return value, and frames dropped
  under load. That is right for a camera and wrong for a corpus, so it is a
  second entry point rather than a flag on the batch one.
- [ ] It belongs with the app surface (`apps/`), against the same `schema.ts`,
  and it is what makes the streaming architecture choice (`StreamingGRU`) pay
  off. Needs the deployment target decision that §9.8 left open.

### 10.3 POPSIGN one part at a time — `sb.extract.popsign_cycle` — DEPRECATED 2026-09-22

**Kept for history; not pursued.** POPSIGN is deprecated (§2); the two open
items below (running the cycle, regenerating the train manifest) are closed
as not-pursued, not completed. `popsign_cycle.py` stays in `sb-extract`,
marked paused rather than deleted.

- [x] download → extract → **verify** → delete, resumable at part and clip level.
  ~870 GB does not fit; the landmarks are ~14 GB, so the video is a transient
  input.
- [x] Deletion is gated on verification, not on the extractor exiting 0: every
  clip must have a `done` unit, the npz must exist, and a seeded sample must
  pass the spec. A part deleted while partly extracted costs a ~220 GB
  re-download to notice.
- [x] ~~Not yet run. Start with `--part test`...~~ — **deprecated 2026-09-22,
  not pursued**: POPSIGN's extracted test split no longer exists on disk to
  verify against.
- [x] ~~`train.csv` still describes 1 of 4 parts...~~ — **deprecated
  2026-09-22, not pursued**: the manifest and its underlying npz have been
  deleted along with the rest of POPSIGN's extracted data.

### 10.4 Per-stage environments — `ops/envs.ps1`

- [x] `uv sync --package <member>` with `UV_PROJECT_ENVIRONMENT` gives one env
  per stage. **Measured**: an `sb-mlops`-only environment drops torch,
  tensorflow, mediapipe and opencv — so the "`sb-mlops` must not import
  `sb-recognize`" invariant is now executable rather than merely documented.
- [x] Fixed a break this introduced: `packages/*` matched the new Deno package,
  which has no `pyproject.toml`, and that fails **every** `uv sync`. Excluded.
- [ ] The default `.venv` is still the fat one and is what notebooks use. Decide
  whether the notebook kernels should move to `.venvs/train`, which would make
  the isolation real for the surface that actually trains.

### 10.5 Kaggle variation = architecture

- [x] Variation is now just the architecture (`gru`, `bilstm`,
  `conv1d-transformer`); subset, coords, score, params, regime and an
  UNFINISHED marker moved into the **version note**, with the full record in the
  `meta.json` uploaded beside the weights.
- [x] `sb-sync rescheme` added for the migration, since Kaggle has no rename: it
  pulls each affected run back (sha256-verified, because checkpoints are pruned
  locally), re-uploads under the new handle, and re-prunes. Dry run by default.
- [ ] **The old 16 variations are still on the model page.** `rescheme` never
  deletes remotely. Remove them by hand once the new ones look right.
- [ ] Consequence to live with: versions of one variation are no longer
  all-else-equal, so a version list mixes subsets and cannot be read as a
  learning curve. Restores are unaffected — the manifest pins an exact
  `<variation>/<version>` handle per run.

---

---

## 11. Streaming Confidence & Reset (live per-frame prediction, "Plan 2")

**Promoted from Backlog (2026-09-22)**, where it sat as an unscoped remark
since 2026-09-19. The user's own live-inference architecture (MediaPipe →
streaming model with per-frame updates and a reset signal → periodic
LLM reader) supplies the missing pieces (a concrete reset trigger, a UX
framing) the original remark lacked; this section replaces that stub with a
phased plan, ordered so the untested assumption gets checked before anything
is retrained.

**Original problem, unchanged.** `sb.recognize.architectures.StreamingGRU`/
`StreamingLSTM` (and `sb.recognize.interp.models.LandmarkRNN`) are trained
with a loss at the **last valid frame only** — they can technically run
frame-by-frame at inference (the recurrent state is causal), but nothing
supervises the running prediction to be meaningful mid-sequence.

**Scope decisions made when this was promoted:**
- **`bilstm` is excluded entirely, permanently** — a bidirectional model's
  backward pass has already read the whole future sequence at every
  position, so there is no causal per-frame readout to expose (see
  `sb.recognize.architectures.BiLSTM`'s docstring). Reference-only, same as
  §4.1. `cnn`/`dnn` stay in as **no-memory controls** (`cnn`: finite
  receptive field, not unbounded state; `dnn`: exactly memory-free per
  frame) — useful baselines, not reset candidates.
- **"Confidence reset" and "hidden-state reset" are two different things.**
  For `gru`/`lstm`, reset means clearing the model's own carried recurrent
  state (`sb.recognize.streaming.RecurrentSession.reset`). For `cnn`/`dnn`
  there is no persistent state to reset — an external accumulator would be
  needed if either is ever used as a live "running confidence", which is not
  built here.
- **The reset MECHANISM and the reset DECISION-MAKER are developed and
  validated separately.** `AcceptTrigger` (rule-based: threshold + hold
  duration) proves the mechanism works before an LLM (the user's Plan 2 §3,
  reading confidence scores and matching a next-word suggestion) becomes the
  thing that decides when to pull it — see the cross-reference to §8 below.

### 11.1 Phase A/B/D — run 2026-09-22, 0 failures

`experiments/recognition/gislr.3.streaming.confidence-eval.ipynb` (new) +
`sb.recognize.streaming` (new module: `per_frame_probs`, `RecurrentSession`,
`build_synthetic_stream`, `AcceptTrigger`) + `forward_all` added to
`StreamingGRU`/`StreamingLSTM`/`CausalConv1D` in `architectures.py` (mirrors
the existing `LandmarkRNN.forward_all` convention and its same
last-frame-only-supervision caveat). Full results:
`docs/reports/streaming-confidence.md`.

- [x] **Phase A, run**: per-frame true-class confidence binned by relative
  position within a clip (early/mid/late). The naive aggregate looked weak
  (`gru` 0.05/0.14/0.21) but conflated two regimes — splitting by whether a
  segment starts fresh or carries a prior sign's un-reset state (§3b, added
  after the first run) shows a **fresh** start already accumulates strong
  evidence (`gru` 0.16 → 0.42 → **0.58**, matching the whole-video confidence
  numbers in `five-arch-benchmark.md`), while a **contaminated** one stays
  near-zero throughout (`gru` 0.01 → 0.05 → 0.09). The last-frame-only-
  supervision concern that motivated this phase is real in principle but is
  **not** the dominant effect measured here.
- [x] **Phase B, run**: oracle reset (at the true boundary) vs. no reset —
  bleed-through drops **19.6→0.8 frames** (`gru`) / **16.1→0.4** (`lstm`);
  re-acquisition latency drops **41.4→23.3** (`gru`) / **36.6→21.5** (`lstm`).
  Reset is not a marginal improvement, it removes most of the effect.
- [x] **Mechanism verified on real data, not just synthetically**:
  `RecurrentSession.step` reproduces `per_frame_probs`'s batch computation to
  **1.7e-06** (`gru`) / **1.0e-06** (`lstm`) max abs diff (checked on CPU —
  GPU cuDNN dispatches a different kernel for single-step vs whole-sequence
  calls, producing ~5e-4 divergence that is a numerics artifact, not a
  correctness question).
- [x] **Phase D, run**: `AcceptTrigger` sweep over oracle-reset streams.
  `gru` dominates `lstm` at every `(tau, hold_frames)` setting. Real
  precision/coverage trade, no single winner — `tau=0.7, hold=5` is a
  reasonable middle ground for `gru` (0.858 precision, 51/120 segments
  missed). See the report §2.3 for the full sweep.
- [ ] **Caveat carried forward, not yet addressed**: clips are fed at native
  length, not subsampled to `MAX_SEQ_LEN=128` like training (10/120 sampled
  clips, 8.3%, exceed it). Confirmed not the explanation for the
  fresh/contaminated gap (`dnn` is equally exposed and shows almost none of
  it), but still an open mismatch worth closing before trusting absolute
  confidence numbers.

### 11.2 Phase C — per-frame-supervised retrain (downgraded, 2026-09-22)

- [-] **Superseded (2026-09-24): §12.3 trained per-frame-supervised continuous models (C1/C2).** **No longer the assumed prerequisite.** §11.1's freshness split showed
  clean per-frame confidence is already well-calibrated and accumulates
  sensibly on its own — the blocker was carried-over state, not the
  training objective. Downgraded from "next step" to **optional refinement,
  revisit only if a live reset-wired system still under-performs** after
  §11.2's actual lever (reset, not retraining) is in place.
- [-] **Superseded by §12.3 (2026-09-24).** If revisited anyway: add an auxiliary per-frame loss term (target =
  the video's static label at every frame, weight ramped from 0 over the
  first ~15-20% of the sequence, small coefficient e.g. 0.1-0.3) alongside
  the existing last-frame loss — never replacing it, to avoid regressing the
  canonical `gru`/`lstm` accuracy numbers. New config keys in a
  `gislr.streaming-confidence.json`-style file (never hyperparameters
  hardcoded in a cell, per this repo's convention), `gru`/`lstm` only.

### 11.3 Phase E — LLM as reset decision-maker (blocked on §8)

**Superseded 2026-09-24 by §12.6** (fused LLM + recognizer acceptance); §8's
scope question is answered and §12.1 supplies sentence-level data.

- [-] **Superseded by §12.6 (2026-09-24).** **Blocked on §8's still-open scope question and its own open item**:
  where does "next-word suggestion" data come from? Neither GISLR nor
  POPSIGN has sentence-level transcripts. Do not start building this against
  an undefined vocabulary/language-model source.
- [-] **Superseded by §12.6 (2026-09-24).** Once unblocked: swap `AcceptTrigger` for an LLM reading the same
  confidence stream this notebook produces — decoupled from Phase A-D by
  construction, since the reset mechanism is already validated independently
  of who decides to pull it.

Related: §4.1 (BiLSTM accuracy-vs-streaming-viability decision, same
exclusion rule applied here), §8 (LLM scope + the missing sentence-level
data source, now shared by both), §10.2 (livestream mode — this stays a
notebook-based simulation until that's built).

---

## 12. Continuous Signing: Sentence Dataset, Frame-Level Model, Open Vocabulary (2026-09-23, new)

**The user's four experiments, plus pipeline research** — each step gets its
own plan and the user's review *before* it is built. Order agreed 2026-09-23
(it differs from the user's numbering because of dependencies):

| step | user # | what | status |
|---|---|---|---|
| **12.1** | 2 | GISLR-Sentences: sentence corpus + test-derived continuous dataset → Kaggle | **built + run 2026-09-23; upload pending** |
| 12.2 | 3 (baselines) | existing isolated models on 12.1 | **run 2026-09-23** — `docs/reports/sentence-baselines.md` |
| 12.3 | 1 + 4 | continuous frame-level model (null class, add-a-sign head) | **trained 2026-09-23** (C1/C2/C3 + C-open); final eval (§4) re-running 2026-09-24 |
| 12.4 | 3 (rerun) + 4 | 12.3 on 12.1; teach held-out signs (model side of custom signs) | plan pending — **do 1st** |
| 12.5 | — | pipeline structure + **Cloudflare Workers** deployment research | **research done 2026-09-24**; build plan awaiting review |
| 12.6 | — (2026-09-24) | downstream LLM: next-gloss prior fused with recognizer confidence, gloss → fluent English, TTS | plan pending — **do 3rd** |
| 12.7 | — (2026-09-24) | user-facing custom-sign feature (capture → enroll → persist) | plan pending — **do 4th** |

**Decisions the user made (2026-09-23):**
- Sentences written by Claude, validated by script, user reviews samples.
- Null frames: interpolated transitions + hands-absent rest segments (not hard cuts only).
- **Model output contract (12.3):** at every frame, a confidence over *every*
  gloss for the sign in progress since the last reset — once it crosses a
  threshold the sign is accepted and the scores reset for the next sign.
  This is §11's `AcceptTrigger` + `RecurrentSession.reset` loop, now as the
  design target of a model trained for it (so §11.2's per-frame-supervised
  retrain is back in scope, as part of 12.3, not as an optional refinement).
- Kaggle dataset private; sentences built **only from `test.csv`**, using as
  many test clips as possible.

**Facts that shaped the plan (measured 2026-09-23):** GISLR clips are
trimmed to the sign (0 hand-absent lead-in/out frames over 1,500 test
clips; 0.4% hand-absent frames overall; length median 22 / p95 131 / max
405), so every null frame must be synthesized. Each test signer covers
221–249 of the 250 glosses, so one signer per sequence is feasible.
`test.csv` doubles as the canonical val set every existing checkpoint was
early-stopped on — 12.2's baselines carry that selection advantage, and
12.3 must select on a `train.csv`-derived set.

### 12.1 GISLR-Sentences v1 — built + run 2026-09-23 (local), upload pending

`experiments/recognition/gislr.0.dataset.sentences.ipynb` + new
`sb.recognize.sequences` (`corpus.py`, `compose.py`, committed corpus in
`corpus_data/`) + `configs/gislr.sentences.json`.

- [x] Corpus v1: 1,757 sentences, 11 themes, ASL gloss order, length 2–7
  (mean 3.14); every gloss in ≥12 sentences; POS lexicon for all 250
  glosses. The vocabulary has no I/you (ASL points), no `eat` (ASL FOOD
  doubles for it), no `want`/`big`/`small`/`good` — sentences are simple
  and third-person heavy.
- [x] Coverage-driven planner: **all 18,896 test clips placed**, 8.0% of
  slots re-use a clip (flagged), 0 orphans → 6,616 `sentence` sequences
  (1,227 distinct sentences); `control` split = same clips, random order,
  each once (~6.1k sequences).
- [x] Smoke test (40 sequences): every segment bit-identical to its source,
  rest frames hands-NaN/body present, null frames labelled −1; ~0.08 s per
  sequence.
- [x] ~~Local build + `kagglehub.dataset_upload`~~ — **uploading ~16 GB from
  this machine was too slow (2026-09-23)**. Replaced by a Kaggle-side build:
  `gislr.0.dataset.sentences-kaggle.ipynb`, **generated** by
  `python -m sb.recognize.sequences.kaggle_notebook` from the exact source of
  the five modules it needs (registered under their real names, so no
  re-implementation), the config and the corpus files. Verified
  end to end on a 40-sequence slice with every real `sb.*` import blocked:
  identical plan (6,616 / 8.0% / 1,227), round-trip 20/20 at float32 and
  float16. Regenerate it after changing any of those inputs.
- [x] **Local build run (user, 2026-09-23)**: 12,727/12,727 sequences, 0
  failed; round-trip 300/300 exact; 0 segment-consistency issues;
  18,896/18,896 clips used in `sentence` (17,560 once, max 7 uses);
  **12.12 GB** float32; frames 69.1% sign / 7.4% transition / 23.5% rest in
  both splits (matched by construction). `docs/logs/daily/2026-09-23.md`.
- [x] **Fixed in 12.3's training generator (closed 2026-09-24); the dataset v2 is tracked in 12.3's last item.** **Realism flaw found in the exemplar**: at rest the pose wrists stay at
  signing height while the hands are NaN, so rest is trivially separable
  (hands NaN ⇒ null). 12.2 must score rest and transition frames
  separately; 12.3's training generator should synthesize a lowered-hands
  rest (wrists interpolated toward the hips, hand landmarks present) rather
  than copying this. Candidate for a v2 of the dataset.
- [ ] **Publish**: run the Kaggle twin (GISLR_Stratified as input, CPU, Save & Run All),
  then Output → New Dataset `GISLR-Sentences` (private). §3 stops early if
  the projected size exceeds Kaggle's ~20 GB `/kaggle/working` limit
  (~16 GB estimated at float32; `STORAGE_DTYPE = "float16"` halves it).
- [x] **Decided by the user 2026-09-24: keep the gloss `minemy` as is** (no possessive-only
  rewrite). Review the printed sentence sample. Open question from the user
  review: `minemy` (my/mine) is also used as "I" in sentences like
  `MINEMY HUNGRY` because GISLR has no I/me sign. Rewrite those to
  possessive-only, or keep them?
- [ ] Follow-up, not blocking: synthesized null frames are not real
  transitions/rest — any 12.2–12.4 number is an upper bound on real
  continuous signing until validated on real multi-sign video.

### 12.2 Baselines on GISLR-Sentences — built + run 2026-09-23

User-approved decisions: τ/hold chosen on **5 held-out selection signers**
(seeded) of the `sentence` split and every number reported on the other
16; B4 sliding window included for `cnn`/`dnn`/`bilstm` (+ `gru` as a
reference); v1 published as is, with the `minemy` and rest-realism fixes
deferred; the dataset is read via kagglehub
(`sb.core.paths.gislr_sentences_dir`), falling back to the local build
until the Kaggle copy exists.

`experiments/recognition/gislr.3.streaming.sentence-baselines.ipynb` +
`configs/gislr.sentence-baselines.json`; new code:
`sb.recognize.streaming` (`clip_probs`, `first_accept`, the vectorized
`AcceptTrigger`, `decode_stream` with a per-stream forward cache and a
reset/no-reset switch, `collapse_repeats`, `window_probs`, `decode_windows`),
`sb.recognize.sequences.metrics` (edit alignment, GER, latency, and where
insertions land by frame kind), and `sb.recognize.sequences.baselines`
(content-addressed stream feature cache, registry/five-arch loaders,
signer split).

- [x] `first_accept` ≡ `AcceptTrigger`: 0 mismatches over 3,000 random
  cases (+ a real-probability parity cell in the notebook).
- [x] `decode_stream` ≡ the true live loop (`RecurrentSession.step` +
  `AcceptTrigger` + `reset()`, CPU): 0 mismatches over 120 cases each for
  `gru`/`lstm`, with and without reset (also a notebook cell).
- [x] Smoke run of every cell (6 sequences/group): passes; B1 agrees with the
  isolated predictions on 39/39 clips for all six models. Stream feature
  caches built (2 × ~2.1 GB).
- [x] Decoded three real sequences with every model/mode before building the
  sweep. Found: reset-on-accept **re-fires inside long signs** (`lstm`
  accepted one 247-frame clip 8 times), and no-reset repeats one gloss
  throughout. Both are now scored raw and with consecutive duplicates
  collapsed.
- [x] **Run (user, 2026-09-23).** Parity: 0/960 trigger and 0/120 live-loop
  mismatches; B1 agrees with the isolated predictions on 99.96–100% of
  17,829 clips per model. Results (16 evaluation signers, `sentence`
  split): oracle B1 GER **0.221** (`gru_reg`); best streaming **0.507**
  (`gru` sliding window, collapsed); reset-on-accept **0.659** (`lstm`) /
  0.681 (`gru`). Errors are **deletions** (0.30–0.47), not substitutions
  (~0.14) or insertions (~0.05). Sentence ≡ control (±0.01). Full write-up:
  `docs/reports/sentence-baselines.md`.
- [x] **Why (diagnostic, same day):** with the state reset at each TRUE
  sign start, the fixed-hold trigger commits the right gloss on only 60% of
  signs vs 76% isolated. Short signs (<12 frames, 25% of signs) need a
  short hold, and long signs (50+) commit early and wrong. No single hold
  serves both (best 0.617). The remaining drop to the real stream is reset
  timing. This is the design brief for 12.3: per-frame supervision + a
  model-signalled boundary, not a fixed hold.
- [ ] Follow-up, optional: extend the B2 `hold` grid past 8 and `dnn`'s
  τ/window below 0.4/16 (all selected at grid edges). Expected to move
  those rows a few points, not to resolve the length trade-off.
- [ ] Follow-up: once the Kaggle copy is published, confirm
  `gislr_sentences_dir()` resolves it (this run used the local build).

### 12.3 Continuous model — plan approved + built 2026-09-23, **trained 2026-09-23**

The user approved all recommended options:
- **a dedicated boundary head** rather than null-only commits;
- **training on continuous streams with no state reset** (the decoder may
  still reset);
- **all four runs** (C1 GRU, C2 LSTM, C3 CTC, C-open);
- **a separate open-vocabulary run** with 20 glosses held out.

The design brief comes from `docs/reports/sentence-baselines.md` §5.

**Built:**
- `ContinuousRNN` (+ `ContinuousGRU`/`ContinuousLSTM`, `ARCHS` keys
  `gru_continuous`/`lstm_continuous`) in `architectures.py`: a per-frame
  **cosine** gloss+null head (`CosineGlossHead`, with `class_mask` and
  `enroll()` for §12.4) and a sign-boundary head. It keeps the isolated
  read-out, so `sb-evaluate` scores it canonically.
- New package `sb.recognize.continuous`:
  - `data.py`: NaN-preserving `train.csv` clip bank (`clipbank_v1`,
    3.0 GB); a per-epoch random partition into one-signer streams of 1–6
    signs; 0–15-frame interpolated gaps; lowered or hands-absent rest;
    per-frame gloss/null/boundary targets.
  - `train.py`: the driver (auto-resume, registry run, `meta.json` every
    epoch). Selection is on validation-stream segment accuracy, from a
    sign-stratified 5% of `train.csv`.
  - `decode.py`: D1 boundary commit, D1r + reset, D3 null-gated, D4 greedy
    CTC. D2 (the user's literal loop) is `streaming.decode_stream(...,
    exclude=null)`.
- Notebooks: `gislr.1.models.continuous.ipynb` (training, one section per
  run, learning curves, canonical isolated eval) and
  `gislr.3.streaming.continuous-eval.ipynb` (the §12.2 protocol, results
  beside the baseline rows, a by-length analysis). Configs:
  `configs/gislr.continuous.json` and `gislr.continuous-eval.json`.

**Checks:**
- [x] `smoke=N` driver mode (no registry writes) passes for all four runs:
  71,801 train clips (20.5k streams/epoch) and 3,780 validation clips
  (1,097 streams); C-open holds out 20 glosses (66,096 train clips).
- [x] Contract tests, GRU and LSTM:
  - isolated read-out == `forward_all` at the last frame;
  - state_dict round-trips with the mask;
  - masked classes are `-inf` (also under CUDA/AMP);
  - `RecurrentSession` matches the batch forward to 6e-7;
  - D2 matches the live loop (0/6 mismatches);
  - `enroll()` unmasks a class.
- [x] The eval notebook smoke-run end to end on random-init fake runs. Its
  parity cell passes, and every artifact and figure is produced.
- [x] **Rest design corrected before any training.** Measured on training
  frames: hips are out of frame in 97% of frames (median y 1.25), hands are
  never detected below y ≈ 0.9, and each hand is present in only ~30% of
  frames. So "lowered" rest now drops the pose wrists to hip height and
  **blanks each hand once it leaves the frame**, instead of keeping the
  hands visible below the image.
- [x] Bug found and fixed: hands-absent rest blanked only the hands' y
  columns and left x present.
- [x] **Train C1, C2, C3, C-open** (user, 2026-09-23) — all four runs early-stopped,
  canonically evaluated. Results: **C1 GRU 0.7188** / **C2 LSTM 0.7144** /
  **C3 CTC 0.3354** (last-frame CTC readout is not meaningful, evaluate under
  D4 greedy CTC on streams) / **C-open 0.6696** (−20-gloss penalty by design).
  Boundary F1 ~0.42–0.45 for C1/C2/C-open. Run IDs: C1 `1790143122`, C2
  `1790144582`, C3 `1790142624`, C-open `1790146838`. Two orphan runs
  (`1790139949`, `1790141422`) exist with `eval: pending` — superseded by the
  fresh runs the training cells created, not on the leaderboard.
- [x] **Eval notebook: §4 lost to a disconnect (2026-09-24) — re-run finished the same day; results below.** The user's long run
  was interrupted. **No §4 state was saved**: the old cell kept every result in
  memory and wrote `final.json` only after all 120 decode loops, and it had
  reached ~C2/D3 (~70 loops). On disk: `diag_C{1,2,3}.json` and
  `sweep_C{1,2,3}.json` only.
  - **Analysis from what survived** (5 selection signers, `sentence` split):
    best is **C1 D3 collapsed, GER 0.413** (sub 0.301 / del 0.084 / ins 0.029;
    sentence acc 0.223; median latency +2 frames after sign end). The best
    §12.2 baseline on the same signers is 0.595 (`gru` B4 collapsed), so **~30%
    fewer errors**. C2 D3 0.437. The user's literal loop on the new model
    (D2 collapsed) gets 0.546, still deletion-heavy (0.36). C3 (CTC) is best at
    0.672 (D4).
  - **The error type changed.** The baselines lost signs (deletions 0.30–0.47).
    C1 D3 finds them (del 0.084), and what remains is **substitutions ~0.30**,
    close to the isolated classifier's ~25% error. Segmentation is largely
    solved; the rest is classification accuracy, which points back to §7.
  - Reset-on-commit (D1r) hurts C1/C2 (insertions 0.36–0.54). That is
    expected: they were trained on streams without a reset. D4 (CTC decoding)
    on the frame-loss models is meaningless (GER 3.5–4.2).
  - **Grid edges:** C1/C2 D1 picked β=0.7 and min_mass=4 (both the maximum);
    D3 picked ν=0.7 (the maximum). C3 D2 picked τ=0.4, hold=1 (minimum).
  - **Fixed in the notebook (2026-09-24):**
    - §4 rewritten to be **resumable**: one part file per (run, split, frames) in
      `final_parts/`, skip-if-exists, headline part first.
    - One forward pass per sequence now feeds all of a run's chosen settings, so
      §4 is ~10x less work.
    - New **§3b** sweeps `decoders_ext` (D1 β up to 0.9 × min_mass up to 16; D3 ν
      0.8/0.9) and skips combinations §3 already scored.
    - Smoke-tested end to end on 3 sequences per group (120 final rows, 12
      parts + reference; smoke output deleted).
    - Outputs cleared.
  - **Next (user):** re-run the notebook. §§1–3 are cached, so §3b and §4 do the
    work, and an interruption now loses at most one part. Then Claude writes
    `docs/reports/continuous-models.md`.
  - **Re-run finished 2026-09-24 (all 13 parts + `final.json`). Report:
    `docs/reports/continuous-models.md`.** Evaluation signers, `sentence` split:
    **C1 D3 c GER 0.293** (sub 0.213 / del 0.064 / ins 0.017, sentence acc 0.377,
    median latency +1 frame) vs best streaming baseline `gru` B4 c 0.507 (−42%),
    `lstm` B2 c 0.659 (−56%), oracle `gru_reg` B1 0.221. C2 D3 c 0.298, C3 best
    0.546 (D4). Substitutions are at oracle level → classification is the limit.
    Hard-cut flips the decoder: D3 0.546, D1 c 0.486–0.491 (best). Reset decoders
    (D1r/D2) insert on transitions (3.4–5.7 / 1k frames). Signs < 12 frames (25%)
    recognized 54% vs 77–82% for longer.
  - [ ] **Follow-up: D5 = D3 ∪ D1 decoder** (commit at the end of a non-null run
    *or* at a high-β boundary crossing inside one), swept on cached forwards in
    the same notebook — targets hard-cut without losing D3's paused-signing result.
  - [ ] **Follow-up: short signs** (< 12 frames, 46% missed) — feed into §7's
    normalization/augmentation ablations.
  - **Deployment choice (recommendation, not yet user-confirmed):** C1 + D3 c
    (ν=0.7, min_len 4), no state resets.
  - ~~Re-run in progress (observed 2026-09-24 ~08:48)~~ — *superseded: the run finished; see the "Re-run finished" bullet above.* §1 parity passes (C1 has 1
    float near-tie at τ, tolerated). §2 diag and §3 sweeps load from cache. §3b ran.
    §4 had just started (`final C1 sentence with-null` 0/5054). §§5–6 have not run.
    The notebook is **not committed** until the run finishes.
    - **§3b results (selection GER):** the best settings do not change. C1 D3
      collapsed stays **0.413** (ν=0.7, min_len 4). ν=0.8/0.9 score worse, so ν=0.7
      is now an interior optimum. C2 D3 uncollapsed moves to ν=0.8 (0.467). C1 D1
      collapsed moves to β=0.8, min_mass 8 (0.537, interior). D1 non-collapsed still
      picks the new maximum (β=0.9, min_mass 16; C1 0.637, C2 0.646), but it is far
      behind D3, so it is not worth widening again.
- [x] **Parity fixed; the run itself is tracked by the partial-run item above (2026-09-24).** **Run the eval notebook** — first attempt 2026-09-23 stopped at the §1
  parity cell (C1: 1/10 D2-vs-live-loop mismatches). **Not a decoder bug**
  (diagnosed 2026-09-23): a float tie at the threshold. At frame 347 of
  sequence row 2 (τ=0.5, hold=2), the live `step` gives p=0.5000002 and the
  batch forward 0.4999995, so the accept lands one frame apart (348 vs 349).
  The two resync at the next emission. The parity cell now counts such
  divergences as near-ties (`TIE_EPS=1e-5`) and fails only on an unexplained
  mismatch. Verified: C1 0/10 (+1 near-tie), C2 0/10, C3 0/10. The parity cell
  doesn't pass `cache`, so the earlier "exclude + cache" hypothesis was
  wrong. Unblocked — re-run the notebook (user), then write
  `docs/reports/continuous-models.md`. `docs/logs/daily/2026-09-23.md` §8.
- [ ] Follow-up: GISLR-Sentences v2 with the realistic lowered rest
  (pose down, hands out of frame). (`minemy` stays as is — user, 2026-09-24.)

### 12.4–12.7 — the user's three new asks (filed 2026-09-24)

The user asked (2026-09-24) for: (a) the full pipeline deployed on
**Cloudflare Workers**, (b) a **downstream LLM** that fuses its next-word
prediction with the recognizer's confidence to accept signs, turns ASL glosses
into fluent English, and speaks it via **TTS**, (c) **custom signs** a user can
teach the model. Partial overlap found: (a) ⊂ §12.5's outline (nothing
Cloudflare-specific) and §10.2; (b) ⊂ §8 + §11.3 (no gloss→English, no TTS);
(c) ⊂ §12.4 (model side only, no user feature). Filed as extensions, not
duplicates.

**Order and why** (differs from the user's 1/2/3 numbering):
1. **12.4 first** — cheapest and unblocked: C-open (`1790146838`) is trained
   with 20 held-out glosses, `CosineGlossHead.enroll()` exists. It decides
   *how* custom signs work (prototype imprinting vs fine-tune), which 12.5
   and 12.7 both depend on.
2. **12.5 second** — research only, but it sets the hard constraints (where
   each model runs, latency and size budgets, which LLM/TTS are available)
   before 12.6 picks an LLM and builds against it.
3. **12.6 third** — needs 12.3's decoders (done), 12.1's corpus (done) and
   12.5's model/latency choices.
4. **12.7 last** — product feature on top of 12.4's method and 12.5's
   storage/deploy decisions.

### 12.4 Add-a-sign — model side (plan + review needed)

- [ ] Enroll held-out glosses from 1/5/10 examples — prototype imprinting
  (no retraining) vs short fine-tune with replay; new-class accuracy on 12.1
  and forgetting on the base classes. Start from C-open's 20 held-out
  glosses.
- [ ] Also measure what 12.7 needs: accuracy vs number of user examples, and
  whether a *new signer's* few examples transfer (enroll with one signer,
  test on others vs same signer).

### 12.5 Pipeline + Cloudflare Workers deployment research — **research done 2026-09-24**, build plan awaiting review

**Web app built (2026-09-24, the user's request "prepare the web app for the pipelines").**
- **Scope:** sign → speech only. The user chose that; speech → sign stays out until its gloss
  engine (spaCy-bound `rules_v2`) and a sign-video source are decided.
- **`apps/web`** (Vite + TS):
  - camera / video file / held-out replay → Holistic (VIDEO mode) → LiteRT.js step model → TS
    ports of `OnlineDecoder` + `decide` + `Lattice`, and of the KN n-gram and `gloss2en` →
    `speechSynthesis`;
  - UI: uncertain glosses greyed out (< 0.5), auto-speak only when every gloss is ≥ 0.5, the
    lattice's "waiting" shown, sentence end after 45 null frames, a 30 fps resampling clock, a
    mirror toggle, live fps and ms.
- **`apps/web/tools/export.py`** writes the assets and parity fixtures from Python.
  `pipeline.config.json` holds the deployed rule: the lag-2 lattice from §12.6.
- **Parity, all passing:**
  - `npm test`: decoder 60 streams × 6 rules identical; prior max diff 3e-15; gloss → English
    5,724/5,724 identical;
  - `scripts/browser-check.ts` (headless Chrome over CDP): **24/24 replay streams identical to
    Python's TFLite + OnlineDecoder**, GER 0.219 on those 73 signs; Holistic loads and runs
    (GPU delegate).
- **Finding:** an all-NaN frame (nobody in view) gives p(null) ≈ 0.01. The app treats frames
  with no pose as null and resets the state. Filed as deployment-research §4 risk 5.
- **`apps/edge`:**
  - Worker serving `web/dist` as static assets, `/api/health`, and `/api/english` (Workers AI,
    prompt v1, hash = Python's);
  - verified with `wrangler dev --env offline`; the AI route needs `CLOUDFLARE_API_TOKEN`;
  - not deployed.
- **2026-09-25, user decision: Holistic on CPU.** The GPU delegate was slower on their machine. `pipeline.config.json` `holistic_delegate: "CPU"` (`?delegate=GPU` on the URL to compare). The recognizer was already CPU (LiteRT.js WASM). Re-checked: `npm test` passes, headless replay 24/24 identical, Holistic smoke test runs on CPU.
- [x] **Deployed 2026-09-25**: https://signbridge.onecoder1.workers.dev (Workers Free plan). `wrangler deploy` from `apps/edge`; found and fixed a missing `assets.binding` in `wrangler.jsonc` (every genuine 404 was crashing with a 500 — `dev:offline` never exercised the path since `/` and `/speech` always matched a real file).
- [x] **Camera + overlay, live-verified 2026-09-25** (user: "should show the video along with mediapipe overlay live"). Headless Chrome with a fake camera device confirmed the video and overlay canvas were already both showing and correctly stacked; the overlay itself was just too subtle to read as "live" (2px pale dots, face and pose lumped together, no skeleton). Rewrote `drawFrame` (`apps/web/src/main.ts`): faint face mesh, a bright pose skeleton over the model's own ME-126 upper-body subset ({11-16,23,24}), thicker hand skeletons with joint dots. Verified the new drawing code in isolation (synthetic landmarks, headless-Chrome screenshot) and re-ran `npm test` (11/11) + `browser-check.ts` (24/24 replay identical, Holistic smoke `ok: true`) — no regression.
- [ ] **Next (user):** try the live camera with a real person and a few known sentences —
  check mirroring, fps, and whether the new overlay reads as clearly "live."
- [ ] Later: Durable Object session, custom signs (§12.7).

**User decision (2026-09-24):** landmark extraction **and** inference run on the
client. The user asked to "prepare the full sign to speech pipeline" and to research
TFLite exports, Cloudflare Workers and "Cloudflare remote functions".

**Done 2026-09-24:** `docs/reports/deployment-research.md`, plus
`sb.recognize.export.step` (`export_web(run_dir)`), a single-frame, stateful,
**Flex-free** TFLite export of `gru_continuous`. The existing Kaggle export can't
serve live use: it takes the whole clip, its WHILE loop needs Flex, and it has no
continuous heads. C1 `1790143122` → 3.46 MB, 17 builtin ops. TFLite vs PyTorch
prob diff 2.4e-6. **LiteRT.js 2.5.3 in headless Chrome (WASM): 0.20 ms/frame
median, p95 0.30 ms, diff 6e-7.** WebGPU failed to compile headless (no adapter,
probably); not needed. LiteRT.js is browser-only (fails in Node: needs `document`).
The export is at `registry/runs/1790143122/export/web/` (gitignored, not registered
in meta.json). The cosine class matrix is kept out of the graph (`classes.f32`), so
custom signs are a client-side row append. Also observed: **deno 2.9.6 and node 26
are now installed** (§10.1 had them missing).

**Findings that change other sections:**
- **§12.6: the next-gloss prior should run on the client** (n-gram over gloss IDs).
  Workers AI documents no logprobs, so a hosted-LLM prior means k round trips or
  uncalibrated ranking on every sign's accept path. The LLM keeps only gloss →
  English. *Awaiting user confirmation.*
- **§12.7: custom-sign prototypes** are 1 KB rows. DO storage or D1 plus an
  IndexedDB cache; R2 is not needed.

**Open questions for the user (report §9):** ~~(1) which "remote functions"~~ — **answered 2026-09-24: the user meant Workers RPC, and agrees it isn't needed** since extraction + inference are client-side. The browser↔edge link is a plain WebSocket/HTTP to a Worker, and the browser can't call Workers RPC anyway. RPC only appears as the Worker → session-DO method call (`stub.method()`), which is the default DO API, not a design choice. ~~(2)–(4)~~ **answered 2026-09-24:** **Free** plan; **vanilla TypeScript + Vite** (SolidJS or QwikCity later); the prior moves to the client, and the stages stay separate (§12.6).

**Build plan (report §9), nothing past step 0 built:**
- [x] 0. Step export + LiteRT.js browser parity (2026-09-24).
- [ ] 1. `apps/web` live prototype: camera → HolisticLandmarker (LIVE_STREAM) →
  `frameToRows` → LiteRT.js step model → D3 → on-screen glosses. **Measure the
  achieved fps, check mirroring/handedness on known signs**, then do a small live
  accuracy A/B vs the eval numbers (report §4 risks 1–4).
- [ ] 2. `apps/shared-ts`: generated gloss list / ME_132 rows / manifest schema, plus
  a parity script against `sb.core.*`.
- [ ] 3. n-gram prior + fused acceptance, offline notebook on 12.1 streams vs C1 D3
  (→ §12.6).
- [ ] 4. `apps/edge`: Agent (DO) session, Workers AI gloss → English
  (versioned prompt), TTS stream back (→ §12.6).
- [ ] 5. Custom signs, client-side enroll (→ §12.4/§12.7).
- [ ] Follow-up: try fp16 weights for the step model (≈1.7 MB), re-check parity.

Original research questions (answered in the report):

`apps/` was scaffolded 2026-09-24 (READMEs fix the web/edge/shared-ts split and the
contracts; tooling is left to this research). Write up as `docs/reports/deployment-research.md` (the 2026-09-23 chat plan
for this was never written down — start fresh). Research questions:

- [x] **Where does each stage run?** Camera + MediaPipe landmarks almost
  certainly in the browser (MediaPipe Tasks for Web); the recognizer in the
  browser (TF.js / ONNX Runtime Web / TFLite-wasm) vs in a Worker; LLM + TTS
  on Workers AI or via an external API. Per-frame recognition over the
  network costs a round trip every frame — quantify it.
- [x] **Workers constraints** (verify from current Cloudflare docs, don't
  assume): CPU-time and memory limits, bundle size, WASM support, whether
  ONNX Runtime can run inside a Worker, cold starts.
- [x] **Workers AI catalogue**: which LLMs and TTS models are available,
  their latency/pricing, and whether constrained or streaming output is
  supported (12.6 needs a next-gloss distribution over 250 glosses).
- [x] **Session state**: Durable Objects for per-session LLM context and
  WebSocket streaming; where custom-sign prototypes live (KV / D1 / R2 — R2
  was not activated on this account as of 2026-09-04, §9.3).
- [x] **Export path**: the StreamingGRU/`ContinuousGRU` → browser format.
  TFLite export already exists (`sb.recognize.export.keras`); check it
  covers the continuous heads and the cosine head's `class_mask`.
- [x] Output: an architecture diagram, a latency budget per stage, and a
  recommendation. Resolves §10.2's and §9.8's open "deployment target"
  question.

### 12.6 Downstream: next-gloss prediction + fused acceptance + noise rejection, gloss → English, TTS — **experiments built 2026-09-24**

**Demo diagnosis (2026-09-24, the user's question: "very misleading glosses, translation way off").**
Report: `docs/reports/sign-to-speech-demo-analysis.md`. Artifacts: `data/cache/gislr/pipeline_demo/analysis/`.
All 5,054 evaluation-signer sentence streams, from the cached outputs (= the demo's streaming path).
- **Observed:** the committed demo outputs used `rule_default` (rescore λ=0.3, **no floor**), because
  the next-gloss sweep had not run then. A re-run now picks the sweep's best rule (floor + prior).
- **Recognition:**
  - with no floor, 48.2% of sentences contain a wrong or extra gloss, 40.7% are exact, and
    displayed-gloss precision is 77.7%;
  - floor + prior: 25.1% / 33.7% / 88.1%;
  - lattice lag 1: 21.8% / 34.7% / 89.4%;
  - confidence is informative: ≤0.2 → 27% right (1,857 glosses), 0.2–0.3 → 48%, ≥0.8 → 98.5%;
  - of 3,079 substitutions, 7% are documented look-alike pairs, 2% near-meaning, <1% opposites,
    and **about 90% unrelated signs**;
  - worst recall: nap 32%, give 36%, ride/there/bedroom 44%;
  - for a substitution, the true gloss is in the top 3 38% of the time.
- **Translation (rules):**
  - chrF 73.6 / BLEU 57.9 on true glosses vs 55.0 / 39.2 on recognized glosses (389 reference
    streams); 59% of streams get different English;
  - failure patterns even on true glosses: noun lists, locative prepositions, clause/time order,
    `finish`/tense, `hesheit`/`owie`, adjective vs verb.
- [ ] **Fix A1:** pin the demo/client rule in `gislr.pipeline-demo.json` (lattice lag 1 once §2.2's
  evaluation check holds, else floor + prior θ0.3) instead of reading the sweep implicitly.
- [ ] **Fix A2:** show uncertainty in the UI: glosses next to the English, grey out anything under
  0.5, speak only if every gloss ≥ 0.5 or after the user confirms. **Fix A3:** refuse to translate a
  sequence the trigram finds very unlikely when a gloss is also low confidence.
- [ ] **Fix B4:** an LLM gloss → English arm with top-3 alternatives + confidences per sign and a
  meaning guard, scored on *recognized* glosses (needs Cloudflare credentials in `.env`).
- [ ] **Fix B5:** rule-engine patches for the listed patterns (offline fallback), each measured on
  the references. **Fix B6:** add English-from-recognized-glosses chrF to the demo's integration check.
- [ ] **Fix C7/C8:** a better recognizer overall (the errors are a long tail); noise as null; targeted
  data for the worst glosses; **run the demo's `VIDEO_FILE` path on real signing**.
- Next action: the user picks which fixes to do first. Claude's recommendation is A1 + A2 + B6
  (quick), then B4 when credentials exist.

**User decisions (2026-09-24, second round):**
- **Stages stay separate:** MediaPipe extraction → **recognizer + next-gloss prediction**
  (side by side) → **gloss → English** ("the opposite of the rule-based model") → TTS.
  One layered video → text model is **future work** (Backlog).
- Next-gloss prediction: after e.g. `minemy give`, predict every possible next gloss
  with a value. **Accept a sign when the prediction and the current sign confidence
  match.** The recognizer must **reject noise that spans a while.**
- The prior runs **on the client** (accepted; this was the §12.5 proposal).
- Workers **Free** plan; `apps/web` = **vanilla TypeScript + Vite** (SolidJS or
  QwikCity later).

**Built 2026-09-24:**
- `sb.rescore.prior`: interpolated Kneser-Ney n-grams (uni/bi/tri/4-gram; torch-free,
  `to_dict()` for a client port), `UniformLM`, theme-stratified sentence folds and
  theme-out folds, `next_gloss_metrics`, `top_next`. `sb.rescore.neural.GRULM`: a small
  GRU LM (`neural` extra).
- `sb.recognize.continuous.fuse`: D3/D1 segments → recognizer vote `q` → rules `none`
  (θ floor), `rescore` (`q·p^λ`), `agree` (the user's rule: fused top gloss in the
  prior's top-k **and** `q ≥ θ_lo`, or `q ≥ θ_hi` alone), each with a `max_len` noise
  gate. **Parity verified 2026-09-24:** no prior + no gate reproduces C1 D3 c exactly
  on all 5,054 eval streams (GER 0.29338, float16 cache path included).
- `sb.recognize.sequences.noise`: `fidget` / `hold` / `reverse` blocks from same-group
  sign frames, inserted into gaps. `sb.recognize.continuous.cache`: chunked float16
  forward cache.
- `sb.rescore.gloss2en`: reverse rule engine (the inverse of `rules_v2`) + `to_gislr`
  round-trip mapping. `sb.rescore.client`: Workers AI REST client with a JSONL cache.
  `prompts/v1/gloss2en.txt`. `evalset/gloss2en.v1.jsonl`: 132 sentences with draft
  references.
- Notebooks: `experiments/recognition/gislr.4.downstream.next-gloss.ipynb` (config
  `gislr.downstream.json`) and `gislr.4.downstream.gloss-to-english.ipynb` (config
  `gislr.gloss2en.json`).

**State (2026-09-24):**
- `gislr.4.downstream.next-gloss.ipynb`: **built, smoke-tested (20 streams/group, no GRU),
  not run.** It trains the GRU LMs (17 tiny fits, CPU), so **the user runs it**. Expected
  cost: forward about 2–5 min GPU, then the sweep (about 5.5k decodes on about 1.6k
  selection streams, roughly 20–40 min, resumable per part).
- Count-based numbers seen in the smoke (held-out, pooled): the trigram predicts the next
  gloss top-1 11.6%, **top-5 30%**, top-10 39% (sentence-fold), and 27.5% top-5
  (theme-out). Perplexity is 52 vs 251 uniform. In-sample top-5 64% shows how much is
  memorization. The GRU is not yet measured. Demo: `who` → have / that / finish;
  `yesterday grandma` → give.
- `gislr.4.downstream.gloss-to-english.ipynb`: **run 2026-09-24 without the LLM arm**
  (no Cloudflare credentials in `.env`). Eval set (132): **rules_v1 BLEU 60.2 / chrF 75.0 /
  exact 42%** vs identity 7.9 / 46.8 / 7%. Round trip (432 sentences): content recall
  92.5% (identity 93.5%), invented content 5.5%. Caveat: the references and the rules
  have the same author (Claude), so this is a development number until the user reviews
  the references.

**Second pass (2026-09-24, "create the experiments if necessary"):**
- **First real fusion numbers** (quick check, selection signers, clean, trigram
  sentence-fold, D3): recognizer alone GER 0.4133 (= the continuous eval's selection
  value), **rescore λ=0.3 0.401 (best, −3%)**, λ=1.0 0.477. `agree`: k=10 θ_hi=0.3 0.470,
  θ_hi=0.75 **0.730**. `agree` swaps substitutions for deletions (sub 0.301 → 0.184, del
  0.084 → 0.542). It is a **precision mode** ("say nothing rather than a wrong word"), not a
  GER win. The sweep grid is widened to reach its best point (θ_hi down to 0.2, k up to 50;
  λ down to 0.05).
- **New: `sb.recognize.continuous.online.OnlineDecoder`**, the client's frame-by-frame
  D3 + fused-acceptance loop (the reference `apps/web` ports). Parity with
  `fuse.decode_fused`: 0 mismatches over 2,400 random streams (all rules, gate, collapse).
- **New: `gislr.5.pipeline.sign-to-speech.demo.ipynb`, run 2026-09-24**: held-out streams (or
  a video file via the new `sb.extract.holistic.extract_video`) → **browser TFLite step
  model** → `OnlineDecoder` (trigram prior) → rules English → SAPI speech. **Integration
  check on 300 held-out streams: streaming = offline on 293, 7 float near-ties, 0
  unexplained. GER 0.245, sentence accuracy 47%** (rescore λ=0.3). Recognizer 0.06 ms/frame,
  decoder 2 µs/frame, English about 0.01 ms, speech about 0.2 s. The first default rule
  (agree k10 θ_hi .75) gave GER 0.598, which is what triggered the check above.
- **New: `gislr.4.downstream.tts.ipynb`, run 2026-09-24 (SAPI arm)**: both OS voices, Whisper WER
  0.0% / 0.3%, 0.2 s to synthesize 2.2 s of audio (RTF 0.09). The Workers AI arm (MeloTTS,
  Aura-1, Aura-2 on 20 sentences, about 1.6k neurons estimated) needs credentials.
  **Finding:** Aura-2 costs about 120 neurons per sentence, so the Free plan allows about 80 a
  day. **Default speech = the browser's `speechSynthesis`** (free, client-side). Report
  corrected.
- Fixed in `gloss2en`: a leading `if`/`because` ("The if will rain…" → "If it is raining,
  we stay home.").

**Next-gloss notebook: run complete (user, finished ~17:50 2026-09-24).** The notebook was
re-executed from cache by Claude to save complete outputs, since Jupyter had last saved
mid-run. Parity: D3 baseline 0.2934 = the continuous eval. **Results (evaluation signers,
D3, trigram, sentence-fold):**
- **Prior, clean-tuned (rescore λ=0.3, no floor): sentence GER 0.293 → 0.276** (−6%,
  sentence accuracy 37.7% → 40.7%). Control (no grammar) 0.289 → 0.298 (+3%). theme-out shrinks
  the gain to about a third. Bigram, trigram and 4-gram are equal; GRU slightly worse; unigram gives nothing.
- **Noise: recognizer alone 0.982 on noisy sentences (75% of blocks accepted).** A
  confidence floor θ=0.3 gives 0.580 (39% accepted; fidget 46%, reverse 41%, hold 30%) but
  costs clean 0.293 → 0.347 (sub 0.213 → 0.091, del 0.064 → 0.252). Noise-tuned best (prior +
  floor): 0.565 noisy / 0.326 clean. **The prior does nothing against noise. The length gate
  never helps** (no chosen setting uses it). D1 is worse everywhere (best 0.480 clean).
- `agree` ≈ `rescore` once tuned (0.327 vs 0.326 clean): no gain over rescore + floor.
- The demo's 0.245 was an easy 300-stream sample; the same rule on all 5,054 streams is 0.276.
- Report: `docs/reports/sign-to-speech-downstream.md` §2.1 (+2 figures).
- [ ] **Next (proposal, needs the user's OK): retrain C1 with non-sign activity as null**
  (fidget/hold segments in the training composer), then re-run this notebook. 39% noise
  acceptance under the best decoder is too high for a product. A decoder cannot fix it.
- [ ] **Floor recall experiment (2026-09-24, plan awaiting the user's approval):** the user asked
  for fewer missed signs under the confidence floor without more wrong or extra signs.
  **Diagnostic on the selection signers' cached C1 outputs** (Claude, no training; D3 segments
  labelled by frame overlap, a heuristic, not the GER alignment):
  - clean: the θ=0.3 floor rejects 1,620 segments. 511 of them (32%) were **correct**, which is
    10.4% of the 4,895 true signs. The rest are 1,097 wrong and 12 spurious.
  - noisy: it rejects 4,410 segments: 507 correct (11%), 2,727 noise, 1,071 wrong.
  - Rejected segments' true label is in q's top-3 51%, top-5 60% and top-10 70% of the time
    (accepted: 93/95/97%), so re-ranking can recover part of what the floor now deletes.
  - At the floor's error budget, scoring by `peak` (the max per-frame probability of the voted
    class) admits **+122 correct signs on clean** (about −0.025 deletions). On noisy it admits
    **−248**, because noise produces peaky frames. `q` stays the best single noise separator
    (AUC 0.73 below the floor). Uncertain signs and noise need different features, so the
    selector has to be learned on both, or noise removed at the model (the retrain above).
  - Noise fragments into short D3 segments (median 32 frames), which is why the length gate fails.
  - Plan (phases): (0) budget-matched evaluation harness (real alignment labels, risk–coverage
    curves, per-signer bootstrap); (1) no-training arms: `peak`/margin scores, per-class
    shrunk floors, segment-lattice decoding with one segment of look-ahead (top-k + skip,
    n-gram both sides); (2) a tiny learned acceptance selector (logistic regression on segment
    + prior features, the user runs it); (3) repeat on the noise-as-null retrain.
  - **Approved and built (2026-09-24):** the user said to go ahead and to leave long runs to them.
    - `sb.recognize.continuous.select`:
      - `Acceptor` with scores q/peak/margin/qp/class/linear;
      - `class_shifts`, `LogReg` (numpy, exports plain numbers);
      - `label_segments` (GER alignment);
      - `Lattice` (fixed-lag top-k + skip with the prior on both sides) and `decision_delays`.
    - `fuse`: `Segment.peak`, a `Decider` protocol, `Rule.decide`.
    - `OnlineDecoder`: any `Decider` or `Lattice`, running peak, `flush_all`.
    - Parity on 60 streams: 0 mismatches (`Acceptor` = the old `none`/`rescore`; lattice lag 0 = the
      floor; online = batch for peak, lattice lag 1 and 2, and rescore). `ty` clean.
    - Notebook `gislr.4.downstream.acceptance.ipynb` + `configs/gislr.acceptance.json`: labels,
      references and budget (§1), arms §2–§5, budget-matched choice + eval + per-signer bootstrap (§6),
      figures (§7).
    - Smoke run (30 streams/group, 21 s) passed end to end; smoke outputs deleted.
  - [x] **User ran `gislr.4.downstream.acceptance.ipynb` (2026-09-24, run 1, committed c60c5a4).**
    **The θ grid was too coarse** for the 0.002 budget tolerance, so every family was chosen inside the
    budget, with *more* deletions than the floor (e.g. lattice lag 1: selection clean del 0.377 vs 0.368).
    Run-1 evaluation-signer GER (sentence clean / noisy / control clean / control noisy; noise accepted):
    - floor: 0.347 / 0.580 / 0.343 / 0.580; 39%;
    - floor+prior: 0.326 / 0.565 / 0.351 / 0.586; 40%;
    - lattice lag 1: 0.336 / 0.545 / 0.385 / 0.589; 34%;
    - lattice lag 2: 0.335 / 0.542 / 0.384 / 0.591; 33%;
    - linear λ0: 0.348 / 0.548 / 0.368 / 0.565; 36%.
    Run-1 results are saved as `results/final_run1.json` and `deltas_run1.csv`.
  - Corrections to the diagnostic above, from alignment labels:
    - rejected clean segments: 530 correct (33%), 939 wrong, 151 extra;
    - a rejected *wrong* segment has its true gloss in the top 5 only 37% of the time on clean
      (24% noisy). The overlap-based "60%" included correct segments.
  - Linear selector: held-out-signer AUC 0.844 vs `q` 0.816 in-sample. Its weights are margin, log q,
    log peak and (negatively) the null share; the prior features add little.
  - **Refinement (Claude, 2026-09-24):**
    - The notebook gained §5b: two rounds of 9 θ values at each family's budget edge. It ran on
      the selection signers: 612 settings, about 7 min, `sweep_refine0/1.json`.
    - Also in the notebook: shared `eligible()`/`all_sweeps()`; the choice is per family (lattice:
      best k and λ per lag); the per-stream cache is keyed by a rule hash (it was keyed by label,
      which would have gone stale).
    - The binding limit is extra signs on noisy streams (0.0529).
    - Budget-matched Δ missed signs vs the floor (selection signers, clean / noisy):
      - **lattice lag 1** (k5, λ0.3, θ0.263): **−0.024 / −0.016**, 20 frames median delay;
      - lattice lag 2 (k3, λ0.2, θ0.269): −0.022 / −0.020, 35 frames;
      - linear λ0 (θ0.38): −0.006 / −0.028;
      - per-sign floor: −0.011 / −0.010;
      - floor+prior: −0.011 / −0.003;
      - lag 0: −0.010 / −0.004;
      - qp, margin, peak: worse (peak +0.15 clean).
    - Report: `sign-to-speech-downstream.md` §2.2, figure `acceptance_tradeoff.png`.
  - [x] **The user re-ran `gislr.4.downstream.acceptance.ipynb` (2026-09-24, complete).** Evaluation
    signers, Δ vs the floor with per-signer bootstrap 95% CIs:
    - **lattice lag 2** (k5, λ0.2, θ0.2687) **wins on both sentence variants**:
      - clean: missed −0.014 [−0.018, −0.009], wrong −0.012, extra 0.000; GER 0.320 vs 0.347;
        exact 34.3% vs 30.8%; delay 27 frames;
      - noisy: missed −0.007 [−0.011, −0.002], wrong −0.023, extra −0.004, noise −2.1 pts;
        GER 0.547 vs 0.580; delay 53 frames;
    - lattice lag 1 (k5, λ0.3, θ0.2627): clean missed −0.011 (win); noisy missed +0.002 (n.s.),
      but wrong −0.031 and noise −2.9 pts; GER 0.321 / 0.543; delay 18/25 frames;
    - floor+prior: clean missed −0.018, but noisy noise +0.7 pts [+0.1, +1.2], so it fails the
      noise limit;
    - linear: a noise filter (noisy extra −0.020), clean missed +0.008;
    - control (random sequences): lag 2 +0.020 missed (GER 0.363 vs 0.343); lag 1 +0.051 (0.391);
    - selection → evaluation shrinkage is about half (lag 1 clean −0.024 → −0.011).
    Reports updated: downstream §2.2 (evaluation table), summary, recommendation 0 (**lag-2 lattice
    replaces the plain floor**); demo analysis fix A1.
  - [ ] **Next:** pin the lag-2 lattice as the demo/client rule (`gislr.pipeline-demo.json`, demo fix
    A1) and port `LatticeRunner` to TypeScript with the other decoder pieces. Awaiting the user's go.
  - [ ] Phase 3 (retrain C1 with noise as null) still pending the user's go. §2.2 again shows noise is
    the ceiling: any method that recovers real signs also admits more noise.
- **GRU LM result (§1, complete):** lowest held-out perplexity (51.3 vs trigram 52.1), but
  worse ranking (top-5 25.3% vs 30.0%, MRR 0.185 vs 0.207) and worse on unseen topics
  (perplexity 89.1 vs 74.5). **The n-gram is the predictor to ship.**
- Noise blocks: eval 2,410 fidget / 2,600 hold / 2,535 reverse (median 94/96/21 frames).
- Docs updated 2026-09-24: new `docs/reports/sign-to-speech-downstream.md`; `speech-to-sign-audit.md`
  §10; deployment-research and continuous-models follow-ups; daily log 2026-09-24; week 39;
  `docs/README.md` index (it was also missing the landmark-importance, continuous-models,
  deployment-research and speech-to-sign-audit reports).

**Next actions:**
- [x] **User: run `gislr.4.downstream.next-gloss.ipynb`**: done 2026-09-24, results above, then Claude writes
  `docs/reports/downstream-next-gloss.md` (questions in the notebook's §7).
- [ ] **User: add `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN` (Workers AI: Read) to
  `.env`**, then re-run the gloss-to-English notebook's §2–§4 (about 430 calls/model,
  within the Free plan's 10k neurons/day).
- [ ] **User: review `evalset/gloss2en.v1.jsonl` references** (Claude drafts); the
  reviewed version becomes v2.
- [ ] Follow-up: `hesheit` → "they" is a guess. GISLR has one he/she/it sign, and only
  context (or the LLM) can pick.
- [ ] Follow-up: port the winning prior (n-gram tables via `to_dict()`) and the fusion
  rule to `apps/web` TypeScript, with a parity check.

**User decision (2026-09-24), answers §8's scope question:** the roadmap
*is* continuous/sentence-level. The LLM (1) keeps the session's accepted
glosses as context, (2) predicts the next gloss, (3) that prediction is
combined with the recognizer's per-frame confidence and a sign is accepted
when the combined score clears a threshold, (4) turns the accepted gloss
sequence (ASL order, no inflection) into fluent English, (5) TTS speaks it.

- **Accepted 2026-09-24:** the next-gloss prior runs **on the client** (n-gram / small LM over gloss IDs). The hosted LLM keeps only gloss → English.
- [~] **Fused acceptance, offline first** (built 2026-09-24 as n-gram/GRU priors, see above; awaiting the user's run). Original plan text: (notebook, on 12.1 streams with the
  12.3 decoders): LLM next-gloss distribution *constrained to the 250-gloss
  vocabulary* × recognizer confidence (shallow fusion, weight λ tuned on the
  selection signers). Compare GER against 12.3's best decoder (C1 D3). This
  supersedes §11.3 (LLM as reset decision-maker).
- [ ] **Circularity risk**: 12.1's sentences were written by Claude, so a
  Claude/LLM prior will look better on them than on real signing. Hold out
  sentence *templates*/themes, and report the gain as an upper bound.
- [~] (built: uni/bi/tri/4-gram + GRU arms, awaiting run) Compare a small n-gram LM over the 12.1 corpus as a baseline — if it
  gets most of the gain, the LLM earns its place only for step (4).
- [~] (built + rules arm run 2026-09-24; LLM arm needs credentials) **Gloss → English**: prompt design + an eval set (`sb-rescore`'s
  `evalset/` and `prompts/` skeletons are the natural home). Handle GISLR
  gaps (no I/you, `minemy` doubling as "I" — §12.1).
- [ ] **TTS**: pick a model per 12.5 (Workers AI vs external), measure
  end-to-end latency sign-accepted → audio.
- [ ] When to emit: per accepted gloss (streaming, may revise) vs at a pause
  (sentence end = long null/rest run from the boundary head).

### 12.7 Custom signs — user-facing feature (plan + review needed)

- [ ] Capture flow: user records N examples of a new sign + types its gloss;
  landmarks extracted client-side.
- [ ] Enroll via 12.4's winning method (prototype imprinting runs in the
  client, no server training); persist per-user prototypes (storage per
  12.5).
- [ ] The LLM (12.6) must learn the new gloss exists — add it to the
  constrained vocabulary and give it a user-provided English meaning.
- [ ] Guardrails: reject an enrollment that collides with an existing gloss
  (high cosine similarity to its prototype), and a way to delete/re-record.

### 12.8 Live camera: continuous recognition breaks — diagnosis + fixes (2026-09-25, new)

**User report (2026-09-25), first camera test of `apps/web`:** "most sentences are garbage": missing,
extra and wrong glosses. Isolated recognition is good; continuous streaming is very bad. The user asked
for possible solutions.

- [x] **Robustness probes run (Claude, 2026-09-25, inference only):**
  `gislr.3.streaming.live-robustness.ipynb` → `docs/reports/live-streaming-gap.md`. Setup: C1 + D3 on
  400 evaluation-signer streams. Clean GER 0.293, isolated acc 0.737.
  - **Hurt the stream, not isolated** (the user's symptom):
    - 15 fps with repeated frames: GER 0.504, missed 0.353, isolated 0.719;
    - jitter 1%: 0.516, extra 0.165, isolated 0.730;
    - 6 chained sentences with no reset: 0.415, extra 0.107, recall 0.73 → 0.67 along the chain.
  - **Hurt both:**
    - x squashed 0.56 (landscape webcam vs GISLR's portrait framing): 0.651 / isolated 0.416;
    - mirror: 0.990 / 0.032;
    - scale 0.7: 0.375.
  - **App-side fixes measured:**
    - interpolate instead of repeat: 0.504 → 0.365;
    - EMA α 0.5: 0.516 → 0.337;
    - **reframe** (a per-session shoulder/nose affine to GISLR's medians): aspect 0.651 → 0.334 and
      scale 0.375 → 0.334, costing +0.04 on clean;
    - live-like combination: 0.592 → 0.481 with all three.
  - GISLR framing: shoulders span 0.57 of the image width, shoulder mid y 0.61, nose y 0.38. The app
    requests 640×480 landscape.
- [?] **Open question to the user:** what fps did the UI show? Was mirroring checked on a one-handed
  sign? How far from the camera were you? "Isolated works": was that live, or the offline numbers?
- [ ] **Fix 1 (app, no retrain):** reframe calibration; interpolate in `Clock`; EMA α 0.5; a forced
  reset at the next null frame after ~600 frames. TS ports with parity fixtures. Awaiting the user's go.
- [ ] **Fix 2:** landmark recorder in `apps/web` (download frames + timestamps), then 30–50 real known
  sentences as the first real continuous test set. Measures what the probes can't: real transitions and
  rest, and Tasks-vs-legacy Holistic differences.
- [x] **Research + recommendations (Claude, 2026-09-25):** `docs/reports/improvements-research.md` Part A.
  - Main finding: Zuo et al. (EMNLP 2024, online CSLR) got online WER 38.4% → 22.1% by adding a
    background class for co-articulation and training on clips cut from *real* continuous video by
    forced alignment.
  - Our C1 has only seen synthetic transitions, so real recordings are the biggest lever.
  - New candidates:
    - a One Euro filter instead of the EMA;
    - forced-alignment labeling of recorded known sentences;
    - ASLLRP DAI (real continuous ASL with time-aligned glosses);
    - ASL Citizen (83k webcam isolated videos, Deaf signers);
    - a fixed-context model as an alternative to the drifting GRU state.
  - Recommended order: user check (fps/mirror/framing) → app fixes + recorder → C1 v2 retrain gated on
    probes + real recordings.
- [ ] Add a One Euro filter probe to `gislr.3.streaming.live-robustness.ipynb` (compare with EMA α 0.5).
- [ ] **Fix 3 (retrain C1 v2, needs the user's go):** normalized input; composer augmentations (fps
  drop/repeat, speed 0.7–1.5×, jitter, hand dropout, scale/shift/aspect); streams up to ~2k frames;
  noise as null (§12.6 Phase 3). Score it on the probes and on Fix 2's recordings.

**Status check, sign → speech (2026-09-24, re-confirmed ~13:35).** Unchanged: the recognizer half is done offline (C1 D3 GER 0.293), and none of 12.4–12.7 is started. The next step is the §12.4 plan. **Gap check (2026-09-24).** Nothing deployable exists yet:
`apps/` is READMEs only, `sb-rescore` is a skeleton, and
`sb.recognize.export` (`keras.py`/`tflite.py`) has no support for the
continuous model (`ContinuousGRU`'s null/boundary heads, the cosine head's
`class_mask`) — the §12.5 export item is confirmed open, not just unchecked.

Related: §8 (sentence-level data for an LLM — 12.1's corpus is the first
sentence-level artifact in the repo, though synthetic), §11 (reset
mechanism), §10.2 (livestream mode).

---

## 13. Speech → Sign: the merged Maimuna/Raiyan pipeline (2026-09-24, new) — current goal: speech → gloss

The user brought in the team's Colab notebook
`chosen_merged_asl_pipeline_hybrid_1.ipynb`: Whisper large-v3 → rule engine +
fine-tuned T5 → WLASL / signasl.org video / fingerspelling. It is the reverse
direction to the rest of the repo, and it is the scope `sb-synthesize` and
`experiments/synthesis/` were reserved for. **Audit + integration plan:
`docs/reports/speech-to-sign-audit.md`.** The user's instruction: plan and
research first, then integrate it as its own pipeline.

**Audit findings (2026-09-24):**
- [x] Inventory of what actually runs, plus the dead code (ASR normalization
  unused; `best_of` ignored at temperature 0; `jiwer`/BLEU imported but unused;
  the "3-stage evaluation" is not in the notebook).
- [x] Rule engine run alone on the 30 batch sentences (isolated spaCy 3.8
  env). **The T5 refinement causes the worst errors**: #19 you → HE, #25
  her → ME (meaning flips), #20 `HE HE HE HE HE HE HE`, subjects dropped (#2,
  #4, #12). The rule engine alone has none of these. T5 is better at ME,
  time-fronting, IF/BUT/BEFORE/PLEASE, and phrasal verbs.
- [x] Other issues: WLASL clips played untrimmed (`frame_start`/`frame_end`
  ignored) and first-instance only; exact-match lexicon lookup; signasl.org
  scraping (copyright/ToS, spoofed UA); WLASL is C-UDA non-commercial; no
  reference glosses; Colab-only APIs; conflicting checkpoint provenance
  (NCSLGR vs ASLG-PC12).

**Blocked on the authors / user (report §6):**
- [ ] Get `rule_engine.py`, `run_inference.py`, `step7_aslg_transfer.py`,
  `training_metadata.json`, the T5 checkpoint, and the second dataset's
  notebook/model.
- [ ] Which corpus trained the checkpoint (NCSLGR / ASLG-PC12 / both), and
  which split was held out?
- [ ] The 30-sentence CSV. Do reference glosses exist? If not, write them
  (with the user) so it becomes a scored test set.
- [ ] Decisions for the user: drop signasl.org scraping (recommended); renderer
  **A** (fixed WLASL video) first, then **B** (landmark avatar,
  back-scored by our recognizer) as the research arm (recommended); host
  `sb-synthesize` as the package (recommended).

**Vocabulary (2026-09-24, report §8):** GISLR-250 covers 47% of the gloss tokens in
the team's 30 test sentences, and only 1 of 30 sentences completely. Decision
implied: **do not restrict text → gloss to the 250.** Add a lexicon-coverage
metric (ASL-LEX 2,723 / WLASL 2,000 / fingerspell / no sign) instead. The
vocabulary limit only matters for rendering (deferred) and recognition (§12).

**Implementation plan: speech → gloss** (report §5; rendering deferred). Each phase
needs a plan + the user's review before it is built.

*Phase 0 — inputs (still owed by the authors, see above).* **Updated 2026-09-24:**
the user asked to integrate the pipeline without waiting, so Phases 1–2 were built
from the uploaded notebook alone. The rule engines, the guard (on T5's *recorded*
outputs) and ASLG-PC12 are scored now; T5 itself waits for the checkpoint, and team30
uses **draft** references (written by Claude) until the team supplies real ones.

*Phase 1 — `sb-synthesize` package skeleton → code (no training):*
- [x] **Done 2026-09-24.** `sb.synthesize.gloss.rules_v1`: the frozen engine, **byte-identical** to
  `rule_engine.py` (T5's training input). `aslg.0` §4 pins it to the 12
  rule-only outputs recorded in report §2 (12/12).
- [x] **Done 2026-09-24** (27/30 exact on team30's draft refs — a development score, same author). `sb.synthesize.gloss.rules_v2`: the fixed engine: `I`/`me` → `ME`; keep
  `IF`/`BUT`/`BECAUSE`/`BEFORE`/`PLEASE`; keep phrasal particles; front time words;
  cover `never`/`no`/`can't`/`won't`; one spaCy parse per sentence; tokenize commas
  properly.
- [x] **Done 2026-09-24** (local dir only; Kaggle via `sb-sync` once the checkpoint exists; code path smoke-tested with public `t5-small`). `sb.synthesize.gloss.t5`: load from a local dir or Kaggle
  (`ensure_local`-style); `generate()` with `no_repeat_ngram_size` +
  `repetition_penalty`; hyperparameters in a config, not code.
- [x] **Done 2026-09-24**: keeps 19/30 recorded T5 outputs, rejects all of #19/#20/#25/#2/#4/#12/#21/#30, keeps #1/#26/#29/#27. `sb.synthesize.gloss.hybrid`: the **guarded hybrid**. Accept T5 only if
  it keeps the rule gloss's content words and pronoun identity (no you → HE),
  with no n-gram repeated; otherwise use rules_v2. Target: report §2's #19/#20/#25
  fixed, with the T5 wins (#1/#26/#29) kept.
- [x] **Done 2026-09-24** (+ CUDA-12 cuBLAS shim: CTranslate2 needs `cublas64_12.dll`, torch is cu130; `nvidia-cublas-cu12` declared for win32). `sb.synthesize.asr`: faster-whisper wrapper with the notebook's
  parameters, the normalization bug fixed, `best_of` dropped; model size in
  the config (large-v3 locally; turbo is what Workers AI hosts).
- [x] **Done 2026-09-24** (coverage lives in `sb.synthesize.lexicon`). `sb.synthesize.metrics`: BLEU-4, chrF, ROUGE-L, METEOR, gloss WER, ASR WER,
  plus **lexicon coverage** (ASL-LEX/WLASL sign · fingerspell · no-sign).
- [x] **Done 2026-09-24** (+ `sounddevice`, `soundfile`, `huggingface-hub`, `nvidia-cublas-cu12`; spaCy pinned <3.9 with `en_core_web_sm` 3.8.0 by URL; `ty check packages/sb-synthesize` clean; `.venvs/<stage>` isolation not re-checked). Declare deps in `packages/sb-synthesize/pyproject.toml` (spacy +
  `en_core_web_sm`, transformers, sentencepiece, faster-whisper, sacrebleu,
  rouge-score, nltk). **Not via `uv pip install`.** Check that `.venvs/<stage>`
  isolation still holds.

*Phase 2 — data + evaluation notebooks:*
- [x] **Built + run 2026-09-24** (ASLG split v1: 81,088 unique texts → 64,870/8,109/8,109, 0 train/test overlap; WLASL index 2,000 glosses; NCSLGR absent). `experiments/synthesis/aslg.0.dataset.text2gloss.ipynb`: ASLG-PC12
  (HF, 82,709 training pairs) and NCSLGR (1,888 utterances) into
  `data/raw/<dataset>/`, fixed held-out splits, plus the 30 sentences with
  hand-written references (user + team), and an ASL-LEX/WLASL gloss list for the
  coverage metric.
- [x] **Built + run 2026-09-24** — results in report §9. `experiments/synthesis/aslg.1.models.text2gloss.ipynb`: rules-v1 /
  rules-v2 / T5 / guarded hybrid on every test set, with a per-sentence error table.
  Config `experiments/synthesis/configs/aslg.text2gloss.json`.
- [~] **Built 2026-09-24; needs the user's recordings** (§1 recorder, `RECORD = True`). `experiments/synthesis/speech.2.asr.eval.ipynb`: Whisper large-v3 vs
  turbo WER on our own recordings of the 30 sentences (sentence-level speech with
  transcripts; Speech Commands can't do this).
  - **2026-09-24, second pass: the user runs Jupyter on a remote PC with no mic.** The user's
    run of `speech.3` crashed (`PortAudioError: Error querying device -1`: no default
    input device), and `speech.2` ran with 0/30 recordings. **Fixed:**
    `asr.microphone()` + a clear error in `asr.record`. Both notebooks skip recording when
    there is no mic. New `sb.synthesize.tts` (Windows SAPI via PowerShell, no deps, plus
    `add_noise`). `speech.2` §1b speaks team30 with both voices (David, Zira) × clean/20/10/5/0
    dB and scores every source separately. `speech.3` §2b: typed text → speech → ASR →
    gloss.
  - **Second bug found and fixed:** downloading `large-v3-turbo` failed with WinError 1314
    (HF cache symlinks without Developer Mode). `asr.model_dir` now downloads into plain
    `data/external/whisper/<model>/` (large-v3 2.9 GB, turbo 1.6 GB).
  - **Results (synthetic speech: an optimistic bound, run 2026-09-24):** WER about 1% for both
    models from clean down to 10 dB, 1.3–1.8% at 5 dB, 2.8–3.1% at 0 dB. **turbo is 1.7×
    faster** (0.12 s vs 0.20 s median per sentence), at equal accuracy. The only
    clean-audio errors were **digits** ("three o'clock" → "3 o'clock"), and `rules_v2` passed
    them through (`TIME 3`). **Fixed:** `asr.spell_numbers` in `SpeechToGloss` before
    glossing. On clean synthetic speech the ASR cost to gloss WER drops from 0.013 to 0.000.
    The raw transcript is kept for WER.
  - Still open: human recordings (when a mic is available) for the real number.

*Phase 3 — training (user runs it):*
- [ ] Only if Phase 2 says T5 is worth it: retrain T5 on
  (English, rules_v2) → gloss, from the team's `step7_aslg_transfer.py`, via
  the repo's driver pattern (config, auto-resume, registry run). Registry
  records for text2gloss need a `task` field → `meta.json` schema v5
  (`sb.mlops.registry::FIELDS`, then `sb-docs`).
- [ ] T5 checkpoint to Kaggle via `sb-sync`
  (`signbridge-<dataset>/transformers/t5-text2gloss/…` — slug to confirm).

*Phase 4 — demo + deployment:*
- [x] **Built 2026-09-24**, smoke-tested (Windows TTS wav → `tiny.en` on GPU → `TOMORROW IF RAIN ME STAY HOME | YOU HELP ME CAN`). `experiments/synthesis/speech.3.pipeline.demo.ipynb`: record → ASR → gloss,
  local Jupyter, no Colab APIs. **Re-run 2026-09-24 with the no-mic fixes:** §1 skips cleanly,
  and §2b gives "Yesterday my brother and I went to the store. Where is the cat?" →
  `YESTERDAY MY BROTHER AND ME GO STORE | CAT WHERE` (large-v3 0.5 s for 4.9 s of audio).
  Also: the aslg.0/aslg.1 re-run by the user reproduces report §9 exactly (T5 still missing, so
  the hybrids equal the rules).
- [x] **Cloudflare path built 2026-09-25 (user decision: browser-only, stay on the Workers
  Free plan — no Containers, since spaCy has no browser runtime and running Python
  server-side needs a paid plan + Docker).** `apps/web/speech.html` + `src/speech/`:
  - ASR: `apps/edge` `POST /api/asr` → Workers AI `whisper-large-v3-turbo` (mic or a file →
    16 kHz mono WAV → base64; `/api/health` reports `asr: bool`);
  - gloss: `rules_v1`/`rules_v2` ported from `sb.synthesize.gloss` to TypeScript on
    **wink-nlp** (`wink-eng-lite-web-model`, MIT) instead of spaCy — Pyodide has no spaCy
    build (no dependency parse), so each `dep_`/tag test in `rules_v2` became a local
    heuristic (documented in `rules_v2.ts`'s header). Agreement with the Python (spaCy)
    engines on 330 sentences (team30 + a seeded ASLG-PC12 sample), after tuning the
    heuristics on the mismatches: **rules_v1 218/330 (64.8%), rules_v2 251/330 (76.1%)**,
    both 100% (v1) / 93–100% (v2, still tuning) on team30 itself. Not exact — a known gap,
    tracked as agreement floors in `test/speech.test.ts`, not hidden;
  - T5: `sb.synthesize.gloss.export_web` (new) exports the checkpoint to two ONNX graphs
    (encoder, decoder — no KV cache, the client re-reads the whole prefix each beam step),
    parity-gated against PyTorch (max logit diff < 1e-3) before writing anything,
    quantized to int8, cut into < 25 MiB parts for Workers static assets, chained with a
    manifest (`tools/export_speech.py assets`). Browser side: `src/speech/beam.ts` (an
    exact port of transformers 5.x `_beam_search`, the vectorized merge-into-finished-beams
    algorithm) + `src/speech/t5.ts` (tokenizer via `@huggingface/tokenizers`, onnxruntime-web
    in a Worker so it never blocks the page) + `src/speech/loader.ts` (chunk fetch, SHA-256
    check, Cache Storage). `sb.synthesize.gloss.hybrid`'s guard ported exactly
    (`src/speech/guard.ts`);
  - **Parity (dev run, public `google-t5/t5-base`; the team's checkpoint is still not on
    disk — see below): fp32 ONNX beam search is byte-identical to `model.generate()` on
    12/12 sentences (encoder diff 1.1e-6, decoder diff 2.7e-5). int8 quantization is too
    lossy for a general-purpose model at this size** (2/12 identical; decoder logit diff up
    to 4.8, garbled/foreign-language output) — **not yet re-measured on the real
    gloss-refiner checkpoint**, which may tolerate it better (smaller effective vocabulary,
    a narrower output distribution). If int8 stays unusable, fp32 is 620 MB decoder + 419 MB
    encoder (≈ 27 + 18 parts): works under the Free plan's static-assets limits but is a
    slow first load;
  - `apps/edge/wrangler.jsonc`: `ASR_MODEL` var (`none` in `--env offline`); Worker still
    Free plan, still no Durable Object session;
  - `public/_headers` sets COOP/COEP so onnxruntime-web can use several WASM threads when
    the browser allows it; single-threaded (`numThreads=1`) is the automatic fallback
    otherwise, so it still works without those headers (e.g. behind a proxy that strips them).
  - **Deployed 2026-09-25**: `wrangler whoami` shows the `.env` token authenticates fine
    (the `user/tokens/verify` failure earlier was that unrelated endpoint's own scoping, not
    a bad token) — `npx wrangler deploy` from `apps/edge` pushed both pages + the Worker to
    **https://signbridge.onecoder1.workers.dev**. Live-verified: `/`, `/speech`, `/api/health`
    (`llm: true`, `asr: true`), `/api/english`. `/assets/t5/manifest.json` 404s as expected
    (T5 was never exported — no real checkpoint), so the speech page runs rules-only until
    the checkpoint arrives.
  - **Found + fixed while verifying the deploy**: `wrangler.jsonc`'s `assets` block had no
    `binding: "ASSETS"`, so `env.ASSETS` was `undefined` everywhere the Worker calls
    `env.ASSETS.fetch(request)` — any request that fell through to the Worker without
    matching a static asset or `/api/*` (i.e. any real 404) crashed with a 500. Pre-existing
    since the Worker was first built 2026-09-24; `dev:offline` never hit it because `/` and
    `/speech` always matched a real file. Fixed and redeployed; `/nonexistent` now returns a
    clean 404.
  - **The team's checkpoint arrived 2026-09-25 and is live.** `model.safetensors` copied
    into `data/external/t5-text2gloss/thesis_hybrid_dataset1/model/` (still gitignored, per
    `data/`'s rule). Sanity check on 3 sentences with `sb.synthesize.gloss.t5.T5Refiner`
    directly: coherent gloss, e.g. "Can you help me?" → `CAN YOU HELP ME`. Re-ran
    `tools/export_speech.py assets` + `fixtures` against it, then `npm test`:
    - `sb.synthesize.gloss.export_web.export` gained a `ship: "fp32" | "int8"` parameter
      (default **`fp32`**, changed from int8) — **on the real checkpoint, int8 matched
      `generate()` on only 13/25 sentences (52%), some meaning-changing (an invented "HE"
      subject: "She likes coffee." → int8 `HE LIKE COFFEE` vs Python's `LIKE COFFEE`)**,
      against fp32's 25/25 exact. fp32 is a larger download (~1 GB vs ~260 MB, both chunked
      for the 25 MiB Workers cap: 44 vs 12 parts) but exact, so it ships;
    - the manifest now records `precision` so the browser (and anyone reading it) knows
      which graph is loaded;
    - `wrangler deploy` re-run: `/assets/t5/manifest.json` and every `.onnx` part are live
      and fetch 200 (verified by curl on `manifest.json`'s own part list). `/speech` now
      has a real T5 refiner, not just rules.
    - **Verified 2026-09-25 with headless Chrome** (`--use-fake-device-for-media-stream`,
      CDP): the live site actually loads and runs, camera stream reaches 640×480,
      MediaPipe Holistic runs (delegate `CPU`), the recognizer processes frames (fps
      counter moves). One transient run negotiated a degenerate 2×2 stream — reproduced
      once, not on retry, and matched locally too under the same COOP/COEP headers, so it
      looks like Chrome's fake-device warm-up, not a `_headers` (COOP/COEP) interaction;
      not chased further since it didn't reproduce.

  **2026-09-25 (user): reduce speech → gloss inference time.** Checkpoint files moved from
  `C:\Users\Public\Downloads\thesis_hybrid_dataset1\` into the repo
  (`data/external/t5-text2gloss/thesis_hybrid_dataset1/`, gitignored) and the Downloads
  copy deleted, per the user's instruction. (Found and left alone: an unrelated
  `model_v2/` under the same directory — a different-sized `model.safetensors` with no
  `config.json`, timestamped mid-session; provenance unclear, gitignored either way, not
  touched.) Three changes, all benchmarked on the real checkpoint (40 sentences) before
  shipping, none of them guesses:
  - `sb.synthesize.gloss.export_web.export` gained `ship: "fp32" | "int8" | "mixed"`,
    **default now `mixed`** (int8 encoder + fp32 decoder): the encoder runs once per
    sentence, so its quantization error doesn't compound across beam steps the way the
    decoder's does. Measured: 34/40 exact vs `generate()`, **same guard-acceptance rate as
    full fp32 (24/40)**, at ~730 MB vs fp32's ~1 GB (encoder 419 MB → 105 MB int8, decoder
    unchanged). `apps/web/test/speech.test.ts` now runs `fp32`/`int8`/`mixed` as three
    variants (mixed isn't asserted exact, since it isn't meant to be — a future regression
    would still show in its printed identical-count).
  - `experiments/synthesis/configs/aslg.text2gloss.json`'s `guarded` preset (the browser's
    only): `num_beams` 4→2, `max_length` 64→56. Beam=2 was ~16% faster than beam=4 on this
    checkpoint and kept guard-acceptance near-identical (23/40 vs 24/40, 33/40 outputs
    byte-identical to beam=4); beam=1 was ~50% faster but cost more quality (19/40
    accepted). max_length 56 keeps margin above the checkpoint's observed max gloss length
    (55 tokens over 330 sentences, p99 44). The `team` preset (frozen, must reproduce the
    team notebook's exact `generate()` call) is untouched.
  - **Not done: WebGPU.** Considered (onnxruntime-web's webgpu execution provider, with a
    wasm fallback) as the highest-ceiling lever, but not shipped — this session has no way
    to drive a real browser against a GPU backend to verify it actually helps or even
    works (headless Chrome here has no GPU), and shipping unverified browser-only code
    risks breaking what currently works. Left for whoever can test it with a real browser.
  - Re-exported, rebuilt, redeployed; live-verified (`manifest.json` reports
    `"precision": "mixed"`, `"generate": {"max_length": 56, "num_beams": 2, ...}`).
  - **Not measured this session: actual wall-clock latency in a real browser tab.**
    Everything above is a compute-cost argument (fewer beams × fewer steps × a smaller
    download), backed by Python-side timing and quality benchmarks, not a stopwatch on the
    deployed page itself.

  **2026-09-25 (user): camera mode should show the video with a live MediaPipe overlay.**
  Audited first rather than assumed broken: headless Chrome with a fake camera device
  confirmed the video element and the overlay canvas were already both visible, correctly
  stacked (identical bounding rects), and both painting (video: real pixels; canvas:
  correctly resized) — `runLive()` already calls `drawFrame()` every processed frame for
  both `camera` and `file`. The fake device has no human features, so Holistic found
  nothing to draw, which looked like "no overlay" but wasn't a code bug. What **was** worth
  fixing: the overlay itself was barely visible by design — 2px pale-grey dots for all 501
  non-hand landmarks (face + pose lumped together), no pose skeleton. Rewrote
  `apps/web/src/main.ts::drawFrame`: face mesh now faint on purpose (468 points would
  otherwise drown out what matters), a bright pose skeleton over exactly the model's own
  ME-126 upper-body subset ({11-16,23,24} — shoulders/elbows/wrists/hips, from CLAUDE.md),
  and thicker/brighter hand skeletons with joint dots (hands previously had lines only, no
  dots). Verified by feeding the exact new drawing code synthetic landmarks in an isolated
  headless-Chrome render (bypassing detection) — screenshot confirms a clear, high-contrast
  overlay. Re-ran `npm test` (11/11) and `browser-check.ts` (24/24 replay identical to
  Python, Holistic smoke test `ok: true`) — no regression from the rewrite. Deployed with
  the T5 changes above.

**Results + state (2026-09-24, report §9):**
- ASLG-PC12 test (2,000, independent of our refs): **rules_v2 BLEU 36.4 / WER 0.303 vs
  rules_v1 26.3 / 0.347**. team30 (draft refs): team T5 recorded 36.0 / 0.414 — no better
  than its own rules_v1 input (39.3 / 0.428); guard on it 66.3 / 0.224.
- **Provenance clue:** the team's T5 output has no `DESC-`/`X-`/`BE`, which are 21%/5% of
  ASLG tokens, so the checkpoint wasn't trained on raw ASLG-PC12 glosses (NCSLGR, or ASLG
  stripped). Still to confirm with the authors.
- [ ] **User (when a mic is available):** record the 30 sentences in `speech.2.asr.eval.ipynb` §1, then run §2–§3. Until then §1b's synthetic speech stands in (optimistic).
- [ ] **User/team:** review team30's draft references (`evalsets/team30.v1.json`), and
  write a **held-out sentence set not authored by Claude** — rules_v2's team30 score is
  circular until then.
- [ ] When the T5 checkpoint arrives: put it at
  `data/external/t5-text2gloss/thesis_hybrid_dataset1/model/`, delete
  `data/cache/synthesis/text2gloss/aslg_test_hybrid_*.parquet`, re-run `aslg.1`.

**Deferred — rendering** (resumes after speech → gloss): renderer research in
report §7.2 (`spoken-to-signed-translation` + our own lexicon); WLASL clip
fixes (trimming, signer choice, no scraping); back-recognition scoring.

**User questions answered 2026-09-24** (report §7):
- [x] **Dropped (user, 2026-09-24).** Google Speech Commands v0.02: **not useful**. Keyword spotting over 35 words,
  12 of them GISLR glosses; Whisper is already open-vocabulary. At most, its noise
  files could be used for an ASR robustness check. Awaiting the user's agreement.
- [-] **Set aside (user, 2026-09-24): the current goal is speech → gloss; rendering waits.** When it resumes, start here. **Renderer: use a ready-made library instead of clips** (user's idea).
  Recommended: `spoken-to-signed-translation` (MIT, MediaPipe-Holistic poses,
  pluggable lexicon, fingerspelling fallback) + **our own ASL lexicon built from
  GISLR npz** (best exemplar per gloss by recognizer confidence) + a browser skeleton
  viewer. Replaces renderer A/B's clip path. Awaiting the user's go-ahead.

**Links to other work:** §12.6's gloss → English is the reverse of this
section's English → gloss (same corpora, one text↔gloss module); §12.5: Workers
AI hosts both `whisper-large-v3-turbo` (ASR) and `melotts` (TTS), so one
deployment covers both directions; `apps/` gains a speech → sign surface.

---

## Backlog / Someday

- [ ] **Layered end-to-end model (user, 2026-09-24, future work):** group the recognizer and
  the next-gloss predictor (and possibly gloss → English) into one model whose input is
  the live video frame and whose output is text. For now the stages stay separate
  (§12.6) so each can be measured and swapped.
- [ ] `apps/web` framework upgrade: SolidJS or QwikCity, after the vanilla TS + Vite
  prototype (§12.5 step 1).
- [ ] (add unscoped ideas here as they come up, promote to a numbered section once
  they have a concrete plan)

---

*Last updated: September 4, 2026, later still (**the workspace restructure
landed** — §9.8, which had been filed as deferred and was overruled). The repo
is now a **uv workspace**: six packages under `packages/` (`sb-core` as the
seam, `sb-extract`, `sb-recognize`, `sb-mlops`, plus `sb-rescore` and
`sb-synthesize` as scoped skeletons), notebooks as thin drivers in
`experiments/<domain>/`, `registry/` committed at the top level and `data/`
gitignored absolutely, six console scripts in place of the `sys.path`-bootstrapping
scripts, and the project renamed **signbridge** (folder and git remote
unchanged). `features/` became a real split — `cache.py` + `base_v1` +
`firstplace_v1` sharing one substitutable surface. Nothing was rebuilt: 36 GB of
caches and landmarks moved by rename and every cache key is byte-identical.
§9.9 keeps the reasoning this overruled. Open: the notebooks have been parsed
but not re-executed, and the R2 upload still has not run.)*

*Previously: September 4, 2026 (**§9.1-§9.7 executed**, one commit per
stage). meta.json is schema **v4** with a `provenance` block and 42 runs
backfilled to `null`; feature caches are **content-addressed** (14 groups /
30.3 GB moved by rename, not rebuild); checkpoint sync to R2 is built and
verified except the upload itself (**no bucket or credentials yet — the weights
are still single-copy**, §9.3); `LANDMARK_TENSOR` v1 replaced four restatements
of the row layout and validates on write and read; the training stack goes
through a `DatasetSource` so `grep gislr_dir` over the three consumers returns
nothing; `schemas/meta.v4.json`, `index.csv` and the README's schema and
registry tables are all generated by `gen_docs.py --check`; `eval_gru.py` is now
`evaluate.py`. §9.8 (rename, workspace split, registry move, aliases, MLflow,
DVC, empty scaffolding) remains deferred/rejected, unchanged.)*

*Previously: September 4, 2026 (**§9 filed**: an external architecture review
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
`experiments/recognition/gislr.1.models.firstplace.ipynb` + `modules/model/{features,optim,train_fp}.py`
+ `Conv1DTransformer` in `architectures.py`; awaiting the user's run. Correction
filed to §7.2: the reference point is lip landmark 17, not shoulder-centre.)*

*Previously: July 22, 2026 (POPSIGN test-split extraction finished · all 4 train dataset parts downloaded, train manifest regeneration still pending · motion-energy and subset-comparison reports backfilled · filed 5 new remarks: elevated §7.1 semantic-confusion diagnostic, corrected stale §4 architecture-run status + flagged BiLSTM-depth conflict (§4.1), consolidated 1st-place-solution recreation (§4.2), filed landmark-reduction write-up pending draft-paper link (§3.0.1), filed new LLM correction-layer idea pending scope decision (§8))*
