# docs/

Two kinds of document live here: **logs** (time-ordered — what happened when)
and **reports** (standalone — what a test or analysis found).

```
docs/
├── logs/
│   ├── daily/<YYYY-MM-DD>.md     # figures in logs/daily/assets/<YYYY-MM-DD>/
│   └── weekly/<YYYY>-<WW>.md     # week starts SUNDAY; figures in logs/weekly/assets/<YYYY>-<WW>/
└── reports/<topic-slug>.md       # figures in reports/assets/<topic-slug>/
```

| folder | content | naming |
|---|---|---|
| `logs/daily/` | day-by-day work logs and dated narrative | `<YYYY-MM-DD>.md`, figures in `logs/daily/assets/<YYYY-MM-DD>/` |
| `logs/weekly/` | weekly summaries | `<YYYY>-<WW>.md`, e.g. `2026-30.md`; figures in `logs/weekly/assets/<YYYY>-<WW>/` |

**Week numbering.** Weeks run **Sunday → Saturday** (not ISO, which starts Monday), and week 1
is the week containing January 1. So week 30 of 2026 is **2026-07-19 → 2026-07-25**; ISO would
number that same Sunday as the last day of its week 29. Always state the date range in the
weekly file's title so the numbering never has to be re-derived. The running week's file is
written as work happens and marked **in progress** until its Saturday.
| `reports/` | standalone topic reports from a test or analysis — motion-energy, subset-comparison, confidence-tuning, plateau-breakout, … | `<topic-slug>.md`, figures in `reports/assets/<topic-slug>/` |

**Which one to write.** If the document is "here is what I did on this date",
it's a log. If it is "here is what this experiment measured and concluded",
it's a report — and the daily log for that date should link to it rather than
restate it. Long-lived findings belong in `reports/` so they stay findable
without knowing the date they were produced.

Every substantial analysis or experiment gets a write-up here; figures always
live under the matching `assets/` subfolder, never inline-only in notebooks.

## Daily logs

Chronological. `logs/daily/<YYYY-MM-DD>.md`.

| date | log | contents |
|---|---|---|
| 2026-07-15 | [logs/daily/2026-07-15.md](logs/daily/2026-07-15.md) | GISLR landmark motion-energy analysis (z-noise finding, keep/discard recommendation) · 1st-place solution landmark cross-check · GRU training in detail: full-543 baseline vs ME-126 subset (+3.1 pts at half the parameters) |
| 2026-07-16 | [logs/daily/2026-07-16.md](logs/daily/2026-07-16.md) | Landmark-subset discriminability comparison (F-ratio / MI / probe classifier, 3 scopes): **ME-126 wins**; discriminability ≠ motion energy (rho −0.12) · POPSIGN extraction module + driver notebook (resource-capped, resumable) built & validated |
| 2026-07-17 | [logs/daily/2026-07-17.md](logs/daily/2026-07-17.md) | Registry tooling v1: per-run metadata + queryable index · fresh-run-folder policy · training regime **v2-plateau-300** · LSTM / BiLSTM / CausalConv1D benchmark notebooks built · eval script generalized (arch dispatch + xy mode) |
| 2026-07-18 | [logs/daily/2026-07-18.md](logs/daily/2026-07-18.md) | **Repo restructure**: unified `modules/model` training stack · flat epoch-seconds model registry + meta.json schema v2 (registry reset) · `data/{raw,cache,temp,models}` tree + temp-cleanup policy · docs daily/weekly/reports split · single-progress-bar training |
| 2026-07-19 | [logs/daily/2026-07-19.md](logs/daily/2026-07-19.md) | **Plateau diagnosed as overfitting** (train 90–99% vs val ~75%, gap 0.16–0.24) · most-confused pairs are semantic near-synonyms · GISLR **stage-2 evaluation/submission notebook** + arch-generic TFLite export + meta.json **schema v3** (`submission.tested`, submission queue as a query) · `docs/` split into logs/ vs reports/ · POPSIGN **confidence-tuning** first results ([report](reports/confidence-tuning.md)) — thresholds barely matter, the quality proxies are measuring clip padding · POPSIGN extraction driver given a **CLI handoff** (`extract_popsign.py`) · **bulk test-split extraction running** (15,549/33,600, 1 failed, 1.46 videos/s) + the on-disk npz format recorded |
| 2026-08-23 | [logs/daily/2026-08-23.md](logs/daily/2026-08-23.md) | **The 1st-place port diverges at epoch 15** — the step AWP + LateDropout switch on: loss pinned at ln(250), stem weight norm 16→224, `stem_bn.running_var` 4.8→8467, 285 of 300 epochs wasted; the re-run with clipping + frozen BN stats died the same way but cost 9 min, so §5b now ablates the two switches apart · canonical **0.7459** from the surviving epoch-15 checkpoint (below the 0.7565 GRU, and not a measurement of the recipe) · its confused pairs are the **same semantic near-synonyms** the 18-run aggregate found · fixes: `grad_clip: 1.0`, BatchNorm stats frozen during AWP's adversarial pass, and collapse/plateau **stopping conditions** for `fp-onecycle-300` |
| 2026-09-04 | [logs/daily/2026-09-04.md](logs/daily/2026-09-04.md) | **Reproducibility §9.1–§9.7**: meta.json **schema v4** with a provenance block (42 runs backfilled) · **content-addressed** feature caches (30.3 GB moved by rename, not rebuild) · `LANDMARK_TENSOR` v1 spec + validators · the `DatasetSource` seam (no `gislr_dir` left in the training stack) · schema/index/README tables generated from `registry.FIELDS` · `eval_gru.py` → `evaluate.py` · **§9.8 the workspace restructure**: six `packages/sb-*`, `experiments/`, `registry/` and `data/` at the top level, renamed **signbridge** · checkpoint sync given **kaggle/local/s3 backends** after R2 turned out to be unavailable · TODO audit (11 done-but-unticked, 4 obsolete) |
| 2026-09-05 | [logs/daily/2026-09-05.md](logs/daily/2026-09-05.md) | **The plateau is a generalization gap, not a label ceiling** — the diagnosis's GPU arms: symmetric confusion **0.012 on train vs 0.273 on val**, 19 of 20 confusable pairs at exactly zero on training data · per-pair probes near chance (`awake`/`wake` 0.463) and pooled velocity does not rescue them (§8b), which bounds *pooled summaries* rather than the landmarks · §8 rescoring bounded at ~2.7 pts; §7.2/§7.4 are the evidence-backed levers |
| 2026-09-13 | [logs/daily/2026-09-13.md](logs/daily/2026-09-13.md) | **POPSIGN extraction staged** one train part at a time + separate pilot notebook · 4th train part fix · GIF overlay · `gru_deep` · top-5 streaming runs exported to TFLite · Kaggle `asl-signs` code submission found dead |
| 2026-09-14 | [logs/daily/2026-09-14.md](logs/daily/2026-09-14.md) | Confidence-tuning notebook and four reports de-staled after the kagglehub-cache redesign · CLAUDE.md trimmed |
| 2026-09-16 | [logs/daily/2026-09-16.md](logs/daily/2026-09-16.md) | **GISLR moves to GISLR_Stratified npz** — canonical split reset (fixed 80/20, 18,896 val) · landmark-order permutation fix · local test scoring replaces Kaggle submission |
| 2026-09-18 | [logs/daily/2026-09-18.md](logs/daily/2026-09-18.md) | Landmark-importance run written up · per-axis saliency · interactive report |
| 2026-09-19 | [logs/daily/2026-09-19.md](logs/daily/2026-09-19.md) | Feature-discriminability run and write-up · pair-similarity and curated-features notebooks · repo cleanup (motion-energy notebook and Kaggle-submit helpers deleted) |
| 2026-09-21 | [logs/daily/2026-09-21.md](logs/daily/2026-09-21.md) | **Motion-energy notebook rebuilt for GISLR_Stratified npz, run, and written up** — xy-native RMS speed, jitter smoothed before the derivative, new per-joint-angle "change of angles" scope · all three scopes, 0 failed units · global run confirms the pre-npz 50-video sample almost exactly · **BiLSTM re-baseline + curated-feature ablation** run: `bilstm_base` 73.71% (below every current-split `gru`) · curated feature set is the dominant cause of `bilstm_curated`'s ~16pp loss · **five-architecture benchmark** run: `bilstm` 73.92% (offline) > `gru` 73.80% > `lstm` 72.86% > `cnn` 66.96% > `dnn` 64.85% top-1 · `dnn`'s mean true-class confidence (0.24) a quarter of every other arm's despite comparable top-3/5 |
| 2026-09-22 | [logs/daily/2026-09-22.md](logs/daily/2026-09-22.md) | **Live streaming confidence/reset architecture scoped and run** — built and ran `gislr.3.streaming.confidence-eval.ipynb` (Phase A/B/D, 0 failures): fresh-start confidence is already well-calibrated (`gru` late-third 0.58, matching whole-video numbers) · reset cuts bleed-through 96–98% and roughly halves re-acquisition latency · `RecurrentSession`'s incremental step API verified to 1e-6 against the batch readout · the assumed per-frame-supervised retrain downgraded from prerequisite to optional refinement · **repo cleanup: POPSIGN deprecated**, ~6.2GB of extracted data deleted, extraction packages/notebooks marked paused not deleted |

## Weekly logs

Chronological. `logs/weekly/<YYYY>-<WW>.md`.

| week | dates | log | contents |
|---|---|---|---|
| 2026-29 | Jul 12 – Jul 18 | [logs/weekly/2026-29.md](logs/weekly/2026-29.md) | motion energy → discriminability (ME-126 wins on both, rho −0.12 between them) · xy beats xyz · registry v1 → the restructure (registry reset, pre-reset weights gone) · 4 architecture notebooks |
| 2026-30 | Jul 19 – Jul 25 | [logs/weekly/2026-30.md](logs/weekly/2026-30.md) | plateau diagnosed as **overfitting** · TFLite export working for all 4 archs (Keras rebuild) · training consolidated to one notebook + config · POPSIGN **test-split extraction finished** (33,599/33,600) and all 4 train dataset parts now downloaded (train manifest regeneration + bulk run still pending) |
| 2026-35 | Aug 23 – Aug 29 | [logs/weekly/2026-35.md](logs/weekly/2026-35.md) | 1st-place port: recreated, ~35× faster, collapses at epoch 15 twice · weeks 31–34 had no dev work |
| 2026-36 | Aug 30 – Sep 5 | [logs/weekly/2026-36.md](logs/weekly/2026-36.md) | reproducibility §9.1–§9.7 · uv-workspace restructure, rename to signbridge · 42 checkpoints on Kaggle · plateau = generalization gap · TS extractor cannot run on Deno |
| 2026-37 | Sep 6 – Sep 12 | [logs/weekly/2026-37.md](logs/weekly/2026-37.md) | no dev work |
| 2026-38 | Sep 13 – Sep 19 | [logs/weekly/2026-38.md](logs/weekly/2026-38.md) | staged POPSIGN extraction · GISLR_Stratified + canonical split reset · local test scoring · discriminability / pair-similarity / curated-features notebooks |
| 2026-39 | Sep 20 – Sep 26 | [logs/weekly/2026-39.md](logs/weekly/2026-39.md) | *in progress*: motion-energy notebook rebuilt for GISLR_Stratified npz, run end to end (0 failed units), xy-native global run confirms the old sample almost exactly · new joint-angle "change of angles" instrument · BiLSTM re-baseline + curated-feature ablation (feature set dominates the loss) · five-architecture benchmark run (`bilstm`/`gru` lead, `dnn`'s confidence a quarter of the rest) · streaming confidence/reset architecture scoped and run: fresh-start confidence is already usable, un-reset state contamination is the real problem (reset cuts bleed-through 96–98%) · POPSIGN deprecated, extracted data deleted |

## Reports

Standalone; not date-ordered — findable by topic.

| report | contents |
|---|---|
| [reports/motion-energy.md](reports/motion-energy.md) | GISLR per-landmark motion analysis, two runs. 2026-07-15 (pre-npz, xyz): ~92% of pose "motion" is z-axis noise · seeded 50-video samples reproduce the global ranking (rho 0.95+) · ME-126 keep/discard recommendation. **2026-09-21 rerun** (GISLR_Stratified npz, §5): xy-native global run confirms the old sample almost exactly · new joint-angle-change instrument (dominant-hand asymmetry, one elbow reversal) · legs' xy motion vs arms flagged as an open question |
| [reports/subset-comparison.md](reports/subset-comparison.md) | Landmark-subset discriminability (F-ratio / MI / probe classifier, 3 scopes): **ME-126 wins** the 6-subset leaderboard (49.9% global probe) · discriminability ≈ uncorrelated with motion energy (rho −0.12) · probe difficulty profile tracks the trained GRU's (rho 0.640) |
| [reports/confidence-tuning.md](reports/confidence-tuning.md) | POPSIGN extraction-quality threshold sweep (**partial — 2 of 7 arms**): `min_hand_landmarks_confidence` is inert and the *pose* thresholds gate the hands · thresholds move hand detection by only ~0.02 · **the quality proxies are dominated by clip padding** — ~half of every clip is non-signing lead-in/lead-out, and detection is 0.85–0.94 within the signing span |
| [reports/feature-discriminability.md](reports/feature-discriminability.md) | Engineered per-frame feature discriminability (joint angles + kinematics, 15-class scope): **joint angles are the most information-dense feature type** — 56 angle features alone reach 68.3% probe accuracy vs 79.4% for 3,258 position features (~50× more per feature) · top individual feature (`angle_R_palm_facing_mean`) beats every raw landmark coordinate |
| [reports/pair-similarity.md](reports/pair-similarity.md) | Re-tests `plateau-diagnosis.md` §6's confusable-pair separability probes with the angle+kinematics feature set (16 pairs + 8 controls, 5-fold CV): **angles don't rescue what pooling loses** — `corr(confusion_rate, probe_accuracy)` = −0.71, matching §6's −0.72 · `awake`/`wake` stays at chance (0.512) under every feature type |
| [reports/curated-features.md](reports/curated-features.md) | Trains a real DNN + LSTM on the ME-126 + xy + 28-joint-angle recipe (922 features/frame, an 83% cut from the full pipeline): **LSTM 69.09% top-1 / 79.65% top-2**, DNN 57.76% / 72.53% · curation cost the memory-free DNN 9× more accuracy than the LSTM vs their full-543 counterparts (−13.26pp vs −1.46pp) |
| [reports/plateau-diagnosis.md](reports/plateau-diagnosis.md) | **Why accuracy stops at ~75%** (31 canonical runs): semantically similar signs absorb only **10.5% of errors** · the plateau is a **generalization gap**: symmetric confusion is **0.012 on train vs 0.273 on val** · levers are **§7.2 normalization / §7.4 augmentation** |
| [reports/extractor-parity.md](reports/extractor-parity.md) | **The Deno/TypeScript extractor cannot run** (12-clip test run, 0/12): `@mediapipe/tasks-vision` creates a **WebGL** context regardless of `delegate: "CPU"`, and Deno has no WebGL · POPSIGN extraction is **not** blocked — `sb-extract` is native C++ with no GL requirement |
| [reports/bilstm-curated.md](reports/bilstm-curated.md) | BiLSTM on the current split (4 arms, ablation): `bilstm_base` **73.71% top-1** (below every current-split `gru` run) · `bilstm_curated` lost by ~16pp; decomposed into feature set −17.4pp (dominant), architecture +10.6pp (recovers most), batch size −9.1pp |
| [reports/five-arch-benchmark.md](reports/five-arch-benchmark.md) | GRU/LSTM/BiLSTM/CNN/DNN on one feature pipeline: **`bilstm` 73.92% (offline) > `gru` 73.80% > `lstm` 72.86% > `cnn` 66.96% > `dnn` 64.85%** top-1 · new mean-true-class-confidence metric: `dnn` scores 0.2387, a quarter of every other arm's, despite comparable top-3/5 |
| [reports/streaming-confidence.md](reports/streaming-confidence.md) | Is mid-sequence streaming confidence usable, and does resetting hidden state at a sign boundary help? **Yes and yes**: fresh causal start reaches 0.58 mean true-class confidence by its late third · reset cuts bleed-through 96–98% and roughly halves re-acquisition latency · retrain reprioritized from prerequisite to optional |
