# Graph Report - repository  (2026-09-19)

## Corpus Check
- 349 files · ~329,221 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 177 file(s) not represented in the graph (top: .npy 56, .csv 52, .npz 50)

## Summary
- 1293 nodes · 2169 edges · 141 communities (95 shown, 46 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 107 edges (avg confidence: 0.84)
- Token cost: 1,612,309 input · 0 output

## Community Hubs (Navigation)
- Kaggle Checkpoint Artifacts
- Dataset Path Resolution
- TS Extractor CLI
- CLAUDE.md Repo Conventions
- Daily/Weekly Log Digest
- Model Promotion CLI
- Landmark Group Slicing
- 1st-Place Feature Cache
- Frame Quality Heuristics
- Docs Auto-Generation
- Model Registry CLI
- Attention/DNN Model Heads
- Base Feature Cache Pipeline
- Motion-Energy Diagnostic Report
- Training Config Schema
- Recognition Reporting Utils
- Frame Feature Extraction
- MediaPipe Holistic Extraction
- POPSIGN Extraction CLI
- Per-Dataset Video Sources
- Deno Project Config
- FirstPlace Dataset Loader
- Interpretability Training Utils
- AWP/Cosine LR Training Utils
- Generic Cache Key Hashing
- Canonical Eval Runner
- Landmark NPZ I/O
- Landmark Subset Definitions
- Landmark Overlay Rendering
- POPSIGN Download/Extract Cycle
- Canonical Split & Data Source
- Kaggle Submission/Leaderboard
- Landmark Importance Scoring
- TS Extractor NPZ Verification
- Extractor Parity Runner
- Landmarker Worker Pool
- Unified Training Driver
- AWP Optimizer Wrapper
- Landmark Tensor Contract (sb-core)
- Fold-Based Array Datasets
- Gloss Vocabulary Mapping
- Extractor Parity Comparison
- Feature Cache Migration
- LateDropout/MaskedBatchNorm Layers
- Lookahead Optimizer
- Length-Bucketed Batch Sampler
- meta.json Schema (v4)
- MediaPipe Landmark Groups Overview
- sb-core Shared Contracts
- Causal Depthwise Conv Block
- Conv1D-Transformer Trunk Stage
- Workspace Package Map
- Registry Field Descriptions
- Motion-Energy Scope Charts
- Landmark-Importance Result Charts
- Subset ANOVA & Early Run Charts
- In-RAM Array Dataset
- Mid-Batch Run Learning Curves
- Mid-Batch Run Accuracy Charts
- Late-Batch Per-Class Accuracy Charts
- GRU Baseline Accuracy Results
- GRU ME-126 Learning Dynamics
- Per-Category Motion Energy Insight
- BiLSTM Reference Architecture
- Causal Conv1D Architecture
- Streaming Dropout Mechanism
- Conv1D-Transformer Architecture
- StreamingGRU Deployment Model
- StreamingLSTM Reference Model
- Training Config Loader/Validator
- Permutation Importance Method
- GRU Baseline Learning Curves
- GRU ME-126 Accuracy Distribution
- Architecture Spec Registry
- Efficient Channel Attention
- ME_132/FP_118 Run Pair Charts
- Cross-Scope Motion Consistency
- ME-126 vs Baseline Accuracy
- Per-Landmark Saliency Charts
- Confidence-Tuning Threshold Charts
- Rescoring Client (Unimplemented)
- Late Registry Run Charts
- meta.json Field: assets
- meta.json Field: checkpoints
- meta.json Field: coords
- meta.json Field: created
- meta.json Field: dataset
- meta.json Field: feature_dim
- meta.json Field: hyperparameters
- meta.json Field: model_name
- meta.json Field: n_classes
- meta.json Field: n_landmarks
- meta.json Field: n_params
- meta.json Field: notes
- meta.json Field: provenance
- meta.json Field: run_id
- meta.json Field: schema_version
- meta.json Field: split
- meta.json Field: streaming
- meta.json Field: submission
- meta.json Field: subset
- meta.json Field: training
- Jupyter Environment Check
- sb-extract Package Root
- sb-mlops Package Root
- Deployment Export Module
- Named Feature Pipelines
- Interpretability Track Root
- Rescore Package Root
- Synthesize Package Root
- Early ME_126 Run Pair
- ME_126 Run 1784393064 Charts
- ME_126 Run 1784399151 Charts
- FP_118 Run 1784401260 Charts
- FP_118 Run 1784402251 Charts
- Run 1784447187 Chart Pair
- Run 1784453891 Chart Pair
- Runs 1784455294/455964 Charts
- Runs 1784456692/457430 Charts
- Runs 1784459026/459817 Charts
- Run 1787483814 Confusion+Accuracy
- Run 1784388482 Accuracy
- Run 1784389439 Accuracy
- Run 1784447175 Accuracy
- Run 1784447182 Accuracy
- Run 1784447190 Accuracy
- Run 1784449360 Accuracy
- Run 1784449770 Accuracy
- Run 1784450082 Accuracy
- Run 1784451163 Accuracy
- Run 1784451456 Accuracy
- Run 1784451842 Accuracy
- Run 1784454580 Accuracy
- Run 1787494351 Learning Curves
- Run 1789251515 Accuracy
- Run 1789252196 Accuracy
- Run 1789252933 Accuracy
- Run 1789258464 Accuracy

## God Nodes (most connected - your core abstractions)
1. `README.md (signbridge)` - 27 edges
2. `TrainingConfig` - 17 edges
3. `train_firstplace_run()` - 16 edges
4. `KaggleBackend` - 13 edges
5. `load_npz()` - 13 edges
6. `train_run()` - 13 edges
7. `Backend` - 12 edges
8. `build_model()` - 12 edges
9. `env_value()` - 11 edges
10. `extract_dataset()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Resumable Manifest Pattern` --semantically_similar_to--> `sb-extract-ts README`  [INFERRED] [semantically similar]
  docs/reports/motion-energy.md → packages/sb-extract-ts/README.md
- `Landmark-importance interpretability track` --semantically_similar_to--> `ME-126 landmark subset`  [INFERRED] [semantically similar]
  README.md → docs/logs/daily/2026-07-15.md
- `signbridge` --depends_on--> `sb-extract`  [EXTRACTED]
  pyproject.toml → packages/sb-extract/pyproject.toml
- `signbridge` --depends_on--> `sb-rescore`  [EXTRACTED]
  pyproject.toml → packages/sb-rescore/pyproject.toml
- `signbridge` --depends_on--> `sb-synthesize`  [EXTRACTED]
  pyproject.toml → packages/sb-synthesize/pyproject.toml

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Streaming vs offline architecture benchmark family** — concept_streaminggru, concept_streaminglstm, concept_bilstm, concept_causalconv1d, concept_conv1dtransformer [EXTRACTED 1.00]
- **ME-126 subset selection evidence chain** — concept_motion_energy_analysis, concept_discriminability_probe, concept_me_126_subset, concept_fp_118_subset [EXTRACTED 1.00]
- **Registry reproducibility infrastructure (§9)** — concept_meta_json_schema_v4, concept_provenance_block, concept_model_registry, concept_checkpoint_sync_kaggle_backend [EXTRACTED 1.00]
- **Three Independent Methods Converge on ME-126** — docs_reports_motion_energy_me_126_subset, docs_reports_subset_comparison_discriminability_probe, docs_reports_landmark_importance_region_importance [EXTRACTED 1.00]
- **MediaPipe Persistent-Landmarker Resolution-Change Hang Pattern** — docs_reports_confidence_tuning_leaked_native_graph_bug, docs_logs_weekly_2026_30, packages_sb_extract_ts_readme [INFERRED 0.85]
- **GISLR Landmark & Accuracy Research Report Chain** — docs_reports_motion_energy, docs_reports_subset_comparison, docs_reports_landmark_importance, docs_reports_plateau_diagnosis [EXTRACTED 1.00]
- **Cross-group landmark RMS speed comparison** — docs_logs_daily_assets_2026_07_15_global_overview_pose_landmarks, docs_logs_daily_assets_2026_07_15_global_overview_hand_landmarks, docs_logs_daily_assets_2026_07_15_global_overview_face_landmarks, docs_logs_daily_assets_2026_07_15_global_overview_rms_speed [EXTRACTED 1.00]
- **GRU Baseline Training Run (loss, accuracy, LR schedule shown together)** — docs_logs_daily_assets_2026_07_15_gru_baseline_learning_curves, docs_logs_daily_assets_2026_07_15_gru_baseline_loss_curve, docs_logs_daily_assets_2026_07_15_gru_baseline_accuracy_curve, docs_logs_daily_assets_2026_07_15_gru_baseline_lr_schedule [EXTRACTED 1.00]
- **GRU Baseline Per-Class Accuracy Diagnostic** — docs_logs_daily_assets_2026_07_15_gru_baseline_per_class_chart, docs_logs_daily_assets_2026_07_15_gru_baseline_per_class_overall_accuracy, docs_logs_daily_assets_2026_07_15_gru_baseline_per_class_macro_accuracy, docs_logs_daily_assets_2026_07_15_gru_baseline_per_class_worst_classes, docs_logs_daily_assets_2026_07_15_gru_baseline_per_class_250_sign_vocabulary [INFERRED 0.75]
- **GRU ME-126 Training Run (Loss/Accuracy/LR Schedule)** — docs_logs_daily_assets_2026_07_15_gru_me126_learning_curves, docs_logs_daily_assets_2026_07_15_gru_me126_learning_curves_val_accuracy_plateau, docs_logs_daily_assets_2026_07_15_gru_me126_learning_curves_train_val_overfitting_gap, docs_logs_daily_assets_2026_07_15_gru_me126_learning_curves_onecycle_lr_schedule [INFERRED 0.75]
- **Per-category landmark motion analysis for sign 'beside'** — docs_logs_daily_assets_2026_07_15_per_category_example_0_chart, docs_logs_daily_assets_2026_07_15_per_category_example_0_sign_beside, docs_logs_daily_assets_2026_07_15_per_category_example_0_rms_speed_metric, docs_logs_daily_assets_2026_07_15_per_category_example_0_pose_motion_insight, docs_logs_daily_assets_2026_07_15_per_category_example_0_face_landmarks_static_insight [EXTRACTED 1.00]
- **Z-axis motion-energy noise investigation (2026-07-15 assets)** — docs_logs_daily_assets_2026_07_15_per_video_example_0_video_2234689254_radio, docs_logs_daily_assets_2026_07_15_per_video_example_1_video_2649863433_pajamas, docs_logs_daily_assets_2026_07_15_xy_vs_xyz_z_axis_motion_energy_share [INFERRED 0.75]
- **Landmark-subset selection analysis (2026-07-16 assets, scopes A and B)** — docs_logs_daily_assets_2026_07_16_leaderboard_landmark_subset_probe_leaderboard, docs_logs_daily_assets_2026_07_16_scope_a_consistency_within_class_cv, docs_logs_daily_assets_2026_07_16_scope_b_f_ratio_per_landmark_anova_f [INFERRED 0.75]
- **POPSIGN Confidence-Threshold Tuning Analysis** — docs_reports_assets_confidence_tuning_default_vs_strict_threshold_comparison_chart, docs_reports_assets_confidence_tuning_hand_presence_profile_hand_presence_profile_chart, docs_reports_assets_confidence_tuning_window_effect_window_effect_chart [INFERRED 0.80]
- **Landmark-importance interpretability study charts** — docs_reports_assets_landmark_importance_region_ranking_chart, docs_reports_assets_landmark_importance_test_confusion_matrices_chart, docs_reports_assets_landmark_importance_top20_landmarks_chart, docs_reports_assets_landmark_importance_weight_histograms_chart [INFERRED 0.75]
- **Motion-energy analysis charts** — docs_reports_assets_motion_energy_cross_scope_chart, docs_reports_assets_motion_energy_global_overview_chart [INFERRED 0.75]
- **Motion-energy analysis figure set (per-category, per-video, xyz-vs-xy)** — docs_reports_assets_motion_energy_per_category_example_0, docs_reports_assets_motion_energy_per_video_example_0, docs_reports_assets_motion_energy_xy_vs_xyz, docs_reports_assets_motion_energy_per_category_example_0_rms_speed_metric [INFERRED 0.80]
- **Landmark-subset comparison report figure set (leaderboard, consistency, mutual information)** — docs_reports_assets_subset_comparison_leaderboard, docs_reports_assets_subset_comparison_scope_a_consistency, docs_reports_assets_subset_comparison_scope_b_mi [INFERRED 0.80]
- **GISLR subset-comparison training runs (ME_126, ME_132, FP_118)** — registry_runs_1784385530_assets_learning_curves, registry_runs_1784386490_assets_learning_curves, registry_runs_1784387336_assets_learning_curves [INFERRED 0.75]
- **GISLR runs sharing the 250-sign per-class accuracy evaluation protocol** — registry_runs_1784387336_assets_per_class_accuracy_chart, registry_runs_1784388482_assets_per_class_accuracy_chart, registry_runs_1784389439_assets_per_class_accuracy_chart, registry_runs_1784390290_assets_per_class_accuracy_chart, registry_runs_1784393064_assets_per_class_accuracy_chart [INFERRED 0.75]
- **GISLR subset/architecture comparison across ME_132, FP_118, ME_126 runs** — registry_runs_1784393683_assets_learning_curves, registry_runs_1784394315_assets_learning_curves, registry_runs_1784395799_assets_learning_curves [INFERRED 0.75]
- **Runs sharing a step-decay LR schedule with train/val divergence pattern** — registry_runs_1784396516_assets_learning_curves_chart, registry_runs_1784397301_assets_learning_curves_chart, registry_runs_1784399151_assets_learning_curves_chart [INFERRED 0.75]
- **Consistently hard sign classes across ME runs** — registry_runs_1784399741_assets_per_class_accuracy_chart, registry_runs_1784400020_assets_per_class_accuracy_chart, registry_runs_1784400832_assets_per_class_accuracy_chart [INFERRED 0.75]
- **Per-class accuracy comparison across four registry runs** — registry_runs_1784401260_assets_per_class_accuracy_chart, registry_runs_1784402251_assets_per_class_accuracy_chart, registry_runs_1784447175_assets_per_class_accuracy_chart, registry_runs_1784447182_assets_per_class_accuracy_chart [INFERRED 0.75]
- **Canonical GISLR Per-Class Accuracy Evaluations Across Runs** — registry_runs_1784451163_assets_per_class_accuracy_chart, registry_runs_1784451456_assets_per_class_accuracy_chart, registry_runs_1784451842_assets_per_class_accuracy_chart, registry_runs_1784453891_assets_per_class_accuracy_chart, registry_runs_1784454580_assets_per_class_accuracy_chart [INFERRED 0.75]
- **GISLR per-class accuracy report charts (250 signs, overall ~0.73-0.75)** — registry_runs_1784455294_assets_per_class_accuracy, registry_runs_1784455964_assets_per_class_accuracy, registry_runs_1784456692_assets_per_class_accuracy, registry_runs_1784457430_assets_per_class_accuracy, registry_runs_1784459026_assets_per_class_accuracy, registry_runs_1784459817_assets_per_class_accuracy [INFERRED 0.85]
- **Recurring hardest classes (there, give, beside, awake, nap, wake, vacuum) across near-tied ME-126 eval runs** — registry_runs_1789251515_assets_per_class_accuracy_per_class_accuracy, registry_runs_1789252196_assets_per_class_accuracy_per_class_accuracy, registry_runs_1789252933_assets_per_class_accuracy_per_class_accuracy [INFERRED 0.85]
- **Per-Class Accuracy Charts Across GISLR Runs** — registry_runs_1789253641_assets_per_class_accuracy, registry_runs_1789254670_assets_per_class_accuracy, registry_runs_1789255418_assets_per_class_accuracy, registry_runs_1789256241_assets_per_class_accuracy, registry_runs_1789257008_assets_per_class_accuracy, registry_runs_1789257785_assets_per_class_accuracy [INFERRED 0.80]

## Communities (141 total, 46 thin omitted)

### Community 0 - "Kaggle Checkpoint Artifacts"
Cohesion: 0.06
Nodes (38): env_value(), Environment first, repo ``.env`` second, ``default`` last., Backend, cmd_drop_resume(), cmd_prune(), cmd_pull(), cmd_push(), cmd_rescheme() (+30 more)

### Community 1 - "Dataset Path Resolution"
Cohesion: 0.06
Nodes (49): Datasets, cleanup_temp(), clear_dataset_cache(), dataset_cache_dir(), DatasetMap, _find_repo_root(), gislr_dir(), Path (+41 more)

### Community 2 - "TS Extractor CLI"
Cohesion: 0.07
Nodes (43): loadManifest(), main(), Manifest, parseArgs(), saveManifest(), Unit, VIDEO_EXT, walkVideos() (+35 more)

### Community 3 - "CLAUDE.md Repo Conventions"
Cohesion: 0.12
Nodes (41): AWP/LateDropout epoch-15 divergence, BiLSTM (offline-only reference), Canonical GISLR evaluation, Canonical-split reset (2026-09-16, GISLR_Stratified), CausalConv1D, Checkpoint sync via Kaggle Model backend, cnn1d num_layers 5→2 config-sync bug, POPSIGN confidence-threshold tuning (+33 more)

### Community 4 - "Daily/Weekly Log Digest"
Cohesion: 0.07
Nodes (42): 2026-09-18 Daily Log: Landmark-Importance Write-up, Per-Axis (x/y/z/speed) Saliency Diagnostic, Week 29 Summary Log, Fresh-Run-Folder Policy, Registry v1 to Restructure, v2-plateau-300 Training Regime, Week 30 Running Summary Log, cnn1d num_layers Misconfiguration (+34 more)

### Community 5 - "Model Promotion CLI"
Cohesion: 0.09
Nodes (34): datetime, _backed_up(), check(), load_aliases(), main(), promote(), Path, Aliases: which run is champion, and what a deployment fetches. Deployment must… (+26 more)

### Community 6 - "Landmark Group Slicing"
Cohesion: 0.11
Nodes (30): LandmarkGroup, slice, One contiguous block of holistic rows., build_bilstm(), build_cnn1d(), build_keras_model(), build_recurrent(), check_parity() (+22 more)

### Community 7 - "1st-Place Feature Cache"
Cohesion: 0.11
Nodes (27): augment(), build_cache(), cache_dir(), cache_inputs(), cache_key(), collate_fn(), flip_lr(), _interp_time() (+19 more)

### Community 8 - "Frame Quality Heuristics"
Cohesion: 0.13
Nodes (24): _bone_cv(), frame_quality(), _jitter(), load_landmarks(), _longest_false_run(), _present(), DataFrame, ndarray (+16 more)

### Community 9 - "Docs Auto-Generation"
Cohesion: 0.14
Nodes (22): blocks(), _json_type(), main(), meta_json_schema(), _plain(), Regenerate everything in the docs that is derived from code or the registry.…, JSON Schema for one meta.json, generated from registry.FIELDS., Strip the markdown a README cell wants but a JSON Schema description should not… (+14 more)

### Community 10 - "Model Registry CLI"
Cohesion: 0.14
Nodes (23): eval_command(), load_meta(), mark_tested(), migrate_all(), migrate_meta(), new_run_dir(), pointer_run_dir(), Path (+15 more)

### Community 11 - "Attention/DNN Model Heads"
Cohesion: 0.13
Nodes (15): LandmarkAttention, LandmarkDNN, LandmarkRNN, Tensor, Custom DNN / LSTM / GRU for the landmark-importance notebook (TODO §3). All…, Unidirectional LSTM/GRU over attention-gated, projected landmark features —…, One learned scalar gate per landmark, ``sigmoid(logit)`` in (0, 1), multiplied…, ``per_landmark``: (..., n_landmarks, channels) -> same shape, gated. (+7 more)

### Community 12 - "Base Feature Cache Pipeline"
Cohesion: 0.13
Nodes (20): concurrent_futures, build_cache(), cache_dir(), cache_inputs(), cache_key(), load_video(), DataFrame, ndarray (+12 more)

### Community 13 - "Motion-Energy Diagnostic Report"
Cohesion: 0.11
Nodes (22): Per-Category Motion Energy Example (sign 'beside'), MediaPipe Landmark Groups (pose/left_hand/right_hand/face), RMS Speed (Motion-Energy Metric), Per-Video Motion Energy Example (video 2234689254, 'radio'), One-handed signs leave the unused hand channel at zero motion energy, RMS Speed: xyz vs xy-only (z-axis share of motion energy), Z-axis dominates measured motion energy for pose landmarks, Landmark-Subset Probe Leaderboard (+14 more)

### Community 14 - "Training Config Schema"
Cohesion: 0.15
Nodes (8): Any, Hyperparameter deviations this architecture declares — usually empty. Kept…, Resolved HYP for one architecture: shared values + its overrides., Landmark subsets to train for this architecture (global list unless the…, False parks an architecture without deleting its settings., One row per architecture — what the notebook displays so the whole grid (and…, Recorded in every run's meta.json as `training.source`., TrainingConfig

### Community 15 - "Recognition Reporting Utils"
Cohesion: 0.14
Nodes (19): matplotlib_pyplot, Stage 2: landmark tensors -> gloss. Everything that turns a landmark sequence…, comparison_row(), confusion_matrix(), load_history(), plot_confusion(), Path, Run reporting: learning-curve figures into a run's assets/ + comparison rows. (+11 more)

### Community 16 - "Frame Feature Extraction"
Cohesion: 0.18
Nodes (19): build_cache(), build_frame_features(), cache_dir(), cache_inputs(), cache_key(), _center_and_scale(), _derivatives(), load_video() (+11 more)

### Community 17 - "MediaPipe Holistic Extraction"
Cohesion: 0.20
Nodes (18): multiprocessing, benchmark_worker_counts(), default_n_workers(), extract_dataset(), landmarks_root(), load_manifest(), _manifest_path(), pending_jobs() (+10 more)

### Community 18 - "POPSIGN Extraction CLI"
Cohesion: 0.15
Nodes (15): cmd_pilot(), cmd_run(), load_split(), main(), DataFrame, Run POPSIGN landmark extraction (TODO §2.2) as a real script. **Why a script…, The real thing: resumable extraction of a whole split., `--confidence` → a threshold dict. Accepts a named config from the tuning… (+7 more)

### Community 19 - "Per-Dataset Video Sources"
Cohesion: 0.15
Nodes (17): get_source(), Per-dataset extraction adapters. Each dataset differs in exactly three ways —…, One corpus of raw video, as the extractor sees it., VideoSource, artifact_path(), landmarks_dir(), landmarks_root(), manifest() (+9 more)

### Community 20 - "Deno Project Config"
Cohesion: 0.11
Nodes (18): compilerOptions, lib, strict, exports, fmt, lineWidth, imports, @mediapipe/tasks-vision (+10 more)

### Community 21 - "FirstPlace Dataset Loader"
Cohesion: 0.12
Nodes (14): cached_lengths(), drop_empty_frames(), FirstPlaceDataset, mirror_permutation(), preprocess(), Dataset, ndarray, Permutation p such that ``x[:, p]`` is the left/right-swapped subset. Asserts… (+6 more)

### Community 22 - "Interpretability Training Utils"
Cohesion: 0.14
Nodes (18): _frame_mask(), load_split_arrays(), predict_probs_indexed(), Module, ndarray, no_grad, Path, Tensor (+10 more)

### Community 23 - "AWP/Cosine LR Training Utils"
Cohesion: 0.16
Nodes (17): cosine_one_cycle(), frozen_bn_stats(), Run a forward pass without letting it move any BatchNorm running stats. AWP…, Linear warmup then cosine decay to ``lr_min_ratio * lr``, stepped per BATCH…, _build_meta(), _evaluate(), _is_finished(), load_fp_config() (+9 more)

### Community 24 - "Generic Cache Key Hashing"
Cohesion: 0.16
Nodes (17): functools, hashlib, cache_dir(), cache_inputs(), cache_key(), features_root(), _hash_manifest(), manifest_fingerprint() (+9 more)

### Community 25 - "Canonical Eval Runner"
Cohesion: 0.14
Nodes (15): build_model(), Module, The ONLY model-constructor call in the training/eval stack. Architecture-…, evaluate_run(), load_one(), load_video(), load_video_firstplace(), main() (+7 more)

### Community 26 - "Landmark NPZ I/O"
Cohesion: 0.15
Nodes (17): iter_landmark_files(), ndarray, Path, Write one video's landmark tensor. Returns the final path. ``num_frames`` is…, ``(landmarks, meta)`` for one video. Widening float16 -> float32 on read is the…, Every completed npz under a landmarks tree, in a stable order. ``.tmp.npz``…, read_landmark_npz(), write_landmark_npz() (+9 more)

### Community 27 - "Landmark Subset Definitions"
Cohesion: 0.14
Nodes (12): get_subset(), LandmarkSubset, _make(), pose_rows(), ndarray, Canonical registry of MediaPipe Holistic landmark subsets. Single source of…, A named landmark subset: sorted holistic row indices + provenance., Flat feature-column indices for a landmark-major (543 × xyz) frame. ``coords``:… (+4 more)

### Community 28 - "Landmark Overlay Rendering"
Cohesion: 0.18
Nodes (16): annotate(), contact_sheet(), draw_frame(), _px(), ndarray, Path, Render video frames with extracted landmarks drawn on top — the visual half of…, Write one annotated PNG per requested frame index. Seeks directly to each frame… (+8 more)

### Community 29 - "POPSIGN Download/Extract Cycle"
Cohesion: 0.19
Nodes (16): delete_videos(), download(), extract(), load_state(), main(), Path, POPSIGN one-part-at-a-time: download → extract → **verify** → delete. The five…, Remove one part's raw video. Re-downloadable, but slowly. (+8 more)

### Community 30 - "Canonical Split & Data Source"
Cohesion: 0.12
Nodes (15): get_canonical_split(), DataFrame, Path, The canonical split, and the constants every run must agree on. Feature…, GISLR_Stratified's own fixed split (``train.csv`` / ``test.csv``) — identical…, DatasetSource, get_source(), Dataset seam for the training stack (TODO §9.5). Everything in… (+7 more)

### Community 31 - "Kaggle Submission/Leaderboard"
Cohesion: 0.18
Nodes (15): get_duckdb_conn(), kaggle_submit_command(), leaderboard(), Path, query_runs(), Which trained runs still need scoring on the official/held-out test set. Every…, Submit one run's ``export/submission.zip`` and record the result.…, In-memory DuckDB connection — the project's standard loading layer. (+7 more)

### Community 32 - "Landmark Importance Scoring"
Cohesion: 0.17
Nodes (15): attention_weights(), combine_ranking(), gradient_saliency(), _normalized_rank(), DataFrame, ndarray, Per-landmark importance/ranking for a trained DNN/LSTM/GRU (TODO §3). Three…, (N,) values -> (N,) rank in [0, 1], 1 = most important (highest value). (+7 more)

### Community 33 - "TS Extractor NPZ Verification"
Cohesion: 0.17
Nodes (14): argparse, check(), encode_npy(), encode_npz(), main(), ndarray, Does `npz.ts` produce something numpy actually reads? Two modes, because they…, Transcription of npz.ts::toFloat16. (+6 more)

### Community 34 - "Extractor Parity Runner"
Cohesion: 0.21
Nodes (14): main(), DataFrame, Run both extractors over the same handful of clips — the gate in TODO §10.1.…, Mirror the chosen clips into `<label>/<id>.mp4` under VIDEO_DIR. Hardlinked…, `deno run src/cli.ts` at native geometry, against the local .task model.…, Same videos, same model, into `<WORK_DIR>/py/<label>/<id>.npz`., numpy must actually read what `npz.ts` wrote. This is the execution-level check…, `limit` clips that still have video on disk, stable across runs. Sampled one-… (+6 more)

### Community 35 - "Landmarker Worker Pool"
Cohesion: 0.15
Nodes (14): _results(), _extract_one(), _init_worker(), _make_landmarker(), ndarray, Pool initializer: quiet stderr + single-threaded math libs + a landmarker.…, Close the current landmarker and build a fresh one. **Closing is the point.**…, One HolisticLandmarker in VIDEO mode. `confidence` overrides any subset of… (+6 more)

### Community 36 - "Unified Training Driver"
Cohesion: 0.28
Nodes (12): atomic_torch_save(), _atomic_write_json(), _build_meta(), _is_finished(), Path, The unified training driver behind every gislr.1.model.*.ipynb notebook. One…, Train every subset for one architecture, using the shared training config. This…, Train one registry run; returns its run folder. Builds missing feature caches,… (+4 more)

### Community 37 - "AWP Optimizer Wrapper"
Cohesion: 0.18
Nodes (7): contextlib, math, AWP, Optimization pieces the 1st-place recipe needs and torch does not ship. The…, Move every weight along its gradient. Returns False (and changes nothing) if…, Adversarial Weight Perturbation — the reference's main regularizer. Each step,…, torch

### Community 38 - "Landmark Tensor Contract (sb-core)"
Cohesion: 0.20
Nodes (9): dataclasses, os, Reading and writing landmark tensors — the one implementation of the on-disk…, The landmark-tensor contract between extraction (stage 1) and training (stage…, The contract as data — for embedding in a manifest or a run record., spec(), Training configuration: one file is the source of truth for every architecture.…, pathlib (+1 more)

### Community 39 - "Fold-Based Array Datasets"
Cohesion: 0.18
Nodes (9): DataLoader, collate_with_row(), FoldArrayDataset, make_fold_loader(), make_row_tracked_loader(), Dataset, Never shuffled — used wherever a prediction must be scattered back into a full-…, Same flat-cache read as ``base_v1.SubsetArrayDataset``, indexed by an explicit… (+1 more)

### Community 40 - "Gloss Vocabulary Mapping"
Cohesion: 0.17
Nodes (11): json, check_covers(), derive_label_map(), invert(), load_label_map(), Path, The gloss vocabulary: sign name <-> class index. Small, but it belongs in `sb-…, `{sign: index}` from a dataset's official label map. (+3 more)

### Community 41 - "Extractor Parity Comparison"
Cohesion: 0.22
Nodes (9): numpy, compare_one(), main(), Path, Do the Python and TypeScript extractors agree? — measure before switching.…, Structural + geometric comparison of one clip extracted both ways., On-disk read for the GISLR_Stratified npz dataset — shared by both feature…, sb_core_io (+1 more)

### Community 42 - "Feature Cache Migration"
Cohesion: 0.27
Nodes (10): legacy_files(), main(), plan(), Path, Move the flat, name-keyed feature caches into content-addressed directories.…, ``subset_tag`` is not invertible on its own; build the lookup from the…, Shape check: offsets count matches the split, data size matches the frames ×…, One entry per (pipeline, subset, coords, split) group found on disk. (+2 more)

### Community 43 - "LateDropout/MaskedBatchNorm Layers"
Cohesion: 0.28
Nodes (4): LateDropout, MaskedBatchNorm1d, BatchNorm over valid frames only. Padded frames must not enter the batch…, Dropout that stays OFF for the first ``start_step`` optimizer steps. A 0.8…

### Community 44 - "Lookahead Optimizer"
Cohesion: 0.25
Nodes (4): Optimizer, Lookahead, Lookahead (Zhang et al. 2019), ``sync_period=5`` in the reference. Keeps a set…, Call once after every completed optimizer step. Returns True on the steps where…

### Community 45 - "Length-Bucketed Batch Sampler"
Cohesion: 0.32
Nodes (3): LengthBucketedBatchSampler, Batch clips of similar length together. **Why this is not optional.** GISLR…, Re-shuffle for the coming epoch (call alongside ``dataset.set_epoch``).

### Community 46 - "meta.json Schema (v4)"
Cohesion: 0.25
Nodes (7): additionalProperties, description, $id, required, $schema, title, type

### Community 47 - "MediaPipe Landmark Groups Overview"
Cohesion: 0.43
Nodes (7): Face landmark group, GISLR video corpus (94,477 videos), Hand landmark groups (left/right), Global RMS Speed Overview Chart, Pose landmarks show highest and most variable motion magnitude, Pose landmark group, RMS Speed (per-landmark motion metric)

### Community 48 - "sb-core Shared Contracts"
Cohesion: 0.29
Nodes (5): Contracts shared by every stage and both directions. `sb-core` is the seam.…, empty_sequence(), Gloss -> pose sequence — NOT IMPLEMENTED. The one thing fixed in advance:…, An all-absent pose sequence of the right shape — the only thing this module can…, sb_core

### Community 49 - "Causal Depthwise Conv Block"
Cohesion: 0.29
Nodes (4): CausalDWConv1D, Conv1DBlock, Depthwise 1D conv, left-padded by (kernel_size-1)*dilation so frame t never…, The reference's efficient conv block: expand -> causal depthwise -> masked BN…

### Community 50 - "Conv1D-Transformer Trunk Stage"
Cohesion: 0.29
Nodes (4): Conv1DTransformerStage, Pre-norm transformer block using BatchNorm (not LayerNorm) — the reference's…, One stage of the 1st-place trunk: three causal Conv1DBlocks (local, causal…, TransformerBlock

### Community 51 - "Workspace Package Map"
Cohesion: 0.57
Nodes (7): sb-core, sb-extract, sb-mlops, sb-recognize, sb-rescore, sb-synthesize, signbridge

### Community 52 - "Registry Field Descriptions"
Cohesion: 0.29
Nodes (7): description, type, description, type, properties, architecture, metrics

### Community 53 - "Motion-Energy Scope Charts"
Cohesion: 0.33
Nodes (6): Per-video RMS speed chart — video 2234689254 'radio', Per-video RMS speed chart — video 2649863433 'pajamas', Chart: RMS speed xyz vs xy-only — z-axis share of motion energy by landmark group, Chart: Landmark-subset probe leaderboard (ME_126, ME_132, FP_118, HANDS_POSE_50, HANDS_42, FULL_543), Chart: Scope A within-class consistency — CV of rms_speed_xy across 10 videos of 'better', Chart: Scope B per-landmark ANOVA F-ratio (max over descriptors), 10 classes

### Community 54 - "Landmark-Importance Result Charts"
Cohesion: 0.33
Nodes (6): Region-level Landmark Importance Chart, Held-out Test Confusion Matrices (DNN/GRU/LSTM), Top-20 Landmark Importance Ranking (DNN/GRU/LSTM), Per-layer Weight Histograms (DNN/GRU/LSTM), Cross-scope Mean RMS Speed per Landmark Chart, Global Motion Energy Overview Chart

### Community 55 - "Subset ANOVA & Early Run Charts"
Cohesion: 0.53
Nodes (6): Global per-landmark ANOVA F chart (250 classes), ME_126 run 1784385530 learning curves (loss/accuracy/LR), ME_126 run 1784385530 per-class accuracy distribution, ME_132 run 1784386490 learning curves (loss/accuracy/LR), ME_132 run 1784386490 per-class accuracy distribution, FP_118 run 1784387336 learning curves (loss/accuracy/LR)

### Community 56 - "In-RAM Array Dataset"
Cohesion: 0.33
Nodes (3): Dataset, Flat in-RAM cache + offsets; uniform subsample past MAX_SEQ_LEN. In-RAM with…, SubsetArrayDataset

### Community 57 - "Mid-Batch Run Learning Curves"
Cohesion: 0.60
Nodes (6): ME_132 run 1784393683 learning curves (loss/accuracy/LR schedule), ME_132 run 1784393683 per-class accuracy distribution, FP_118 run 1784394315 learning curves (loss/accuracy/LR schedule), FP_118 run 1784394315 per-class accuracy distribution, ME_126 run 1784395799 learning curves (loss/accuracy/LR schedule), ME_126 run 1784395799 per-class accuracy distribution

### Community 58 - "Mid-Batch Run Accuracy Charts"
Cohesion: 0.47
Nodes (6): ME_126 Run 1784399741 Learning Curves, ME_126 Run 1784399741 Per-Class Accuracy, ME_132 Run 1784400020 Learning Curves, ME_132 Run 1784400020 Per-Class Accuracy, ME_132 Run 1784400832 Learning Curves, ME_132 Run 1784400832 Per-Class Accuracy

### Community 59 - "Late-Batch Per-Class Accuracy Charts"
Cohesion: 0.33
Nodes (6): Per-Class Accuracy Chart (run 1789253641), Per-Class Accuracy Chart (run 1789254670), Per-Class Accuracy Chart (run 1789255418), Per-Class Accuracy Chart (run 1789256241), Per-Class Accuracy Chart (run 1789257008), Per-Class Accuracy Chart (run 1789257785)

### Community 60 - "GRU Baseline Accuracy Results"
Cohesion: 0.60
Nodes (5): GISLR 250-Sign Vocabulary, GRU Baseline Per-Class Accuracy Chart, GRU Baseline Macro Accuracy (0.704), GRU Baseline Overall Accuracy (0.706), 15 Worst-Performing Sign Classes (GRU Baseline: give, there, beside, vacuum, nap, wake, awake, puzzle, jeans, ride, mouth, lips, chin, empty, pajamas)

### Community 61 - "GRU ME-126 Learning Dynamics"
Cohesion: 0.50
Nodes (5): GRU ME-126 Learning Curves (Chart), ME-126 Landmark Subset, One-Cycle LR Schedule (peak ~3e-4 at step ~8000), Train/Val Overfitting Gap After Epoch ~20, Validation Accuracy Plateau (~0.73)

### Community 62 - "Per-Category Motion Energy Insight"
Cohesion: 0.70
Nodes (5): RMS Speed per Landmark Category Chart (sign 'beside'), Insight: face/lips landmarks are nearly static, Insight: pose landmarks show highest motion variance, RMS speed per landmark metric, Sign 'beside' example (310 videos)

### Community 65 - "Streaming Dropout Mechanism"
Cohesion: 0.40
Nodes (3): Tensor, Dropout with the reference's ``noise_shape=(None, 1, 1)``: the mask is per-…, _sample_dropout()

### Community 69 - "Training Config Loader/Validator"
Cohesion: 0.50
Nodes (5): load_config(), Path, Fail fast and specifically — a bad config must not surface as a weird error…, Read + validate the training config. Cheap and side-effect free, so every cell…, validate()

### Community 70 - "Permutation Importance Method"
Cohesion: 0.60
Nodes (5): permutation_importance(), no_grad, Tensor, Baseline video-accuracy + (543,) accuracy DROP per landmark on shuffling that…, _video_preds()

### Community 71 - "GRU Baseline Learning Curves"
Cohesion: 0.67
Nodes (4): GRU Baseline Train/Val Accuracy Curve, GRU Baseline Learning Curves Chart, GRU Baseline Train/Val Loss Curve, GRU Baseline LR Schedule

### Community 72 - "GRU ME-126 Accuracy Distribution"
Cohesion: 0.50
Nodes (4): GRU ME-126 Per-Class Accuracy Distribution (Chart), ME-126 Landmark Subset (data source for this chart), GRU ME-126 Overall/Macro Accuracy (73.7% / 73.5%), 15 Worst-Performing Sign Classes (ME-126 GRU)

### Community 73 - "Architecture Spec Registry"
Cohesion: 0.50
Nodes (3): inspect, ArchSpec, The GISLR benchmark architectures — the single definition of every model class,…

### Community 75 - "ME_132/FP_118 Run Pair Charts"
Cohesion: 0.67
Nodes (4): ME_132 run 1784396516 — learning curves (loss/accuracy/LR), ME_132 run 1784396516 — per-class accuracy distribution (overall 0.748), FP_118 run 1784397301 — learning curves (loss/accuracy/LR), FP_118 run 1784397301 — per-class accuracy distribution (overall 0.753)

### Community 76 - "Cross-Scope Motion Consistency"
Cohesion: 1.00
Nodes (3): Cross-scope RMS Speed Comparison Chart, Landmark Motion Profile by Body Region, Cross-Scope Sampling Consistency Finding

### Community 77 - "ME-126 vs Baseline Accuracy"
Cohesion: 1.00
Nodes (3): Baseline Full 543-Landmark Model, Per-Class Accuracy Scatter: ME-126 vs Baseline, ME-126 Landmark Subset

### Community 78 - "Per-Landmark Saliency Charts"
Cohesion: 1.00
Nodes (3): Scope B: Per-Landmark Mutual Information Chart, Global: Per-Landmark ANOVA F Chart (250 classes), Saliency by Axis x Region Chart (DNN, GRU, LSTM)

### Community 79 - "Confidence-Tuning Threshold Charts"
Cohesion: 0.67
Nodes (3): Default vs Strict Pose Threshold Comparison Chart, Hand Presence Profile Over Clip Position Chart, Whole-Clip vs Signing-Span Detection Rate Window Effect Chart

### Community 81 - "Late Registry Run Charts"
Cohesion: 1.00
Nodes (3): Per-Class Accuracy Chart (run 1789558839), Per-Class Accuracy Chart (run 1789559734), Per-Class Accuracy Chart (run 1789560829)

### Community 82 - "meta.json Field: assets"
Cohesion: 0.67
Nodes (3): description, type, assets

### Community 83 - "meta.json Field: checkpoints"
Cohesion: 0.67
Nodes (3): description, type, checkpoints

### Community 84 - "meta.json Field: coords"
Cohesion: 0.67
Nodes (3): description, type, coords

### Community 85 - "meta.json Field: created"
Cohesion: 0.67
Nodes (3): description, type, created

### Community 86 - "meta.json Field: dataset"
Cohesion: 0.67
Nodes (3): description, type, dataset

### Community 87 - "meta.json Field: feature_dim"
Cohesion: 0.67
Nodes (3): description, type, feature_dim

### Community 88 - "meta.json Field: hyperparameters"
Cohesion: 0.67
Nodes (3): description, type, hyperparameters

### Community 89 - "meta.json Field: model_name"
Cohesion: 0.67
Nodes (3): description, type, model_name

### Community 90 - "meta.json Field: n_classes"
Cohesion: 0.67
Nodes (3): description, type, n_classes

### Community 91 - "meta.json Field: n_landmarks"
Cohesion: 0.67
Nodes (3): description, type, n_landmarks

### Community 92 - "meta.json Field: n_params"
Cohesion: 0.67
Nodes (3): description, type, n_params

### Community 93 - "meta.json Field: notes"
Cohesion: 0.67
Nodes (3): description, type, notes

### Community 94 - "meta.json Field: provenance"
Cohesion: 0.67
Nodes (3): provenance, description, type

### Community 95 - "meta.json Field: run_id"
Cohesion: 0.67
Nodes (3): run_id, description, type

### Community 96 - "meta.json Field: schema_version"
Cohesion: 0.67
Nodes (3): schema_version, description, type

### Community 97 - "meta.json Field: split"
Cohesion: 0.67
Nodes (3): split, description, type

### Community 98 - "meta.json Field: streaming"
Cohesion: 0.67
Nodes (3): streaming, description, type

### Community 99 - "meta.json Field: submission"
Cohesion: 0.67
Nodes (3): submission, description, type

### Community 100 - "meta.json Field: subset"
Cohesion: 0.67
Nodes (3): subset, description, type

### Community 101 - "meta.json Field: training"
Cohesion: 0.67
Nodes (3): training, description, type

## Knowledge Gaps
- **149 isolated node(s):** `name`, `version`, `exports`, `extract`, `check` (+144 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 569 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **46 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TrainingConfig` connect `Training Config Schema` to `Canonical Eval Runner`, `Training Config Loader/Validator`, `Landmark Tensor Contract (sb-core)`, `Recognition Reporting Utils`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `build_cache()` connect `Base Feature Cache Pipeline` to `Generic Cache Key Hashing`, `Landmark Subset Definitions`, `Recognition Reporting Utils`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **What connects `name`, `version`, `exports` to the rest of the system?**
  _149 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Kaggle Checkpoint Artifacts` be split into smaller, more focused modules?**
  _Cohesion score 0.05913461538461538 - nodes in this community are weakly interconnected._
- **Should `Dataset Path Resolution` be split into smaller, more focused modules?**
  _Cohesion score 0.05587808417997097 - nodes in this community are weakly interconnected._
- **Should `TS Extractor CLI` be split into smaller, more focused modules?**
  _Cohesion score 0.07312925170068027 - nodes in this community are weakly interconnected._
- **Should `CLAUDE.md Repo Conventions` be split into smaller, more focused modules?**
  _Cohesion score 0.1226215644820296 - nodes in this community are weakly interconnected._