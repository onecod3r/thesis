"""Unified GISLR training stack shared by every gislr.1.model.*.ipynb notebook
and the modules/scripts/ CLIs.

- ``architectures`` — the model classes (StreamingGRU/StreamingLSTM/BiLSTM/
  CausalConv1D) and the ``ARCHS`` registry; the single definition the training
  notebooks AND the eval script load state_dicts against.
- ``data``          — canonical split, per-subset feature caches, in-RAM dataset.
- ``registry``      — run folders (data/models/<epoch-seconds>/), meta.json
  writing (schema: README.md §"Model registry"), asset registration.
- ``config``        — the shared training config (src/config/gislr.training.json):
  one source of truth for hyperparameters across every architecture.
- ``train``         — the training driver (auto-resume, early stopping,
  single-progress-bar reporting).
- ``report``        — learning-curve plotting into a run's assets/.
- ``export``        — deployment export: run -> ONNX -> TF SavedModel -> TFLite
  -> submission.zip, arch-generic (driven by gislr.2.models.evaluation.ipynb).
- ``submission``    — the submission queue: DuckDB over every meta.json,
  "untested runs for dataset X, capped at the daily limit", + the Kaggle call.
"""

from modules.model.architectures import ARCHS, ArchSpec, build_model
from modules.model.data import (
    MAX_SEQ_LEN,
    ROWS_PER_FRAME,
    SEED,
    build_subset_cache,
    get_canonical_split,
    load_label_map,
    subset_tag,
)
from modules.model.export import export_run
from modules.model.registry import (
    CKPT_BEST,
    CKPT_LAST,
    META_GLOB,
    SUBMISSION_DEFAULT,
    eval_command,
    load_meta,
    mark_tested,
    migrate_all,
    pointer_run_dir,
    resolve_run_dir,
    write_meta,
)
from modules.model.submission import leaderboard, submit_run, untested_runs
from modules.model.report import (
    comparison_row,
    confusion_matrix,
    load_history,
    plot_confusion,
    save_learning_curves,
)
from modules.model.config import TrainingConfig, load_config
from modules.model.train import train_from_config, train_run

__all__ = [
    "ARCHS",
    "ArchSpec",
    "build_model",
    "MAX_SEQ_LEN",
    "ROWS_PER_FRAME",
    "SEED",
    "build_subset_cache",
    "get_canonical_split",
    "load_label_map",
    "subset_tag",
    "CKPT_BEST",
    "CKPT_LAST",
    "META_GLOB",
    "SUBMISSION_DEFAULT",
    "eval_command",
    "export_run",
    "leaderboard",
    "load_meta",
    "mark_tested",
    "migrate_all",
    "pointer_run_dir",
    "resolve_run_dir",
    "submit_run",
    "untested_runs",
    "write_meta",
    "comparison_row",
    "confusion_matrix",
    "load_history",
    "plot_confusion",
    "save_learning_curves",
    "TrainingConfig",
    "load_config",
    "train_from_config",
    "train_run",
]
