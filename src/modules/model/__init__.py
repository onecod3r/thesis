"""Unified GISLR training stack shared by every gislr.1.model.*.ipynb notebook
and the modules/scripts/ CLIs.

- ``architectures`` — the model classes (StreamingGRU/StreamingLSTM/BiLSTM/
  CausalConv1D) and the ``ARCHS`` registry; the single definition the training
  notebooks AND the eval script load state_dicts against.
- ``data``          — canonical split, per-subset feature caches, in-RAM dataset.
- ``registry``      — run folders (data/models/<epoch-seconds>/), meta.json
  writing (schema: README.md §"Model registry"), asset registration.
- ``train``         — the training driver (auto-resume, early stopping,
  single-progress-bar reporting).
- ``report``        — learning-curve plotting into a run's assets/.
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
from modules.model.registry import (
    CKPT_BEST,
    CKPT_LAST,
    eval_command,
    load_meta,
    pointer_run_dir,
    resolve_run_dir,
    write_meta,
)
from modules.model.report import comparison_row, save_learning_curves
from modules.model.train import train_run

__all__ = [
    "ARCHS", "ArchSpec", "build_model",
    "MAX_SEQ_LEN", "ROWS_PER_FRAME", "SEED",
    "build_subset_cache", "get_canonical_split", "load_label_map", "subset_tag",
    "CKPT_BEST", "CKPT_LAST", "eval_command", "load_meta", "pointer_run_dir",
    "resolve_run_dir", "write_meta",
    "comparison_row", "save_learning_curves",
    "train_run",
]
