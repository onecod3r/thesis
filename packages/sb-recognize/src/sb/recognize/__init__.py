"""Stage 2: landmark tensors -> gloss.

Everything that turns a landmark sequence into a prediction: the model classes,
the named feature pipelines, both training drivers, the canonical evaluation and
the deployment export. Run records themselves live in `sb-mlops`, which this
package depends on and which deliberately does not depend back — the registry
must be readable without importing torch.

- ``architectures`` — the model classes (StreamingGRU / StreamingLSTM / BiLSTM /
  CausalConv1D / Conv1DTransformer) and the ``ARCHS`` registry: the single
  definition the training drivers AND the eval script load state_dicts against.
- ``data``          — the canonical split and the constants every run agrees on.
- ``features``      — named, versioned input pipelines (``base_v1``,
  ``firstplace_v1``) over a shared content-addressed cache.
- ``sources``       — the dataset seam: dir resolver, label map, split, sample
  reader per dataset. Adding POPSIGN is an entry here, not a second driver.
- ``config``        — the shared training config: one source of truth for
  hyperparameters across every architecture.
- ``train``         — the training driver (auto-resume, early stopping, one
  progress bar per run).
- ``train_firstplace`` — the 1st-place recipe (fixed-length cosine, AWP,
  Lookahead, collapse/plateau stops); same registry, split and schema.
- ``optim``         — Lookahead, AWP and the cosine one-cycle torch lacks.
- ``evaluate``      — the canonical per-class evaluation, all architectures.
- ``report``        — learning curves, confusion matrices into a run's assets/.
- ``export``        — PyTorch -> native Keras rebuild -> TFLite, parity-gated.
"""

from sb.recognize.architectures import ARCHS, ArchSpec, build_model
from sb.recognize.config import TrainingConfig, load_config
from sb.recognize.data import (
    MAX_SEQ_LEN,
    N_VAL,
    ROWS_PER_FRAME,
    SEED,
    get_canonical_split,
    load_label_map,
    subset_tag,
)
from sb.recognize.export.tflite import export_run
from sb.recognize.features.base_v1 import build_cache as build_subset_cache
from sb.recognize.report import (
    comparison_row,
    confusion_matrix,
    load_history,
    plot_confusion,
    save_learning_curves,
)
from sb.recognize.sources import SOURCES, DatasetSource, get_source
from sb.recognize.train import train_from_config, train_run
from sb.recognize.train_firstplace import load_fp_config, train_firstplace

__all__ = [
    "ARCHS",
    "ArchSpec",
    "build_model",
    "TrainingConfig",
    "load_config",
    "MAX_SEQ_LEN",
    "N_VAL",
    "ROWS_PER_FRAME",
    "SEED",
    "build_subset_cache",
    "get_canonical_split",
    "load_label_map",
    "subset_tag",
    "export_run",
    "comparison_row",
    "confusion_matrix",
    "load_history",
    "plot_confusion",
    "save_learning_curves",
    "DatasetSource",
    "SOURCES",
    "get_source",
    "train_from_config",
    "train_run",
    "load_fp_config",
    "train_firstplace",
]
