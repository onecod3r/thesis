"""The canonical split, and the constants every run must agree on.

Feature construction and caching moved out to :mod:`sb.recognize.features` when
the pipelines were named and versioned — what is left here is the part that is
**not** pipeline-specific and must never vary: the split and the asserted
val-set size.

**Canonical split reset (2026-09-16).** GISLR moved from a live download of
the ``asl-signs`` *competition* parquet to a self-produced Kaggle *dataset*,
``GISLR_Stratified`` (pre-converted npz, `sb.core.paths.gislr_dir`), which
ships its own fixed 80/20 split (``train.csv``/``test.csv``, stratified on
``sign`` only, upstream seed 42 — not participant-disjoint) instead of the
90/10 split this module used to compute itself. This is a comparability reset
like the 2026-07-18 registry reset: the 37 runs canonically evaluated on the
old 9,448-video 90/10 split are historical references only, not comparable to
anything evaluated on this one. Every run and :mod:`sb.recognize.evaluate`
reproduce THIS split identically; a new run displaces a leaderboard entry only
on this same split and metric.
"""

from pathlib import Path

import pandas as pd

from sb.core.schema import N_LANDMARKS
from sb.core.subsets import subset_tag  # re-exported: the run tag lives with the subsets
from sb.core.vocab import check_covers, load_label_map  # re-exported: callers ask data for both
from sb.recognize.features.base_v1 import MAX_SEQ_LEN

SEED = 42  # the upstream split's seed (GISLR_Stratified's producing notebook) — recorded for provenance; not applied here, the split is read as given
ROWS_PER_FRAME = N_LANDMARKS  # 543 — the landmark schema owns this number
N_VAL = 18896  # canonical val-set size for GISLR_Stratified's fixed test split — asserted, never assumed

__all__ = [
    "SEED", "ROWS_PER_FRAME", "MAX_SEQ_LEN", "N_VAL",
    "load_label_map", "get_canonical_split", "subset_tag",
]


def get_canonical_split(
    data_dir: Path, sign2idx: dict[str, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GISLR_Stratified's own fixed split (``train.csv`` / ``test.csv``) —
    identical to every leaderboard run and to sb.recognize.evaluate (the
    canonical evaluation). Deterministic and cheap, so consumers call it
    instead of sharing live state."""
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "test.csv")
    check_covers(sign2idx, pd.concat([train_df["sign"], val_df["sign"]]).unique())
    train_df["label"] = train_df["sign"].map(sign2idx)
    val_df["label"] = val_df["sign"].map(sign2idx)
    assert len(val_df) == N_VAL, "val-set size drifted — leaderboard comparability broken"
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
