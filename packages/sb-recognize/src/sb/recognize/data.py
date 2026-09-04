"""The canonical split, and the constants every run must agree on.

Feature construction and caching moved out to :mod:`sb.recognize.features` when
the pipelines were named and versioned — what is left here is the part that is
**not** pipeline-specific and must never vary: the split, its seed, and the
asserted val-set size.

The canonical split (stratified 90/10, ``random_state=42`` -> 9,448-video val
set) is THE comparability requirement. Every run and
:mod:`sb.recognize.evaluate` reproduce it identically; a new run displaces a
leaderboard entry only on this same split and metric.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from sb.core.schema import N_LANDMARKS
from sb.core.vocab import check_covers, load_label_map  # re-exported: callers ask data for both
from sb.recognize.features.base_v1 import MAX_SEQ_LEN

SEED = 42  # canonical project seed (split + training)
ROWS_PER_FRAME = N_LANDMARKS  # 543 — the landmark schema owns this number
N_VAL = 9448  # canonical val-set size — asserted, never assumed

__all__ = [
    "SEED", "ROWS_PER_FRAME", "MAX_SEQ_LEN", "N_VAL",
    "load_label_map", "get_canonical_split", "subset_tag",
]


def get_canonical_split(
    data_dir: Path, sign2idx: dict[str, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 90/10 split, random_state=42 — identical to every leaderboard
    run and to sb.recognize.evaluate (the canonical evaluation).
    Deterministic and cheap, so consumers call it instead of sharing live state."""
    df = pd.read_csv(data_dir / "train.csv")
    check_covers(sign2idx, df["sign"].unique())
    df["label"] = df["sign"].map(sign2idx)
    tr, va = train_test_split(df, test_size=0.1, stratify=df["sign"], random_state=SEED)
    assert len(va) == N_VAL, "val-set size drifted — leaderboard comparability broken"
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def subset_tag(name: str, coords: str = "xyz") -> str:
    """Run tag: 'ME_126' -> 'me126' (+'-xy' when z is dropped).

    This is the *human* handle — registry pointer-file keys, progress-bar
    labels, run notes. It is deliberately not the feature-cache identifier: a
    name cannot notice that its subset's indices changed, which is what
    :func:`sb.recognize.features.cache.cache_key` is for.
    """
    tag = name.lower().replace("_", "")
    return tag if coords == "xyz" else f"{tag}-{coords}"
