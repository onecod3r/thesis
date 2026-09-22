"""Dataset seam for the training stack (TODO §9.5).

Everything in ``modules/model`` resolved GISLR by hand: ``train.py`` and
``train_fp.py`` both called ``gislr_dir()``, ``data.load_label_map`` read
GISLR's ``sign_to_prediction_index_map.json``, ``get_canonical_split`` read its
``train.csv``, and the feature-cache root was the literal string ``"gislr"``.
That is the coupling that doubles when POPSIGN training starts — not the
``<dataset>.<stage>.<topic>.ipynb`` notebook names, which are a deliberate
convention and stay as they are.

A :class:`DatasetSource` bundles the four things the training stack actually
needs from a dataset:

1. **where it lives** — lazily, so importing this module downloads nothing;
2. **its label map** — sign → class index;
3. **its canonical split** — the comparability contract, seeded and asserted;
4. **how to read one sample** — npz for both GISLR (since 2026-09-16) and POPSIGN.

plus the identity used for cache addressing and provenance (name, upstream ref,
manifest filename).

Adding a second dataset is then a new entry in :data:`SOURCES`, not a second
copy of the training driver. What it will need before its numbers are
comparable to anything: a split function with a fixed seed **and an asserted
val size**, exactly as GISLR's has (``data.get_canonical_split`` asserts
``N_VAL``) — a split that can silently drift is not a benchmark.

**POPSIGN is deprecated (2026-09-22, TODO §2)** and was never actually added
here — ``get_source("popsign")`` still fails; GISLR remains the only
registered source. This seam is kept general on principle, not because a
second dataset is imminent.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sb.core import vocab
from sb.recognize import data as D
from sb.recognize.features import base_v1, cache


@dataclass(frozen=True)
class DatasetSource:
    """One dataset, as the training stack sees it."""

    name: str
    kaggle_ref: str | None
    manifest: str  # the split manifest, fingerprinted into provenance
    resolve_dir: Callable[[], Path]
    label_map: Callable[[Path], dict[str, int]]
    canonical_split: Callable[[Path, dict[str, int]], tuple[pd.DataFrame, pd.DataFrame]]
    read_sample: Callable[..., np.ndarray]
    sample_path: Callable[[Path, pd.Series], Path]

    @property
    def features_root(self) -> Path:
        """``data/cache/<dataset>/features`` — one subtree per dataset, which is
        the existing data-placement policy and stays that way."""
        return cache.features_root(self.name)


def _gislr_dir() -> Path:
    from sb.core.paths import gislr_dir

    return gislr_dir()


def _gislr_read_sample(path, rows, coords="xyz", *, keep_nan: bool = False):
    """One video's npz → ``(T, len(rows), len(coords))``.

    ``keep_nan`` picks the pipeline: the base stack zeroes NaN at cache-build
    time, the 1st-place stack must keep it (its normalization is NaN-aware and
    two augmentations write NaN as a mask).
    """
    if keep_nan:
        from sb.recognize.features import firstplace_v1

        return firstplace_v1.load_video_raw(path, rows, coords)
    return base_v1.load_video(path, rows, coords)


GISLR = DatasetSource(
    name="gislr",
    kaggle_ref="bracu23101281/gislr-stratified",
    manifest="train.csv",
    resolve_dir=_gislr_dir,
    label_map=vocab.load_label_map,
    canonical_split=D.get_canonical_split,
    read_sample=_gislr_read_sample,
    sample_path=lambda data_dir, row: data_dir / row["npz_relpath"],
)

SOURCES: dict[str, DatasetSource] = {GISLR.name: GISLR}


def get_source(name: str) -> DatasetSource:
    """The dataset a run trains on, by the name recorded in its meta.json."""
    if name not in SOURCES:
        raise KeyError(
            f"unknown dataset {name!r}; registered: {sorted(SOURCES)}. "
            "Adding one means a DatasetSource entry here (dir resolver, label "
            "map, seeded canonical split with an asserted val size, sample "
            "reader) — not a second training driver."
        )
    return SOURCES[name]
