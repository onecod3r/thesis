"""Data loading for the sign-vs-gap detector (TODO §17): reads GISLR-GapCorpus
from disk.

Unlike :mod:`sb.recognize.continuous.data` (composes streams on the fly, in
RAM, every epoch), this corpus is already materialized on disk by
:mod:`sb.recognize.sequences.gapcorpus_kaggle_notebook` -- many GB, several
random-neighbour passes over every GISLR clip -- so reading whole sequences
from their npz files per batch is the natural fit, not an on-the-fly composer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sb.recognize.sequences.compose import read_sequence


class GapCorpus(Dataset):
    """One split (``"train"`` or ``"test"``) of GISLR-GapCorpus: raw
    ``landmarks (T, 543, 3)`` (NaN-preserving) + ``gap (T,)`` 0/1, read lazily
    per index from ``root/<split>.csv`` and ``root/<npz_relpath>``."""

    def __init__(self, root: Path | str, split: str):
        self.root = Path(root)
        self.index = pd.read_csv(self.root / f"{split}.csv", keep_default_na=False)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        row = self.index.iloc[index]
        arr = read_sequence(self.root / row["npz_relpath"])
        return {"x": arr["landmarks"].astype(np.float32), "gap": arr["gap"].astype(np.float32),
                "seq_id": row["seq_id"]}


def collate(batch: list[dict]) -> dict:
    """Pad a list of sequences to the batch's max length. ``x`` is padded
    with NaN (not 0): :class:`~sb.recognize.architectures.KinematicFrontend`
    reads NaN as "not detected", so a padded frame correctly looks absent
    rather than looking like a real (0, 0) landmark; the loss masks it out
    regardless."""
    T = max(len(b["gap"]) for b in batch)
    B = len(batch)
    x = np.full((B, T, 543, 3), np.nan, dtype=np.float32)
    gap = np.zeros((B, T), np.float32)
    mask = np.zeros((B, T), bool)
    for i, b in enumerate(batch):
        n = len(b["gap"])
        x[i, :n] = b["x"]
        gap[i, :n] = b["gap"]
        mask[i, :n] = True
    return {"x": torch.from_numpy(x), "gap": torch.from_numpy(gap), "mask": torch.from_numpy(mask),
            "lengths": torch.tensor([len(b["gap"]) for b in batch]),
            "seq_id": [b["seq_id"] for b in batch]}
