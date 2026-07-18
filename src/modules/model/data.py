"""GISLR data layer: canonical split, per-subset feature caches, in-RAM dataset.

The feature caches live at ``data/cache/gislr/features/`` — one flat float32
array + frame-offset index per (split, subset, coords), decoded from the raw
parquet once and shared by every architecture notebook. Cache builds are
resumable policy-wise: skipped when both files exist, written atomically
(temp file + ``os.replace``), so an interrupt never leaves a half-written cache.

The canonical split (stratified 90/10, ``random_state=42`` → 9,448-video val
set) is THE leaderboard comparability requirement — identical here and in
``modules/scripts/eval_gru.py``.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from modules.paths import CACHE_DIR

SEED = 42  # canonical project seed (split + training)
ROWS_PER_FRAME = 543  # holistic rows per frame (GISLR parquet layout)
MAX_SEQ_LEN = 128  # uniform-subsample cap, identical across every run
N_VAL = 9448  # canonical val-set size — asserted, never assumed

FEATURES_DIR = CACHE_DIR / "gislr" / "features"


def load_label_map(data_dir: Path) -> dict[str, int]:
    import json

    return json.loads((data_dir / "sign_to_prediction_index_map.json").read_text())


def get_canonical_split(
    data_dir: Path, sign2idx: dict[str, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 90/10 split, random_state=42 — identical to every leaderboard
    run and to modules/scripts/eval_gru.py (the canonical evaluation).
    Deterministic and cheap, so consumers call it instead of sharing live state."""
    df = pd.read_csv(data_dir / "train.csv")
    missing = set(df["sign"].unique()) - set(sign2idx)
    assert not missing, f"signs missing from the official label map: {missing}"
    df["label"] = df["sign"].map(sign2idx)
    tr, va = train_test_split(df, test_size=0.1, stratify=df["sign"], random_state=SEED)
    assert len(va) == N_VAL, "val-set size drifted — leaderboard comparability broken"
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def subset_tag(name: str, coords: str = "xyz") -> str:
    """Cache/run tag: 'ME_126' -> 'me126' (+'-xy' when z is dropped) — matches
    the historical cache naming so existing caches are reused, not rebuilt."""
    tag = name.lower().replace("_", "")
    return tag if coords == "xyz" else f"{tag}-{coords}"


def load_video_subset(path, rows: np.ndarray, coords: str = "xyz") -> np.ndarray:
    """One parquet -> (T, len(rows), len(coords)) float32, NaN->0.
    Row selection happens here so caches only ever hold the subset's data."""
    cols = list(coords)
    table = pq.read_table(path, columns=cols)
    data = np.column_stack([table.column(c).to_numpy() for c in cols])
    n = data.shape[0] // ROWS_PER_FRAME
    arr = data.reshape(n, ROWS_PER_FRAME, len(cols))[:, rows, :].astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_subset_cache(
    df: pd.DataFrame, prefix: str, subset, coords: str, data_dir: Path, progress=None
) -> tuple[Path, Path]:
    """Decode every parquet of one split once into one flat float32 array +
    frame offsets under data/cache/gislr/features/. Skip-if-exists; atomic.

    ``progress``: optional callable(done, total) for single-bar reporting.
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    tag = subset_tag(subset.name, coords)
    data_path = FEATURES_DIR / f"{prefix}_{tag}_data.npy"
    off_path = FEATURES_DIR / f"{prefix}_{tag}_offsets.npy"
    if data_path.exists() and off_path.exists():
        return data_path, off_path

    t0 = time.time()
    paths = [data_dir / p for p in df["path"]]
    rows = subset.array
    chunks, offsets = [], [0]
    with ThreadPoolExecutor(12) as ex:
        for i, arr in enumerate(
            ex.map(lambda p: load_video_subset(p, rows, coords), paths)
        ):
            chunks.append(arr.reshape(-1))
            offsets.append(offsets[-1] + arr.shape[0])
            if progress is not None and i % 500 == 0:
                progress(i, len(paths))
    flat = np.concatenate(chunks)
    for target, payload in (
        (data_path, flat),
        (off_path, np.asarray(offsets, dtype=np.int64)),
    ):
        tmp = target.with_suffix(".tmp.npy")
        np.save(tmp, payload)
        os.replace(tmp, target)
    print(
        f"{prefix}/{tag}: cached {len(df)} videos, {flat.nbytes / 1e9:.2f} GB "
        f"({time.time() - t0:.0f}s)"
    )
    return data_path, off_path


class SubsetArrayDataset(Dataset):
    """Flat in-RAM cache + offsets; uniform subsample past MAX_SEQ_LEN.

    In-RAM with num_workers=0 is deliberate: it trains GISLR at ~0.3 min/epoch
    and sidesteps Windows spawn-pickling entirely."""

    def __init__(self, df, data_path, off_path, feature_dim, max_seq_len=MAX_SEQ_LEN):
        self.labels = df["label"].to_numpy()
        self.data = np.load(data_path)  # ~5 GB for an ME-126-sized train split
        self.offsets = np.load(off_path)
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len
        assert len(self.labels) == len(self.offsets) - 1, "cache/split mismatch"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        d = self.feature_dim
        arr = self.data[self.offsets[i] * d : self.offsets[i + 1] * d].reshape(-1, d)
        T = arr.shape[0]
        if T > self.max_seq_len:
            arr = arr[np.linspace(0, T - 1, self.max_seq_len).astype(int)]
            T = self.max_seq_len
        return torch.from_numpy(np.ascontiguousarray(arr)), T, int(self.labels[i])


def collate_fn(batch):
    batch.sort(key=lambda x: x[1], reverse=True)  # enforce_sorted packing
    feats, lengths, labels = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    padded = torch.zeros(len(feats), int(lengths[0]), feats[0].shape[1])
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = f
    return padded, lengths, labels
