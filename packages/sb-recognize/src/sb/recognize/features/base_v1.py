"""`base_v1` — the default feature pipeline: row-select, NaN -> 0, subsample.

One flat float32 array + a frame-offset index per (split, subset, coords),
decoded from the GISLR_Stratified npz files once and shared by every
architecture. NaN becomes 0 at cache-build time and clips longer than
``MAX_SEQ_LEN`` are uniformly subsampled at read time.

Contrast with :mod:`sb.recognize.features.firstplace_v1`, which must keep NaN
alive to training time and crops rather than subsamples. The two are separate
pipelines with separate cache addresses, not variants of one.

Bump ``PIPELINE_VERSION`` whenever a change here alters the bytes this produces:
every key moves with it, so no stale array can be reused under an address that
no longer describes it.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sb.recognize.features import cache
from sb.recognize.features.gislr_stratified import load_npz

PIPELINE = "base_v1"
PIPELINE_VERSION = 1
NAN_POLICY = "zero"

MAX_SEQ_LEN = 128  # uniform-subsample cap, identical across every run


def cache_inputs(subset, coords: str, data_dir: Path | str, dataset: str = "gislr") -> dict:
    """This pipeline's identity bound to one (subset, coords, dataset)."""
    return cache.cache_inputs(
        pipeline=PIPELINE,
        pipeline_version=PIPELINE_VERSION,
        nan_policy=NAN_POLICY,
        subset=subset,
        coords=coords,
        data_dir=data_dir,
        dataset=dataset,
    )


def cache_key(subset, coords: str, data_dir: Path | str, dataset: str = "gislr") -> str:
    return cache.cache_key(cache_inputs(subset, coords, data_dir, dataset))


def cache_dir(subset, coords: str, data_dir: Path | str, dataset: str = "gislr") -> Path:
    return cache.cache_dir(cache_inputs(subset, coords, data_dir, dataset))


def load_video(path, rows: np.ndarray, coords: str = "xyz") -> np.ndarray:
    """One GISLR_Stratified npz -> (T, len(rows), len(coords)) float32, NaN->0.
    Row selection happens here so caches only ever hold the subset's data."""
    arr = load_npz(path, rows, coords)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_cache(
    df: pd.DataFrame, prefix: str, subset, coords: str, data_dir: Path,
    progress=None, dataset: str = "gislr"
) -> tuple[Path, Path]:
    """Decode every parquet of one split once into ``features/base_v1/<key>/``.
    Skip-if-exists; atomic.

    Skip-if-exists is only safe because the directory is content-addressed: a
    hit means these exact inputs were cached before, not merely that something
    with the same subset name was.

    ``progress``: optional callable(done, total) for single-bar reporting.
    """
    from sb.recognize.data import subset_tag

    inputs = cache_inputs(subset, coords, data_dir, dataset)
    root = cache.cache_dir(inputs)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / f"{prefix}_data.npy"
    off_path = root / f"{prefix}_offsets.npy"
    if data_path.exists() and off_path.exists():
        return data_path, off_path

    t0 = time.time()
    paths = [data_dir / p for p in df["npz_relpath"]]
    rows = subset.array
    chunks, offsets = [], [0]
    with ThreadPoolExecutor(12) as ex:
        for i, arr in enumerate(ex.map(lambda p: load_video(p, rows, coords), paths)):
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
    cache.write_sidecar(root, inputs)
    print(
        f"{prefix}/{subset_tag(subset.name, coords)} [{root.name}]: cached "
        f"{len(df)} videos, {flat.nbytes / 1e9:.2f} GB ({time.time() - t0:.0f}s)"
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
