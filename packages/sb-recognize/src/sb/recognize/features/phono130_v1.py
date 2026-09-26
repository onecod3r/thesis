"""`phono130_v1` — phonological features only: every clip's 543 landmarks become the 130
per-frame features of :mod:`sb.recognize.phonology` (TODO §3.10), and nothing else reaches the
model.

Same interface and cache layout as :mod:`sb.recognize.features.base_v1` (one flat float32 array
+ frame offsets per split, in-RAM dataset, uniform subsample past ``MAX_SEQ_LEN``), so the training
driver, ``sb-evaluate`` and the dataset class are shared. The subset and coords arguments only
address the cache: extraction always reads all 543 landmarks in xyz.

Bump ``PIPELINE_VERSION`` whenever :func:`sb.recognize.phonology.model_features` changes the
bytes it produces.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from sb.recognize import phonology as PH
from sb.recognize.features import cache
from sb.recognize.features.base_v1 import MAX_SEQ_LEN, SubsetArrayDataset, collate_fn

PIPELINE = "phono130_v1"
PIPELINE_VERSION = 2  # 2: inputs clipped to +-10 (phonology.CLIP)
NAN_POLICY = "zero"
WORKERS = 12

__all__ = ["PIPELINE", "SubsetArrayDataset", "build_cache", "cache_key", "collate_fn", "feature_dim", "load_video"]


def feature_dim(subset=None, coords: str = "xyz") -> int:
    return PH.F


def cache_inputs(subset, coords: str, data_dir: Path | str, dataset: str = "gislr") -> dict:
    return cache.cache_inputs(pipeline=PIPELINE, pipeline_version=PIPELINE_VERSION, nan_policy=NAN_POLICY,
                              subset=subset, coords=coords, data_dir=data_dir, dataset=dataset)


def cache_key(subset, coords: str, data_dir: Path | str, dataset: str = "gislr") -> str:
    return cache.cache_key(cache_inputs(subset, coords, data_dir, dataset))


def _features(path: str) -> np.ndarray:
    return PH.model_features(PH.load(path))


def load_video(path) -> tuple[np.ndarray, int]:
    """One clip for evaluation: ``((T, 130), T)`` with the training subsample past ``MAX_SEQ_LEN``."""
    arr = _features(str(path))
    T = len(arr)
    if T > MAX_SEQ_LEN:
        arr = arr[np.linspace(0, T - 1, MAX_SEQ_LEN).astype(int)]
        T = MAX_SEQ_LEN
    return arr, T


def build_cache(df: pd.DataFrame, prefix: str, subset, coords: str, data_dir: Path,
                progress=None, dataset: str = "gislr") -> tuple[Path, Path]:
    """Extract every clip of one split once into ``features/phono130_v1/<key>/``.
    Skip-if-exists (content-addressed); atomic."""
    inputs = cache_inputs(subset, coords, data_dir, dataset)
    root = cache.cache_dir(inputs)
    root.mkdir(parents=True, exist_ok=True)
    data_path, off_path = root / f"{prefix}_data.npy", root / f"{prefix}_offsets.npy"
    if data_path.exists() and off_path.exists():
        return data_path, off_path
    t0 = time.time()
    paths = [str(data_dir / p) for p in df["npz_relpath"]]
    chunks, offsets = [], [0]
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, arr in enumerate(ex.map(_features, paths, chunksize=64)):
            chunks.append(arr.reshape(-1))
            offsets.append(offsets[-1] + arr.shape[0])
            if progress is not None and i % 500 == 0:
                progress(i, len(paths))
    flat = np.concatenate(chunks).astype(np.float32)
    for target, payload in ((data_path, flat), (off_path, np.asarray(offsets, dtype=np.int64))):
        tmp = target.with_suffix(".tmp.npy")
        np.save(tmp, payload)
        os.replace(tmp, target)
    cache.write_sidecar(root, inputs)
    print(f"{prefix}/{PIPELINE} [{root.name}]: cached {len(df)} videos, {flat.nbytes / 1e9:.2f} GB "
          f"({time.time() - t0:.0f}s)")
    return data_path, off_path
