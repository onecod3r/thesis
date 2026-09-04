"""GISLR data layer: canonical split, per-subset feature caches, in-RAM dataset.

The feature caches live at ``data/cache/gislr/features/<pipeline>/<key>/`` — one
flat float32 array + frame-offset index per (split, subset, coords), decoded
from the raw parquet once and shared by every architecture notebook. Cache
builds are resumable policy-wise: skipped when both files exist, written
atomically (temp file + ``os.replace``), so an interrupt never leaves a
half-written cache.

**``<key>`` is a content address, not a name** (TODO §9.2). It hashes everything
that determined the cache's bytes: the pipeline and its version, the subset's
actual index array, coords, the NaN policy, and the dataset manifest the split
came from. The old flat ``<split>_<tag>_data.npy`` layout keyed on the subset's
*name*, so editing an index list in ``subsets.py`` left the tag unchanged and
every later run silently trained on the previous array — skip-if-exists made
that failure completely quiet. Under a content address the edited subset simply
addresses a different directory, and two runs are comparable on inputs exactly
when their ``feature_cache_key`` matches (recorded in meta.json's provenance
block).

The canonical split (stratified 90/10, ``random_state=42`` → 9,448-video val
set) is THE leaderboard comparability requirement — identical here and in
``modules/scripts/eval_gru.py``.
"""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
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

# ---- feature-cache identity (TODO §9.2) ------------------------------------
# Bump PIPELINE_VERSION whenever this module's cache-build path changes the
# bytes it produces; every key moves with it, so no stale array can be reused
# under a name that no longer describes it.
PIPELINE = "base_v1"  # row-select then NaN -> 0 at build time
PIPELINE_VERSION = 1
NAN_POLICY = "zero"
CACHE_KEY_FILE = "cache_key.json"  # sidecar: what a cache directory holds


def load_label_map(data_dir: Path) -> dict[str, int]:
    return json.loads((data_dir / "sign_to_prediction_index_map.json").read_text())


@lru_cache(maxsize=8)
def _hash_manifest(path_str: str, size: int, mtime: float) -> str | None:
    from modules.model.provenance import sha256_file

    return sha256_file(path_str)


def manifest_fingerprint(data_dir: Path | str, manifest: str = "train.csv") -> str | None:
    """Fingerprint of the split manifest a cache's rows came from.

    GISLR is a Kaggle *competition* download with no version number, so this
    hash is what actually distinguishes one copy of the dataset from another.
    Memoized on (path, size, mtime) — it is read on every cache-key computation.
    """
    path = Path(data_dir) / manifest
    if not path.is_file():
        return None
    st = path.stat()
    return _hash_manifest(str(path), st.st_size, st.st_mtime)


def cache_inputs(
    subset,
    coords: str,
    data_dir: Path | str,
    *,
    pipeline: str = PIPELINE,
    pipeline_version: int = PIPELINE_VERSION,
    nan_policy: str = NAN_POLICY,
    dataset: str = "gislr",
) -> dict:
    """Everything that determines a feature cache's bytes.

    ``indices_sha256`` is the point: it hashes the subset's actual index array,
    so a redefinition of ME_126 that keeps the name changes the key.
    """
    indices = np.asarray(subset.array, dtype=np.int64)
    return {
        "pipeline": pipeline,
        "pipeline_version": pipeline_version,
        "dataset": dataset,
        "subset": subset.name,
        "n_landmarks": int(indices.size),
        "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "coords": coords,
        "nan_policy": nan_policy,
        "rows_per_frame": ROWS_PER_FRAME,
        "split_strategy": "stratified 90/10",
        "split_seed": SEED,
        "manifest_sha256": manifest_fingerprint(data_dir),
    }


def feature_cache_key(subset, coords: str, data_dir: Path | str, **kw) -> str:
    """16 hex chars addressing one feature cache — the value recorded as
    `provenance.feature_cache_key` and the directory it lives in."""
    payload = json.dumps(
        cache_inputs(subset, coords, data_dir, **kw),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_dir(subset, coords: str, data_dir: Path | str, **kw) -> Path:
    """``data/cache/<dataset>/features/<pipeline>/<key>/`` for these inputs."""
    return (
        FEATURES_DIR
        / kw.get("pipeline", PIPELINE)
        / feature_cache_key(subset, coords, data_dir, **kw)
    )


def write_cache_sidecar(root: Path, inputs: dict, **extra) -> Path:
    """Write ``cache_key.json`` next to the arrays it describes.

    Without it a content-addressed directory is an opaque hash; with it, a
    cache whose key no config asks for any more can still say what it was.
    """
    path = root / CACHE_KEY_FILE
    payload = {"key": root.name, **inputs, **extra}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


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
    """Run tag: 'ME_126' -> 'me126' (+'-xy' when z is dropped).

    This is the *human* handle — registry pointer-file keys, progress-bar
    labels, run notes. It is deliberately no longer the feature-cache
    identifier: a name cannot notice that its subset's indices changed, which
    is what :func:`feature_cache_key` is for (TODO §9.2).
    """
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
    frame offsets under ``features/<pipeline>/<key>/``. Skip-if-exists; atomic.

    Skip-if-exists is only safe because the directory is content-addressed: a
    hit means these exact inputs were cached before, not merely that something
    with the same subset name was.

    ``progress``: optional callable(done, total) for single-bar reporting.
    """
    inputs = cache_inputs(subset, coords, data_dir)
    root = FEATURES_DIR / PIPELINE / feature_cache_key(subset, coords, data_dir)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / f"{prefix}_data.npy"
    off_path = root / f"{prefix}_offsets.npy"
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
    write_cache_sidecar(root, inputs)
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
