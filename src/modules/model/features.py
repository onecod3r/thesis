"""1st-place feature pipeline: normalization + lag differences + augmentation.

This is the *input side* of the Kaggle GISLR 1st-place solution (hoyso48),
ported from ``gislr.0.competition.entry.1st.ipynb`` (recovered from git history
at ``fd1c7aa``, TODO §4.2). It is deliberately separate from
``modules.model.data`` because it is incompatible with it in three ways, none
of them cosmetic:

1. **NaN must survive to training time.** ``data.load_video_subset`` calls
   ``nan_to_num`` at cache-build time, but the 1st-place normalization is
   NaN-aware (mean/std over *detected* landmarks only) and two of its
   augmentations *write* NaN as a masking value. A cache with zeros where the
   NaNs were cannot reproduce it, so this module keeps its own cache
   (``*_nan_data.npy``), built once per (split, subset).
2. **No uniform subsample.** ``SubsetArrayDataset`` resamples every clip to
   ``MAX_SEQ_LEN=128``; the 1st place keeps the native frame rate and
   random-crops to ``MAX_LEN=384``, which is what makes its temporal
   augmentations meaningful.
3. **Features are 6 channels per landmark**, not 2: normalized xy, the lag-1
   difference and the lag-2 difference, concatenated.

**Streaming caveat (read before reusing this for a deployment model).** Two
parts of the reference pipeline are NOT causal:

- the normalization statistics are whole-sequence (mean of the reference
  landmark over every frame, std over every frame and landmark), and
- the lag differences are *forward* (``dx[t] = x[t+1] - x[t]``), so frame t
  reads frames t+1 and t+2.

The forward differences are a 2-frame lookahead and flip to causal with
``diff_mode="backward"`` (a knob here). The normalization does not: a streaming
version needs a running estimator, which is TODO §7.2's job. This module
reproduces the reference faithfully by default — see §0 of
``gislr.1.models.firstplace.ipynb`` for why that is the right starting point.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from modules.model import data as D
from modules.model.data import FEATURES_DIR, ROWS_PER_FRAME, subset_tag

# Feature-cache identity for THIS pipeline (TODO §9.2). Bump the version when a
# change here alters the cached bytes; the key moves and nothing stale is reused.
PIPELINE = "firstplace_v1"
PIPELINE_VERSION = 1
NAN_POLICY = "preserve"  # the whole reason this pipeline has its own cache

MAX_LEN = 384  # 1st-place CFG.max_len — a temporal CROP bound, not a subsample
REF_LANDMARK = 17  # raw holistic row used as the translation reference (a lip
# point). Verbatim from the reference ``Preprocess``; note it is NOT the
# shoulder-centre TODO §7.2 assumes — see the notebook's §0 discussion.
CHANNELS_PER_LANDMARK = 6  # xy + lag-1 dxy + lag-2 dxy


# ============================================================
# Left/right mirror map (for the flip_lr augmentation)
# ============================================================
# Raw holistic index pairs, verbatim from the reference notebook. Anything not
# named here is midline (lip centre, nose bridge) and mirrors to itself.
_LHAND = list(range(468, 489))
_RHAND = list(range(522, 543))
_LLIP = [84, 181, 91, 146, 61, 185, 40, 39, 37, 87, 178, 88, 95, 78, 191, 80, 81, 82]
_RLIP = [314, 405, 321, 375, 291, 409, 270, 269, 267, 317, 402, 318, 324, 308, 415,
         310, 311, 312]
_LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
_REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
_LPOSE = [513, 505, 503, 501]
_RPOSE = [512, 504, 502, 500]
_LNOSE = [98]
_RNOSE = [327]

MIRROR_PAIRS: dict[int, int] = {}
for _left, _right in ((_LHAND, _RHAND), (_LLIP, _RLIP), (_LEYE, _REYE),
                      (_LPOSE, _RPOSE), (_LNOSE, _RNOSE)):
    for _a, _b in zip(_left, _right):
        MIRROR_PAIRS[_a] = _b
        MIRROR_PAIRS[_b] = _a


def mirror_permutation(rows: np.ndarray) -> np.ndarray:
    """Permutation p such that ``x[:, p]`` is the left/right-swapped subset.

    Asserts the subset is *closed* under mirroring — if a landmark's mirror sits
    outside the subset the flip would silently drop it, corrupting the
    augmentation rather than failing. (FP_118 and ME_126 are both closed.)
    """
    pos = {int(r): i for i, r in enumerate(rows)}
    perm = np.arange(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        m = MIRROR_PAIRS.get(int(r))
        if m is None:
            continue  # midline landmark — mirrors to itself
        assert m in pos, (
            f"landmark {r} mirrors to {m}, which is not in this subset — "
            "flip_lr would corrupt it; use a mirror-closed subset")
        perm[i] = pos[m]
    return perm


# ============================================================
# NaN-preserving feature cache
# ============================================================

def load_video_raw(path: Path, rows: np.ndarray, coords: str = "xy") -> np.ndarray:
    """One parquet -> (T, len(rows), len(coords)) float32 **with NaN intact**.

    The NaNs are the whole point (see module docstring); that is the only
    difference from ``data.load_video_subset``.
    """
    cols = list(coords)
    table = pq.read_table(path, columns=cols)
    data = np.column_stack([table.column(c).to_numpy() for c in cols])
    n = data.shape[0] // ROWS_PER_FRAME
    return data.reshape(n, ROWS_PER_FRAME, len(cols))[:, rows, :].astype(np.float32)


def build_nan_cache(df, prefix: str, subset, coords: str, data_dir: Path,
                    progress=None, dataset: str = "gislr") -> tuple[Path, Path]:
    """Flat float32 array + frame offsets, NaNs preserved. Skip-if-exists,
    atomic writes — same contract as ``data.build_subset_cache``.

    The two pipelines can no longer collide by construction rather than by
    filename convention: this one addresses ``features/firstplace_v1/<key>/``
    and its key carries ``nan_policy="preserve"`` (TODO §9.2).
    """
    kw: dict[str, Any] = dict(pipeline=PIPELINE, pipeline_version=PIPELINE_VERSION,
                              nan_policy=NAN_POLICY, dataset=dataset)
    inputs = D.cache_inputs(subset, coords, data_dir, **kw)
    root = D.cache_dir(subset, coords, data_dir, **kw)
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
            ex.map(lambda p: load_video_raw(p, rows, coords), paths)
        ):
            chunks.append(arr.reshape(-1))
            offsets.append(offsets[-1] + arr.shape[0])
            if progress is not None and i % 500 == 0:
                progress(i, len(paths))
    flat = np.concatenate(chunks)
    for target, payload in ((data_path, flat),
                            (off_path, np.asarray(offsets, dtype=np.int64))):
        tmp = target.with_suffix(".tmp.npy")
        np.save(tmp, payload)
        os.replace(tmp, target)
    D.write_cache_sidecar(root, inputs)
    print(f"{prefix}/{subset_tag(subset.name, coords)} [{root.name}] "
          f"(nan-preserving): cached {len(df)} videos, "
          f"{flat.nbytes / 1e9:.2f} GB ({time.time() - t0:.0f}s)")
    return data_path, off_path


# ============================================================
# Augmentation — operates on raw (T, P, 2) with NaN, before preprocessing
# ============================================================

def _interp_time(x: np.ndarray, new_t: int) -> np.ndarray:
    """Linear resample along time. NaN propagates into any window it touches,
    matching the reference ``tf.image.resize`` behaviour."""
    t = x.shape[0]
    if new_t == t or t < 2:
        return x
    src = np.linspace(0.0, t - 1, new_t)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo).astype(np.float32)[:, None, None]
    return x[lo] * (1.0 - w) + x[hi] * w


def resample(x, rng, rate=(0.5, 1.5)):
    """Speed the signing up or down — the reference's strongest augmentation."""
    r = rng.uniform(*rate)
    return _interp_time(x, max(1, int(round(r * x.shape[0]))))


def flip_lr(x, perm):
    """Mirror the signer: x -> 1-x (GISLR coordinates are normalized to [0,1])
    *and* swap every left/right landmark pair."""
    x = x.copy()
    x[..., 0] = 1.0 - x[..., 0]
    return x[:, perm]


def spatial_random_affine(x, rng, scale=(0.8, 1.2), shear=(-0.15, 0.15),
                          shift=(-0.1, 0.1), degree=(-30, 30)):
    """Scale -> shear -> rotate -> shift, in the reference's order (the order
    matters: the shear displaces the rotation centre)."""
    center = np.array([0.5, 0.5], dtype=np.float32)
    if scale is not None:
        x = rng.uniform(*scale) * x
    if shear is not None:
        shear_x = shear_y = rng.uniform(*shear)
        if rng.random() < 0.5:
            shear_x = 0.0
        else:
            shear_y = 0.0
        x = x @ np.array([[1.0, shear_x], [shear_y, 1.0]], dtype=np.float32)
        center = center + np.array([shear_y, shear_x], dtype=np.float32)
    if degree is not None:
        rad = np.deg2rad(rng.uniform(*degree))
        c, s = np.cos(rad), np.sin(rad)
        rot = np.array([[c, s], [-s, c]], dtype=np.float32)
        x = (x - center) @ rot + center
    if shift is not None:
        x = x + rng.uniform(*shift)
    return x.astype(np.float32)


def temporal_crop(x, rng, length=MAX_LEN):
    t = x.shape[0]
    high = int(np.clip(t - length, 1, length))
    return x[int(rng.integers(0, high)):][:length]


def temporal_mask(x, rng, size=(0.2, 0.4)):
    """Blank a contiguous span of frames to NaN — simulates a detection dropout."""
    t = x.shape[0]
    n = int(t * rng.uniform(*size))
    if n < 1:
        return x
    off = int(rng.integers(0, int(np.clip(t - n, 1, t))))
    x = x.copy()
    x[off:off + n] = np.nan
    return x


def spatial_mask(x, rng, size=(0.2, 0.4)):
    """Blank a random axis-aligned xy box to NaN — simulates occlusion."""
    off_x, off_y = rng.random(), rng.random()
    s = rng.uniform(*size)
    m = ((off_x < x[..., 0]) & (x[..., 0] < off_x + s)
         & (off_y < x[..., 1]) & (x[..., 1] < off_y + s))
    x = x.copy()
    x[m] = np.nan
    return x


def augment(x, rng, perm, max_len=MAX_LEN):
    """The reference ``augment_fn``: same operations, probabilities and order.
    Applied to raw coordinates *before* normalization."""
    if rng.random() < 0.8:
        x = resample(x, rng)
    if rng.random() < 0.5:
        x = flip_lr(x, perm)
    x = temporal_crop(x, rng, max_len)
    if rng.random() < 0.75:
        x = spatial_random_affine(x, rng)
    if rng.random() < 0.5:
        x = temporal_mask(x, rng)
    if rng.random() < 0.5:
        x = spatial_mask(x, rng)
    return x


# ============================================================
# Preprocess — normalization + lag differences
# ============================================================

def drop_empty_frames(x: np.ndarray) -> np.ndarray:
    """Reference ``filter_nans_tf``: drop frames where *every* kept landmark is
    undetected. Keeps at least one frame so downstream shapes stay valid."""
    keep = ~np.isnan(x).all(axis=(1, 2))
    return x[keep] if keep.any() else x[:1]


def preprocess(x: np.ndarray, ref_idx: int, max_len: int = MAX_LEN,
               diff_mode: str = "forward") -> np.ndarray:
    """(T, P, 2) raw+NaN -> (T', 6P) float32 model input.

    Reference-point translation (mean position of ``REF_LANDMARK`` over the
    clip), per-channel scale normalization, then position + lag-1 + lag-2
    differences.

    ``diff_mode``:
      - ``"forward"``  — reference behaviour, ``dx[t] = x[t+1] - x[t]`` (reads
        the future; harmless for this offline model, a 2-frame lookahead
        otherwise).
      - ``"backward"`` — causal, ``dx[t] = x[t] - x[t-1]``. Streaming-safe.
    """
    assert diff_mode in ("forward", "backward"), diff_mode

    # NaN-aware mean/std, written out rather than via np.nanmean: an all-NaN
    # reference landmark is a NORMAL case here (the signer's lip point is never
    # detected in that clip) and np.nanmean answers it with a RuntimeWarning
    # per call — 85k of them per epoch. Doing the masked sums directly is both
    # quiet and slightly cheaper.
    ref = x[:, ref_idx:ref_idx + 1, :]
    seen = ~np.isnan(ref)
    n_seen = seen.sum(axis=(0, 1), keepdims=True)
    # translation reference: where this signer's reference landmark sits,
    # averaged over the clip (0.5 = frame centre when it is never detected)
    mean = np.where(n_seen > 0,
                    np.where(seen, ref, 0.0).sum(axis=(0, 1), keepdims=True)
                    / np.maximum(n_seen, 1),
                    0.5).astype(np.float32)
    # scale reference: per-channel spread about that centre over the whole clip
    # — this is what removes signer size / distance to camera
    sq = (x - mean) ** 2
    ok = ~np.isnan(sq)
    n_ok = ok.sum(axis=(0, 1), keepdims=True)
    std = np.sqrt(np.where(ok, sq, 0.0).sum(axis=(0, 1), keepdims=True)
                  / np.maximum(n_ok, 1))
    std = np.where(~np.isfinite(std) | (std < 1e-6), 1.0, std).astype(np.float32)
    x = ((x - mean) / std).astype(np.float32)

    x = x[:max_len]
    t, p, _ = x.shape
    zero = np.zeros_like(x[:1])
    if t <= 2:
        dx = dx2 = np.zeros_like(x)
    elif diff_mode == "forward":
        dx = np.concatenate([x[1:] - x[:-1], zero], 0)
        dx2 = np.concatenate([x[2:] - x[:-2], zero, zero], 0)
    else:
        dx = np.concatenate([zero, x[1:] - x[:-1]], 0)
        dx2 = np.concatenate([zero, zero, x[2:] - x[:-2]], 0)

    out = np.concatenate([x.reshape(t, 2 * p), dx.reshape(t, 2 * p),
                          dx2.reshape(t, 2 * p)], axis=-1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# Dataset + collate
# ============================================================

class FirstPlaceDataset(Dataset):
    """In-RAM NaN-preserving cache -> augment -> preprocess, per sample.

    Augmentation is per-``__getitem__`` and seeded from ``(seed, epoch, i)``, so
    an epoch is reproducible and a resumed run does not replay the exact
    augmentations of the epoch it resumes into. ``num_workers=0`` for the same
    Windows-spawn reason as ``data.SubsetArrayDataset``; the augmentation is
    small-array numpy and does not bottleneck the GPU.
    """

    def __init__(self, df, data_path, off_path, subset, *, augment_data: bool,
                 max_len: int = MAX_LEN, diff_mode: str = "forward",
                 seed: int = 42, mmap: bool = False):
        self.labels = df["label"].to_numpy()
        # The ~3 GB cache is opened LAZILY, per process, and deliberately not
        # held as an attribute at construction time: with num_workers > 0,
        # Windows spawn pickles the dataset to each worker, and a 3 GB array in
        # the pickle overruns the pipe ("OSError: [Errno 22] Invalid argument",
        # then a truncated-pickle error in the child). Storing only the path
        # keeps the pickle tiny; each worker mmaps the file itself and they
        # share one page cache.
        self._data_path = Path(data_path)
        self._data: np.ndarray | None = None
        self.mmap = mmap
        self.offsets = np.load(off_path)
        self.n_landmarks = len(subset)
        self.perm = mirror_permutation(subset.array)
        self.ref_idx = int(np.searchsorted(subset.array, REF_LANDMARK))
        assert subset.array[self.ref_idx] == REF_LANDMARK, (
            f"reference landmark {REF_LANDMARK} is not in subset {subset.name} — "
            "the 1st-place normalization cannot be reproduced on it")
        self.augment_data = augment_data
        self.max_len = max_len
        self.diff_mode = diff_mode
        self.seed = seed
        self.epoch = 0
        assert len(self.labels) == len(self.offsets) - 1, "cache/split mismatch"

    @property
    def data(self) -> np.ndarray:
        """The feature cache, opened on first use in whichever process asks.

        ``mmap`` for the worker path (shared page cache, no per-worker copy);
        a plain read in-process, which is faster once the pages are warm.
        """
        if self._data is None:
            self._data = np.load(self._data_path,
                                 mmap_mode="r" if self.mmap else None)
        return self._data

    def set_epoch(self, epoch: int) -> None:
        """Re-seed augmentation for the coming epoch (call once per epoch)."""
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        d = self.n_landmarks * 2
        flat = self.data[self.offsets[i] * d: self.offsets[i + 1] * d]
        arr = flat.reshape(-1, self.n_landmarks, 2)
        arr = drop_empty_frames(arr)
        if self.augment_data:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            arr = augment(arr, rng, self.perm, self.max_len)
        else:
            arr = arr[:self.max_len]
        feats = preprocess(arr, self.ref_idx, self.max_len, self.diff_mode)
        tensor = torch.from_numpy(np.ascontiguousarray(feats))
        return tensor, feats.shape[0], int(self.labels[i])


class LengthBucketedBatchSampler(Sampler[list[int]]):
    """Batch clips of similar length together.

    **Why this is not optional.** GISLR clip lengths are extremely skewed —
    median 22 frames, mean 38, p99 219, max 537. Padding to the batch max
    therefore does almost nothing at batch 128: the max of 128 draws lands deep
    in the tail, so the observed median padded length was **277**, i.e. ~6.4x
    more compute than the frames actually carry. Measured cost of a training
    epoch is dominated by that padding, and attention makes it worse than
    linear (O(T^2) in the padded length).

    Batching by length instead brings the padded total to within ~1.2x of the
    frames themselves — a measured **5.9x** at batch 128 and **6.9x** at 256.

    Randomness is preserved in two ways, because sorting the whole epoch by
    length would be a serious training bug (the LR schedule and the BatchNorm
    statistics would both see a systematic length ramp):

    1. indices are shuffled, then sorted only *within* a pool of
       ``pool_batches`` batches — so a clip's batch-mates vary every epoch;
    2. the resulting batches are shuffled again, so length is uncorrelated with
       position in the epoch.

    Lengths are the cached (pre-augmentation) ones. ``resample`` then scales a
    clip by 0.5-1.5x, which blurs the buckets but does not undo them — the
    ordering is approximately preserved, and the 5.9x above is measured against
    the cached lengths.
    """

    def __init__(self, lengths, batch_size: int, pool_batches: int = 50,
                 shuffle: bool = True, drop_last: bool = True, seed: int = 42):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.pool = batch_size * pool_batches
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self._n = len(self._build(0))

    def set_epoch(self, epoch: int) -> None:
        """Re-shuffle for the coming epoch (call alongside ``dataset.set_epoch``)."""
        self.epoch = int(epoch)

    def _build(self, epoch: int) -> list[np.ndarray]:
        n = len(self.lengths)
        rng = np.random.default_rng((self.seed, epoch))
        order = rng.permutation(n) if self.shuffle else np.arange(n)
        batches: list[np.ndarray] = []
        for start in range(0, n, self.pool):
            chunk = order[start:start + self.pool]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for b in range(0, len(chunk), self.batch_size):
                batch = chunk[b:b + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)  # length must not correlate with epoch position
        return batches

    def __iter__(self):
        for batch in self._build(self.epoch):
            yield batch.tolist()

    def __len__(self):
        return self._n


def cached_lengths(off_path) -> np.ndarray:
    """Per-clip frame counts straight from a cache's offset array — what
    :class:`LengthBucketedBatchSampler` buckets on, without touching the data."""
    return np.diff(np.load(off_path))


def collate_fn(batch):
    """Pad to the *batch* max, not to MAX_LEN.

    Only actually cheap when paired with :class:`LengthBucketedBatchSampler` —
    with randomly composed batches the max sits in the tail of a very skewed
    length distribution and this saves almost nothing (see that class).
    Numerically identical to fixed-length padding either way, because every op
    in ``Conv1DTransformer`` is mask-aware.

    Returns ``(padded, lengths, labels)`` — the same contract as
    ``data.collate_fn``, so the model, the training driver and
    ``modules/scripts/evaluate.py`` all keep one calling convention. Unlike
    ``data.collate_fn`` there is no descending-length sort, because nothing here
    uses packed sequences.
    """
    feats, lengths, labels = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    t_max = int(lengths.max())
    padded = torch.zeros(len(feats), t_max, feats[0].shape[1])
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = f
    return padded, lengths, labels
