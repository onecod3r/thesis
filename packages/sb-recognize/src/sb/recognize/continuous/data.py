"""Training streams for continuous models, composed on the fly from isolated
``train.csv`` clips (TODO §12.3).

Same idea as :mod:`sb.recognize.sequences.compose` (the GISLR-Sentences test
set), but in **feature space** and randomized every epoch:

- **Clip bank** (:class:`ClipBank`): every ``train.csv`` clip as
  ``(T, feature_dim)`` float32 for one (subset, coords), **NaN kept**, so the
  synthesized frames follow the same NaN rules as the test set. NaN becomes 0
  only when a batch is formed, exactly where ``base_v1`` zeroes it.
  Content-addressed under ``features/clipbank_v1/<key>/``.
- **Epoch plan** (:func:`epoch_sequences`): a fresh random partition of the
  training clips into one-signer, random-order sequences of 1-6 signs, so
  every clip is seen once per epoch. Random order is deliberate: the model
  gets no language prior, so the ``sentence``/``control`` comparison on
  GISLR-Sentences stays meaningful.
- **Composer** (:func:`compose`): gaps of 0-15 interpolated frames between
  signs (0 = back-to-back), and rest at both ends of one of two kinds:

  - **lowered** (the realistic one): the pose wrists, hand points and elbows
    travel down to hip height, and each hand stays visible only while its
    wrist is still inside the frame (``hand_visible_max``), then goes NaN.
    Measured on the training frames (2026-09-23): hips are out of frame in
    97% of frames (median y 1.25), hand landmarks are never detected below
    y ~0.9, and each hand is present in only ~30% of frames even during
    signing -- a resting hand is below the frame, so MediaPipe extrapolates
    the pose and drops the hand.
  - **hands-absent**: the edge frame held with both hands NaN and the pose
    left at signing height -- GISLR-Sentences v1's rest, kept in the mix so
    the model also handles that test set.

Per-frame targets: the gloss label on sign frames and ``null_index`` on
every synthesized frame, plus a **boundary** target that is 1 on the frames
around each sign's end.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sb.core.schema import GROUPS
from sb.recognize.features import cache
from sb.recognize.features.gislr_stratified import load_npz

PIPELINE = "clipbank_v1"
PIPELINE_VERSION = 1
_COORD_INDEX = {"x": 0, "y": 1, "z": 2}
_POSE0 = next(g.offset for g in GROUPS if g.name == "pose")
_HANDS = [np.arange(g.offset, g.offset + g.size) for g in GROUPS if g.name in ("left_hand", "right_hand")]


# ---------------------------------------------------------------------------
# clip bank
# ---------------------------------------------------------------------------

class ClipBank:
    """Every clip of one split as NaN-preserving features: a flat
    ``(frames, feature_dim)`` float32 array + offsets, row ``i`` = row ``i``
    of the split dataframe."""

    def __init__(self, data: np.ndarray, offsets: np.ndarray):
        self.data, self.offsets = data, offsets

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def clip(self, i: int) -> np.ndarray:
        return self.data[self.offsets[i]:self.offsets[i + 1]]

    def length(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    @staticmethod
    def inputs(subset, coords: str, data_dir: Path) -> dict:
        return cache.cache_inputs(pipeline=PIPELINE, pipeline_version=PIPELINE_VERSION,
                                  nan_policy="keep", subset=subset, coords=coords, data_dir=data_dir)

    @classmethod
    def build(cls, df: pd.DataFrame, prefix: str, subset, coords: str, data_dir: Path,
              progress=None) -> "ClipBank":
        """Load from the content-addressed cache, or decode every npz once
        (atomic, skip-if-exists)."""
        inputs = cls.inputs(subset, coords, data_dir)
        root = cache.cache_dir(inputs)
        paths = {n: root / f"{prefix}_{n}.npy" for n in ("data", "offsets")}
        if all(p.is_file() for p in paths.values()):
            return cls(np.load(paths["data"]), np.load(paths["offsets"]))
        root.mkdir(parents=True, exist_ok=True)
        rows = subset.array
        t0 = time.time()
        chunks, offsets = [], [0]
        with ThreadPoolExecutor(12) as ex:
            it = ex.map(lambda p: load_npz(Path(data_dir) / p, rows, coords), df["npz_relpath"])
            for i, arr in enumerate(it):
                chunks.append(arr.reshape(len(arr), -1).astype(np.float32))
                offsets.append(offsets[-1] + len(arr))
                if progress is not None:
                    progress(i + 1, len(df))
        arrays = {"data": np.concatenate(chunks), "offsets": np.asarray(offsets, np.int64)}
        for n, arr in arrays.items():
            tmp = root / f"{prefix}_{n}.tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, paths[n])
        cache.write_sidecar(root, inputs)
        print(f"clip bank {prefix} [{root.name}]: {len(df)} clips, "
              f"{arrays['data'].nbytes / 1e9:.2f} GB ({time.time() - t0:.0f}s)")
        return cls(arrays["data"], arrays["offsets"])


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def holdout_glosses(signs, n: int, seed: int) -> list[str]:
    """``n`` glosses drawn with ``seed`` from the sorted vocabulary -- left out
    of training entirely for the open-vocabulary run (TODO §12.4)."""
    if n <= 0:
        return []
    vocab = sorted(set(signs))
    return sorted(np.random.default_rng(seed).choice(vocab, n, replace=False).tolist())


def train_val_indices(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Clip-level split of ``train.csv``, stratified on sign: the validation
    streams used for checkpoint selection. ``test.csv`` is never used for
    selection here."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    tr, va = train_test_split(idx, test_size=val_fraction, random_state=seed, stratify=df["sign"])
    return np.sort(tr), np.sort(va)


# ---------------------------------------------------------------------------
# feature layout (which columns are which landmarks)
# ---------------------------------------------------------------------------

@dataclass
class Layout:
    """Column indices of one (rows, coords) feature vector that the rest
    synthesis touches: the ``y`` columns it lowers (x is kept), and each
    hand's full column set, which it blanks when the hand leaves the frame."""

    n_features: int
    hand_cols: list[np.ndarray]    # per hand: every column of its landmarks (all coords)
    hand_y: list[np.ndarray]       # per hand: y columns of its landmarks
    hand_wrist_y: list[int | None]  # per hand: y column of its wrist (hand landmark 0)
    pose_side_y: list[np.ndarray]  # per side: pose wrist + hand-point y columns
    pose_wrist_y: list[int | None]
    elbow_y: list[int | None]
    hip_y: list[int]

    @classmethod
    def of(cls, rows: np.ndarray, coords: str) -> "Layout":
        if "y" not in coords:
            raise ValueError("rest synthesis needs the y coordinate")
        pos = {int(r): i for i, r in enumerate(rows)}
        yi = coords.index("y")
        nc = len(coords)

        def ycol(row: int) -> int | None:
            return pos[row] * nc + yi if row in pos else None

        def ycols(rs) -> np.ndarray:
            return np.array([c for r in rs if (c := ycol(int(r))) is not None], dtype=np.int64)

        def allcols(rs) -> np.ndarray:
            return np.array([pos[int(r)] * nc + k for r in rs if int(r) in pos for k in range(nc)],
                            dtype=np.int64)

        def pose(k: int) -> int:  # pose landmark k -> canonical row
            return _POSE0 + k

        return cls(
            n_features=len(rows) * nc,
            hand_cols=[allcols(h) for h in _HANDS],
            hand_y=[ycols(h) for h in _HANDS],
            hand_wrist_y=[ycol(int(h[0])) for h in _HANDS],
            pose_side_y=[ycols([pose(k) for k in (15, 17, 19, 21)]), ycols([pose(k) for k in (16, 18, 20, 22)])],
            pose_wrist_y=[ycol(pose(15)), ycol(pose(16))],
            elbow_y=[ycol(pose(13)), ycol(pose(14))],
            hip_y=[c for c in (ycol(pose(23)), ycol(pose(24))) if c is not None],
        )


# ---------------------------------------------------------------------------
# composer
# ---------------------------------------------------------------------------

def _lowered(frame: np.ndarray, lay: Layout, default_drop: float) -> np.ndarray:
    """``frame`` with both hands (and the pose wrists/hand points, elbows
    half-way) moved down to hip height. NaN stays NaN."""
    out = frame.copy()
    hips = frame[lay.hip_y] if lay.hip_y else np.array([])
    hip = float(np.nanmean(hips)) if hips.size and np.isfinite(hips).any() else None

    def drop(y_ref: int | None) -> float:
        y = frame[y_ref] if y_ref is not None else np.nan
        return max(hip - y, 0.05) if hip is not None and np.isfinite(y) else default_drop

    for cols, wrist in zip(lay.hand_y, lay.hand_wrist_y):
        if cols.size:
            out[cols] += drop(wrist)
    for cols, wrist, elbow in zip(lay.pose_side_y, lay.pose_wrist_y, lay.elbow_y):
        d = drop(wrist)
        if cols.size:
            out[cols] += d
        if elbow is not None:
            out[elbow] += d / 2
    return out


def _rest(edge: np.ndarray, n: int, lowered: bool, entering: bool, lay: Layout,
          cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """``n`` rest frames next to ``edge``. Hands-absent: edge held, hands NaN.
    Lowered: the arms travel between ``edge`` and the lowered pose over the
    first/last ``rest_ramp_frames`` (toward the sign when ``entering``), and
    each hand is NaN on every frame where its wrist is below
    ``hand_visible_max`` (out of frame)."""
    if lowered:
        low = _lowered(edge, lay, cfg["rest_default_drop"])
        k = min(cfg["rest_ramp_frames"], n)
        t = np.clip(np.arange(1, n + 1) / (k + 1), 0, 1)[:, None]  # 0 -> 1 over the ramp
        frames = edge[None] + t * (low - edge)[None]  # leaving: edge -> low
        if entering:
            frames = frames[::-1]  # arriving: low -> edge
        for cols, wrist in zip(lay.hand_cols, lay.hand_wrist_y):
            if wrist is not None:
                gone = ~(frames[:, wrist] <= cfg["hand_visible_max"])  # below the frame, or already NaN
                frames[np.ix_(gone, cols)] = np.nan
    else:
        frames = np.repeat(edge[None], n, axis=0)
        for cols in lay.hand_cols:
            frames[:, cols] = np.nan
    frames = frames + rng.normal(0.0, cfg["rest_jitter"], size=frames.shape).astype(np.float32)
    return frames.astype(np.float32)


def compose(clips: list[np.ndarray], labels: list[int], null_index: int, lay: Layout,
            cfg: dict, rng: np.random.Generator) -> dict:
    """One training stream from ``clips`` (in order): ``x`` (T, F) float32 with
    NaN, ``y`` (T,) gloss or ``null_index``, ``b`` (T,) boundary target,
    ``segments`` (n, 2) end-exclusive."""
    g0, g1 = cfg["gap_frames"]
    r0, r1 = cfg["rest_frames"]
    parts, ys = [], []

    def add(arr, label):
        parts.append(arr)
        ys.append(np.full(len(arr), label, np.int64))

    lowered = rng.random() < cfg["p_lowered_rest"]
    add(_rest(clips[0][0], int(rng.integers(r0, r1 + 1)), lowered, True, lay, cfg, rng), null_index)
    segments = []
    t = len(parts[0])
    for i, (clip, lab) in enumerate(zip(clips, labels)):
        if i > 0:
            g = int(rng.integers(g0, g1 + 1))
            if g:
                a, b = clips[i - 1][-1], clip[0]
                w = (np.arange(1, g + 1, dtype=np.float32) / (g + 1))[:, None]
                add(a[None] + w * (b - a)[None], null_index)
                t += g
        segments.append((t, t + len(clip)))
        add(clip, lab)
        t += len(clip)
    add(_rest(clips[-1][-1], int(rng.integers(r0, r1 + 1)), lowered, False, lay, cfg, rng), null_index)
    y = np.concatenate(ys)
    b = np.zeros(len(y), np.float32)
    for s, e in segments:
        b[max(s, e - cfg["boundary_before"]):min(len(y), e + cfg["boundary_after"])] = 1.0
    return {"x": np.concatenate(parts).astype(np.float32), "y": y, "b": b,
            "segments": np.asarray(segments, np.int64)}


def epoch_sequences(clip_idx: np.ndarray, participants: np.ndarray, lengths: np.ndarray,
                    cfg: dict, rng: np.random.Generator) -> list[list[int]]:
    """A random partition of ``clip_idx`` into one-signer, random-order
    sequences of ``signs_per_seq`` signs, each at most ``max_frames`` of
    signing. Every clip appears exactly once."""
    lo, hi = cfg["signs_per_seq"]
    out: list[list[int]] = []
    for p in np.unique(participants[clip_idx]):
        mine = rng.permutation(clip_idx[participants[clip_idx] == p])
        i = 0
        while i < len(mine):
            n = int(rng.integers(lo, hi + 1))
            seq, frames = [], 0
            while i < len(mine) and len(seq) < n and (not seq or frames + lengths[mine[i]] <= cfg["max_frames"]):
                seq.append(int(mine[i]))
                frames += lengths[mine[i]]
                i += 1
            out.append(seq)
    order = rng.permutation(len(out))
    return [out[k] for k in order]


def batches(seqs: list[list[int]], lengths: np.ndarray, batch_size: int,
            rng: np.random.Generator) -> list[list[list[int]]]:
    """Group sequences of similar signing length into batches (less padding),
    then shuffle the batch order."""
    order = sorted(range(len(seqs)), key=lambda k: sum(lengths[c] for c in seqs[k]))
    groups = [[seqs[k] for k in order[i:i + batch_size]] for i in range(0, len(order), batch_size)]
    return [groups[k] for k in rng.permutation(len(groups))]


def collate(streams: list[dict]) -> dict:
    """Pad a list of composed streams into tensors. NaN -> 0 here (the
    ``base_v1`` policy, applied after synthesis). Padding is masked by
    ``mask``; ``y`` padding is -100 (ignored by cross-entropy)."""
    T = max(len(s["y"]) for s in streams)
    F = streams[0]["x"].shape[1]
    B = len(streams)
    x = np.zeros((B, T, F), np.float32)
    y = np.full((B, T), -100, np.int64)
    b = np.zeros((B, T), np.float32)
    mask = np.zeros((B, T), bool)
    for i, s in enumerate(streams):
        n = len(s["y"])
        x[i, :n] = np.nan_to_num(s["x"], nan=0.0, posinf=0.0, neginf=0.0)
        y[i, :n] = s["y"]
        b[i, :n] = s["b"]
        mask[i, :n] = True
    return {"x": torch.from_numpy(x), "y": torch.from_numpy(y), "b": torch.from_numpy(b),
            "mask": torch.from_numpy(mask),
            "lengths": torch.tensor([len(s["y"]) for s in streams]),
            "segments": [s["segments"] for s in streams],
            "labels": [s["y"][s["segments"][:, 0]] for s in streams]}
