"""``landmark_interp_v1`` — the engineered-feature pipeline for the landmark-
importance notebook (``gislr.1.models.landmark-importance.ipynb``, TODO §3).

Unlike ``base_v1`` (raw xy/xyz, subset-selected), this pipeline keeps **all
543** landmarks and, per frame, replaces raw coordinates with:

- a **per-landmark channel block** (10 channels x 543 landmarks = 5430):
  translation-invariant position (anchored at mid-shoulder), scale-invariant
  (divided by inter-shoulder distance), first-derivative velocity,
  second-derivative acceleration, and speed magnitude — see
  :func:`build_frame_features`.
- a small **relational block** (12 scalars): inter-hand fingertip/wrist
  distances and hand-to-face-anchor distances. These describe a relationship
  between landmarks, not one landmark, so they sit outside the per-landmark
  block the attention gate (``sb.recognize.interp.models.LandmarkAttention``)
  operates on.

The two blocks are concatenated into one flat ``(T, 5442)`` array per frame so
the existing generic cache/dataset machinery
(``sb.recognize.features.base_v1.SubsetArrayDataset`` / ``collate_fn``) can be
reused unchanged; ``sb.recognize.interp.models`` un-concatenates them by a
fixed split point (:data:`PER_LANDMARK_DIM`).
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from sb.core.subsets import EYES_NOSE_36, LIPS_40, SUBSETS, pose_rows
from sb.recognize.features import cache
from sb.recognize.features.gislr_stratified import load_npz

PIPELINE = "landmark_interp_v1"
PIPELINE_VERSION = 1
NAN_POLICY = "zero"  # NaN enters at the normalization step (missing anchor/landmark), zeroed after

FULL_543 = SUBSETS["FULL_543"]
CHANNELS_PER_LANDMARK = 10  # x,y,z (normalized) | vx,vy,vz | ax,ay,az | speed
PER_LANDMARK_DIM = len(FULL_543) * CHANNELS_PER_LANDMARK  # 5430

# ---- anchors used by the relational block --------------------------------
# Holistic rows: pose shoulders (11, 12) for centering/scale; hand landmark
# indices are MediaPipe Hand order (0=wrist, 4/8/12/16/20=thumb..pinky tips),
# offset into the left_hand (468) / right_hand (522) holistic blocks.
_LEFT_SHOULDER, _RIGHT_SHOULDER = pose_rows([11, 12])
_HAND_POINTS = {"wrist": 0, "thumb_tip": 4, "index_tip": 8, "middle_tip": 12,
                "ring_tip": 16, "pinky_tip": 20}
LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET = 468, 522
CHIN_ROW = 152  # MediaPipe face-mesh chin landmark; face rows == holistic rows 0-467
EPS = 1e-6

RELATIONAL_NAMES: list[str] = (
    [f"inter_hand_{name}" for name in _HAND_POINTS]  # 6
    + [f"{hand}_index_tip_to_{anchor}" for hand in ("left", "right")
       for anchor in ("lips", "chin", "nose")]  # 6
)
RELATIONAL_DIM = len(RELATIONAL_NAMES)  # 12
FEATURE_DIM = PER_LANDMARK_DIM + RELATIONAL_DIM  # 5442


def _center_and_scale(raw: np.ndarray) -> np.ndarray:
    """(T, 543, 3) raw xyz -> (T, 543, 3) mid-shoulder-centered, inter-shoulder
    -scaled, NaN -> 0. NaN from an undetected anchor/shoulder propagates through
    the arithmetic and is zeroed at the end, same policy as ``base_v1``."""
    anchor = np.nanmean(raw[:, [_LEFT_SHOULDER, _RIGHT_SHOULDER], :], axis=1)  # (T, 3)
    local = raw - anchor[:, None, :]
    shoulder_dist = np.linalg.norm(
        raw[:, _LEFT_SHOULDER, :] - raw[:, _RIGHT_SHOULDER, :], axis=-1
    )  # (T,)
    valid = np.isfinite(shoulder_dist) & (shoulder_dist > EPS)
    fallback = np.nanmedian(shoulder_dist[valid]) if valid.any() else 1.0
    if not np.isfinite(fallback) or fallback <= EPS:
        fallback = 1.0
    scale = np.where(valid, shoulder_dist, fallback)
    normed = local / scale[:, None, None]
    return np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)


def _derivatives(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """position (T,543,3) -> (velocity, acceleration, speed); t=0 has zero
    velocity/acceleration by construction (no t=-1 frame to difference against)."""
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    acc = np.zeros_like(pos)
    acc[1:] = vel[1:] - vel[:-1]
    speed = np.linalg.norm(vel, axis=-1)  # (T, 543)
    return vel, acc, speed


def _relational_block(pos: np.ndarray) -> np.ndarray:
    """(T, 543, 3) normalized positions -> (T, RELATIONAL_DIM) distances."""
    T = pos.shape[0]
    dists = np.zeros((T, RELATIONAL_DIM), dtype=np.float32)
    col = 0
    for offset in _HAND_POINTS.values():
        lp = pos[:, LEFT_HAND_OFFSET + offset, :]
        rp = pos[:, RIGHT_HAND_OFFSET + offset, :]
        dists[:, col] = np.linalg.norm(lp - rp, axis=-1)
        col += 1
    lips_center = pos[:, LIPS_40, :].mean(axis=1)
    nose_center = pos[:, EYES_NOSE_36, :].mean(axis=1)  # eyes/nose anchor block
    chin = pos[:, CHIN_ROW, :]
    for hand_offset in (LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET):
        tip = pos[:, hand_offset + _HAND_POINTS["index_tip"], :]
        for anchor in (lips_center, chin, nose_center):
            dists[:, col] = np.linalg.norm(tip - anchor, axis=-1)
            col += 1
    return np.nan_to_num(dists, nan=0.0, posinf=0.0, neginf=0.0)


def build_frame_features(raw: np.ndarray) -> np.ndarray:
    """(T, 543, 3) raw xyz -> (T, FEATURE_DIM) flat engineered features.

    ``raw`` must already be in canonical holistic row order (what
    ``gislr_stratified.load_npz`` returns for ``rows=FULL_543.array``).
    """
    pos = _center_and_scale(raw)
    vel, acc, speed = _derivatives(pos)
    per_landmark = np.concatenate(
        [pos, vel, acc, speed[..., None]], axis=-1
    ).astype(np.float32)  # (T, 543, 10)
    relational = _relational_block(pos)  # (T, 12)
    T = raw.shape[0]
    return np.concatenate(
        [per_landmark.reshape(T, PER_LANDMARK_DIM), relational], axis=-1
    ).astype(np.float32)


def load_video(path: Path) -> np.ndarray:
    """One GISLR_Stratified npz -> (T, FEATURE_DIM) engineered features."""
    raw = load_npz(path, FULL_543.array, coords="xyz")  # (T, 543, 3), NaN preserved
    return build_frame_features(raw)


def cache_inputs(data_dir: Path | str, dataset: str = "gislr") -> dict:
    return cache.cache_inputs(
        pipeline=PIPELINE,
        pipeline_version=PIPELINE_VERSION,
        nan_policy=NAN_POLICY,
        subset=FULL_543,
        coords="xyz",
        data_dir=data_dir,
        dataset=dataset,
    )


def cache_key(data_dir: Path | str, dataset: str = "gislr") -> str:
    return cache.cache_key(cache_inputs(data_dir, dataset))


def cache_dir(data_dir: Path | str, dataset: str = "gislr") -> Path:
    return cache.cache_dir(cache_inputs(data_dir, dataset))


def build_cache(
    df: pd.DataFrame, prefix: str, data_dir: Path, progress=None, dataset: str = "gislr"
) -> tuple[Path, Path]:
    """Decode + engineer-feature every video of one split once into
    ``features/landmark_interp_v1/<key>/``. Skip-if-exists; atomic — same
    pattern as ``base_v1.build_cache``.
    """
    inputs = cache_inputs(data_dir, dataset)
    root = cache.cache_dir(inputs)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / f"{prefix}_data.npy"
    off_path = root / f"{prefix}_offsets.npy"
    if data_path.exists() and off_path.exists():
        return data_path, off_path

    t0 = time.time()
    paths = [data_dir / p for p in df["npz_relpath"]]
    chunks, offsets = [], [0]
    with ThreadPoolExecutor(8) as ex:  # heavier per-video compute than base_v1 -> fewer workers
        for i, arr in enumerate(ex.map(load_video, paths)):
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
        f"{prefix}/landmark_interp_v1 [{root.name}]: cached {len(df)} videos, "
        f"{flat.nbytes / 1e9:.2f} GB ({time.time() - t0:.0f}s)"
    )
    return data_path, off_path
