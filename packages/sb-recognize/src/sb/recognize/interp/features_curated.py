"""``landmark_curated_v1`` — the highest-contributing-feature pipeline for
the curated-features training notebook
(``gislr.1.models.curated-features.ipynb``, TODO §3.5).

Puts together the three winning conclusions from the prior interpretability
experiments into one per-frame feature vector, rather than the full-543/xyz
input ``landmark_interp_v1`` uses:

1. **ME-126 landmark subset** (`sb.core.subsets.SUBSETS["ME_126"]`) — the
   subset motion-energy, subset-comparison and landmark-importance all
   independently converged on (lips + hands + eyes/nose + upper-body pose).
2. **xy only, z dropped** — TODO §3.1's established default; z carries
   mostly noise for pose/most landmarks.
3. **28 engineered joint angles** — the single most information-dense
   feature type found by the feature-discriminability track
   (`docs/reports/feature-discriminability.md`).

**Ordering matters**: normalization and angle computation happen on the
*full* 543-landmark, 3D-normalized tensor, BEFORE subsetting to ME-126 and
dropping z. Two of the 28 angles (`{L,R}_palm_facing`) are a cross-product
z-component and would be wrong computed from xy alone; every angle's named
vertices already live inside ME-126 (`UPPER_BODY_POSE_8` covers every
arm/wrist angle, `HANDS_42` covers every finger angle), so this doesn't
smuggle in landmarks the trained model's raw-coordinate block doesn't see —
it only means "z is dropped from the raw coordinate channel, not from the
geometry the angles are derived from." The relational block (inter-hand /
hand-to-face distances) is likewise a derived quantity, computed on the full
3D positions same as `landmark_interp_v1`.

Per frame: position(xy) + velocity(xy) + acceleration(xy) + speed = 7
channels x 126 landmarks = 882, + 28 angles, + 12 relational = **922**
(``FEATURE_DIM``), an 83% reduction from ``landmark_interp_v1``'s 5,442.
Column order: ``[per_landmark(882), angles(28), relational(12)]`` —
``sb.recognize.interp.models.split_features`` un-concatenates by this same
order when ``angle_dim > 0``.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from sb.core.subsets import SUBSETS
from sb.recognize.features import cache
from sb.recognize.features.gislr_stratified import load_npz
from sb.recognize.interp.geometry import RELATIONAL_DIM, RELATIONAL_NAMES
from sb.recognize.interp.geometry import center_and_scale, derivatives, relational_block
from sb.recognize.interp.kinematics import ANGLE_NAMES, N_ANGLES, compute_angles

PIPELINE = "landmark_curated_v1"
PIPELINE_VERSION = 1
NAN_POLICY = "zero"

FULL_543 = SUBSETS["FULL_543"]
ME_126 = SUBSETS["ME_126"]
CHANNELS_PER_LANDMARK = 7  # x,y | vx,vy | ax,ay | speed
N_LANDMARKS = len(ME_126)  # 126
PER_LANDMARK_DIM = N_LANDMARKS * CHANNELS_PER_LANDMARK  # 882
ANGLE_DIM = N_ANGLES  # 28
FEATURE_DIM = PER_LANDMARK_DIM + ANGLE_DIM + RELATIONAL_DIM  # 922


def build_frame_features(raw: np.ndarray) -> np.ndarray:
    """(T, 543, 3) raw xyz -> (T, FEATURE_DIM) curated engineered features.

    ``raw`` must already be in canonical holistic row order (what
    ``gislr_stratified.load_npz`` returns for ``rows=FULL_543.array``).
    """
    pos_full = center_and_scale(raw)  # (T, 543, 3) -- full landmarks, still 3D
    angles = compute_angles(pos_full)  # (T, ANGLE_DIM) -- needs 3D (palm-facing)
    relational = relational_block(pos_full)  # (T, RELATIONAL_DIM) -- derived, still 3D

    pos_xy = pos_full[:, ME_126.array, :2]  # (T, 126, 2) -- subset + z dropped HERE
    vel, acc, speed = derivatives(pos_xy)
    per_landmark = np.concatenate(
        [pos_xy, vel, acc, speed[..., None]], axis=-1
    ).astype(np.float32)  # (T, 126, 7)

    T = raw.shape[0]
    return np.concatenate(
        [per_landmark.reshape(T, PER_LANDMARK_DIM), angles, relational], axis=-1
    ).astype(np.float32)


def load_video(path: Path) -> np.ndarray:
    """One GISLR_Stratified npz -> (T, FEATURE_DIM) curated engineered features."""
    raw = load_npz(path, FULL_543.array, coords="xyz")  # (T, 543, 3), NaN preserved
    return build_frame_features(raw)


def cache_inputs(data_dir: Path | str, dataset: str = "gislr") -> dict:
    return cache.cache_inputs(
        pipeline=PIPELINE,
        pipeline_version=PIPELINE_VERSION,
        nan_policy=NAN_POLICY,
        subset=ME_126,
        coords="xy",
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
    ``features/landmark_curated_v1/<key>/``. Skip-if-exists; atomic — same
    pattern as ``features.build_cache``/``base_v1.build_cache``.
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
    with ThreadPoolExecutor(8) as ex:
        for i, arr in enumerate(ex.map(load_video, paths)):
            chunks.append(arr.reshape(-1))
            offsets.append(offsets[-1] + arr.shape[0])
            if progress is not None:
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
        f"{prefix}/landmark_curated_v1 [{root.name}]: cached {len(df)} videos, "
        f"{flat.nbytes / 1e9:.2f} GB ({time.time() - t0:.0f}s)"
    )
    return data_path, off_path
