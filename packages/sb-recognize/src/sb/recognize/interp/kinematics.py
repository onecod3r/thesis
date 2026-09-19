"""``landmark_kinematics_v1`` — per-video engineered-feature descriptors for
the feature-discriminability notebook
(``gislr.0.dataset.feature-discriminability.ipynb``, TODO §3.4).

Unlike ``features.py`` (``landmark_interp_v1``, a per-*frame* tensor built as
model input), this pipeline reduces one ``(T, 543, 3)`` npz straight down to a
**flat, named, per-video** descriptor dict — there is no training consumer
here, so there's no fixed channel-count/model-input discipline to keep, only
"describe this video with numbers a discriminability metric can rank."

Pipeline, per video:

1. **Reindex + interpolate** every gap (a single undetected landmark or a
   fully-undetected frame) linearly over the frame axis — same policy the
   motion-energy notebook established. This is the answer to "a dropped frame
   shouldn't fake a velocity spike": rather than deleting frames and dividing
   by a reconstructed gap, the frame count is *preserved* and the gap is
   filled with a straight-line interpolant, so a derivative computed across
   it reads as steady motion, not a jump. ``n_valid_frames`` (frames with at
   least one detected landmark, pre-interpolation) is kept as a diagnostic
   descriptor.
2. **Normalize**: ``geometry.center_and_scale`` — same mid-shoulder anchor +
   inter-shoulder scale as ``landmark_interp_v1``.
3. **Smooth**: Savitzky-Golay (window/polyorder configurable, default 7/2 —
   same constants as the motion-energy notebook) on normalized position.
   **Jitter** = normalized − smoothed (the high-frequency residual), RMS'd
   per region.
4. **Derivatives**: velocity/acceleration/speed from the *smoothed* position
   (dt=1 throughout, since step 1 already made the frame axis contiguous).
5. **Rolling variance**: windowed variance of per-landmark speed, region-
   averaged.
6. **Joint angles**: arccos of the normalized dot product at named skeletal
   vertices (arm, wrist orientation, per-finger flexion, palm-facing) — see
   :data:`ANGLE_NAMES` / :func:`compute_angles`.
7. **Relational distances**: ``geometry.relational_block``, unchanged.
8. **Reduce**: mean + std over time for every per-landmark/per-angle/
   relational signal; region means for jitter/rolling-variance — one flat
   ``dict[str, float]`` per video (:func:`video_descriptors`).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from sb.core.schema import GROUPS, N_LANDMARKS
from sb.core.subsets import SUBSETS, pose_rows
from sb.recognize.features.gislr_stratified import load_npz
from sb.recognize.interp.geometry import (
    LEFT_HAND_OFFSET,
    RIGHT_HAND_OFFSET,
    RELATIONAL_NAMES,
    center_and_scale,
    relational_block,
)

PIPELINE = "landmark_kinematics_v1"
PIPELINE_VERSION = 1

FULL_543 = SUBSETS["FULL_543"]
EPS = 1e-6

# ---- joint-angle vertex definitions (holistic row indices) ---------------
# Pose: shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24 (MediaPipe Pose).
_L_SHOULDER, _R_SHOULDER, _L_ELBOW, _R_ELBOW, _L_HIP, _R_HIP, _L_POSE_WRIST, _R_POSE_WRIST = (
    pose_rows([11, 12, 13, 14, 23, 24, 15, 16])
)

# Hand: MediaPipe Hand indices, offset into the holistic left/right hand blocks.
# base = wrist(0) for the four fingers, CMC(1) for the thumb (its own proximal joint).
_FINGER_CHAINS: dict[str, tuple[int, int, int, int]] = {
    "thumb": (1, 2, 3, 4),    # CMC, MCP, IP, TIP
    "index": (0, 5, 6, 7),    # wrist, MCP, PIP, DIP
    "middle": (0, 9, 10, 11),
    "ring": (0, 13, 14, 15),
    "pinky": (0, 17, 18, 19),
}
_HAND_WRIST, _HAND_INDEX_MCP, _HAND_PINKY_MCP, _HAND_MIDDLE_MCP = 0, 5, 17, 9


def _finger_angle_specs(offset: int, side: str) -> list[tuple[str, int, int, int]]:
    """(name, a, vertex, c) triples for both flex angles of every finger."""
    specs = []
    for finger, (base, j1, j2, j3) in _FINGER_CHAINS.items():
        specs.append((f"{side}_{finger}_mcp_flex", offset + base, offset + j1, offset + j2))
        specs.append((f"{side}_{finger}_pip_flex", offset + j1, offset + j2, offset + j3))
    return specs


ANGLE_SPECS: list[tuple[str, int, int, int]] = (
    [
        ("L_shoulder_angle", _L_HIP, _L_SHOULDER, _L_ELBOW),
        ("R_shoulder_angle", _R_HIP, _R_SHOULDER, _R_ELBOW),
        ("L_elbow_angle", _L_SHOULDER, _L_ELBOW, _L_POSE_WRIST),
        ("R_elbow_angle", _R_SHOULDER, _R_ELBOW, _R_POSE_WRIST),
        ("L_wrist_angle", _L_POSE_WRIST, LEFT_HAND_OFFSET + _HAND_WRIST,
         LEFT_HAND_OFFSET + _HAND_MIDDLE_MCP),
        ("R_wrist_angle", _R_POSE_WRIST, RIGHT_HAND_OFFSET + _HAND_WRIST,
         RIGHT_HAND_OFFSET + _HAND_MIDDLE_MCP),
    ]
    + _finger_angle_specs(LEFT_HAND_OFFSET, "L")
    + _finger_angle_specs(RIGHT_HAND_OFFSET, "R")
)
ANGLE_NAMES: list[str] = [name for name, *_ in ANGLE_SPECS] + ["L_palm_facing", "R_palm_facing"]
N_ANGLES = len(ANGLE_NAMES)  # 6 + 20 + 2 = 28

ROW_TYPE = np.empty(N_LANDMARKS, dtype=object)
for _g in GROUPS:
    ROW_TYPE[_g.slice] = _g.name


def _angle_deg(pos: np.ndarray, a: int, vertex: int, c: int) -> np.ndarray:
    """(T,) angle at `vertex` between rays to `a` and `c`, degrees."""
    va = pos[:, a, :] - pos[:, vertex, :]
    vc = pos[:, c, :] - pos[:, vertex, :]
    denom = np.linalg.norm(va, axis=-1) * np.linalg.norm(vc, axis=-1)
    cos = np.divide(np.sum(va * vc, axis=-1), denom,
                    out=np.zeros_like(denom), where=denom > EPS)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _palm_facing_deg(pos: np.ndarray, offset: int) -> np.ndarray:
    """(T,) angle between the palm normal and the local +z (forward) axis."""
    wrist = pos[:, offset + _HAND_WRIST, :]
    v1 = pos[:, offset + _HAND_INDEX_MCP, :] - wrist
    v2 = pos[:, offset + _HAND_PINKY_MCP, :] - wrist
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal, axis=-1)
    nz = np.divide(normal[:, 2], norm, out=np.zeros_like(norm), where=norm > EPS)
    return np.degrees(np.arccos(np.clip(nz, -1.0, 1.0)))


def compute_angles(pos: np.ndarray) -> np.ndarray:
    """(T, 543, 3) normalized positions -> (T, N_ANGLES) degrees."""
    cols = [_angle_deg(pos, a, v, c) for _, a, v, c in ANGLE_SPECS]
    cols.append(_palm_facing_deg(pos, LEFT_HAND_OFFSET))
    cols.append(_palm_facing_deg(pos, RIGHT_HAND_OFFSET))
    return np.stack(cols, axis=-1)


def _reindex_interpolate(raw: np.ndarray) -> tuple[np.ndarray, int]:
    """(T, 543, 3) raw, NaN where undetected -> (filled, n_valid_frames).

    Every gap (a single undetected landmark or a fully-undetected frame) is
    linearly interpolated over the frame axis, `limit_direction="both"` so
    leading/trailing gaps extrapolate flat. A landmark never detected in the
    whole video stays NaN here — `center_and_scale`'s own NaN->0 handles it.
    `n_valid_frames` counts frames with at least one detected landmark,
    measured on the raw (pre-interpolation) data.
    """
    T = raw.shape[0]
    frame_valid = ~np.isnan(raw).all(axis=(1, 2))
    flat = pd.DataFrame(raw.reshape(T, -1))
    filled = flat.interpolate(method="linear", limit_direction="both", axis=0).to_numpy()
    return filled.reshape(raw.shape), int(frame_valid.sum())


def _smooth(pos: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing along the frame axis; no-op below 3 frames."""
    T = pos.shape[0]
    if T < 3:
        return pos.copy()
    w = min(window, T if T % 2 == 1 else T - 1)
    po = min(polyorder, w - 1)
    flat = pos.reshape(T, -1)
    allnan = np.isnan(flat).all(axis=0)
    sm = flat.copy()
    if (~allnan).any():
        sm[:, ~allnan] = savgol_filter(flat[:, ~allnan], w, po, axis=0)
    return sm.reshape(pos.shape)


def _derivatives(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """smoothed position (T,543,3) -> (velocity, speed); dt=1 (contiguous frames)."""
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    speed = np.linalg.norm(vel, axis=-1)  # (T, 543)
    return vel, speed


def _region_means(per_landmark: np.ndarray) -> dict[str, float]:
    """(543,) per-landmark scalar -> {region_name: mean}, NaN-safe."""
    out = {}
    for g in GROUPS:
        vals = per_landmark[g.slice]
        out[g.name] = float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
    return out


def video_descriptors(
    raw: np.ndarray,
    savgol_window: int = 7,
    savgol_polyorder: int = 2,
    rolling_var_window: int = 5,
) -> dict[str, float]:
    """One GISLR_Stratified video's (T, 543, 3) raw xyz -> a flat, named
    per-video descriptor dict (mean/std over time per signal)."""
    T = raw.shape[0]
    detection_rate = (~np.isnan(raw[:, :, 0])).mean(axis=0)  # (543,), pre-interpolation

    filled, n_valid_frames = _reindex_interpolate(raw)
    pos = center_and_scale(filled)
    smoothed = _smooth(pos, savgol_window, savgol_polyorder)
    jitter_rms = np.sqrt(np.mean((pos - smoothed) ** 2, axis=0))  # (543, 3) -> per-landmark

    vel, speed = _derivatives(smoothed)
    rolling_var = (
        pd.DataFrame(speed).rolling(rolling_var_window, min_periods=1).var(ddof=0)
        .fillna(0.0).to_numpy()
    )  # (T, 543)

    angles = compute_angles(smoothed)         # (T, N_ANGLES)
    relational = relational_block(smoothed)   # (T, RELATIONAL_DIM)

    desc: dict[str, float] = {
        "n_frames": float(T),
        "valid_frame_frac": float(n_valid_frames) / T if T else 0.0,
    }

    for i in range(N_LANDMARKS):
        rtype = ROW_TYPE[i]
        for ci, coord in enumerate("xyz"):
            desc[f"{rtype}_{i}_{coord}_mean"] = float(np.mean(smoothed[:, i, ci]))
            desc[f"{rtype}_{i}_{coord}_std"] = float(np.std(smoothed[:, i, ci]))
        desc[f"{rtype}_{i}_speed_mean"] = float(np.mean(speed[:, i]))
        desc[f"{rtype}_{i}_speed_std"] = float(np.std(speed[:, i]))
        desc[f"{rtype}_{i}_detection_rate"] = float(detection_rate[i])

    for name, jitter_region_mean in _region_means(jitter_rms.mean(axis=-1)).items():
        desc[f"jitter_rms_{name}"] = jitter_region_mean
    for name, var_region_mean in _region_means(rolling_var.mean(axis=0)).items():
        desc[f"rolling_var_speed_{name}"] = var_region_mean

    for i, name in enumerate(ANGLE_NAMES):
        desc[f"angle_{name}_mean"] = float(np.mean(angles[:, i]))
        desc[f"angle_{name}_std"] = float(np.std(angles[:, i]))

    for i, name in enumerate(RELATIONAL_NAMES):
        desc[f"relational_{name}_mean"] = float(np.mean(relational[:, i]))
        desc[f"relational_{name}_std"] = float(np.std(relational[:, i]))

    return desc


def load_video_descriptors(path: Path | str, **kwargs) -> dict[str, float]:
    """One GISLR_Stratified npz -> its `video_descriptors()` dict."""
    raw = load_npz(path, FULL_543.array, coords="xyz")  # (T, 543, 3), NaN preserved
    return video_descriptors(raw, **kwargs)
