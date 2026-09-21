"""``landmark_motion_energy_v1`` — per-landmark RMS speed and per-joint-angle
RMS angular speed, for the motion-energy diagnostic notebook
(``gislr.0.dataset.motion-energy.ipynb``, TODO §1).

Rebuilt 2026-09-21 for the GISLR_Stratified npz dataset (the previous version
read raw ``asl-signs`` parquet via DuckDB, deleted 2026-09-19 when GISLR moved
off that download — see ``docs/reports/motion-energy.md``). Three design
changes from the old pipeline, closing TODO §1.8 / §7.7:

1. **z dropped before the speed computation, not after.** The old notebook
   computed xyz RMS speed and only decomposed xyz-vs-xy afterward on a
   50-video sample; that sample found ~92% of pose "motion" was z-axis depth
   jitter (motion-energy report §2.3). Here every landmark's motion energy is
   computed directly on xy (:func:`landmark_motion_energy` takes whatever
   ``raw`` it's given — the caller drops z before calling it), so the global
   scope is xy-native rather than needing a second pass.
2. **Jitter is smoothed out before the derivative, not measured after it.**
   Savitzky-Golay (same window/polyorder as the original, and as
   ``kinematics.video_descriptors``'s ``jitter_rms``) runs on the filled
   position series before the frame-to-frame difference, so a landmark that
   quivers in place scores near-zero motion instead of the raw per-frame
   noise magnitude.
3. **Joint angles + their rate of change are new** — reuses
   ``kinematics.compute_angles`` (28 angles: shoulder/elbow/wrist flex +
   10 finger joints per hand + palm-facing) rather than re-deriving joint
   geometry, and adds :func:`joint_angle_motion`'s RMS angular speed (degrees
   per frame) as the angle analogue of per-landmark RMS speed.

Angles are always computed in xyz (articulation is a 3D quantity — collapsing
a bent elbow to 2D loses exactly the bend), even though landmark motion
energy is computed on whatever coords the caller passes. This is a deliberate
asymmetry, not an oversight: see the notebook's title cell.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sb.core.schema import GROUPS, N_LANDMARKS
from sb.core.subsets import SUBSETS
from sb.recognize.features.gislr_stratified import load_npz
from sb.recognize.interp.geometry import center_and_scale, reindex_interpolate, smooth_savgol
from sb.recognize.interp.kinematics import ANGLE_NAMES, compute_angles

PIPELINE = "landmark_motion_energy_v1"
PIPELINE_VERSION = 1

FULL_543 = SUBSETS["FULL_543"]

ROW_TYPE = np.empty(N_LANDMARKS, dtype=object)
for _g in GROUPS:
    ROW_TYPE[_g.slice] = _g.name


def load_positions(path: Path | str, coords: str) -> np.ndarray:
    """One GISLR_Stratified npz -> (T, 543, len(coords)) raw positions, NaN
    preserved where undetected. Fixed to the full 543-landmark subset — every
    scope in the motion-energy notebook analyzes all landmarks, never a
    subset (subset selection is a downstream keep/discard decision, not an
    input to this measurement)."""
    return load_npz(path, FULL_543.array, coords=coords)


def landmark_motion_energy(raw: np.ndarray, window: int = 7, polyorder: int = 2) -> pd.DataFrame:
    """(T, 543, C) raw positions, NaN where undetected -> tidy per-landmark
    RMS speed, in whatever coordinate subspace ``raw``'s last axis holds
    (pass xy to drop z, xyz to keep it).

    NaN policy (unchanged from the pre-npz notebook): a transition t->t+1
    counts toward a landmark's RMS only if that landmark was detected
    (non-NaN) at BOTH t and t+1 in the RAW data — interpolation fills gaps so
    Savitzky-Golay has a contiguous series to smooth, but never manufactures
    scored motion across a gap. Landmarks with zero valid transitions get
    rms_speed=NaN.
    """
    T, L, _C = raw.shape
    assert L == N_LANDMARKS, f"expected {N_LANDMARKS} landmark rows, got {L}"
    valid = ~np.isnan(raw).any(axis=-1)  # (T, L) — per-landmark, per-frame
    filled, _frame_valid, _n = reindex_interpolate(raw)
    smoothed = smooth_savgol(filled, window, polyorder)  # jitter removed before velocity
    vel = np.diff(smoothed, axis=0)  # (T-1, L, C)
    speed2 = (vel ** 2).sum(axis=-1)  # (T-1, L)
    vt = valid[1:] & valid[:-1]  # both endpoints observed
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(np.where(vt, speed2, 0.0).sum(axis=0) / vt.sum(axis=0))
    return pd.DataFrame({
        "type": ROW_TYPE,
        "landmark_index": np.arange(N_LANDMARKS),
        "rms_speed": rms,
        "n_valid_transitions": vt.sum(axis=0),
    })


def joint_angle_motion(raw_xyz: np.ndarray, window: int = 7, polyorder: int = 2) -> pd.DataFrame:
    """(T, 543, 3) raw xyz positions -> tidy per-joint-angle RMS angular
    speed ("change of angles", degrees/frame) — the angle analogue of
    :func:`landmark_motion_energy`.

    Pipeline: reindex+interpolate -> ``center_and_scale`` (angles are already
    scale/translation invariant, but this keeps the exact same normalized,
    jitter-smoothed position series ``kinematics.video_descriptors`` computes
    its angles from, so values are directly comparable to the
    feature-discriminability notebook's ``angle_*_mean/std`` columns) ->
    Savitzky-Golay smooth -> ``compute_angles`` -> frame-to-frame angle
    difference -> RMS. No NaN masking needed beyond that: interpolation has
    already filled every gap before the angle is computed, and an angle
    time series has no per-landmark detection concept.
    """
    filled, _frame_valid, _n = reindex_interpolate(raw_xyz)
    pos = center_and_scale(filled)
    smoothed = smooth_savgol(pos, window, polyorder)
    angles = compute_angles(smoothed)  # (T, N_ANGLES) degrees
    dtheta = np.diff(angles, axis=0)  # (T-1, N_ANGLES)
    rms = np.sqrt(np.mean(dtheta ** 2, axis=0))
    return pd.DataFrame({
        "angle_name": ANGLE_NAMES,
        "rms_angular_speed": rms,
        "angle_mean": angles.mean(axis=0),
        "angle_std": angles.std(axis=0),
    })
