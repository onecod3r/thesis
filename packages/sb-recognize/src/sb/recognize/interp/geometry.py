"""Shared normalization/relational geometry for ``sb.recognize.interp``
pipelines — mid-shoulder-centered, inter-shoulder-scaled positions and the
hand/face relational distances, used by both ``features.py``
(``landmark_interp_v1``) and ``kinematics.py`` (``landmark_kinematics_v1``).

Extracted from ``features.py`` (2026-09-19) so the two pipelines don't
diverge on what "normalized" means — same anchor, same scale, same NaN
policy.
"""

import numpy as np

from sb.core.subsets import EYES_NOSE_36, LIPS_40, pose_rows

# Holistic rows: pose shoulders (11, 12) for centering/scale; hand landmark
# indices are MediaPipe Hand order (0=wrist, 4/8/12/16/20=thumb..pinky tips),
# offset into the left_hand (468) / right_hand (522) holistic blocks.
LEFT_SHOULDER, RIGHT_SHOULDER = pose_rows([11, 12])
HAND_POINTS = {"wrist": 0, "thumb_tip": 4, "index_tip": 8, "middle_tip": 12,
               "ring_tip": 16, "pinky_tip": 20}
LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET = 468, 522
CHIN_ROW = 152  # MediaPipe face-mesh chin landmark; face rows == holistic rows 0-467
EPS = 1e-6

RELATIONAL_NAMES: list[str] = (
    [f"inter_hand_{name}" for name in HAND_POINTS]  # 6
    + [f"{hand}_index_tip_to_{anchor}" for hand in ("left", "right")
       for anchor in ("lips", "chin", "nose")]  # 6
)
RELATIONAL_DIM = len(RELATIONAL_NAMES)  # 12


def center_and_scale(raw: np.ndarray) -> np.ndarray:
    """(T, 543, 3) raw xyz -> (T, 543, 3) mid-shoulder-centered, inter-shoulder
    -scaled, NaN -> 0. NaN from an undetected anchor/shoulder propagates through
    the arithmetic and is zeroed at the end, same policy as ``base_v1``."""
    anchor = np.nanmean(raw[:, [LEFT_SHOULDER, RIGHT_SHOULDER], :], axis=1)  # (T, 3)
    local = raw - anchor[:, None, :]
    shoulder_dist = np.linalg.norm(
        raw[:, LEFT_SHOULDER, :] - raw[:, RIGHT_SHOULDER, :], axis=-1
    )  # (T,)
    valid = np.isfinite(shoulder_dist) & (shoulder_dist > EPS)
    fallback = np.nanmedian(shoulder_dist[valid]) if valid.any() else 1.0
    if not np.isfinite(fallback) or fallback <= EPS:
        fallback = 1.0
    scale = np.where(valid, shoulder_dist, fallback)
    normed = local / scale[:, None, None]
    return np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)


def derivatives(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """position (T, L, C) -> (velocity, acceleration, speed); t=0 has zero
    velocity/acceleration by construction (no t=-1 frame to difference
    against). Dimension-agnostic in the last axis (C=2 for xy, 3 for xyz)."""
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    acc = np.zeros_like(pos)
    acc[1:] = vel[1:] - vel[:-1]
    speed = np.linalg.norm(vel, axis=-1)  # (T, L)
    return vel, acc, speed


def relational_block(pos: np.ndarray) -> np.ndarray:
    """(T, 543, 3) normalized positions -> (T, RELATIONAL_DIM) distances."""
    T = pos.shape[0]
    dists = np.zeros((T, RELATIONAL_DIM), dtype=np.float32)
    col = 0
    for offset in HAND_POINTS.values():
        lp = pos[:, LEFT_HAND_OFFSET + offset, :]
        rp = pos[:, RIGHT_HAND_OFFSET + offset, :]
        dists[:, col] = np.linalg.norm(lp - rp, axis=-1)
        col += 1
    lips_center = pos[:, LIPS_40, :].mean(axis=1)
    nose_center = pos[:, EYES_NOSE_36, :].mean(axis=1)  # eyes/nose anchor block
    chin = pos[:, CHIN_ROW, :]
    for hand_offset in (LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET):
        tip = pos[:, hand_offset + HAND_POINTS["index_tip"], :]
        for anchor in (lips_center, chin, nose_center):
            dists[:, col] = np.linalg.norm(tip - anchor, axis=-1)
            col += 1
    return np.nan_to_num(dists, nan=0.0, posinf=0.0, neginf=0.0)
