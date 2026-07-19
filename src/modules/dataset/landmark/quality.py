"""Extraction-quality proxies for POPSIGN landmark npz files.

POPSIGN has **no ground-truth landmarks**, so extraction quality cannot be
measured directly — everything here is a *proxy*, and each one is chosen because
a known failure mode of MediaPipe on this footage makes it worse:

- **detection rate** — fraction of frames where a landmark group is present at
  all. A threshold set too high shows up here first (the detector simply gives
  up on hard frames).
- **presence rate for hands specifically** — signs live in the hands; a config
  with great face detection and absent hands is useless for this project.
- **longest gap** — 30 scattered missing frames and one 30-frame blackout are
  the same detection rate but very different data. Gaps break motion features.
- **jitter** — median frame-to-frame displacement of a present landmark.
  Detector flicker (re-detecting slightly differently each frame) inflates this
  and pollutes exactly the velocity features TODO §7.3 wants to add.
- **bone-length CV** — coefficient of variation of a rigid segment
  (shoulder→elbow). Real anatomy keeps it near-constant within a clip, so
  variance is a direct read on landmark instability, independent of how much the
  signer actually moved.

The composite ``quality_score`` combines them with explicit, tunable weights
(``DEFAULT_WEIGHTS``) — the weighting is a judgement call, so it is data, not
logic, and the ranking can be re-derived without re-extracting anything.

Row layout is the GISLR-compatible holistic order written by
``extraction.py``: face 0-467, left_hand 468-488, pose 489-521, right_hand
522-542.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# holistic row blocks (see extraction.GROUP_LAYOUT)
GROUPS: dict[str, slice] = {
    "face": slice(0, 468),
    "left_hand": slice(468, 489),
    "pose": slice(489, 522),
    "right_hand": slice(522, 543),
}

# pose rows are offset by 489; MediaPipe pose indices 11/13 = left shoulder/elbow
POSE_OFFSET = 489
LEFT_SHOULDER, LEFT_ELBOW = POSE_OFFSET + 11, POSE_OFFSET + 13
RIGHT_SHOULDER, RIGHT_ELBOW = POSE_OFFSET + 12, POSE_OFFSET + 14

# higher = better for the first two, lower = better for the rest (see score_frame)
DEFAULT_WEIGHTS: dict[str, float] = {
    "hand_rate": 3.0,        # hands carry the sign — weighted hardest
    "pose_rate": 1.0,
    "face_rate": 0.5,        # face matters least for recognition
    "jitter": -2.0,          # flicker pollutes velocity features
    "bone_cv": -1.5,         # rigid-segment instability
    "gap": -1.0,             # long blackouts break motion features
}


def load_landmarks(npz_path: Path) -> np.ndarray:
    """(T, 543, 3) float32 with NaN preserved — NaN *is* the signal here."""
    with np.load(npz_path) as d:
        return d["landmarks"].astype(np.float32)


def _present(arr: np.ndarray, block: slice) -> np.ndarray:
    """Per-frame presence of a landmark group: any non-NaN row in the block."""
    return ~np.isnan(arr[:, block, 0]).all(axis=1)


def _longest_false_run(mask: np.ndarray) -> int:
    """Longest consecutive stretch where `mask` is False (the detection gap)."""
    longest = current = 0
    for present in mask:
        current = 0 if present else current + 1
        longest = max(longest, current)
    return longest


def _jitter(arr: np.ndarray, block: slice) -> float:
    """Median frame-to-frame displacement over frames where the group is present.

    Uses xy only: z is largely noise for pose landmarks (the ~92% finding in
    docs/logs/daily/2026-07-15.md), and including it would measure that noise
    rather than detector stability.
    """
    xy = arr[:, block, :2]
    delta = np.linalg.norm(np.diff(xy, axis=0), axis=-1)  # (T-1, L)
    finite = delta[np.isfinite(delta)]
    return float(np.median(finite)) if finite.size else float("nan")


def _bone_cv(arr: np.ndarray, a: int, b: int) -> float:
    """Coefficient of variation of the |a-b| distance — a rigid bone should hold
    it constant, so CV is instability with the signer's motion divided out."""
    seg = np.linalg.norm(arr[:, a, :2] - arr[:, b, :2], axis=-1)
    seg = seg[np.isfinite(seg) & (seg > 0)]
    if seg.size < 2:
        return float("nan")
    return float(np.std(seg) / np.mean(seg))


def frame_quality(arr: np.ndarray) -> np.ndarray:
    """Per-frame quality in [0, 1] — how many groups were detected, hand-weighted.

    Used to pick which frames to render in the visual check: the best and worst
    frames of a clip are the ones worth looking at.
    """
    hands = (_present(arr, GROUPS["left_hand"]).astype(float)
             + _present(arr, GROUPS["right_hand"]).astype(float)) / 2
    pose = _present(arr, GROUPS["pose"]).astype(float)
    face = _present(arr, GROUPS["face"]).astype(float)
    return (3 * hands + 1 * pose + 0.5 * face) / 4.5


def video_metrics(npz_path: Path) -> dict:
    """Every quality proxy for one extracted video (one row of the scoring table)."""
    arr = load_landmarks(npz_path)
    n_frames = int(arr.shape[0])
    if n_frames == 0:
        return {"n_frames": 0, "face_rate": 0.0, "pose_rate": 0.0,
                "left_hand_rate": 0.0, "right_hand_rate": 0.0, "hand_rate": 0.0,
                "any_hand_rate": 0.0, "jitter_hand": float("nan"),
                "jitter_pose": float("nan"), "bone_cv": float("nan"),
                "longest_gap_frames": 0, "longest_gap_frac": 1.0,
                "nan_frac": 1.0}

    present = {g: _present(arr, s) for g, s in GROUPS.items()}
    hand_any = present["left_hand"] | present["right_hand"]
    bone_l = _bone_cv(arr, LEFT_SHOULDER, LEFT_ELBOW)
    bone_r = _bone_cv(arr, RIGHT_SHOULDER, RIGHT_ELBOW)
    gap = _longest_false_run(hand_any)

    return {
        "n_frames": n_frames,
        "face_rate": float(present["face"].mean()),
        "pose_rate": float(present["pose"].mean()),
        "left_hand_rate": float(present["left_hand"].mean()),
        "right_hand_rate": float(present["right_hand"].mean()),
        # mean of the two hands: a config that finds one hand always and the
        # other never is not as good as one that finds both most of the time
        "hand_rate": float((present["left_hand"].mean()
                            + present["right_hand"].mean()) / 2),
        "any_hand_rate": float(hand_any.mean()),
        "jitter_hand": float(np.nanmean([_jitter(arr, GROUPS["left_hand"]),
                                         _jitter(arr, GROUPS["right_hand"])])),
        "jitter_pose": _jitter(arr, GROUPS["pose"]),
        "bone_cv": float(np.nanmean([bone_l, bone_r])),
        "longest_gap_frames": int(gap),
        "longest_gap_frac": float(gap / n_frames),
        "nan_frac": float(np.isnan(arr[:, :, 0]).mean()),
    }


def _z(series: pd.Series) -> pd.Series:
    """Z-score with a zero-variance guard (a metric identical across configs
    must contribute nothing, not NaN)."""
    std = series.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def score(df: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Add `quality_score` to a per-(config, video) metrics table.

    Metrics are z-scored **across the whole table** first, so the score is a
    relative ranking of the configs actually tried — it is not an absolute
    quality measure and shouldn't be compared across different sweeps.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    df = df.copy()
    components = {
        "hand_rate": df["hand_rate"],
        "pose_rate": df["pose_rate"],
        "face_rate": df["face_rate"],
        "jitter": df["jitter_hand"],
        "bone_cv": df["bone_cv"],
        "gap": df["longest_gap_frac"],
    }
    total = pd.Series(0.0, index=df.index)
    for name, values in components.items():
        z = _z(values.fillna(values.median()))
        df[f"z_{name}"] = z
        total = total + weights[name] * z
    df["quality_score"] = total
    return df
