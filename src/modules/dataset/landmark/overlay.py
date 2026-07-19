"""Render video frames with extracted landmarks drawn on top — the visual half
of the confidence-tuning test (TODO §2.3).

The numeric proxies in ``quality.py`` can say "this config detects hands in 94%
of frames"; they cannot say whether those detections are *on the hands*. A
detector that confidently tracks the wrong region scores well on every proxy.
So every tuning verdict gets a human look at the frames, and specifically at the
**worst-scoring** ones — the best frames of a bad config still look fine, which
is exactly why looking only at good frames is misleading.

Drawing is deliberately plain OpenCV (no mediapipe drawing_utils): the npz is
already a plain (T, 543, 3) array in holistic row order, and the connection sets
are the small subset of the skeleton actually relevant here.
"""

from pathlib import Path

import numpy as np

from modules.dataset.landmark.quality import GROUPS, POSE_OFFSET

# BGR, OpenCV order
COLORS = {
    "face": (180, 180, 180),
    "pose": (0, 220, 255),
    "left_hand": (255, 120, 0),
    "right_hand": (0, 160, 255),
}

# hand skeleton: MediaPipe's 21-point topology (wrist 0, then 5 fingers x 4)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
# upper-body pose only — legs are irrelevant to signing and clutter the frame
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]


def _px(point, w: int, h: int):
    """Normalized landmark -> integer pixel, or None when undetected."""
    if not np.isfinite(point[0]) or not np.isfinite(point[1]):
        return None
    return int(round(float(point[0]) * w)), int(round(float(point[1]) * h))


def draw_frame(image: np.ndarray, frame_landmarks: np.ndarray,
               draw_face: bool = True, radius: int = 2) -> np.ndarray:
    """Draw one frame's (543, 3) landmarks onto a copy of `image` (BGR)."""
    import cv2

    out = image.copy()
    h, w = out.shape[:2]

    if draw_face:
        face = frame_landmarks[GROUPS["face"]]
        for p in face[::4]:  # every 4th point — 468 dots would hide the image
            px = _px(p, w, h)
            if px:
                cv2.circle(out, px, 1, COLORS["face"], -1, cv2.LINE_AA)

    pose = frame_landmarks[GROUPS["pose"]]
    for a, b in POSE_CONNECTIONS:
        pa, pb = _px(pose[a], w, h), _px(pose[b], w, h)
        if pa and pb:
            cv2.line(out, pa, pb, COLORS["pose"], 2, cv2.LINE_AA)
    for p in pose[:25]:
        px = _px(p, w, h)
        if px:
            cv2.circle(out, px, radius, COLORS["pose"], -1, cv2.LINE_AA)

    for side in ("left_hand", "right_hand"):
        hand = frame_landmarks[GROUPS[side]]
        for a, b in HAND_CONNECTIONS:
            pa, pb = _px(hand[a], w, h), _px(hand[b], w, h)
            if pa and pb:
                cv2.line(out, pa, pb, COLORS[side], 2, cv2.LINE_AA)
        for p in hand:
            px = _px(p, w, h)
            if px:
                cv2.circle(out, px, radius, COLORS[side], -1, cv2.LINE_AA)
    return out


def annotate(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Stamp small caption lines top-left (config, frame index, score)."""
    import cv2

    out = image.copy()
    for i, text in enumerate(lines):
        y = 18 + i * 16
        cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 3, cv2.LINE_AA)      # outline for readability
        cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_frames(video_path: Path, landmarks: np.ndarray, frame_indices,
                  out_dir: Path, prefix: str, captions=None,
                  draw_face: bool = True) -> list[Path]:
    """Write one annotated PNG per requested frame index.

    Seeks directly to each frame rather than decoding the whole clip — the
    selected frames are scattered across the video and typically few.
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    written = []
    try:
        for idx in frame_indices:
            idx = int(idx)
            if idx >= len(landmarks):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                continue
            drawn = draw_frame(bgr, landmarks[idx], draw_face=draw_face)
            if captions:
                drawn = annotate(drawn, list(captions.get(idx, [])))
            path = out_dir / f"{prefix}_f{idx:05d}.png"
            cv2.imwrite(str(path), drawn)
            written.append(path)
    finally:
        cap.release()
    return written


def contact_sheet(image_paths, ncols: int = 10, thumb_w: int = 220):
    """Tile PNGs into one figure for inline display.

    Contact sheets are how 100 frames get reviewed without 100 inline images
    bloating the notebook (the repo has been burned by that before — a notebook
    once hit 17 MB of cell outputs).
    """
    import cv2
    import matplotlib.pyplot as plt

    paths = list(image_paths)
    if not paths:
        raise ValueError("no images to tile")
    nrows = int(np.ceil(len(paths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 2.1))
    for ax, path in zip(np.ravel(axes), paths):
        img = cv2.imread(str(path))
        scale = thumb_w / img.shape[1]
        img = cv2.resize(img, (thumb_w, int(img.shape[0] * scale)))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_axis_off()
    for ax in np.ravel(axes)[len(paths):]:
        ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    return fig
