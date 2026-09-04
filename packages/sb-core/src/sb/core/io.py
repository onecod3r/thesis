"""Reading and writing landmark tensors — the one implementation of the on-disk
unit both pipeline stages exchange.

`sb-extract` writes these files and `sb-recognize` reads them; before the
restructure the write lived in the extractor and the read lived in the quality
scorer, so the format was agreed by convention rather than by code. Here they
are the same two functions, and both validate against :mod:`sb.core.schema`.

Every write is atomic (temp file + ``os.replace``): an interrupted extraction
must never leave a half-written npz that a later resumable pass would treat as
done.
"""

import os
from pathlib import Path

import numpy as np

from sb.core import schema


def write_landmark_npz(
    path: Path | str,
    landmarks: np.ndarray,
    *,
    fps: float,
    validate: bool = True,
) -> Path:
    """Write one video's landmark tensor. Returns the final path.

    ``num_frames`` is derived, never passed: it is exactly ``landmarks.shape[0]``
    and a caller that could disagree with the array is a caller that will.
    """
    path = Path(path)
    if validate:
        schema.validate_tensor(landmarks, where=path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        landmarks=landmarks,
        fps=np.float32(fps),
        num_frames=np.int32(landmarks.shape[0]),
    )
    os.replace(tmp, path)
    return path


def read_landmark_npz(
    path: Path | str,
    *,
    dtype=np.float32,
    validate: bool = True,
) -> tuple[np.ndarray, dict]:
    """``(landmarks, meta)`` for one video.

    Widening float16 -> float32 on read is the default because every consumer
    computes in float32; pass ``dtype=None`` to keep the stored precision.
    """
    path = Path(path)
    with np.load(path) as d:
        arr = d["landmarks"]
        meta = {"fps": float(d["fps"]), "num_frames": int(d["num_frames"])}
    if dtype is not None:
        arr = arr.astype(dtype)
    if validate:
        schema.validate_tensor(arr, dtype=None, where=path.name)
        if meta["num_frames"] != arr.shape[0]:
            raise schema.SpecError(
                f"{path.name}: num_frames={meta['num_frames']} but landmarks hold "
                f"{arr.shape[0]} frames")
    return arr, meta


def iter_landmark_files(root: Path | str):
    """Every completed npz under a landmarks tree, in a stable order.

    ``.tmp.npz`` staging files are skipped: opening one means reading a video
    that is still being written.
    """
    for path in sorted(Path(root).rglob("*.npz")):
        if not path.name.endswith(".tmp.npz"):
            yield path
