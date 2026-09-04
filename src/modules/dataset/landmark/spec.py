"""The landmark-tensor contract between extraction (stage 1) and training (stage 2).

This is the interface both datasets and both pipeline stages agree on, and until
now it was a paragraph of README prose plus the same 543-row layout restated in
``extraction.GROUP_LAYOUT``, ``quality.GROUPS`` and ``subsets``' docstring. It is
load-bearing for a cross-dataset claim — POPSIGN's extracted npz is written in
*GISLR holistic row order* precisely so that ``subsets.py`` indices apply to it
unchanged — and nothing checked it (TODO §9.4).

Row layout (543 rows/frame, xyz each)::

    rows   0-467  face      (468 face-mesh landmarks; row == face-mesh index)
    rows 468-488  left_hand (21)
    rows 489-521  pose      (33; holistic row = 489 + pose index)
    rows 522-542  right_hand(21)

On-disk unit (one video, POPSIGN extraction)::

    <root>/data/raw/popsign/<split>/<label>/<video_id>.npz
        landmarks   (T, 543, 3) float16, NaN where undetected
        fps         float32 scalar
        num_frames  int32 scalar   == landmarks.shape[0]

The label is carried by the path, not stored inside the file.

**NaN is data, not corruption.** An undetected group is NaN for that frame, and
that is what ``quality.py``'s detection-rate proxies measure and what the
1st-place feature pipeline normalizes over. A validator that rejected NaN would
reject every real file.

Cheap by design: :func:`validate_tensor` does O(1) structural checks by default,
so it can sit on the extraction write path for 30k+ videos (measured: 0.3 us a
call). The value scan (infinities) is opt-in and used by the npz reader, which
is a diagnostic path.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPEC_VERSION = 1

N_LANDMARKS = 543  # holistic rows per frame
N_COORDS = 3  # x, y, z
STORAGE_DTYPE = np.float16  # what extraction writes
MISSING_VALUE = "nan"  # how an undetected landmark is represented


@dataclass(frozen=True)
class LandmarkGroup:
    """One contiguous block of holistic rows."""

    name: str
    offset: int
    size: int
    result_attr: str  # the MediaPipe HolisticLandmarkerResult attribute

    @property
    def slice(self) -> slice:
        return slice(self.offset, self.offset + self.size)


GROUPS: tuple[LandmarkGroup, ...] = (
    LandmarkGroup("face", 0, 468, "face_landmarks"),
    LandmarkGroup("left_hand", 468, 21, "left_hand_landmarks"),
    LandmarkGroup("pose", 489, 33, "pose_landmarks"),
    LandmarkGroup("right_hand", 522, 21, "right_hand_landmarks"),
)

GROUP_SLICES: dict[str, slice] = {g.name: g.slice for g in GROUPS}
POSE_OFFSET: int = GROUP_SLICES["pose"].start  # holistic row of pose landmark 0

# npz keys every extracted file carries, and the dtype each must have
NPZ_KEYS: dict[str, type] = {
    "landmarks": np.float16,
    "fps": np.float32,
    "num_frames": np.int32,
}

assert sum(g.size for g in GROUPS) == N_LANDMARKS, "group sizes must tile the tensor"
assert [g.offset for g in GROUPS] == list(
    np.cumsum([0, *(g.size for g in GROUPS[:-1])])
), "groups must be contiguous and in order"


class SpecError(ValueError):
    """A landmark tensor or npz violates LANDMARK_TENSOR_V1."""


def spec() -> dict:
    """The contract as data — for embedding in a manifest or a run record."""
    return {
        "spec": "LANDMARK_TENSOR",
        "version": SPEC_VERSION,
        "n_landmarks": N_LANDMARKS,
        "n_coords": N_COORDS,
        "storage_dtype": np.dtype(STORAGE_DTYPE).name,
        "missing_value": MISSING_VALUE,
        "row_order": "gislr-holistic",
        "groups": [
            {"name": g.name, "offset": g.offset, "size": g.size} for g in GROUPS
        ],
        "npz_keys": {k: np.dtype(v).name for k, v in NPZ_KEYS.items()},
    }


def validate_tensor(
    arr: np.ndarray,
    *,
    dtype: type | None = STORAGE_DTYPE,
    check_values: bool = False,
    where: str = "landmarks",
) -> np.ndarray:
    """Structural check of a ``(T, 543, 3)`` landmark tensor; returns it.

    ``dtype=None`` accepts any floating type — loaders legitimately widen
    float16 to float32. ``check_values=True`` additionally scans for infinities;
    it is O(size), so the extraction write path leaves it off.

    T == 0 is valid: a video whose frames all failed to decode is a real,
    recorded outcome, not a malformed file.
    """
    if not isinstance(arr, np.ndarray):
        raise SpecError(f"{where}: expected ndarray, got {type(arr).__name__}")
    if arr.ndim != 3:
        raise SpecError(f"{where}: expected (T, {N_LANDMARKS}, {N_COORDS}), got {arr.shape}")
    if arr.shape[1] != N_LANDMARKS or arr.shape[2] != N_COORDS:
        raise SpecError(
            f"{where}: expected (T, {N_LANDMARKS}, {N_COORDS}), got {arr.shape} — "
            "row order is GISLR holistic (face 0-467, left_hand 468-488, "
            "pose 489-521, right_hand 522-542); a different row count means the "
            "subsets.py indices no longer apply to this file"
        )
    if dtype is not None and arr.dtype != np.dtype(dtype):
        raise SpecError(f"{where}: expected dtype {np.dtype(dtype).name}, got {arr.dtype}")
    if dtype is None and arr.dtype.kind != "f":
        raise SpecError(f"{where}: expected a floating dtype, got {arr.dtype}")
    if check_values and arr.size and np.isinf(arr).any():
        # inf cannot come out of the pipeline and silently destroys any
        # normalization downstream. An all-NaN tensor, by contrast, is a real
        # extraction outcome (nothing detected in that clip) and is a QUALITY
        # signal for quality.py to score, never a spec violation.
        raise SpecError(f"{where}: contains infinities (undetected must be NaN)")
    return arr


def validate_npz(path: Path | str, *, check_values: bool = True) -> dict:
    """Validate one extracted npz against the spec; returns a small summary.

    Reads `landmarks` fully, so this is a diagnostic/verification path rather
    than something to call per-sample in a training loop.
    """
    path = Path(path)
    with np.load(path) as d:
        missing = [k for k in NPZ_KEYS if k not in d]
        if missing:
            raise SpecError(f"{path.name}: missing npz key(s) {missing}")
        arr = d["landmarks"]
        fps = d["fps"]
        num_frames = d["num_frames"]
        validate_tensor(arr, check_values=check_values, where=path.name)
        for key, want in (("fps", np.float32), ("num_frames", np.int32)):
            got = {"fps": fps, "num_frames": num_frames}[key]
            if np.dtype(got.dtype) != np.dtype(want):
                raise SpecError(
                    f"{path.name}: {key} dtype {got.dtype}, expected {np.dtype(want).name}")
        if int(num_frames) != arr.shape[0]:
            raise SpecError(
                f"{path.name}: num_frames={int(num_frames)} but landmarks hold "
                f"{arr.shape[0]} frames")
        detected = {
            g.name: float(1.0 - np.isnan(arr[:, g.slice, 0]).all(axis=1).mean())
            if arr.shape[0] else 0.0
            for g in GROUPS
        }
    return {
        "path": str(path),
        "n_frames": int(num_frames),
        "fps": float(fps),
        "dtype": str(arr.dtype),
        "detection_rate": detected,
    }
