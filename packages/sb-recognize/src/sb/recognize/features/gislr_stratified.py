"""On-disk read for the GISLR_Stratified npz dataset — shared by both feature
pipelines (``base_v1`` zeroes NaN, ``firstplace_v1`` preserves it, ``evaluate``
reads it directly), so the layout fix below lives in one place.

**Row-order mismatch.** The producing notebook wrote ``landmarks`` as
``pose[0:33], face[33:501], left_hand[501:522], right_hand[522:543]`` — NOT
this repo's canonical gislr-holistic order (``sb.core.schema.GROUPS``:
``face[0:468], left_hand[468:489], pose[489:522], right_hand[522:543]``) that
every ``sb.core.subsets`` index array (ME_126, FP_118, ...) is defined
against. Reading the file as-is would silently hand a subset the wrong
physical landmarks. ``CANONICAL_TO_STORED`` is the fixed permutation that
undoes this at read time, so callers keep indexing with canonical subset
arrays and get back the landmarks those indices actually name.
"""

from pathlib import Path

import numpy as np

from sb.core.schema import N_LANDMARKS, validate_tensor

_COORD_INDEX = {"x": 0, "y": 1, "z": 2}

CANONICAL_TO_STORED = np.empty(N_LANDMARKS, dtype=np.int64)
CANONICAL_TO_STORED[0:468] = np.arange(33, 501)  # face
CANONICAL_TO_STORED[468:489] = np.arange(501, 522)  # left_hand
CANONICAL_TO_STORED[489:522] = np.arange(0, 33)  # pose
CANONICAL_TO_STORED[522:543] = np.arange(522, 543)  # right_hand (already aligned)


def load_npz(path: Path | str, rows: np.ndarray, coords: str) -> np.ndarray:
    """One GISLR_Stratified npz -> ``(T, len(rows), len(coords))`` float32.

    ``rows`` are canonical gislr-holistic indices (a subset's ``.array``);
    they're remapped onto the npz's own on-disk order before selecting. NaN is
    preserved — callers decide whether to zero it (``base_v1`` does,
    ``firstplace_v1`` must not).
    """
    path = Path(path)
    with np.load(path) as d:
        arr = d["landmarks"]
    validate_tensor(arr, dtype=None, where=path.name)
    mapped_rows = CANONICAL_TO_STORED[np.asarray(rows)]
    coord_idx = [_COORD_INDEX[c] for c in coords]
    return arr[:, mapped_rows, :][:, :, coord_idx].astype(np.float32)
