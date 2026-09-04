"""Gloss -> pose sequence — NOT IMPLEMENTED.

The one thing fixed in advance: whatever generates poses here emits the tensor
``sb.core.schema`` defines — (T, 543, 3), GISLR holistic row order, NaN for
absent landmarks — the same contract recognition consumes. That is the point of
declaring the seam before there is code behind it: two incompatible landmark
layouts in one repo is the failure `sb-core` exists to prevent.

Nothing else about this direction is decided (TODO §8).
"""

from sb.core import schema


def empty_sequence(n_frames: int):
    """An all-absent pose sequence of the right shape — the only thing this
    module can honestly do today, and a schema-valid starting point."""
    import numpy as np

    arr = np.full((n_frames, schema.N_LANDMARKS, schema.N_COORDS), np.nan,
                  dtype=schema.STORAGE_DTYPE)
    return schema.validate_tensor(arr)
