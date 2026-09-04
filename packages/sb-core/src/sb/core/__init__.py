"""Contracts shared by every stage and both directions.

`sb-core` is the seam. Recognition (landmarks -> gloss) and synthesis
(gloss -> pose) sit on opposite sides of it, extraction feeds it, and the
rescoring layer reads its vocabulary — so a change to the landmark layout or
the label space breaks the build instead of silently corrupting data.

Nothing here imports torch, mediapipe or tensorflow: `sb-core` must stay cheap
enough for anything to depend on.

- ``schema``   — LANDMARK_TENSOR v1: row layout, dtype, NaN policy, validators
- ``subsets``  — the landmark-subset registry (FULL_543, ME_126, FP_118, ...)
- ``vocab``    — sign name <-> class index
- ``io``       — atomic landmark-npz read/write, schema-checked
- ``paths``    — the repo tree and lazy dataset resolution
"""

from sb.core import io, paths, schema, subsets, vocab

__all__ = ["io", "paths", "schema", "subsets", "vocab"]
