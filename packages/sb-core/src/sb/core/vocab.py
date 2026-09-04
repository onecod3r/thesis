"""The gloss vocabulary: sign name <-> class index.

Small, but it belongs in `sb-core` rather than in the recognizer, because every
direction needs the same mapping: recognition emits indices, the rescoring layer
reasons over names, and synthesis will take a name and produce a pose sequence.
A second copy of the map is a second chance for the label spaces to disagree.

GISLR ships the map as `sign_to_prediction_index_map.json` (250 classes), which
is authoritative — the competition's own indices. A dataset without one gets a
map derived deterministically from its sorted label set, so the index is stable
across machines.
"""

import json
from pathlib import Path

LABEL_MAP_FILE = "sign_to_prediction_index_map.json"


def load_label_map(data_dir: Path, filename: str = LABEL_MAP_FILE) -> dict[str, int]:
    """`{sign: index}` from a dataset's official label map."""
    return json.loads((Path(data_dir) / filename).read_text(encoding="utf-8"))


def derive_label_map(signs) -> dict[str, int]:
    """`{sign: index}` from an arbitrary label set — sorted, so two machines
    building it from the same data agree."""
    return {sign: i for i, sign in enumerate(sorted(set(signs)))}


def invert(sign2idx: dict[str, int]) -> dict[int, str]:
    """`{index: sign}` — what a prediction gets rendered through."""
    return {v: k for k, v in sign2idx.items()}


def check_covers(sign2idx: dict[str, int], signs) -> None:
    """Fail loudly when a split contains a sign the map does not know.

    Silently dropping such a row would shift every later index, which is the
    kind of corruption that surfaces as a mysterious accuracy drop.
    """
    missing = sorted(set(signs) - set(sign2idx))
    if missing:
        raise KeyError(f"signs missing from the label map: {missing}")
