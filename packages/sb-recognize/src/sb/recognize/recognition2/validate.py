"""Recognition v2 (TODO §14): score the no-pose rule engine (`rules.py`) against ASL-LEX's own
per-gloss ground truth, restricted to the 233 GISLR glosses that map into ASL-LEX.

Reuses `sb.recognize.aslex` read-only: `gloss_entries` to find each gloss's ASL-LEX `EntryID`
variant(s), and `signdata`/`gloss_codes`'s own column access pattern to read ground truth. Ground
truth for an ambiguous gloss (several EntryID variants that disagree on a parameter) is left out of
that parameter's test, the same policy `aslex.gloss_codes` uses.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from sb.recognize.aslex import gloss_entries, signdata
from sb.recognize.recognition2.rules import PARAMETERS, NOT_COMPUTABLE, _norm_cell

# rules.py parameter -> the ASL-LEX signdata.csv column it's checked against.
GROUND_TRUTH_COLUMN = {
    "sign_type": "SignType.2.0", "handshape": "Handshape.2.0",
    "marked_handshape": "MarkedHandshape.2.0", "selected_fingers": "SelectedFingers.2.0",
    "flexion": "Flexion.2.0", "flexion_change": "FlexionChange.2.0", "spread": "Spread.2.0",
    "spread_change": "SpreadChange.2.0", "thumb_position": "ThumbPosition.2.0",
    "thumb_contact": "ThumbContact.2.0", "ulnar_rotation": "UlnarRotation.2.0",
    "movement": "Movement.2.0", "repeated_movement": "RepeatedMovement.2.0",
    "major_location": "MajorLocation.2.0", "minor_location": "MinorLocation.2.0",
    "second_minor_location": "SecondMinorLocation.2.0", "contact": "Contact.2.0",
    "non_dominant_handshape": "NonDominantHandshape.2.0",
}
# category counts among the 233 ASL-LEX-mapped GISLR glosses (this conversation, 2026-09-26/27) --
# for a chance-level baseline in the agreement table; kept as data, not re-derived per run, since it
# should reflect the full lexicon's category inventory, not just whichever gloss sample is scored.
CHANCE_CATEGORIES = {
    "sign_type": 5, "handshape": 39, "marked_handshape": 2, "selected_fingers": 9, "flexion": 6,
    "flexion_change": 2, "spread": 2, "spread_change": 2, "thumb_position": 2, "thumb_contact": 2,
    "ulnar_rotation": 2, "movement": 6, "repeated_movement": 2, "major_location": 5,
    "minor_location": 29, "second_minor_location": 21, "contact": 2, "non_dominant_handshape": 31,
}


def ground_truth_for_gloss(gloss: str, entries: dict[str, tuple[str, list[str]]],
                            data: pd.DataFrame) -> dict[str, str | None]:
    """One gloss's ASL-LEX ground truth per parameter, ``None`` where the gloss doesn't map or its
    matched variants disagree (mirrors `aslex.gloss_codes`'s ambiguity policy, applied to all 18
    parameters instead of the 13 in `aslex.PARAMETERS`)."""
    if gloss not in entries:
        return dict.fromkeys(PARAMETERS)
    _, ids = entries[gloss]
    rows = data.loc[[i for i in ids if i in data.index]]
    out: dict[str, str | None] = {}
    for param, col in GROUND_TRUTH_COLUMN.items():
        vals = {_norm_cell(v) for v in rows[col] if _norm_cell(v) is not None}
        out[param] = vals.pop() if len(vals) == 1 else None
    return out


def agreement_table(records: list[dict]) -> pd.DataFrame:
    """``records``: one dict per scored clip, each with ``"truth"`` and ``"pred"`` sub-dicts (both
    keyed by :data:`PARAMETERS`, as produced by :func:`ground_truth_for_gloss` /
    `rules.codes`). Returns one row per parameter: how many clips had ground truth available, how
    many of those the rule engine could even attempt (not `NOT_COMPUTABLE`), and agreement among the
    attempted ones -- three different denominators kept separate on purpose, since collapsing them
    would hide whether a low score is "wrong" or "structurally unreachable without pose"."""
    rows = []
    for param in PARAMETERS:
        truths = [r["truth"].get(param) for r in records]
        preds = [r["pred"].get(param) for r in records]
        has_truth = [i for i, v in enumerate(truths) if v is not None]
        attempted = [i for i in has_truth if preds[i] != NOT_COMPUTABLE]
        agree = sum(1 for i in attempted if str(preds[i]).lower() == str(truths[i]).lower())
        rows.append({
            "parameter": param, "n_with_ground_truth": len(has_truth),
            "n_attempted": len(attempted),
            "coverage": len(attempted) / len(has_truth) if has_truth else float("nan"),
            "agreement": agree / len(attempted) if attempted else float("nan"),
            "chance": 1 / CHANCE_CATEGORIES[param],
        })
    return pd.DataFrame(rows)


def most_common_pred(preds: list[str]) -> str:
    """Per-gloss aggregation of several clips' codes: the modal prediction, ties broken by first
    occurrence (deterministic, not random)."""
    return Counter(preds).most_common(1)[0][0]
