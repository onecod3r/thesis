"""Committed text -> gloss evaluation sets (TODO §13).

``team30`` (v1): the team's 30 "qualitative unseen" sentences, with the T5
hybrid's recorded output from their notebook and **draft** reference glosses
(unreviewed, see the JSON's ``reference_status``). Small enough to commit; the
large corpora (ASLG-PC12, NCSLGR) live under ``data/`` like every dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_DIR = Path(__file__).parent


def load_meta(name: str = "team30", version: str = "v1") -> dict:
    return json.loads((_DIR / f"{name}.{version}.json").read_text(encoding="utf-8"))


def load(name: str = "team30", version: str = "v1") -> pd.DataFrame:
    """One row per sentence: ``id, english, hybrid_recorded, reference_draft``."""
    return pd.DataFrame(load_meta(name, version)["items"])
