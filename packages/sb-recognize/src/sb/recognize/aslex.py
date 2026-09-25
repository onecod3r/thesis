"""ASL-LEX 2.0 phonological codes for the 250 GISLR glosses (TODO §3.8).

ASL-LEX 2.0 (Caselli et al. 2017; Sehyr et al. 2021, CC BY-NC 4.0) codes 2,723
ASL signs for handshape, selected fingers, flexion, location, movement, sign
type and more. ``signdata.csv`` is downloaded once from OSF (project
``zpha4``) into ``data/external/asl_lex/``.

**Mapping a GISLR gloss to ASL-LEX entries** (:func:`gloss_entries`):

1. **exact**: the gloss equals an ``EntryID`` or ``LemmaID`` with digits and
   punctuation removed. That can match several variants (``cat``, ``cat_2``,
   ``cat_3``), and GISLR does not say which one its signers used.
2. **synonym**: :data:`SYNONYMS`, only where the ASL sign is very likely the
   same (DAD = FATHER, GARBAGE = TRASH, FOOD = EAT). A pair known to be a
   different sign (LOOK vs SEE, SAY vs TELL, SLEEPY vs TIRED) is left
   unmapped, as are glosses ASL-LEX lacks.

:func:`gloss_codes` keeps a parameter value for a gloss only when **every**
matched variant agrees on it. Otherwise the value is ``None`` (ambiguous),
and the gloss is left out of that parameter's test.
"""

from __future__ import annotations

import re
import urllib.request

import pandas as pd

from sb.core.paths import DATA_DIR

SIGNDATA_URL = "https://osf.io/download/9nygd/"  # ASL-LEX 2.0 signdata.csv (OSF zpha4)
ASLLEX_DIR = DATA_DIR / "external" / "asl_lex"

#: GISLR gloss -> ASL-LEX EntryIDs of the same ASL sign (checked by meaning; see module doc)
SYNONYMS: dict[str, list[str]] = {
    "alligator": ["crocodile"], "another": ["other"], "callonphone": ["telephone"], "dad": ["father"],
    "food": ["eat_1", "eat_2"], "garbage": ["trash"], "grandma": ["grandmother"], "grandpa": ["grandfather"],
    "haveto": ["must"], "hesheit": ["he"], "into": ["in"], "kitty": ["cat", "cat_2", "cat_3"],
    "minemy": ["mine", "my"], "mom": ["mother"], "nuts": ["nut", "nut_2"], "owie": ["hurt"],
    "potty": ["toilet"], "store": ["shop_1", "shop_2"], "tooth": ["teeth"], "weus": ["we", "us"],
}

#: the parameters tested, and the ASL-LEX 2.0 column of each (first morpheme)
PARAMETERS: dict[str, str] = {
    "sign_type": "SignType.2.0",
    "major_location": "MajorLocation.2.0",
    "minor_location": "MinorLocation.2.0",
    "contact": "Contact.2.0",
    "movement": "Movement.2.0",
    "repeated_movement": "RepeatedMovement.2.0",
    "selected_fingers": "SelectedFingers.2.0",
    "flexion": "Flexion.2.0",
    "flexion_change": "FlexionChange.2.0",
    "spread": "Spread.2.0",
    "thumb_position": "ThumbPosition.2.0",
    "ulnar_rotation": "UlnarRotation.2.0",
    "handshape": "Handshape.2.0",
}


def signdata() -> pd.DataFrame:
    """ASL-LEX 2.0 ``signdata.csv`` (downloaded on first use)."""
    path = ASLLEX_DIR / "signdata.csv"
    if not path.is_file():
        ASLLEX_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(SIGNDATA_URL, tmp)
        tmp.replace(path)
    return pd.read_csv(path, encoding="latin-1")


def _norm(s) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def gloss_entries(glosses, data: pd.DataFrame | None = None) -> dict[str, tuple[str, list[str]]]:
    """``{gloss: (how, [EntryID, ...])}`` for every gloss that maps;
    ``how`` is ``"exact"`` or ``"synonym"``."""
    d = signdata() if data is None else data
    by_key: dict[str, list[str]] = {}
    for col in ("EntryID", "LemmaID"):
        for entry, v in zip(d["EntryID"], d[col]):
            key = _norm(v)
            if entry not in by_key.setdefault(key, []):
                by_key[key].append(entry)
    out = {}
    for g in glosses:
        if _norm(g) in by_key:
            out[g] = ("exact", by_key[_norm(g)])
        elif g in SYNONYMS:
            out[g] = ("synonym", SYNONYMS[g])
    return out


def gloss_codes(glosses, data: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per GISLR gloss: how it mapped, its variants, and each
    :data:`PARAMETERS` value (``None`` where unmapped or variants disagree)."""
    d = signdata() if data is None else data
    d = d.copy().set_index("EntryID")
    ent = gloss_entries(glosses, d.reset_index())
    rows = []
    for g in glosses:
        how, ids = ent.get(g, ("unmapped", []))
        row = {"gloss": g, "mapping": how, "entries": " ".join(ids)}
        for name, col in PARAMETERS.items():
            vals = {str(v) for v in d.loc[ids, col] if pd.notna(v)} if ids else set()
            row[name] = vals.pop() if len(vals) == 1 else None
        rows.append(row)
    return pd.DataFrame(rows)
