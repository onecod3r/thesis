"""Text <-> gloss corpora (TODO §13).

**ASLG-PC12** (Othman & Jemni 2012; Hugging Face ``achrafothman/aslg_pc12``):
87,710 English/gloss pairs from Europarl. The glosses are **synthetic,
rule-generated** (``DESC-`` adjective and ``X-`` pronoun markers, ``BE`` kept),
so it tests "reproduce that rule system", not "produce real ASL" (report §3).
It ships one ``train`` split; :func:`aslg_splits` makes fixed, seeded
train/val/test splits **by unique English text** (6,587 texts repeat, so a
row-level split would leak).

**NCSLGR** (Boston University, real signing, ~1,888 utterances) has no
scriptable download (licence form). Place an export at
``data/raw/ncslgr/ncslgr.csv`` with ``english`` and ``gloss`` columns;
:func:`ncslgr` returns ``None`` until then.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sb.core.paths import CACHE_DIR, RAW_DIR

ASLG_REPO = "achrafothman/aslg_pc12"
ASLG_FILE = "data/train-00000-of-00001.parquet"
ASLG_RAW = RAW_DIR / "aslg_pc12"
ASLG_CACHE = CACHE_DIR / "aslg_pc12"
NCSLGR_CSV = RAW_DIR / "ncslgr" / "ncslgr.csv"


def aslg_raw() -> pd.DataFrame:
    """Download (once) and load ASLG-PC12: ``text, gloss`` (whitespace/BOM stripped)."""
    path = ASLG_RAW / ASLG_FILE
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        hf_hub_download(ASLG_REPO, ASLG_FILE, repo_type="dataset", local_dir=ASLG_RAW)
    df = pd.read_parquet(path)
    for c in ("text", "gloss"):
        df[c] = df[c].astype(str).str.replace("﻿", "", regex=False).str.strip()
    return df[(df.text != "") & (df.gloss != "")].reset_index(drop=True)


def aslg_splits(fractions: dict[str, float], seed: int, version: str = "v1") -> dict[str, pd.DataFrame]:
    """Fixed splits by unique text, cached under ``data/cache/aslg_pc12/splits/<version>/``.
    A text with several glosses keeps its first one (the duplicates are
    near-identical re-renders of the same rule output)."""
    out = ASLG_CACHE / "splits" / version
    names = list(fractions)
    if all((out / f"{n}.parquet").is_file() for n in names):
        return {n: pd.read_parquet(out / f"{n}.parquet") for n in names}
    df = aslg_raw().drop_duplicates("text", keep="first").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    cuts = np.cumsum([int(round(fractions[n] * len(df))) for n in names])[:-1]
    parts = dict(zip(names, np.split(order, cuts)))
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for n, idx in parts.items():
        d = df.iloc[np.sort(idx)].reset_index(drop=True)
        tmp = out / f"{n}.tmp.parquet"
        d.to_parquet(tmp, index=False)
        os.replace(tmp, out / f"{n}.parquet")
        res[n] = d
    info = {"source": f"hf:{ASLG_REPO}/{ASLG_FILE}", "seed": seed, "fractions": fractions,
            "unique_texts": len(df), "sizes": {n: len(d) for n, d in res.items()}}
    (out / "split_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return res


def ncslgr() -> pd.DataFrame | None:
    if not NCSLGR_CSV.is_file():
        return None
    df = pd.read_csv(NCSLGR_CSV)
    missing = {"english", "gloss"} - set(df.columns)
    if missing:
        raise ValueError(f"{NCSLGR_CSV} needs columns english, gloss (missing {missing})")
    return df.rename(columns={"english": "text"})


def split_info(version: str = "v1") -> dict | None:
    p = ASLG_CACHE / "splits" / version / "split_info.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
