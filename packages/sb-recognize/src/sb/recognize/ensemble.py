"""Ensembles of registry runs on the canonical split (TODO §3.10): average the members'
probabilities per clip. For streaming models the same average works per frame, one recurrent state
per member, so an ensemble of streaming models still streams.

Each member's full probabilities come from :func:`sb.recognize.evaluate.evaluate_run` with
``probs_out`` (cached under ``data/cache/gislr/val_probs/<run_id>.npy``); nothing is trained.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sb.core.paths import CACHE_DIR, MODELS_DIR
from sb.mlops import registry as R

PROBS_DIR = CACHE_DIR / "gislr" / "val_probs"


def probs(run_id: int | str) -> tuple[np.ndarray, np.ndarray]:
    """``(probs (n_val, n_classes) float32, labels)`` of one run, computed once and cached."""
    from sb.recognize.evaluate import evaluate_run

    path = PROBS_DIR / f"{run_id}.npy"
    run_dir = MODELS_DIR / str(run_id)
    if not path.exists():
        evaluate_run(run_dir, verbose=False, probs_out=path)
    labels = np.load(run_dir / "assets" / "val_predictions.npz")["labels"]
    return np.load(path).astype(np.float32), labels


def evaluate(members: dict[str, int], sizes=(1, 2, 3, 4), weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Top-1 / top-5 / macro accuracy of every combination of ``members`` (name -> run id) of the
    given sizes, probabilities averaged (optionally weighted)."""
    P = {n: probs(r) for n, r in members.items()}
    y = next(iter(P.values()))[1]
    assert all((v[1] == y).all() for v in P.values()), "members scored on different splits"
    rows = []
    for k in sizes:
        for combo in combinations(members, k):
            w = [(weights or {}).get(n, 1.0) for n in combo]
            avg = sum(wi * P[n][0] for wi, n in zip(w, combo)) / sum(w)
            top = np.argsort(-avg, axis=1)[:, :5]
            hit = top[:, 0] == y
            rows.append({"members": " + ".join(combo), "n": k,
                         "params": sum(R.load_meta(MODELS_DIR / str(members[n]))["n_params"] for n in combo),
                         "top1": float(hit.mean()), "top5": float((top == y[:, None]).any(1).mean()),
                         "macro": float(pd.Series(hit).groupby(y).mean().mean())})
    return pd.DataFrame(rows).sort_values(["n", "top1"], ascending=[True, False]).reset_index(drop=True)
