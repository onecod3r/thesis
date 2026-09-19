"""Per-feature discriminability metrics for
``gislr.0.dataset.feature-discriminability.ipynb`` (TODO §3.4).

Generalizes the ANOVA-F / probe-classifier pair the (now-broken, TODO §0.1)
``subset-comparison`` notebook used per-landmark into reusable functions that
work on any named flat feature matrix — here, ``kinematics.video_descriptors``'
per-video vectors — plus one new metric the old pair didn't produce: a
**tolerance-band overlap** score, answering directly "feature values that are
close within a class should be tolerated as a match, unless they start
overlapping another class's range."

All functions take ``X (n_videos, n_features)`` and ``y (n_videos,)`` sign
labels; nothing here reads from disk or knows about landmarks — that's
``kinematics.py``'s job.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


def descriptor_matrix(rows: list[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    """List of per-video descriptor dicts (``kinematics.video_descriptors``
    output) -> (X (n, n_features) float32 NaN->0, feature_names).

    Column order is the union of keys across all rows, sorted for
    determinism — a dict is not an ordered contract on its own.
    """
    df = pd.DataFrame(rows).reindex(sorted({k for r in rows for k in r}), axis=1)
    return np.nan_to_num(df.to_numpy(dtype=np.float32), nan=0.0), list(df.columns)


def f_ratio(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """ANOVA F-ratio per feature (between/within-class variance)."""
    with warnings.catch_warnings():   # constant features (e.g. a never-detected landmark) warn
        warnings.simplefilter("ignore")
        F, p = f_classif(X, y)
    return pd.DataFrame({
        "feature": feature_names,
        "F": np.nan_to_num(F, nan=0.0, posinf=0.0),
        "p": p,
    }).sort_values("F", ascending=False, ignore_index=True)


def tolerance_bands(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], k: float = 1.5
) -> pd.DataFrame:
    """Per (class, feature) `mean ± k*std` tolerance band.

    Long format: class, feature, mean, std, lo, hi. `k` controls how wide a
    class's "normal range" is — 1.5 is a loose default (about the middle of a
    typical "not an outlier" convention); tune per how strict a match should be.
    """
    classes = np.unique(y)
    rows = []
    for c in classes:
        mask = y == c
        mu = X[mask].mean(axis=0)
        sd = X[mask].std(axis=0)
        rows.append(pd.DataFrame({
            "class": c, "feature": feature_names,
            "mean": mu, "std": sd, "lo": mu - k * sd, "hi": mu + k * sd,
        }))
    return pd.concat(rows, ignore_index=True)


def band_overlap(bands: pd.DataFrame) -> pd.DataFrame:
    """Per-feature tolerance-band collision across every class pair.

    For each feature, pairwise 1D interval-overlap fraction between every two
    classes' `[lo, hi]` bands (``tolerance_bands`` output), where overlap
    fraction = `overlap_width / min(width_a, width_b)` (0 = bands don't touch,
    1 = the narrower band sits entirely inside the wider one). Returns, per
    feature: `mean_overlap` (how often, on average, this feature's per-class
    "normal range" collides with another class's) and `worst_pair_overlap`
    (its single worst collision) — a **low** score on both means the feature
    has a tight within-class tolerance that rarely mixes with another label,
    which is exactly the property worth ranking features by.
    """
    out = []
    for feat, g in bands.groupby("feature", sort=False):
        lo, hi = g["lo"].to_numpy(), g["hi"].to_numpy()
        width = np.maximum(hi - lo, 1e-12)
        n = len(lo)
        overlaps = []
        for i in range(n):
            for j in range(i + 1, n):
                ov = min(hi[i], hi[j]) - max(lo[i], lo[j])
                if ov > 0:
                    overlaps.append(ov / min(width[i], width[j]))
                else:
                    overlaps.append(0.0)
        overlaps = np.asarray(overlaps) if overlaps else np.zeros(1)
        out.append({
            "feature": feat,
            "mean_overlap": float(overlaps.mean()),
            "worst_pair_overlap": float(overlaps.max()),
        })
    return pd.DataFrame(out).sort_values("mean_overlap", ignore_index=True)


def probe_classifier(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    val_frac: float = 0.10,
    max_iter: int = 200,
    tol: float = 1e-3,
) -> dict:
    """Multinomial logistic probe: standardize, stratified split, fit, score.

    Same recipe the (now-broken) subset-comparison notebook used as its
    headline subset score — here it's the headline score for a *feature set*
    (e.g. "angles only" vs "positions only") rather than a landmark subset.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_frac, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=max_iter, tol=tol)
    clf.fit(scaler.transform(X_tr), y_tr)
    pred = clf.predict(scaler.transform(X_val))
    return {
        "acc": float(accuracy_score(y_val, pred)),
        "macro_acc": float(balanced_accuracy_score(y_val, pred)),
        "n_train": len(y_tr),
        "n_val": len(y_val),
    }


def probe_classifier_cv(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    n_splits: int = 5,
    max_iter: int = 200,
    tol: float = 1e-3,
) -> dict:
    """K-fold cross-validated multinomial logistic probe: mean/std accuracy.

    Same standardize-then-fit recipe as :func:`probe_classifier`, but averaged
    over `n_splits` stratified folds instead of a single held-out split — the
    lower the sample count (e.g. a two-class pair with a few hundred videos),
    the more a single split's accuracy is noise. Matches the 5-fold-CV binary
    separability-probe methodology `docs/reports/plateau-diagnosis.md` §6 used
    for `awake`/`wake` and friends, so results are directly comparable.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, macro_accs = [], []
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=max_iter, tol=tol)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        pred = clf.predict(scaler.transform(X[val_idx]))
        accs.append(accuracy_score(y[val_idx], pred))
        macro_accs.append(balanced_accuracy_score(y[val_idx], pred))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "macro_acc_mean": float(np.mean(macro_accs)),
        "macro_acc_std": float(np.std(macro_accs)),
        "n_splits": n_splits,
    }


def nearest_neighbor_margins(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Per-video same-class vs cross-class nearest-neighbor Euclidean distance.

    A small, visual diagnostic (Scope A) making the tolerance idea concrete:
    a video whose same-class neighbor is much closer than its nearest
    cross-class neighbor sits comfortably inside its class's tolerance band;
    one where the two distances are close is a mixing risk.
    """
    n = len(y)
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    same = y[:, None] == y[None, :]
    rows = []
    for i in range(n):
        same_d = d[i][same[i]]
        cross_d = d[i][~same[i]]
        rows.append({
            "video": i,
            "class": y[i],
            "nearest_same_class": float(same_d.min()) if same_d.size else np.nan,
            "nearest_cross_class": float(cross_d.min()) if cross_d.size else np.nan,
        })
    out = pd.DataFrame(rows)
    out["margin"] = out["nearest_cross_class"] - out["nearest_same_class"]
    return out
