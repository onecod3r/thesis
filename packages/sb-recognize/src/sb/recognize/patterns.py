"""Sign "patterns": hand-built kinematic descriptors per clip, and whether they
are tight within a gloss and far apart across glosses (TODO §3.8).

No model is trained anywhere in this module. Every number is a statistic of
descriptors or a nearest-template match.

Per clip (``clip_descriptors``), for one axis setting (``"x"``, ``"y"``,
``"z"``, ``"xy"``, ``"xyz"``):

1. **Reference point**: every point minus the mid-shoulder point of its frame.
2. **Scale**: divided by the clip's median shoulder width, measured in xy.
   One scale serves every axis setting. The shoulder width along y or z alone
   is about 0, so a per-axis scale is undefined.
3. **Angles** (2-D/3-D settings only; undefined on one axis): each side's
   elbow (shoulder–elbow–wrist), shoulder (other shoulder–shoulder–elbow) and
   wrist (elbow–wrist–middle-finger knuckle).
4. **Pair distances** between hand centroids, and from each hand to the face
   (nose tip) and to the chest (mid-shoulder).
5. **Null frames dropped**: a frame with no shoulders or no hand at all. A
   hand missing in a kept frame stays NaN, and every statistic is NaN-aware.
6. **Variance of displacement, velocity, acceleration and jerk** (orders
   0–3) of every point (per axis) and every pair distance, plus the mean of
   each distance and angle.
7. **Touches**: contact onsets between the two hands and between each hand
   and the face. A contact is the closest pair of points, one from each part,
   below ``touch_thr`` shoulder widths, with hysteresis (it ends above
   ``1.5 × touch_thr``). On one axis a "touch" means the parts coincide along
   that axis only.

Heavy-tailed statistics (variances, touch counts) are ``log1p``-compressed.

Caveat for ``z``: MediaPipe measures each part's depth relative to that part.
Pose depth is relative to the hips, hand depth to its own wrist, and face depth
to the face centre. So a z distance between two parts mixes two depth scales.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sb.recognize.features.gislr_stratified import load_npz

P0, LH0, RH0 = 489, 468, 522  # holistic row of pose landmark 0, left hand 0, right hand 0
FACE = {"nose": 1, "chin": 152, "forehead": 10, "upper_lip": 13, "lower_lip": 14, "cheek_r": 234, "cheek_l": 454}

# the rows one clip is loaded with, in this order
ROWS = np.array([P0 + k for k in (11, 12, 13, 14, 15, 16)]
                + list(range(LH0, LH0 + 21)) + list(range(RH0, RH0 + 21))
                + list(FACE.values()), dtype=np.int64)
_I = {int(r): i for i, r in enumerate(ROWS)}
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = (_I[P0 + k] for k in (11, 12, 13, 14, 15, 16))
LH = np.array([_I[LH0 + k] for k in range(21)])
RH = np.array([_I[RH0 + k] for k in range(21)])
FACE_IDX = np.array([_I[r] for r in FACE.values()])
NOSE = _I[FACE["nose"]]

AXES = {"x": [0], "y": [1], "z": [2], "xy": [0, 1], "xyz": [0, 1, 2]}
ORDERS = ("disp", "vel", "acc", "jerk")  # 0th-3rd derivative
ANGLES = ("l_elbow", "r_elbow", "l_shoulder", "r_shoulder", "l_wrist", "r_wrist")
PAIRS = ("lh_rh", "rh_face", "lh_face", "rh_chest", "lh_chest")
TOUCH_PAIRS = ("lh_rh", "rh_face", "lh_face")
POINTS = ("l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
          "lh_centroid", "rh_centroid", "lh_index_tip", "rh_index_tip", "nose")
FAMILIES = ("angle", "distance", "touch", "kinematics")


def load_clip(path: Path | str) -> np.ndarray:
    """One GISLR_Stratified npz -> ``(T, len(ROWS), 3)`` float32, NaN kept."""
    return load_npz(path, ROWS, "xyz")


# ---------------------------------------------------------------------------
# per-clip descriptors
# ---------------------------------------------------------------------------

def normalize(clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Steps 1, 2 and 5: ``(kept frames, keep mask)``. Kept frames are
    mid-shoulder-centred and divided by the clip's median xy shoulder width."""
    sh = clip[:, [L_SH, R_SH]]
    has_sh = ~np.isnan(sh).any(axis=(1, 2))
    has_hand = ~np.isnan(clip[:, np.r_[LH, RH], 0]).all(axis=1)
    keep = has_sh & has_hand
    if keep.sum() == 0:
        return clip[:0], keep
    c = clip[keep]
    mid = c[:, [L_SH, R_SH]].mean(axis=1, keepdims=True)
    width = np.nanmedian(np.linalg.norm(c[:, L_SH, :2] - c[:, R_SH, :2], axis=-1))
    return (c - mid) / max(float(width), 1e-6), keep


def _angle(a: np.ndarray, v: np.ndarray, b: np.ndarray) -> np.ndarray:
    u, w = a - v, b - v
    cos = (u * w).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _points(n: np.ndarray) -> dict[str, np.ndarray]:
    """``(T, 3)`` per named point; hand points NaN when the hand is missing."""
    return {"l_shoulder": n[:, L_SH], "r_shoulder": n[:, R_SH], "l_elbow": n[:, L_EL], "r_elbow": n[:, R_EL],
            "l_wrist": n[:, L_WR], "r_wrist": n[:, R_WR],
            "lh_centroid": n[:, LH].mean(1), "rh_centroid": n[:, RH].mean(1),
            "lh_index_tip": n[:, LH[8]], "rh_index_tip": n[:, RH[8]], "nose": n[:, NOSE]}


def _order_vars(s: np.ndarray) -> list[float]:
    """Variance of a ``(T,)`` or ``(T, k)`` signal and of its 1st-3rd
    differences, NaN-aware (a difference touching a NaN frame is NaN)."""
    out = []
    for _ in ORDERS:
        v = np.nanvar(s, axis=0) if np.isfinite(s).sum() >= 2 else np.nan
        out.append(float(np.nanmean(v)) if np.ndim(v) else float(v))
        s = np.diff(s, axis=0)
    return out


def _contacts(d: np.ndarray, thr: float) -> tuple[int, float]:
    """Onsets of ``d < thr`` with hysteresis (exit above ``1.5 thr``), and
    the fraction of observed frames in contact. NaN frames are skipped."""
    on, n, frames, seen = False, 0, 0, 0
    for v in d:
        if not np.isfinite(v):
            continue
        seen += 1
        if not on and v < thr:
            on, n = True, n + 1
        elif on and v > 1.5 * thr:
            on = False
        frames += on
    return n, frames / seen if seen else np.nan


def _min_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Closest pair between point sets ``(T, m, k)`` and ``(T, n, k)`` -> ``(T,)``."""
    d = np.linalg.norm(a[:, :, None] - b[:, None], axis=-1).reshape(len(a), -1)
    out = np.full(len(a), np.nan)
    seen = ~np.isnan(d).all(1)
    if seen.any():
        out[seen] = np.nanmin(d[seen], axis=1)
    return out


def clip_descriptors(clip: np.ndarray, axes: str, touch_thr: float = 0.12) -> dict[str, float]:
    """Steps 1-7 for one clip and one axis setting -> ``{feature: value}``.
    Features are named ``<family>:<name>``; an all-NaN clip gives all NaN."""
    ax = AXES[axes]
    n, keep = normalize(clip)
    out: dict[str, float] = {"meta:n_frames": float(len(clip)), "meta:n_kept": float(keep.sum()),
                             "meta:lh_present": float(np.nan), "meta:rh_present": float(np.nan)}
    if len(n) >= 1:
        out["meta:lh_present"] = float((~np.isnan(n[:, LH, 0]).all(1)).mean())
        out["meta:rh_present"] = float((~np.isnan(n[:, RH, 0]).all(1)).mean())
    pts = {k: v[:, ax] for k, v in _points(n).items()}

    if len(ax) >= 2:
        mcp = lambda h: n[:, h[9]][:, ax]  # noqa: E731 -- middle-finger knuckle
        angles = {"l_elbow": _angle(pts["l_shoulder"], pts["l_elbow"], pts["l_wrist"]),
                  "r_elbow": _angle(pts["r_shoulder"], pts["r_elbow"], pts["r_wrist"]),
                  "l_shoulder": _angle(pts["r_shoulder"], pts["l_shoulder"], pts["l_elbow"]),
                  "r_shoulder": _angle(pts["l_shoulder"], pts["r_shoulder"], pts["r_elbow"]),
                  "l_wrist": _angle(pts["l_elbow"], pts["l_wrist"], mcp(LH)),
                  "r_wrist": _angle(pts["r_elbow"], pts["r_wrist"], mcp(RH))}
        for name, a in angles.items():
            out[f"angle:{name}_mean"] = float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
            for o, v in zip(ORDERS, _order_vars(a)):
                out[f"angle:{name}_{o}_var"] = v

    dist = {"lh_rh": np.linalg.norm(pts["lh_centroid"] - pts["rh_centroid"], axis=-1),
            "rh_face": np.linalg.norm(pts["rh_centroid"] - pts["nose"], axis=-1),
            "lh_face": np.linalg.norm(pts["lh_centroid"] - pts["nose"], axis=-1),
            "rh_chest": np.linalg.norm(pts["rh_centroid"], axis=-1),  # chest = the origin
            "lh_chest": np.linalg.norm(pts["lh_centroid"], axis=-1)}
    for name, d in dist.items():
        out[f"distance:{name}_mean"] = float(np.nanmean(d)) if np.isfinite(d).any() else np.nan
        for o, v in zip(ORDERS, _order_vars(d)):
            out[f"distance:{name}_{o}_var"] = v

    lh, rh, face = n[:, LH][:, :, ax], n[:, RH][:, :, ax], n[:, FACE_IDX][:, :, ax]
    for name, (a, b) in {"lh_rh": (lh, rh), "rh_face": (rh, face), "lh_face": (lh, face)}.items():
        cnt, frac = _contacts(_min_dist(a, b), touch_thr) if len(n) else (np.nan, np.nan)
        out[f"touch:{name}_count"] = float(cnt)
        out[f"touch:{name}_frac"] = float(frac)

    for name, p in pts.items():
        for k, a in zip(axes, ax):
            s = p[:, ax.index(a)]
            for o, v in zip(ORDERS, _order_vars(s)):
                out[f"kinematics:{name}_{k}_{o}_var"] = v
    return out


def file_descriptors(path: str, axes: tuple[str, ...], touch_thr: float) -> dict[str, float]:
    """Every axis setting for one npz, columns ``<axes>|<family>:<name>``.
    Module-level so a process pool can pickle it."""
    clip = load_clip(path)
    out: dict[str, float] = {}
    for a in axes:
        out.update({f"{a}|{k}": v for k, v in clip_descriptors(clip, a, touch_thr).items()})
    return out


def dominant_swap(table, axes: str):
    """Mirror the left/right naming of the clips whose left hand is seen
    more than the right, so "r" means the dominant hand in every clip.
    Every descriptor here is invariant to reflecting x, so relabeling the
    columns is the whole mirror."""
    import re

    t = table.copy()
    flip = (t[f"{axes}|meta:lh_present"].fillna(0) > t[f"{axes}|meta:rh_present"].fillna(0)).to_numpy()
    pre = f"{axes}|"
    names = [c for c in t.columns if c.startswith(pre)]
    swap = {"l_": "r_", "r_": "l_", "lh_": "rh_", "rh_": "lh_"}

    def partner(c: str) -> str:
        fam, name = c[len(pre):].split(":", 1)
        name = re.sub(r"(?<![a-z])(lh_|rh_|l_|r_)", lambda m: swap[m.group(1)], name)
        if name.startswith("lh_rh") or name.startswith("rh_lh"):
            name = "lh_rh" + name[5:]  # symmetric pair keeps its name
        return f"{pre}{fam}:{name}"

    src = {c: partner(c) for c in names}
    vals = t.loc[flip, [src[c] if src[c] in t.columns else c for c in names]].to_numpy()
    t.loc[flip, names] = vals
    return t, flip


# ---------------------------------------------------------------------------
# the matrix every test runs on
# ---------------------------------------------------------------------------

def feature_matrix(table, family: str | tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """Columns of ``table`` (a DataFrame of ``clip_descriptors`` rows) in the
    given families -> a standardized ``(n, d)`` matrix and its column names.

    Variances and touch counts are ``log1p``-compressed first, each column is
    z-scored over every clip, and a NaN (e.g. a hand never seen) becomes 0,
    the population mean. Columns that are NaN in more than 95% of clips or
    constant are dropped.
    """
    fams = (family,) if isinstance(family, str) else family
    cols = [c for c in table.columns if c.split(":")[0] in fams]
    if not cols:
        return np.zeros((len(table), 0)), []
    X = table[cols].to_numpy(np.float64, copy=True)
    heavy = np.array([c.endswith("_var") or c.endswith("_count") for c in cols])
    X[:, heavy] = np.log1p(np.clip(X[:, heavy], 0, None))
    ok = (np.isnan(X).mean(0) <= 0.95)
    X, cols = X[:, ok], [c for c, k in zip(cols, ok) if k]
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    ok = sd > 1e-9
    X = np.nan_to_num((X[:, ok] - mu[ok]) / sd[ok])
    return X, [c for c, k in zip(cols, ok) if k]


def fisher_ratio(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per column: between-gloss variance of the gloss means / mean
    within-gloss variance (the ANOVA F without degrees of freedom)."""
    classes = np.unique(y)
    means = np.stack([X[y == c].mean(0) for c in classes])
    within = np.stack([X[y == c].var(0) for c in classes]).mean(0)
    return means.var(0) / np.maximum(within, 1e-12)


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def similarity_report(X: np.ndarray, y: np.ndarray) -> dict:
    """Cosine similarity within vs across glosses on standardized
    descriptors (one row per clip, ``y`` = gloss).

    - ``intra``: mean similarity of two clips of the same gloss;
    - ``inter``: mean similarity of two clips of different glosses;
    - ``nearest_inter``: per gloss, mean similarity of its clips to the
      clips of the single most similar other gloss, averaged over glosses;
    - ``frac_glosses_separable``: share of glosses whose intra similarity
      exceeds their ``nearest_inter`` (tighter than their closest rival);
    - ``silhouette``: mean cosine silhouette with glosses as clusters.
    """
    U = _unit(X)
    classes = np.unique(y)
    idx = [np.flatnonzero(y == c) for c in classes]
    C = np.stack([U[i].mean(0) for i in idx])  # mean of unit vectors
    n = np.array([len(i) for i in idx], float)
    # mean pairwise similarity between groups a, b = C_a · C_b (a != b)
    G = C @ C.T
    # within: (|sum|^2 - n) / (n (n - 1))
    S = C * n[:, None]
    intra_g = ((S * S).sum(1) - n) / (n * (n - 1))
    off = G.copy()
    np.fill_diagonal(off, -np.inf)
    nearest = off.max(1)
    w = np.outer(n, n)
    np.fill_diagonal(w, 0)
    inter = float((G * w).sum() / w.sum())

    # silhouette with cosine distance, from the same group means (exact):
    # mean distance of clip i to group b = 1 - U_i · C_b (own group: excluding i)
    sims = U @ C.T  # (N, K)
    own = np.searchsorted(classes, y)
    a = 1 - (sims[np.arange(len(y)), own] * n[own] - 1) / np.maximum(n[own] - 1, 1)
    sims[np.arange(len(y)), own] = -np.inf
    b = 1 - sims.max(1)
    sil = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    per_gloss_sil = np.array([sil[i].mean() for i in idx])
    return {"intra": float(np.average(intra_g, weights=n)), "inter": inter,
            "nearest_inter": float(nearest.mean()), "frac_glosses_separable": float((intra_g > nearest).mean()),
            "silhouette": float(sil.mean()),
            "per_gloss": {"gloss": classes.tolist(), "intra": intra_g.tolist(), "nearest_inter": nearest.tolist(),
                          "nearest_gloss": classes[off.argmax(1)].tolist(), "silhouette": per_gloss_sil.tolist()}}


def nearest_template(X_ref: np.ndarray, y_ref: np.ndarray, X: np.ndarray, y: np.ndarray, top: int = 5) -> dict:
    """One template per gloss (the mean standardized descriptor of its
    ``X_ref`` clips); each clip of ``X`` goes to the most cosine-similar
    template. Top-1/top-``top`` accuracy. No parameters are fitted beyond
    the per-gloss means."""
    classes = np.unique(y_ref)
    T = _unit(np.stack([X_ref[y_ref == c].mean(0) for c in classes]))
    order = np.argsort(-(_unit(X) @ T.T), axis=1)[:, :top]
    truth = np.searchsorted(classes, y)
    return {"top1": float((order[:, 0] == truth).mean()), f"top{top}": float((order == truth[:, None]).any(1).mean()),
            "chance_top1": 1 / len(classes)}
