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


# ---------------------------------------------------------------------------
# phonological descriptors: handshape, orientation, location, movement, sign type
# ---------------------------------------------------------------------------

# (a, vertex, c) per finger joint, hand-landmark numbering (0 = wrist)
_FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
            "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
PHONOLOGY_FAMILIES = ("handshape", "orientation", "location", "movement", "signtype")


def _joint_angles(h: np.ndarray) -> dict[str, np.ndarray]:
    """``(T, 21, 3)`` hand -> 15 flexion angles (3 per finger, degrees).
    180 = straight."""
    out = {}
    for f, (a, b, c, d) in _FINGERS.items():
        chain = (0, a, b, c, d)
        for k in range(3):
            p, v, q = chain[k], chain[k + 1], chain[k + 2]
            out[f"flex_{f}_{k + 1}"] = _angle(h[:, p], h[:, v], h[:, q])
    return out


def _thirds_delta(s: np.ndarray) -> float:
    """Mean of the last third minus mean of the first third (NaN-aware):
    a change over the sign, e.g. a hand that opens or rotates."""
    s = s[np.isfinite(s)]
    if len(s) < 3:
        return np.nan
    k = len(s) // 3
    return float(s[-k:].mean() - s[:k].mean())


def _stats(prefix: str, s: np.ndarray, out: dict, which=("mean", "std", "delta")) -> None:
    ok = np.isfinite(s)
    for w in which:
        if w == "mean":
            out[f"{prefix}_mean"] = float(s[ok].mean()) if ok.any() else np.nan
        elif w == "std":
            out[f"{prefix}_std"] = float(s[ok].std()) if ok.sum() >= 2 else np.nan
        elif w == "delta":
            out[f"{prefix}_delta"] = _thirds_delta(s)
        elif w == "min":
            out[f"{prefix}_min"] = float(s[ok].min()) if ok.any() else np.nan


def _reversals(v: np.ndarray, min_speed: float) -> int:
    """Direction changes of a 1-D velocity, ignoring near-still frames."""
    s = np.sign(v[np.abs(v) > min_speed])
    return int((np.diff(s) != 0).sum()) if len(s) > 1 else 0


def phonology_descriptors(clip: np.ndarray, near: float = 0.35) -> dict[str, float]:
    """The four ASL phonological parameters (+ sign type) for one clip, per
    hand, on the kept frames of :func:`normalize`.

    - **handshape**: 15 finger flexion angles, fingertip-to-wrist distances ÷
      palm size, 4 spread angles and the thumb-index tip gap. 3-D, since hand
      depth is consistent within a hand.
    - **orientation**: the palm normal and the wrist→middle-knuckle direction.
    - **location** (xy): the hand centroid's distance to the nose, chin,
      forehead, mouth, same-side shoulder, chest and other hand (mean, min,
      fraction of frames within ``near`` shoulder widths), and its mean
      position/spread.
    - **movement** (xy): path length, net displacement, straightness, extent,
      direction reversals (repetition), turning per unit path, mean speed.
    - **signtype**: each hand's presence, the inter-hand distance and the
      correlation of the two hands' (mirrored) velocities (symmetric vs
      alternating).

    Each statistic is ``mean`` / ``std`` over frames, or ``delta`` = last third −
    first third (a change during the sign, e.g. FlexionChange).

    **Left hand = mirrored.** The left hand's features are computed on a copy
    with x negated, so a left-handed signer's dominant hand looks like a
    right-handed signer's. :func:`dominant_swap`'s column swap is then an exact
    mirror of the clip.
    """
    n, _ = normalize(clip)
    out: dict[str, float] = {}
    if len(n) == 0:
        return out
    face_pts = {"nose": [NOSE], "chin": [_I[FACE["chin"]]], "forehead": [_I[FACE["forehead"]]],
                "mouth": [_I[FACE["upper_lip"]], _I[FACE["lower_lip"]]]}
    cents, vels = {}, {}
    for side, hand_idx, other_idx, sh in (("r", RH, LH, R_SH), ("l", LH, RH, L_SH)):
        m = n.copy()
        if side == "l":
            m[..., 0] *= -1
        h = m[:, hand_idx]  # (T, 21, 3)
        present = ~np.isnan(h[:, 0, 0])
        p = f"{side}h_"
        out[f"signtype:{p}present"] = float(present.mean())
        # handshape
        palm = np.linalg.norm(h[:, 9] - h[:, 0], axis=-1)
        palm = np.where(palm > 1e-6, palm, np.nan)
        feats = _joint_angles(h)
        for f, (_, _, _, tip) in _FINGERS.items():
            feats[f"tip_{f}"] = np.linalg.norm(h[:, tip] - h[:, 0], axis=-1) / palm
        dirs = {f: h[:, j[3]] - h[:, j[0]] for f, j in _FINGERS.items()}
        for a, b in (("thumb", "index"), ("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
            feats[f"spread_{a}_{b}"] = _angle(dirs[a], np.zeros_like(dirs[a]), dirs[b])
        feats["thumb_index_gap"] = np.linalg.norm(h[:, 4] - h[:, 8], axis=-1) / palm
        for k, s in feats.items():
            _stats(f"handshape:{p}{k}", s, out)
        # orientation
        normal = np.cross(h[:, 5] - h[:, 0], h[:, 17] - h[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
        point = h[:, 9] - h[:, 0]
        point /= np.maximum(np.linalg.norm(point, axis=-1, keepdims=True), 1e-9)
        for name, vec in (("palm", normal), ("point", point)):
            for k, ax in enumerate("xyz"):
                _stats(f"orientation:{p}{name}_{ax}", vec[:, k], out, ("mean", "delta"))
            if present.sum() >= 2:
                out[f"orientation:{p}{name}_spread"] = float(np.linalg.norm(np.nanstd(vec[present], axis=0)))
        # location (xy)
        c = np.nanmean(h[:, :, :2], axis=1) if present.any() else np.full((len(h), 2), np.nan)
        c[~present] = np.nan
        anchors = {k: np.nanmean(m[:, idx, :2], axis=1) for k, idx in face_pts.items()}
        anchors["shoulder"] = m[:, sh, :2]
        anchors["chest"] = np.zeros((len(m), 2))
        oh = m[:, other_idx, :2]
        anchors["other_hand"] = np.nanmean(oh, axis=1) if (~np.isnan(oh[:, 0, 0])).any() else np.full((len(m), 2), np.nan)
        for k, a in anchors.items():
            d = np.linalg.norm(c - a, axis=-1)
            _stats(f"location:{p}to_{k}", d, out, ("mean", "min"))
            ok = np.isfinite(d)
            out[f"location:{p}near_{k}_frac"] = float((d[ok] < near).mean()) if ok.any() else np.nan
        for k, ax in enumerate("xy"):
            _stats(f"location:{p}pos_{ax}", c[:, k], out, ("mean", "std"))
        # movement (xy), on a 3-frame moving average
        cs = c.copy()
        if len(cs) >= 3:
            cs[1:-1] = (c[:-2] + c[1:-1] + c[2:]) / 3
        v = np.diff(cs, axis=0)
        step = np.linalg.norm(v, axis=-1)
        okv = np.isfinite(step)
        path = float(step[okv].sum()) if okv.any() else np.nan
        seen = np.flatnonzero(np.isfinite(cs[:, 0]))
        net = cs[seen[-1]] - cs[seen[0]] if len(seen) >= 2 else np.array([np.nan, np.nan])
        out[f"movement:{p}path"] = path
        out[f"movement:{p}net_x"], out[f"movement:{p}net_y"] = float(net[0]), float(net[1])
        out[f"movement:{p}net"] = float(np.linalg.norm(net))
        out[f"movement:{p}straightness"] = (  # > 1 only across detection gaps
            float(min(np.linalg.norm(net) / path, 1.0)) if path and path > 1e-6 else np.nan)
        for k, ax in enumerate("xy"):
            col = cs[:, k][np.isfinite(cs[:, k])]
            out[f"movement:{p}extent_{ax}"] = float(np.ptp(col)) if len(col) else np.nan
            out[f"movement:{p}reversals_{ax}"] = float(_reversals(v[okv, k], 0.01)) if okv.any() else np.nan
        vv = v[okv]
        if len(vv) >= 2:
            ang = np.arctan2(vv[:, 1], vv[:, 0])
            turn = np.abs(np.angle(np.exp(1j * np.diff(ang))))
            out[f"movement:{p}turning"] = float(turn.sum() / max(path, 1e-6))
        out[f"movement:{p}speed"] = float(step[okv].mean()) if okv.any() else np.nan
        cents[side], vels[side] = c, v
    # sign type: how the two hands relate (symmetric names survive the swap)
    d = np.linalg.norm(cents["r"] * [1, 1] - cents["l"] * [-1, 1], axis=-1)  # back to the image frame
    _stats("signtype:lh_rh_dist", d, out, ("mean", "std"))
    both = np.isfinite(vels["r"]).all(1) & np.isfinite(vels["l"]).all(1)
    out["signtype:lh_rh_both_frac"] = float((np.isfinite(cents["r"][:, 0]) & np.isfinite(cents["l"][:, 0])).mean())
    if both.sum() >= 3:
        a, b = vels["r"][both].ravel(), vels["l"][both].ravel()  # both in their own mirrored frame
        out["signtype:lh_rh_velcorr"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    return out


def file_phonology(path: str) -> dict[str, float]:
    """:func:`phonology_descriptors` for one npz, columns ``ph|<family>:<name>``,
    plus the hand-presence columns :func:`dominant_swap` reads."""
    clip = load_clip(path)
    out = {f"ph|{k}": v for k, v in phonology_descriptors(clip).items()}
    n, _ = normalize(clip)
    out["ph|meta:lh_present"] = float((~np.isnan(n[:, LH, 0]).all(1)).mean()) if len(n) else np.nan
    out["ph|meta:rh_present"] = float((~np.isnan(n[:, RH, 0]).all(1)).mean()) if len(n) else np.nan
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


# ---------------------------------------------------------------------------
# patterns per phonological parameter (ASL-LEX), not per sign
# ---------------------------------------------------------------------------

def _balanced(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    return float(np.mean([(pred[truth == c] == c).mean() for c in range(k) if (truth == c).any()]))


def parameter_test(X: np.ndarray, gloss: np.ndarray, is_ref: np.ndarray, values: dict,
                   min_glosses: int = 5, n_folds: int = 5, n_perm: int = 200, seed: int = 0) -> dict | None:
    """Do clips group by one phonological parameter's value (e.g. major
    location = Head / Neutral / Body / Hand), **across signs**?

    ``values`` maps gloss -> value (``None`` = unknown). Values held by fewer
    than ``min_glosses`` glosses are dropped. Every score is **gloss-disjoint**:
    a template never contains the sign it is tested on, so a pass means the
    parameter's pattern generalizes to signs it has not seen.

    - ``gloss_bacc``: each gloss's mean descriptor (from ``is_ref`` clips) goes
      to the nearest value template built from all *other* glosses
      (leave-one-gloss-out, cosine); balanced accuracy over values.
    - ``gloss_bacc_null`` / ``p``: the same score with values shuffled across
      glosses ``n_perm`` times; ``p`` = share of shuffles at least as good.
    - ``clip_bacc``: templates from ``is_ref`` clips of the other folds'
      glosses; every non-``is_ref`` clip of the held-out glosses is matched to
      them (``n_folds`` gloss folds).
    - ``silhouette``: cosine silhouette of ``is_ref`` clips grouped by value.
    """
    values = {g: v for g, v in values.items() if isinstance(v, str)}  # None / NaN = unknown
    val = np.array([values.get(g) for g in gloss], dtype=object)
    counts: dict = {}
    for g, v in values.items():
        if v is not None and g in set(gloss):
            counts[v] = counts.get(v, 0) + 1
    kept = sorted(v for v, c in counts.items() if c >= min_glosses)
    if len(kept) < 2:
        return None
    code = {v: i for i, v in enumerate(kept)}
    rng = np.random.default_rng(seed)

    # gloss prototypes (unit mean of ref clips)
    gl = sorted(g for g, v in values.items() if v in code and g in set(gloss))
    U = _unit(X)
    P = _unit(np.stack([U[(gloss == g) & is_ref].mean(0) for g in gl]))
    y = np.array([code[values[g]] for g in gl])
    k = len(kept)

    def logo(yv):
        S = np.zeros((k, P.shape[1]))
        np.add.at(S, yv, P)
        n = np.bincount(yv, minlength=k).astype(float)
        C = S[None] - P[:, None] * (np.arange(k)[None, :, None] == yv[:, None, None])  # drop self
        n_ex = n[None] - (np.arange(k)[None] == yv[:, None])
        C = C / np.maximum(n_ex[..., None], 1)
        sim = np.einsum("gd,gkd->gk", P, C / np.maximum(np.linalg.norm(C, axis=-1, keepdims=True), 1e-12))
        return _balanced(sim.argmax(1), yv, k)

    obs = logo(y)
    null = np.array([logo(rng.permutation(y)) for _ in range(n_perm)])

    # clip level, gloss-disjoint folds
    fold = {g: i % n_folds for i, g in enumerate(rng.permutation(gl))}
    fclip = np.array([fold.get(g, -1) for g in gloss])
    yc = np.array([code.get(v, -1) if v is not None else -1 for v in val])
    preds, truths = [], []
    for f in range(n_folds):
        ref = is_ref & (fclip >= 0) & (fclip != f)
        tst = ~is_ref & (fclip == f)
        T = _unit(np.stack([U[ref & (yc == c)].mean(0) for c in range(k)]))
        preds.append((U[tst] @ T.T).argmax(1))
        truths.append(yc[tst])
    clip_bacc = _balanced(np.concatenate(preds), np.concatenate(truths), k)
    sel = is_ref & (yc >= 0)
    sil = similarity_report(X[sel], yc[sel])["silhouette"]
    return {"values": kept, "glosses_per_value": [counts[v] for v in kept], "n_glosses": len(gl),
            "chance": 1 / k, "gloss_bacc": obs, "gloss_bacc_null": float(null.mean()),
            "p": float((null >= obs).mean()), "clip_bacc": clip_bacc, "silhouette": sil}
