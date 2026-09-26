"""Statistics for the per-sample phonology experiment (TODO §3.10): how alike clips of one gloss
are, how different glosses are, with room for tolerance. No model is fitted: every number is a
statistic, a distance, or a nearest-template / nearest-neighbour match.

Continuous level (``Z`` = standardized clip summaries, NaN -> 0 = the train mean):
:func:`pair_stats`, :func:`tolerance_agreement`, :func:`fisher`, :func:`silhouette`,
:func:`separable_share`, :func:`centroid_eval`, :func:`knn_eval`, :func:`effective_dims`.

Discrete level (per-clip phonological codes, ``sb.recognize.phonology.CODES``):
:func:`code_consistency`, :func:`combo_match`, :func:`combo_uniqueness`, :func:`aslex_agreement`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# continuous
# ---------------------------------------------------------------------------

def standardize(X: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score every column with the ``train`` rows' mean/std; NaN -> 0 (the train mean).
    Returns ``(Z float32, keep)`` where ``keep`` drops constant or all-NaN columns."""
    Xt = X[train].astype(np.float32)
    mu = np.nanmean(Xt, axis=0)
    sd = np.nanstd(Xt, axis=0)
    keep = np.isfinite(mu) & (sd > 1e-6)
    Z = (X[:, keep].astype(np.float32) - mu[keep]) / sd[keep]
    return np.nan_to_num(Z, nan=0.0), keep


def sample_pairs(y: np.ndarray, n: int, rng: np.random.Generator, same: bool,
                 group: np.ndarray | None = None, same_group: bool | None = None) -> np.ndarray:
    """``(n, 2)`` row pairs with equal (``same``) or different labels ``y``. With ``group``
    (e.g. signer), also require the groups to be equal/different (``same_group``)."""
    by = pd.Series(np.arange(len(y))).groupby(y).apply(np.asarray).to_dict()
    out = []
    while len(out) < n:
        m = 4 * (n - len(out))
        a = rng.integers(0, len(y), m)
        if same:
            b = np.array([rng.choice(by[y[i]]) if len(by[y[i]]) >= 2 else i for i in a])
        else:
            b = rng.integers(0, len(y), m)
        ok = (a != b) & ((y[a] == y[b]) if same else (y[a] != y[b]))
        if group is not None and same_group is not None:
            ok &= (group[a] == group[b]) if same_group else (group[a] != group[b])
        out.extend(zip(a[ok], b[ok]))
    return np.array(out[:n])


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random same-gloss pair is more similar than a random different-gloss pair)."""
    r = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _describe(x: np.ndarray, prefix: str) -> dict:
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {f"{prefix}_mean": float(x.mean()), f"{prefix}_std": float(x.std()),
            **{f"{prefix}_p{p}": float(v) for p, v in zip((5, 25, 50, 75, 95), q)}}


def pair_similarity(Z: np.ndarray, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity and Euclidean distance / sqrt(dims) of each pair."""
    cos, dist = [], []
    for s in range(0, len(pairs), 5000):  # batches: the raw baseline has 6,516 columns
        a, b = Z[pairs[s:s + 5000, 0]], Z[pairs[s:s + 5000, 1]]
        cos.append((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9))
        dist.append(np.linalg.norm(a - b, axis=1) / np.sqrt(Z.shape[1]))
    return np.concatenate(cos), np.concatenate(dist)


def pair_stats(Z: np.ndarray, same: np.ndarray, diff: np.ndarray) -> dict:
    """Intra- vs inter-gloss pair statistics: cosine and distance distributions, Cohen's d,
    ROC AUC, and the ratio of mean inter to mean intra distance ("confusability" inverse)."""
    cs, ds = pair_similarity(Z, same)
    cd, dd = pair_similarity(Z, diff)
    pooled = np.sqrt((cs.var() + cd.var()) / 2) + 1e-9
    return {**_describe(cs, "intra_cos"), **_describe(cd, "inter_cos"),
            **_describe(ds, "intra_dist"), **_describe(dd, "inter_dist"),
            "cohen_d_cos": float((cs.mean() - cd.mean()) / pooled),
            "auc_cos": _auc(cs, cd), "auc_dist": _auc(-ds, -dd),
            "dist_ratio_inter_intra": float(dd.mean() / (ds.mean() + 1e-9))}


def tolerance_agreement(Z: np.ndarray, pairs: np.ndarray, taus=(0.25, 0.5, 1.0)) -> dict:
    """Share of features on which the two clips agree within ``tau`` standard deviations: the
    tolerant, feature-by-feature notion of "the same phonology"."""
    hits = np.zeros(len(taus))
    for s in range(0, len(pairs), 5000):
        diff = np.abs(Z[pairs[s:s + 5000, 0]] - Z[pairs[s:s + 5000, 1]])
        hits += [(diff <= t).sum() for t in taus]
    return {f"agree@{t}": float(h / (len(pairs) * Z.shape[1])) for t, h in zip(taus, hits)}


def fisher(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Per column: between-gloss variance / mean within-gloss variance (NaN-aware), eta^2
    (share of variance explained by the gloss), and the missing rate."""
    df = pd.DataFrame(X.astype(np.float32))
    g = df.groupby(y)
    means, vars_, counts = g.mean(), g.var(ddof=0), g.count()
    within = (vars_ * counts).sum() / counts.sum()
    grand = df.mean()
    between = (((means - grand) ** 2) * counts).sum() / counts.sum()
    return pd.DataFrame({"fisher": between / (within + 1e-12),
                         "eta2": between / (between + within + 1e-12),
                         "missing": df.isna().mean()})


def silhouette(Z: np.ndarray, y: np.ndarray, per_class: int, rng: np.random.Generator) -> float:
    """Cosine silhouette with glosses as clusters, on ``per_class`` clips per gloss."""
    from sklearn.metrics import silhouette_score

    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), min(per_class, (y == c).sum()), replace=False)
                          for c in np.unique(y)])
    return float(silhouette_score(Z[idx], y[idx], metric="cosine"))


def _unit_t(Z: np.ndarray) -> torch.Tensor:
    t = torch.as_tensor(Z, device=DEV, dtype=torch.float16 if DEV.type == "cuda" else torch.float32)
    return torch.nn.functional.normalize(t.float(), dim=1).to(t.dtype)


def centroids(Z: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.stack([Z[y == c].mean(axis=0) for c in classes])


def centroid_eval(Ztr, ytr, Zte, yte, classes: np.ndarray, k: int = 5) -> dict:
    """Nearest mean template (cosine), templates from train, scored on test: top-1/top-k and
    per-class accuracy. No fitted weights beyond class means."""
    C = _unit_t(centroids(Ztr, ytr, classes))
    pos = {c: i for i, c in enumerate(classes)}
    tgt = torch.as_tensor([pos[c] for c in yte], device=DEV)
    hits1, hitsk, preds = [], [], []
    for s in range(0, len(Zte), 4096):
        sims = _unit_t(Zte[s:s + 4096]) @ C.T
        top = sims.topk(k, dim=1).indices
        t = tgt[s:s + 4096, None]
        hits1.append((top[:, :1] == t).any(1))
        hitsk.append((top == t).any(1))
        preds.append(top[:, 0])
    h1 = torch.cat(hits1).cpu().numpy()
    per = pd.Series(h1).groupby(yte).mean()
    return {"top1": float(h1.mean()), f"top{k}": float(torch.cat(hitsk).float().mean()),
            "per_class": per, "pred": classes[torch.cat(preds).cpu().numpy()]}


def knn_eval(Ztr, ytr, Zte, yte) -> float:
    """1-nearest-neighbour (cosine) top-1, train -> test."""
    A = _unit_t(Ztr)
    ytr_t = torch.as_tensor(pd.factorize(pd.Series(np.r_[ytr, yte]))[0], device=DEV)
    ytr_c, yte_c = ytr_t[: len(ytr)], ytr_t[len(ytr):]
    hit = []
    for s in range(0, len(Zte), 2048):
        nn = (_unit_t(Zte[s:s + 2048]) @ A.T).argmax(1)
        hit.append(ytr_c[nn] == yte_c[s:s + 2048])
    return float(torch.cat(hit).float().mean())


def separable_share(Ztr, ytr, Zte, yte, classes: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Per gloss: mean cosine of its held-out (test) clips to its own train centroid vs to the
    nearest other gloss's centroid. Separable = own > nearest other. Returns the share and the
    per-gloss table. Held out, so a clip never counts toward the centroid it is compared with."""
    C = _unit_t(centroids(Ztr, ytr, classes))
    rows = []
    for i, c in enumerate(classes):
        s = (_unit_t(Zte[yte == c]) @ C.T).float().mean(0).cpu().numpy()
        other = np.delete(s, i)
        j = int(np.argmax(other))
        rows.append({"gloss": c, "own": float(s[i]), "nearest_other": float(other[j]),
                     "nearest_gloss": np.delete(classes, i)[j], "margin": float(s[i] - other[j])})
    t = pd.DataFrame(rows)
    return float((t.margin > 0).mean()), t


def effective_dims(Z: np.ndarray, rng: np.random.Generator, n: int = 20000, levels=(0.8, 0.9, 0.95)) -> dict:
    """PCA components needed for each share of variance (a sample of ``n`` rows)."""
    idx = rng.choice(len(Z), min(n, len(Z)), replace=False)
    X = torch.as_tensor(Z[idx], device=DEV, dtype=torch.float32)
    X = X - X.mean(0)
    s = torch.linalg.svdvals(X) ** 2
    cum = (s.cumsum(0) / s.sum()).cpu().numpy()
    return {f"pca{int(l * 100)}": int(np.searchsorted(cum, l) + 1) for l in levels}


# ---------------------------------------------------------------------------
# discrete
# ---------------------------------------------------------------------------

def code_consistency(codes: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Per code: number of values, share of 'na', mean within-gloss modal share (how often a
    gloss's clips agree on its most common value), within-gloss normalized entropy, the modal
    share expected by chance (the global most common value), and NMI(code, gloss)."""
    from sklearn.metrics import normalized_mutual_info_score

    rows = []
    for k in codes.columns:
        v = codes[k].astype(str)
        g = pd.DataFrame({"v": v, "y": y})
        modal = g.groupby("y").v.agg(lambda s: s.value_counts(normalize=True).iloc[0])
        ent = g.groupby("y").v.agg(lambda s: _norm_entropy(s))
        rows.append({"code": k, "n_values": v.nunique(), "na": float((v == "na").mean()),
                     "within_modal_share": float(modal.mean()), "within_entropy": float(ent.mean()),
                     "chance_modal_share": float(v.value_counts(normalize=True).iloc[0]),
                     "nmi_gloss": float(normalized_mutual_info_score(y, v))})
    return pd.DataFrame(rows).set_index("code")


def _norm_entropy(s: pd.Series) -> float:
    p = s.value_counts(normalize=True).to_numpy()
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def encode(codes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Codes -> ``(int matrix, na mask)`` for fast pair matching."""
    M = np.stack([pd.factorize(codes[c].astype(str))[0] for c in codes.columns], axis=1)
    na = (codes.astype(str) == "na").to_numpy()
    return M, na


def mismatches(M: np.ndarray, na: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Number of codes on which each pair differs, counting only codes defined for both."""
    a, b = pairs[:, 0], pairs[:, 1]
    defined = ~na[a] & ~na[b]
    return ((M[a] != M[b]) & defined).sum(1)


def combo_match(M, na, same: np.ndarray, diff: np.ndarray, tolerances=range(0, 7)) -> pd.DataFrame:
    """P(two clips' code combinations match with at most m differing codes), same vs different
    gloss, for each tolerance m."""
    ms, md = mismatches(M, na, same), mismatches(M, na, diff)
    return pd.DataFrame({"tolerance": list(tolerances),
                         "p_match_same": [float((ms <= m).mean()) for m in tolerances],
                         "p_match_diff": [float((md <= m).mean()) for m in tolerances]}
                        ).assign(ratio=lambda d: d.p_match_same / d.p_match_diff.where(d.p_match_diff > 0))


def modal_combos(codes: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Each gloss's most common value per code (its 'phonological signature'), and the share of
    its clips matching that signature exactly and within 1-3 differing codes."""
    sig = codes.astype(str).groupby(y).agg(lambda s: s.value_counts().index[0])
    M = codes.astype(str).to_numpy()
    S = sig.loc[y].to_numpy()
    defined = (M != "na") & (S != "na")
    mis = ((M != S) & defined).sum(1)
    share = pd.DataFrame({f"within_{m}": pd.Series(mis <= m).groupby(y).mean() for m in (0, 1, 2, 3)})
    return sig.join(share)


def combo_uniqueness(sig: pd.DataFrame, codes: list[str], tolerances=range(0, 5)) -> pd.DataFrame:
    """For each tolerance m: share of glosses whose signature is within m codes of no other
    gloss's signature (unique), and the mean number of other glosses within m."""
    S = sig[codes].to_numpy()
    defined = S != "na"
    D = ((S[:, None] != S[None]) & defined[:, None] & defined[None]).sum(-1)
    np.fill_diagonal(D, 10**6)
    return pd.DataFrame([{"tolerance": m, "unique_share": float(((D <= m).sum(1) == 0).mean()),
                          "mean_neighbours": float((D <= m).sum(1).mean())} for m in tolerances])


# ASL-LEX value -> our code value, per comparable parameter
ASLEX_MAP = {
    "sign_type": ("sign_type", {"OneHanded": "one", "SymmetricalOrAlternating": "two_moving",
                                "AsymmetricalSameHandshape": "base", "AsymmetricalDifferentHandshape": "base"}),
    "major_location": ("major_location", {"Head": "head", "Hand": "hand", "Body": "body_neutral",
                                          "Neutral": "body_neutral"}),
    "minor_location": ("minor_location", {"Forehead": "forehead", "Eye": "eyes", "CheekNose": ("cheek", "nose"),
                                          "Mouth": "mouth", "Chin": "chin"}),
    "contact": ("contact", {"1": "1", "0": "0"}),
    "repeated_movement": ("repeated", {"1": "1", "0": "0"}),
    "flexion_change": ("handshape_change", {"1.0": "1", "0.0": "0"}),
    "ulnar_rotation": ("orientation_change", {"1": "1", "0": "0"}),
    "movement": ("movement", {"Straight": "straight", "Curved": "curved", "Circular": "circular",
                              "BackAndForth": "back_and_forth"}),
    "selected_fingers": ("selected_fingers", {v: v for v in ("imrp", "i", "im", "t", "ip", "p", "m")}),
    "flexion": ("flexion", {"FullyOpen": "open", "Flat": "bent", "Bent": "bent", "Curved": "curved",
                            "FullyClosed": "closed"}),
    "spread": ("spread", {"1.0": "1", "0.0": "0"}),
    "thumb_position": ("thumb", {"Open": "open", "Closed": "closed"}),
}


def aslex_agreement(codes: pd.DataFrame, y: np.ndarray, lex: pd.DataFrame) -> pd.DataFrame:
    """Per comparable parameter: clip-level agreement of our code with the gloss's ASL-LEX code
    (glosses whose ASL-LEX variants agree and whose value maps), the same with the gloss's modal
    code instead of each clip's, and a baseline that always answers the most common lexicon value."""
    rows = []
    for lp, (ours, table) in ASLEX_MAP.items():
        want = np.array([str(w) for w in lex[lp].reindex(y).to_numpy()])
        ok = np.isin(want, list(table))
        if ok.sum() == 0:
            continue
        got = codes[ours].astype(str).to_numpy()
        target = [table[w] if o else None for w, o in zip(want, ok)]
        hit = np.array([(g in t) if isinstance(t, tuple) else g == t for g, t in zip(got, target)], dtype=bool)
        modal = pd.Series(got).groupby(y).agg(lambda s: s.value_counts().index[0]).reindex(y).to_numpy()
        hit_modal = np.array([(g in t) if isinstance(t, tuple) else g == t for g, t in zip(modal, target)], dtype=bool)
        tv = pd.Series([str(t) for t, o in zip(target, ok) if o])
        per_value = pd.Series(hit[ok]).groupby(tv.to_numpy()).mean()
        rows.append({"aslex": lp, "ours": ours, "clips": int(ok.sum()),
                     "glosses": int(pd.Series(y[ok]).nunique()), "lexicon_values": int(tv.nunique()),
                     "clip_agreement": float(hit[ok].mean()), "gloss_modal_agreement": float(hit_modal[ok].mean()),
                     "majority_baseline": float(tv.value_counts(normalize=True).iloc[0]),
                     "balanced_agreement": float(per_value.mean()), "balanced_chance": 1.0 / tv.nunique()})
    return pd.DataFrame(rows).set_index("aslex")


def signature_accuracy(codes: pd.DataFrame, y: np.ndarray, sig: pd.DataFrame, cols: list[str]) -> dict:
    """Classify each clip as the gloss whose signature (modal codes, from other clips) differs from
    its codes in the fewest defined codes. Ties are split: a clip tied between k glosses counts
    1/k if the right one is among them. Returns the expected top-1, the share of clips whose
    gloss is among the tied best, and the mean tie size."""
    S = sig[cols].astype(str).to_numpy()
    glosses = sig.index.to_numpy()
    pos = {g: i for i, g in enumerate(glosses)}
    M = codes[cols].astype(str).to_numpy()
    hit, tie = [], []
    for s in range(0, len(M), 2000):
        m = M[s:s + 2000]
        D = ((m[:, None] != S[None]) & (m[:, None] != "na") & (S[None] != "na")).sum(-1)
        best = D == D.min(1, keepdims=True)
        true = np.array([pos[g] for g in y[s:s + 2000]])
        hit.append(best[np.arange(len(m)), true])
        tie.append(best.sum(1))
    h, k = np.concatenate(hit), np.concatenate(tie)
    return {"top1_expected": float((h / k).mean()), "in_best_set": float(h.mean()), "mean_tie": float(k.mean())}
