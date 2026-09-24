"""Next-gloss predictors: given the glosses accepted so far, a probability
for every gloss that could come next (TODO §12.6, stage 2).

The recognizer answers "what does this segment look like?"; these answer
"what is likely to be signed next?". :mod:`sb.recognize.continuous.fuse`
combines the two to accept or reject a sign.

Every predictor has the same surface, so the fusion code never knows which
one it has:

- ``vocab`` -- the glosses, then :data:`EOS` last;
- ``dist(history)`` -- probabilities over ``vocab`` (sums to 1);
- ``gloss_dist(history)`` -- the same without ``EOS``, renormalized: what
  fusion uses, since a segment is always some gloss.

:class:`NgramLM` is interpolated Kneser-Ney with one absolute discount. It
is count tables and needs no torch, so it can ship to the client as JSON
(:meth:`NgramLM.to_dict`). The neural arm is :mod:`sb.rescore.neural`.

The corpus is small (1,757 sentences, mean 3.1 glosses) and written by
Claude to templates, so a model scored on sentences it was fitted on
memorizes them. :func:`sentence_folds` and :func:`theme_folds` give the two
held-out protocols the experiments use.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

BOS, EOS = "<s>", "</s>"


class NextGlossModel:
    """Shared surface; subclasses implement :meth:`dist`."""

    vocab: list[str]

    def dist(self, history: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def gloss_dist(self, history: Sequence[str]) -> np.ndarray:
        p = self.dist(history)[:-1]
        return p / p.sum()


class UniformLM(NextGlossModel):
    """No prior: every gloss equally likely. The 'recognizer only' arm."""

    def __init__(self, glosses: Sequence[str]):
        self.vocab = list(glosses) + [EOS]

    def dist(self, history: Sequence[str]) -> np.ndarray:
        return np.full(len(self.vocab), 1.0 / len(self.vocab))


class NgramLM(NextGlossModel):
    """Interpolated Kneser-Ney n-gram over gloss tokens.

    The highest order uses raw counts; lower orders use continuation counts
    (how many distinct contexts a gloss follows). Every level interpolates
    down to a uniform floor, so no gloss ever gets probability 0, which
    matters because fusion takes its log. ``order=1`` is a discounted
    unigram over the uniform floor.
    """

    def __init__(self, glosses: Sequence[str], order: int = 3, discount: float = 0.75):
        assert order >= 1 and 0 < discount < 1
        self.vocab = list(glosses) + [EOS]
        self.index = {g: i for i, g in enumerate(self.vocab)}
        self.order, self.discount = order, discount
        # tables[k]: context tuple of length k-1 -> Counter(next token), k = 1..order
        self.tables: dict[int, dict[tuple[str, ...], Counter]] = {}
        self._memo: dict[tuple[int, tuple[str, ...]], np.ndarray] = {}

    def fit(self, sentences: Sequence[Sequence[str]]) -> "NgramLM":
        n = self.order
        raw: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        grams: set[tuple[str, ...]] = set()
        for s in sentences:
            unknown = [g for g in s if g not in self.index]
            assert not unknown, f"glosses outside the vocabulary: {unknown}"
            toks = [BOS] * (n - 1) + list(s) + [EOS]
            for i in range(n - 1, len(toks)):
                ctx = tuple(toks[i - n + 1:i])
                raw[ctx][toks[i]] += 1
                for k in range(1, n):  # every shorter gram ending at i, for continuation counts
                    grams.add(tuple(toks[i - k:i + 1]))
        self.tables = {n: dict(raw)}
        for k in range(1, n):
            cont: dict[tuple[str, ...], Counter] = defaultdict(Counter)
            # continuation count of (ctx, w) at level k = distinct left extensions u of (u, ctx, w)
            for g in grams:
                if len(g) == k + 1:
                    cont[g[1:-1]][g[-1]] += 1
            self.tables[k] = dict(cont)
        self._memo.clear()
        return self

    def _level(self, k: int, ctx: tuple[str, ...]) -> np.ndarray:
        if k == 0:
            return np.full(len(self.vocab), 1.0 / len(self.vocab))
        key = (k, ctx)
        if key in self._memo:
            return self._memo[key]
        lower = self._level(k - 1, ctx[1:])
        cnt = self.tables.get(k, {}).get(ctx)
        if not cnt:
            out = lower
        else:
            total = sum(cnt.values())
            vec = np.zeros(len(self.vocab))
            for w, c in cnt.items():
                vec[self.index[w]] = max(c - self.discount, 0.0) / total
            out = vec + (self.discount * len(cnt) / total) * lower
        self._memo[key] = out
        return out

    def dist(self, history: Sequence[str]) -> np.ndarray:
        toks = [BOS] * (self.order - 1) + list(history)
        ctx = tuple(toks[len(toks) - (self.order - 1):]) if self.order > 1 else ()
        return self._level(self.order, ctx)

    def to_dict(self) -> dict:
        """The count tables as plain JSON, for a client-side port."""
        return {"format": "kn-ngram/1", "order": self.order, "discount": self.discount,
                "vocab": self.vocab,
                "tables": {str(k): {" ".join(ctx): dict(c) for ctx, c in t.items()}
                           for k, t in self.tables.items()}}


# ============================================================
# Held-out protocols
# ============================================================

def sentence_folds(ids: Sequence[str], themes: Sequence[str], k: int, seed: int) -> dict[str, int]:
    """``{sentence id: fold}``, ``k`` folds stratified by theme: each theme's
    sentences are shuffled (seeded) and dealt round-robin."""
    rng = np.random.default_rng(seed)
    by: dict[str, list[str]] = defaultdict(list)
    for i, t in zip(ids, themes):
        by[t].append(i)
    out: dict[str, int] = {}
    offset = 0
    for t in sorted(by):
        members = [by[t][j] for j in rng.permutation(len(by[t]))]
        for j, sid in enumerate(members):
            out[sid] = (j + offset) % k
        offset += len(members)
    return out


def theme_folds(ids: Sequence[str], themes: Sequence[str]) -> dict[str, int]:
    """``{sentence id: fold}`` with one fold per theme (sorted theme order):
    the predictor never sees the held-out sentence's topic."""
    order = {t: i for i, t in enumerate(sorted(set(themes)))}
    return {i: order[t] for i, t in zip(ids, themes)}


# ============================================================
# Intrinsic evaluation
# ============================================================

def next_gloss_metrics(model: NextGlossModel, sentences: Sequence[Sequence[str]],
                       ks: Sequence[int] = (1, 5, 10)) -> dict:
    """Score a predictor on held-out sentences.

    - ``perplexity`` over every position including the end-of-sentence
      token (standard LM perplexity);
    - ``top{k}``: share of *gloss* positions whose true next gloss is among
      the ``k`` most probable (from ``gloss_dist``, EOS excluded, since
      fusion only ever ranks glosses);
    - ``mrr``: mean reciprocal rank of the true next gloss;
    - ``first_top{k}`` / ``later_top{k}``: the same split into the first
      position (no history) and later ones, since the first gloss of a
      sentence is where a prior can help least.
    """
    nll, n_tok = 0.0, 0
    hits = {k: [] for k in ks}
    first = {k: [] for k in ks}
    rr = []
    idx = {g: i for i, g in enumerate(model.vocab)}
    for s in sentences:
        for i, g in enumerate(list(s) + [EOS]):
            p = model.dist(s[:i])
            nll -= math.log(max(float(p[idx[g]]), 1e-300))
            n_tok += 1
            if g == EOS:
                continue
            q = model.gloss_dist(s[:i])
            rank = int((q > q[idx[g]]).sum()) + 1
            rr.append(1.0 / rank)
            for k in ks:
                hits[k].append(rank <= k)
                if i == 0:
                    first[k].append(rank <= k)
    out = {"n_sentences": len(sentences), "n_positions": len(rr),
           "perplexity": math.exp(nll / max(n_tok, 1)), "mrr": float(np.mean(rr)) if rr else float("nan")}
    for k in ks:
        out[f"top{k}"] = float(np.mean(hits[k])) if hits[k] else float("nan")
        out[f"first_top{k}"] = float(np.mean(first[k])) if first[k] else float("nan")
        later = [h for h, f in zip(hits[k], _first_mask(sentences)) if not f]
        out[f"later_top{k}"] = float(np.mean(later)) if later else float("nan")
    return out


def _first_mask(sentences: Sequence[Sequence[str]]) -> list[bool]:
    return [i == 0 for s in sentences for i in range(len(s))]


def top_next(model: NextGlossModel, history: Sequence[str], k: int = 10) -> list[tuple[str, float]]:
    """The ``k`` most probable next tokens (EOS included, so "end of
    sentence" shows up when it is likely), with probabilities."""
    p = model.dist(history)
    order = np.argsort(-p)[:k]
    return [(model.vocab[i], float(p[i])) for i in order]
