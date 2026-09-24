"""Accepting more real signs than a single confidence floor, at no more wrong
or extra signs (TODO §12.6, floor-recall experiment, 2026-09-24).

A floor ``q_max >= theta`` (:mod:`sb.recognize.continuous.fuse`) does two jobs
with one number: it rejects **noise** (fidgets, held hands) and it rejects
**uncertain real signs**. On the selection signers' clean sentences the θ=0.3
floor threw away 511 correct segments (10.4% of the signs) to drop 1,097
wrong ones. The true sign of a rejected segment was in ``q``'s top 5 60% of
the time. This module holds the alternatives the experiment compares, all at
the floor's error budget:

- :class:`Acceptor`: the same one-segment decision with another **score**.
  ``q`` (the floor itself, fused with the prior when ``lam > 0``), ``peak``
  (the gloss's highest single-frame probability), ``margin`` (top-1 minus
  top-2), ``qp`` (geometric mean of ``q`` and ``peak``), ``class`` (the floor
  shifted per gloss by how often that gloss is right, :func:`class_shifts`),
  and ``linear`` (a small logistic regression over :data:`FEATURES`,
  :class:`LogReg`).
- :class:`Lattice`: **wait before deciding.** An uncertain segment is held
  until ``lag`` more segments have closed. The best path over each held
  segment's top-``k`` glosses or "skip" is then chosen with the prior on both
  sides, and only the oldest held segment is committed. ``lag=0`` is a greedy
  rescore with an unnormalized floor.

Everything is a :class:`~sb.recognize.continuous.fuse.Decider` (``Acceptor``)
or has ``decode``/``runner`` (``Lattice``), so batch decoding and
:class:`~sb.recognize.continuous.online.OnlineDecoder` share one code path.
The client ports the same objects. ``LogReg`` exports to a dict of plain
numbers, one dot product per segment.

Emission frames stay the segment's own last frame even when the lattice
commits later. Alignment and noise attribution depend on position, not on
when the decision was made. :func:`decision_delays` reports the extra wait.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from sb.recognize.continuous.fuse import Emission, Prior, Segment, fused
from sb.recognize.sequences import metrics as M

SCORES = ("q", "peak", "margin", "qp", "class", "linear")
FEATURES = ("log_q", "margin", "log_peak", "entropy", "null_mean", "log_len",
            "log_prior", "prior_rank", "hist_len", "q_agrees")


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


# ---------------------------------------------------------------- features


def features(seg: Segment, c: int, f: np.ndarray, p: np.ndarray | None, history: tuple[int, ...]) -> np.ndarray:
    """:data:`FEATURES` for accepting class ``c`` of ``seg``. ``f`` is the
    distribution ``c`` was picked from (``q``, or ``q`` fused with the prior).
    ``p`` is the prior's next-gloss distribution (``None``: uniform). The
    prior terms are relative to uniform, so they are 0 with no prior."""
    q = seg.q
    n = len(q)
    rest = np.delete(f, c)
    peak = seg.peak[c] if seg.peak is not None else q[c]
    if p is None:
        log_prior, rank = 0.0, 0.0
    else:
        log_prior = float(np.log(max(p[c] * n, 1e-12)))
        rank = float(np.log1p((p > p[c]).sum()))
    return np.array([
        np.log(max(f[c], 1e-12)),
        f[c] - rest.max(),
        np.log(max(float(peak), 1e-12)),
        float(-(q * np.log(q + 1e-12)).sum()),
        1.0 - seg.mass / max(seg.length, 1),
        np.log(seg.length),
        log_prior,
        rank,
        float(min(len(history), 3)),
        float(q.argmax() == c),
    ])


@dataclass
class LogReg:
    """L2 logistic regression on standardized features (Newton steps).
    ``prob(x)`` is the estimated probability that accepting is correct."""
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    b: float
    names: tuple[str, ...] = FEATURES

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> LogReg:
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = np.c_[(X - mu) / sd, np.ones(len(X))]
        beta = np.zeros(Z.shape[1])
        reg = np.full(Z.shape[1], l2)
        reg[-1] = 0.0  # no penalty on the bias
        for _ in range(iters):
            pr = 1 / (1 + np.exp(-Z @ beta))
            g = Z.T @ (pr - y) + reg * beta
            H = (Z * (pr * (1 - pr))[:, None]).T @ Z + np.diag(reg)
            step = np.linalg.solve(H, g)
            beta -= step
            if np.abs(step).max() < 1e-8:
                break
        return cls(mu, sd, beta[:-1], float(beta[-1]))

    def prob(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(((x - self.mu) / self.sd) @ self.w + self.b)))

    def to_dict(self) -> dict:
        return {"features": list(self.names), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "w": self.w.tolist(), "b": self.b}


# ---------------------------------------------------------------- labels


def label_segments(segments: list[Segment], labels) -> list[dict]:
    """Each segment's ``argmax q`` aligned (Levenshtein, as GER does) against
    the reference, with every segment accepted and nothing collapsed:
    ``{"pred", "op" (match/sub/ins), "true" (reference class for match/sub,
    else -1)}``. Training labels for :class:`LogReg`, :func:`class_shifts`
    and the diagnostics; every reported number comes from real decoding."""
    hyp = [int(s.q.argmax()) for s in segments]
    ref = [int(x) for x in labels]
    out = [{"pred": h, "op": M.INS, "true": -1} for h in hyp]
    for op, i, j in M.align(ref, hyp):
        if j is not None:
            out[j]["op"] = op
            out[j]["true"] = ref[i] if i is not None else -1
    return out


def class_shifts(preds: np.ndarray, correct: np.ndarray, n_classes: int, alpha: float = 20.0) -> np.ndarray:
    """Per-gloss log-odds shift: how much more (or less) often a segment voted
    as gloss ``c`` is right than the average segment, shrunk to 0 with
    ``alpha`` pseudo-segments at the global rate (few examples -> no shift)."""
    r = float(correct.mean())
    n = np.bincount(preds, minlength=n_classes).astype(float)
    k = np.bincount(preds, weights=correct.astype(float), minlength=n_classes)
    return _logit((k + alpha * r) / (n + alpha)) - _logit(r)


# ---------------------------------------------------------------- greedy acceptors


@dataclass(frozen=True)
class Acceptor:
    """Accept ``argmax`` of ``q`` (or of ``q`` fused with the prior at weight
    ``lam``) when its ``score`` reaches ``theta``. ``score="q"`` with
    ``lam=0`` is the plain floor, and with ``lam>0`` it is ``fuse``'s
    ``rescore``. For ``class`` and ``linear``, ``theta`` is a probability,
    so it reads like the plain floor's."""
    score: str = "q"
    theta: float = 0.0
    lam: float = 0.0
    gamma: float = 1.0  # class: weight on the per-gloss shift
    max_len: int | None = None
    shifts: np.ndarray | None = field(default=None, repr=False, compare=False)
    model: LogReg | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        assert self.score in SCORES, self.score
        assert self.score != "class" or self.shifts is not None, "class score needs shifts"
        assert self.score != "linear" or self.model is not None, "linear score needs a model"

    def as_dict(self) -> dict:
        return {"score": self.score, "theta": self.theta, "lam": self.lam, "gamma": self.gamma,
                "max_len": self.max_len}

    def pick(self, seg: Segment, history: tuple[int, ...], prior: Prior | None):
        """``(class, distribution it was picked from, prior or None)``."""
        if self.lam > 0 and prior is not None:
            p = prior(history)
            f = fused(seg.q, p, self.lam)
        else:
            p, f = (prior(history) if (prior is not None and self.score == "linear") else None), seg.q
        return int(f.argmax()), f, p

    def value(self, seg: Segment, c: int, f: np.ndarray, p: np.ndarray | None, history: tuple[int, ...]) -> float:
        if self.score == "q":
            return float(f[c])
        peak = float(seg.peak[c]) if seg.peak is not None else float(f[c])
        if self.score == "peak":
            return peak
        if self.score == "margin":
            return float(f[c] - np.delete(f, c).max())
        if self.score == "qp":
            return float(np.sqrt(f[c] * peak))
        if self.score == "class":
            assert self.shifts is not None
            return float(1 / (1 + np.exp(-(_logit(f[c]) + self.gamma * self.shifts[c]))))
        assert self.model is not None
        return float(self.model.prob(features(seg, c, f, p, history)[None])[0])

    def decide(self, seg: Segment, history: tuple[int, ...], prior: Prior | None) -> tuple[int, float] | None:
        if self.max_len is not None and seg.length > self.max_len:
            return None
        c, f, p = self.pick(seg, history, prior)
        return (c, float(f[c])) if self.value(seg, c, f, p, history) >= self.theta else None


# ---------------------------------------------------------------- lattice with look-ahead


@dataclass(frozen=True)
class Lattice:
    """Fixed-lag decoding over segments. Accepting gloss ``c`` for a segment
    gains ``log q_c + lam·log(C·p(c | path so far)) - log theta``; skipping it
    gains 0. With ``lag=0`` and no prior this is the plain floor ``q_c >=
    theta``. With ``lag=L`` a segment is decided once ``L`` later segments
    have closed, over the best path through all of them, so a sign's
    successor can vouch for it."""
    theta: float = 0.3
    lam: float = 0.3
    k: int = 5
    lag: int = 1
    max_len: int | None = None

    def as_dict(self) -> dict:
        return {"score": "lattice", "theta": self.theta, "lam": self.lam, "k": self.k, "lag": self.lag,
                "max_len": self.max_len}

    def runner(self, prior: Prior | None) -> LatticeRunner:
        return LatticeRunner(self, prior)

    def decode(self, segments: list[Segment], prior: Prior | None) -> list[Emission]:
        run = self.runner(prior)
        out = [e for seg in segments for e in run.push(seg)]
        return out + run.flush()


class LatticeRunner:
    """The streaming state of a :class:`Lattice`: accepted history and the
    segments still waiting. ``push`` returns at most one emission (the oldest
    waiting segment, once ``lag`` segments follow it); ``flush`` decides the
    rest at the end of the sentence."""

    def __init__(self, lat: Lattice, prior: Prior | None):
        self.lat, self.prior = lat, prior
        self.history: tuple[int, ...] = ()
        self.pending: list[tuple[Segment, list[tuple[int, float]]]] = []

    def _options(self, seg: Segment) -> list[tuple[int, float]]:
        """``(class, log q_c - log theta)`` for the top-k glosses; empty when
        the noise gate rejects the segment outright."""
        if self.lat.max_len is not None and seg.length > self.lat.max_len:
            return []
        top = np.argsort(-seg.q)[: self.lat.k]
        lt = np.log(self.lat.theta)
        return [(int(c), float(np.log(max(seg.q[c], 1e-12)) - lt)) for c in top]

    def _prior_gain(self, hist: tuple[int, ...], c: int) -> float:
        if self.prior is None or self.lat.lam == 0:
            return 0.0
        p = self.prior(hist)
        return self.lat.lam * float(np.log(max(p[c] * len(p), 1e-12)))

    def _best(self) -> list[int | None]:
        """The best decision (class or ``None`` = skip) for every pending segment."""
        best, best_path = -np.inf, None
        for path in itertools.product(*[[None] + opts for _, opts in self.pending]):
            hist, total = self.history, 0.0
            for opt in path:
                if opt is None:
                    continue
                c, g = opt
                total += g + self._prior_gain(hist, c)
                hist = hist + (c,)
            if total > best:
                best, best_path = total, path
        assert best_path is not None
        return [None if o is None else o[0] for o in best_path]

    def _commit(self, choice: int | None) -> list[Emission]:
        seg, _ = self.pending.pop(0)
        if choice is None:
            return []
        self.history = self.history + (choice,)
        return [(choice, seg.end - 1, float(seg.q[choice]))]

    def push(self, seg: Segment) -> list[Emission]:
        self.pending.append((seg, self._options(seg)))
        if len(self.pending) <= self.lat.lag:
            return []
        return self._commit(self._best()[0])

    def flush(self) -> list[Emission]:
        out: list[Emission] = []
        while self.pending:
            out += self._commit(self._best()[0])
        return out


def decision_delays(segments: list[Segment], lag: int) -> list[int]:
    """Frames between a segment's end and the end of the segment that lets
    it be decided (``lag`` segments later). The last segments of a stream are
    decided at the sentence's end, counted here as the last segment's end."""
    n = len(segments)
    return [segments[min(i + lag, n - 1)].end - s.end for i, s in enumerate(segments)]
