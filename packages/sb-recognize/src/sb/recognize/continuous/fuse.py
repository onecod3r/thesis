"""Fused acceptance: the recognizer's per-segment evidence combined with a
next-gloss prior, plus rejection of long noise (TODO §12.6, stage 2).

The stream is first cut into candidate segments by a decoder that already
works (D3: runs of non-null frames; D1: boundary-head crossings). Each
segment has the recognizer's **vote** ``q`` over the glosses, the same
non-null-weighted mass ``decode.segment_vote`` uses. A rule then decides,
with the prior ``p = prior(history)`` from :mod:`sb.rescore.prior`, which
gloss (if any) to accept:

- ``none``: accept ``argmax q`` if ``q_max >= theta``. With ``theta=0`` and
  no length gate this is exactly D3/D1, which is the parity check.
- ``rescore``: shallow fusion ``f ∝ q · p^lam``, accept ``argmax f`` if its
  normalized score ``>= theta``.
- ``agree``: the user's rule (2026-09-24), "if the prediction and the current
  sign confidence match, accept". Accept the fused top gloss when it is
  among the prior's ``k`` most likely next glosses **and** the recognizer
  gives it at least ``theta_lo``. Strong visual evidence alone
  (``q_max >= theta_hi``) still accepts ``argmax q``, so the prior can never
  block a sign the model is sure of. Otherwise reject.

**Noise gate:** a segment longer than ``max_len`` frames is rejected
whatever its scores. "Noise spanning for more than a while" (fidgeting,
holding the hands up, talking) comes out of D3 as one long non-null run.
Real signs are rarely that long: the GISLR clip length p95 is 131 frames.
The sweep prices the real signs this loses.

A rejected segment leaves the history unchanged. An accepted one appends to
it. History is per stream: one GISLR-Sentences stream is one sentence.

Everything here is numpy on precomputed per-frame outputs (``gp``, ``bp``),
so the sweeps never rerun the model. Anything with a ``decide(seg, history,
prior)`` method can stand in for a :class:`Rule` (``sb.recognize.continuous.select``
adds scores other than ``q``). ``prior`` is any callable
``history (tuple of class ids) -> probabilities over the glosses``. This
module does not import ``sb.rescore``; the notebook adapts the predictors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

Emission = tuple[int, int, float]
Prior = Callable[[tuple[int, ...]], np.ndarray]


@dataclass(frozen=True)
class Segment:
    start: int
    end: int  # exclusive
    q: np.ndarray  # (C,) recognizer vote over glosses, sums to 1
    mass: float  # non-null probability mass (frames' worth)
    peak: np.ndarray | None = None  # (C,) each gloss's highest single-frame probability in the segment

    @property
    def length(self) -> int:
        return self.end - self.start


def vote(gp: np.ndarray, null_index: int) -> tuple[np.ndarray, float]:
    """``gp[s:e]`` -> (vote over glosses summing to 1, non-null mass). The
    argmax equals ``decode.segment_vote``'s class. The share is not its
    confidence: that divides by the mass, which leaves the vector short of
    1, and fusion needs a distribution."""
    w = 1.0 - gp[:, null_index]
    score = (w[:, None] * gp[:, :null_index]).sum(0)
    mass = float(w.sum())
    return score / max(float(score.sum()), 1e-12), mass


def segments_d3(gp: np.ndarray, nu: float, min_len: int, null_index: int) -> list[Segment]:
    """D3's segments: maximal runs with null probability below ``nu``, at
    least ``min_len`` frames."""
    sign = gp[:, null_index] < nu
    edges = np.flatnonzero(np.diff(np.r_[False, sign, False].astype(np.int8)))
    out = []
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s >= min_len:
            q, m = vote(gp[s:e], null_index)
            out.append(Segment(int(s), int(e), q, m, gp[s:e, :null_index].max(0)))
    return out


def segments_d1(gp: np.ndarray, bp: np.ndarray, beta: float, min_mass: float,
                null_index: int) -> list[Segment]:
    """D1's segments: from one boundary crossing (``bp`` rising through
    ``beta``) to the next, dropped when they hold less than ``min_mass``."""
    up = bp >= beta
    cross = np.flatnonzero(up[1:] & ~up[:-1]) + 1
    out, start = [], 0
    for t in cross:
        q, m = vote(gp[start:t + 1], null_index)
        if m >= min_mass:
            out.append(Segment(int(start), int(t) + 1, q, m, gp[start:t + 1, :null_index].max(0)))
        start = int(t) + 1
    return out


@dataclass(frozen=True)
class Rule:
    mode: str = "none"  # none | rescore | agree
    lam: float = 0.0  # prior weight (rescore, agree)
    theta: float = 0.0  # min accepted confidence (none: q; rescore: fused)
    k: int = 5  # agree: fused top gloss must be in the prior's top k
    theta_lo: float = 0.0  # agree: min recognizer share for a prior-backed accept
    theta_hi: float = 1.01  # agree: recognizer share that accepts without the prior
    max_len: int | None = None  # noise gate: reject longer segments

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def decide(self, seg: Segment, history: tuple[int, ...], prior: Prior | None) -> tuple[int, float] | None:
        return decide(seg, history, prior, self)


class Decider(Protocol):
    """One segment + the accepted history -> ``(class, confidence)`` or ``None``."""

    def decide(self, seg: Segment, history: tuple[int, ...], prior: Prior | None) -> tuple[int, float] | None: ...


def fused(q: np.ndarray, p: np.ndarray, lam: float) -> np.ndarray:
    """``q · p^lam``, normalized, computed in log space."""
    lf = np.log(np.maximum(q, 1e-12)) + lam * np.log(np.maximum(p, 1e-12))
    lf -= lf.max()
    f = np.exp(lf)
    return f / f.sum()


def decide(seg: Segment, history: tuple[int, ...], prior: Prior | None, rule: Rule) -> tuple[int, float] | None:
    """One segment -> ``(class, confidence)`` to accept, or ``None``."""
    if rule.max_len is not None and seg.length > rule.max_len:
        return None
    q = seg.q
    if rule.mode == "none":
        c = int(q.argmax())
        return (c, float(q[c])) if q[c] >= rule.theta else None
    assert prior is not None, f"rule {rule.mode!r} needs a prior"
    p = prior(history)
    f = fused(q, p, rule.lam)
    if rule.mode == "rescore":
        c = int(f.argmax())
        return (c, float(f[c])) if f[c] >= rule.theta else None
    if rule.mode == "agree":
        c_q = int(q.argmax())
        if q[c_q] >= rule.theta_hi:
            return c_q, float(q[c_q])
        c = int(f.argmax())
        in_top = int((p > p[c]).sum()) < rule.k
        if in_top and q[c] >= rule.theta_lo:
            return c, float(f[c])
        return None
    raise ValueError(f"unknown rule mode {rule.mode!r}")


def decode_fused(segments: list[Segment], prior: Prior | None, rule: Decider) -> list[Emission]:
    """Segments in stream order -> emissions ``(class, frame, confidence)``,
    committed at each accepted segment's last frame. ``rule`` is a
    :class:`Rule` or any other :class:`Decider`."""
    out: list[Emission] = []
    history: tuple[int, ...] = ()
    for seg in segments:
        d = rule.decide(seg, history, prior)
        if d is None:
            continue
        c, conf = d
        out.append((c, seg.end - 1, conf))
        history = history + (c,)
    return out


def noise_false_accepts(emissions: list[Emission], blocks: list[tuple[int, int]], slack: int = 15) -> dict:
    """How many injected noise blocks produced an emission: one committed at
    a frame in ``[start, end + slack)`` (D3 commits when a run ends, and a
    noise run can merge with the frames right after it)."""
    hit = 0
    n_em = 0
    for a, b in blocks:
        k = sum(1 for _, t, _ in emissions if a <= t < b + slack)
        n_em += k
        hit += k > 0
    return {"blocks": len(blocks), "blocks_accepted": hit, "emissions_in_noise": n_em}
