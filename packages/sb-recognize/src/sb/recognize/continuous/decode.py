"""Stream decoders for continuous models (TODO §12.3).

All of them turn a model's per-frame outputs into ``[(class, frame,
confidence), ...]``, the same emission format
:mod:`sb.recognize.sequences.metrics` scores, so the continuous models and
the §12.2 baselines are measured identically.

- :func:`decode_boundary` (D1): commit where the **boundary head** rises
  through ``beta``, emitting the gloss with the most non-null-weighted
  probability mass since the previous commit. No hold, no fixed length;
  this is the decoder the model was designed for.
- :func:`decode_boundary_reset` (D1r): the same, with the recurrent state
  reset after every commit (the user's "reset for the next sign").
- ``streaming.decode_stream(..., exclude=null_index)`` (D2): the user's
  literal loop (top gloss >= tau for ``hold`` frames -> emit + reset), now on
  a model trained per frame. Null can never be accepted.
- :func:`decode_null` (D3): commit at the end of every run of non-null
  frames. Cannot split back-to-back signs; there to show why the boundary
  head exists.
- :func:`decode_ctc` (D4): greedy CTC (per-frame argmax, repeats collapsed,
  null dropped), for the CTC-trained run.

D1, D3 and D4 are functions of one forward pass over the whole stream. D1r
and D2 restart the model at each commit, using the same per-stream forward
cache as ``decode_stream``.
"""

from __future__ import annotations

import numpy as np
import torch

Emission = tuple[int, int, float]


@torch.no_grad()
def frame_outputs(model, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """``(T, F)`` -> per-frame gloss+null probabilities ``(T, C + 1)`` and
    boundary probabilities ``(T,)``, from a fresh state."""
    model.eval()
    xb = x.unsqueeze(0).to(next(model.parameters()).device)
    gl, bl = model.forward_frames(xb)
    return (torch.softmax(gl.float(), -1).squeeze(0).cpu().numpy(),
            torch.sigmoid(bl.float()).squeeze(0).cpu().numpy())


def segment_vote(gp: np.ndarray, null_index: int) -> tuple[int, float, float]:
    """The gloss with the most non-null-weighted probability over ``gp``'s
    frames: ``(class, confidence, non-null mass)``. Confidence is that
    gloss's share of the mass."""
    w = 1.0 - gp[:, null_index]
    score = (w[:, None] * gp[:, :null_index]).sum(0)
    c = int(score.argmax())
    mass = float(w.sum())
    return c, float(score[c] / max(mass, 1e-9)), mass


def _crossings(bp: np.ndarray, beta: float) -> np.ndarray:
    """Frames where ``bp`` rises through ``beta`` (never frame 0)."""
    up = bp >= beta
    return np.flatnonzero(up[1:] & ~up[:-1]) + 1


def decode_boundary(gp: np.ndarray, bp: np.ndarray, beta: float, min_mass: float,
                    null_index: int) -> list[Emission]:
    """D1: one pass, no reset. A crossing with less than ``min_mass`` frames'
    worth of non-null probability since the last commit emits nothing (a
    boundary fired over null frames) but still moves the commit point."""
    out: list[Emission] = []
    start = 0
    for t in _crossings(bp, beta):
        c, conf, mass = segment_vote(gp[start:t + 1], null_index)
        if mass >= min_mass:
            out.append((c, int(t), conf))
        start = t + 1
    return out


def decode_boundary_reset(model, x: torch.Tensor, beta: float, min_mass: float, null_index: int,
                          cache: dict | None = None) -> list[Emission]:
    """D1r: as D1, but the model restarts from a fresh state after every
    commit. ``cache`` memoizes the forward from each restart frame."""

    def from_frame(p0: int):
        if cache is not None and p0 in cache:
            return cache[p0]
        o = frame_outputs(model, x[p0:])
        if cache is not None:
            cache[p0] = o
        return o

    out: list[Emission] = []
    pos, T = 0, len(x)
    while pos < T - 1:
        gp, bp = from_frame(pos)
        cross = _crossings(bp, beta)
        if not len(cross):
            break
        t = int(cross[0])
        c, conf, mass = segment_vote(gp[: t + 1], null_index)
        if mass >= min_mass:
            out.append((c, pos + t, conf))
        pos += t + 1
    return out


def decode_null(gp: np.ndarray, nu: float, min_len: int, null_index: int) -> list[Emission]:
    """D3: every maximal run of frames with null probability below ``nu``,
    at least ``min_len`` long, emits its segment vote at its last frame."""
    sign = gp[:, null_index] < nu
    out: list[Emission] = []
    edges = np.flatnonzero(np.diff(np.r_[False, sign, False].astype(np.int8)))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s >= min_len:
            c, conf, _ = segment_vote(gp[s:e], null_index)
            out.append((c, int(e - 1), conf))
    return out


def decode_ctc(gp: np.ndarray, null_index: int) -> list[Emission]:
    """D4: greedy CTC -- per-frame argmax, a new emission whenever the
    argmax changes to a non-null class."""
    top = gp.argmax(1)
    out: list[Emission] = []
    prev = null_index
    for t, c in enumerate(top):
        if c != prev and c != null_index:
            out.append((int(c), t, float(gp[t, c])))
        prev = c
    return out
