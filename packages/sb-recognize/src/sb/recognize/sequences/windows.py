"""Window and multi-model decoding for continuous signing (TODO §12.8, the user's proposal of 2026-09-26).

The user's idea: instead of a model that reads the stream frame by frame, feed fixed windows of about a
sign's length (1-3 s) to **two models offset from each other by ~500 ms**, combine their predictions, and
fuse the result with an independent next-gloss model. This module builds every piece so the idea can be
scored next to C1 on the same streams, plus the two variants Claude proposed:

- **windows** (:func:`window_probs`, :func:`combine_offset`, :func:`window_segments`): an isolated-sign
  model reads each window from a fresh state; its last-frame readout is the window's prediction. Two
  models (or one model twice) at an offset give a merged, time-ordered series of predictions, each the
  average of the two models' latest windows. Runs of the same confident top gloss become one segment;
- **per-frame ensembles** (:func:`average`): two continuous models' per-frame gloss+null probabilities
  averaged, then decoded like one model (D3);
- **staggered restarts** (:func:`staggered`): two copies of one recurrent model, each restarted from zero
  every ``period`` frames, the second offset by half a period, combined per frame. One copy always has
  recent context, and no single restart decides a sign.

Segments are :class:`sb.recognize.continuous.fuse.Segment`, so the same fusion rules (next-gloss prior,
thresholds) apply to every arm unchanged.
"""

from __future__ import annotations

import numpy as np
import torch

from sb.recognize.continuous.fuse import Segment

# Defaults mirror `apps/web/src/pipeline/movement.ts`'s `MovementGate` exactly (TODO
# §16.1/§16.2) -- these are the numbers the live browser heuristic actually uses, not a
# separately-tuned offline choice, so `movement_segments` below tests the real thing.
MOVEMENT_POINT_NOISE_EPS = 0.0015
MOVEMENT_SMOOTHING_ALPHA = 0.3
MOVEMENT_STILL_THRESHOLD = 0.003
MOVEMENT_STILL_FRAMES = 10


def movement_segments(x: np.ndarray, min_len: int = 1,
                      point_noise_eps: float = MOVEMENT_POINT_NOISE_EPS,
                      smoothing_alpha: float = MOVEMENT_SMOOTHING_ALPHA,
                      still_threshold: float = MOVEMENT_STILL_THRESHOLD,
                      still_frames: int = MOVEMENT_STILL_FRAMES) -> list[tuple[int, int]]:
    """Cut a stream into candidate-sign spans, the offline (whole-clip-in-hand) sibling of
    `movement.ts`'s live per-frame gate -- for the record-then-recognize BiLSTM mode
    (§16.2), whose only segmentation signal is stillness (unlike the continuous GRU
    family, there is no learned null/boundary head here at all).

    `x` is `(T, F)`, `F = 2 * n_landmarks` (xy pairs, already NaN -> 0, the model's own
    input convention -- no z, so nothing to drop here unlike the browser side, which
    drops z from a raw (543, 3) frame). The frame-to-frame math (sub-jitter clamp, EMA,
    streak count) is a literal port of `MovementGate.push`, frame by frame, so behaviour
    matches the browser's rather than merely resembling it.

    A cut lands at the frame where a `still_frames`-long stillness streak is first
    satisfied (the frame the live gate would force a run closed on); segments are the
    spans between consecutive cuts (and clip start/end), each at least `min_len` long --
    shorter ones (e.g. an opening stillness run with nothing before it) are dropped."""
    T = x.shape[0]
    if T < 2:
        return [(0, T)] if T >= min_len else []
    xy = x.reshape(T, -1, 2)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=-1)  # (T-1, n_landmarks)
    d = np.where(d < point_noise_eps, 0.0, d)
    per_frame = d.mean(axis=1)  # (T-1,), frame t's movement vs frame t-1

    ema, streak = 0.0, 0
    cuts = []
    for t, m in enumerate(per_frame, start=1):  # t = the frame just arrived (1..T-1)
        ema = smoothing_alpha * float(m) + (1.0 - smoothing_alpha) * ema
        streak = streak + 1 if ema < still_threshold else 0
        if streak == still_frames:  # the instant `MovementGate.push` would first return True
            cuts.append(t)

    bounds = [0, *cuts, T]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] - bounds[i] >= min_len]


@torch.no_grad()
def window_probs(model, x: torch.Tensor, win: int, stride: int, offset: int = 0,
                 batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Windows of ``win`` frames ending every ``stride`` frames from ``offset + win - 1`` -> ``(ends,
    probs (n, C))``: each window read from a fresh state at its last frame (``forward_full``). A stream
    shorter than ``offset + win`` gets one window over all of it."""
    T = len(x)
    ends = np.arange(offset + win - 1, T, stride)
    if len(ends) == 0:
        ends = np.array([T - 1])
        wins = x.unsqueeze(0)
    else:
        idx = torch.as_tensor(ends[:, None] - np.arange(win - 1, -1, -1)[None], device=x.device)
        wins = x[idx]  # (n, win, F)
    dev = next(model.parameters()).device
    out = [torch.softmax(model.forward_full(wins[i:i + batch].to(dev)).float(), -1).cpu()
           for i in range(0, len(wins), batch)]
    return ends, torch.cat(out).numpy()


def combine_offset(ends_a: np.ndarray, pa: np.ndarray, ends_b: np.ndarray, pb: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Two window series -> one, time-ordered: at every window end of either model, the mean of each
    model's latest window so far (only one model's, until the other has produced a window)."""
    ev = sorted([(int(t), 0, i) for i, t in enumerate(ends_a)] + [(int(t), 1, i) for i, t in enumerate(ends_b)])
    last: list[np.ndarray | None] = [None, None]
    src = (pa, pb)
    ends, probs = [], []
    for t, m, i in ev:
        last[m] = src[m][i]
        have = [v for v in last if v is not None]
        ends.append(t)
        probs.append(np.mean(have, axis=0))
    return np.asarray(ends), np.stack(probs)


def window_segments(ends: np.ndarray, probs: np.ndarray, tau: float, win: int, min_run: int = 1
                    ) -> list[Segment]:
    """Runs of consecutive predictions with the same top gloss at probability >= ``tau`` (at least
    ``min_run`` of them) -> segments from the first window's start to the last window's end, with the
    run's mean prediction as the vote ``q``."""
    top = probs.argmax(1)
    ok = probs.max(1) >= tau
    out, i, n = [], 0, len(ends)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1] and top[j + 1] == top[i]:
            j += 1
        if j - i + 1 >= min_run:
            q = probs[i:j + 1].mean(0)
            out.append(Segment(max(int(ends[i]) - win + 1, 0), int(ends[j]) + 1, q / q.sum(),
                               float(j - i + 1), probs[i:j + 1].max(0)))
        i = j + 1
    return out


def average(*gps: np.ndarray) -> np.ndarray:
    """Per-frame probabilities of several models over the same stream, averaged."""
    return np.mean(np.stack(gps), axis=0)


@torch.no_grad()
def restarted(model, x: torch.Tensor, period: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame gloss+null probabilities of one copy that starts at frame 0 and restarts from zero
    state at ``offset`` (or ``period`` when ``offset`` is 0) and every ``period`` frames after, plus
    each frame's age (frames since that copy's last restart)."""
    T = len(x)
    cuts = [0] + [c for c in range(offset or period, T, period) if c > 0] + [T]
    dev = next(model.parameters()).device
    parts, age = [], np.empty(T, np.int64)
    for s, e in zip(cuts[:-1], cuts[1:]):
        gl, _ = model.forward_frames(x[s:e].unsqueeze(0).to(dev))
        parts.append(torch.softmax(gl.float(), -1).squeeze(0).cpu().numpy())
        age[s:e] = np.arange(e - s)
    return np.concatenate(parts), age


def staggered(model, x: torch.Tensor, period: int, combine: str = "mean") -> np.ndarray:
    """Two copies restarted every ``period`` frames, half a period apart, combined per frame:
    ``mean`` of both, or ``older`` = the copy that has run longer since its restart (it has context
    while the other is warming up)."""
    ga, aa = restarted(model, x, period, 0)
    gb, ab = restarted(model, x, period, period // 2)
    if combine == "mean":
        return (ga + gb) / 2
    if combine == "older":
        return np.where((aa >= ab)[:, None], ga, gb)
    raise ValueError(f"unknown combine {combine!r}")
