"""Frame-by-frame confidence readout and reset mechanics for causal models.

Supports the streaming-confidence-eval notebook (Plan 2 / TODO Backlog "Live,
per-frame-updating streaming prediction with reset-on-accept"). Two separate
concerns live here, deliberately kept apart:

- **Confidence readout**: `per_frame_probs` (fast, batch, used for bulk
  analysis) and `RecurrentSession` (the true incremental step-by-step API a
  live caller would actually use) must agree numerically -- the notebook's
  parity cell checks this before trusting either.
- **Reset**: `RecurrentSession.reset()` clears the model's own carried hidden
  state (the only architectures with one, `gru`/`lstm`); `AcceptTrigger` is a
  rule-based stand-in for whatever eventually decides WHEN to call it (a
  fixed threshold here, an LLM later) -- kept separate so the reset mechanism
  can be validated independently of who pulls the trigger.

`bilstm` is excluded everywhere in this module: it is offline-only by
construction (`sb.recognize.architectures.BiLSTM`'s docstring) and has no
causal per-frame readout to expose.
"""

from __future__ import annotations

import numpy as np
import torch

RECURRENT_ARCHS = ("gru", "lstm", "gru_continuous", "lstm_continuous",
                   "gru_phono", "gru_phono_raw", "gru_continuous_phono")  # *_phono: input_norm carries the front-end
FRAME_LOCAL_ARCHS = ("cnn", "dnn")
STREAMING_ARCHS = RECURRENT_ARCHS + FRAME_LOCAL_ARCHS


@torch.no_grad()
def per_frame_probs(model, arch: str, x: torch.Tensor) -> np.ndarray:
    """``(T, feature_dim)`` -> ``(T, num_classes)`` softmax probabilities, one
    vector per frame, using whatever causal readout the architecture has.
    Batch-1, unpadded -- an analysis tool (TODO Plan 2 Phase A), not the
    training or export contract.

    Every architecture here is trained with a loss at the LAST valid frame
    only (`LandmarkDNN`'s frame-masked loss is the one exception -- see its
    own docstring), so a mid-sequence confidence vector was never directly
    supervised for `gru`/`lstm`/`cnn`. Treat these as evidence, not ground
    truth, until a per-frame-supervised retrain (TODO Plan 2 Phase C).
    """
    if arch not in STREAMING_ARCHS:
        raise ValueError(
            f"{arch!r} has no causal per-frame readout to analyze here -- it is "
            f"either offline-only (bilstm/conv1d_transformer) or unsupported. "
            f"Supported: {STREAMING_ARCHS}"
        )
    model.eval()
    xb = x.unsqueeze(0).to(next(model.parameters()).device)  # (1, T, F)
    logits = model(xb) if arch == "dnn" else model.forward_all(xb)  # (1, T, C)
    return torch.softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()


class RecurrentSession:
    """True frame-by-frame incremental inference for a causal recurrent model
    (`gru`/`lstm` only -- the two architectures with an actual internal
    hidden state to carry and reset). One frame in, one confidence vector
    out; `reset()` drops the carried state, simulating the UX-facing "sign
    accepted, start fresh" event from the streaming Backlog item.

    This is the reference implementation of the actual live contract --
    Phase A/B's bulk analysis uses the faster `per_frame_probs` batch path
    instead and must cross-check it against this class for numerical parity
    (see the confidence-eval notebook's parity cell): a padding-free forward
    over a whole sequence and stepping through it one frame at a time are the
    same computation, and if they disagree one of the two is wrong.
    """

    def __init__(self, model, arch: str, device: torch.device | None = None):
        if arch not in RECURRENT_ARCHS:
            raise ValueError(f"RecurrentSession only supports {RECURRENT_ARCHS}, got {arch!r}")
        self.model = model.eval()
        self.arch = arch
        self.device = device or next(model.parameters()).device
        self._rnn = model.gru if hasattr(model, "gru") else model.lstm
        self.state = None

    def reset(self) -> None:
        """Drop the carried hidden state -- the next `step` behaves as if no
        frame had ever been seen."""
        self.state = None

    @torch.no_grad()
    def step(self, frame: torch.Tensor) -> np.ndarray:
        """One frame, shape ``(feature_dim,)``, -> that frame's confidence
        vector, shape ``(num_classes,)``. Applies the model's own
        `input_norm` (stateless LayerNorm, safe per-frame) then the recurrent
        cell with whatever state is currently carried."""
        x = self.model.input_norm(frame.to(self.device)).view(1, 1, -1)
        out, self.state = self._rnn(x, self.state)
        logits = self.model.head(out.squeeze(0).squeeze(0))
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()


def build_synthetic_stream(
    clips: list[np.ndarray],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Concatenate isolated-sign clips end-to-end into one synthetic
    continuous stream -- neither GISLR nor POPSIGN has real multi-sign
    sequences (the same gap TODO §8 flags for a sentence-context LLM), so
    this is the stand-in continuous-stream fixture for evaluating reset
    behavior (TODO Plan 2 Phase B).

    Returns the concatenated ``(T_total, feature_dim)`` array, the frame
    index where each clip STARTS (``boundaries[0] == 0`` always), and each
    clip's length in frames. No gap/silence is inserted between clips: an
    untrimmed transition is the harder, more honest case for a reset
    mechanism than a padded one.
    """
    boundaries, lengths, offset = [], [], 0
    for clip in clips:
        boundaries.append(offset)
        lengths.append(len(clip))
        offset += len(clip)
    return np.concatenate(clips, axis=0), boundaries, lengths


@torch.no_grad()
def clip_probs(model, arch: str, x: torch.Tensor) -> np.ndarray:
    """``(T, feature_dim)`` -> ``(num_classes,)``: the model's own whole-clip
    prediction, exactly as it was trained and evaluated -- the last-frame
    readout for every sequence model (``forward_full``, incl. the offline
    ``bilstm``), the softmax average over frames for the memory-free ``dnn``.
    Used for oracle-segmented decoding (TODO §12.2 mode B1)."""
    model.eval()
    xb = x.unsqueeze(0).to(next(model.parameters()).device)
    if arch == "dnn":
        return torch.softmax(model(xb).float(), dim=-1).mean(1).squeeze(0).cpu().numpy()
    return torch.softmax(model.forward_full(xb).float(), dim=-1).squeeze(0).cpu().numpy()


def first_accept(probs: np.ndarray, tau: float, hold: int, exclude: int | None = None) -> tuple[int, int] | None:
    """Vectorized :class:`AcceptTrigger` from a fresh streak: the first frame
    ``k`` at which the top class has stayed ``>= tau`` for ``hold``
    consecutive frames, as ``(k, class)``; ``None`` if it never does.

    Same rule as feeding ``probs`` row by row into a new ``AcceptTrigger``
    and stopping at its first return (checked in the sentence-baselines
    notebook), without a Python loop per frame -- the difference between
    minutes and hours for a threshold sweep over a whole dataset.

    ``exclude`` (e.g. the null column of a continuous model) is a class that
    can never be accepted: a frame whose top class it is counts as not held.
    """
    T = len(probs)
    if T == 0:
        return None
    top = probs.argmax(1)
    held = probs[np.arange(T), top] >= tau
    if exclude is not None:
        held &= top != exclude
    cont = np.zeros(T, bool)
    cont[1:] = held[1:] & held[:-1] & (top[1:] == top[:-1])
    idx = np.arange(T)
    run_start = np.maximum.accumulate(np.where(cont, 0, idx))
    streak = np.where(held, idx - run_start + 1, 0)
    hits = np.flatnonzero(streak >= hold)
    if not len(hits):
        return None
    k = int(hits[0])
    return k, int(top[k])


def decode_stream(
    model, arch: str, x: torch.Tensor, tau: float, hold: int, *, reset: bool = True,
    cache: dict | None = None, exclude: int | None = None,
) -> list[tuple[int, int, float]]:
    """Run a causal model over a whole continuous stream and emit a gloss
    every time the accept rule fires: ``[(class, frame, confidence), ...]``.

    ``reset=True`` is the reset-on-accept loop (TODO §12.2 mode B2): after an
    accept at frame ``k`` the model restarts from a fresh state at ``k + 1``,
    which is exactly ``RecurrentSession.reset()`` -- computed as a fresh
    batch forward from ``k + 1`` rather than stepping frame by frame, since
    the two agree to 1e-6 (§11.1). ``reset=False`` (mode B3) keeps the state
    and only clears the trigger's streak, as ``AcceptTrigger`` does after
    firing -- the no-reset contamination baseline.

    ``cache`` (one dict per stream, shared across calls) memoizes the
    fresh-state forward from each start frame: it depends only on where the
    state was reset, not on ``tau``/``hold``, so a threshold sweep over one
    stream reuses most of its forwards.
    """

    def from_frame(p0: int) -> np.ndarray:
        if cache is not None and p0 in cache:
            return cache[p0]
        p = per_frame_probs(model, arch, x[p0:])
        if cache is not None:
            cache[p0] = p
        return p

    out: list[tuple[int, int, float]] = []
    pos, T = 0, len(x)
    probs = from_frame(0)  # the no-reset stream; also the first reset segment
    while pos < T:
        p = from_frame(pos) if reset and pos > 0 else probs[pos:]
        hit = first_accept(p, tau, hold, exclude)
        if hit is None:
            break
        k, cls = hit
        out.append((cls, pos + k, float(p[k, cls])))
        pos += k + 1
    return out


@torch.no_grad()
def window_probs(
    model, arch: str, x: torch.Tensor, win: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Classify every ``win``-frame window of a stream (step ``stride``) as an
    isolated clip: ``(n_windows, num_classes)`` probabilities and each
    window's last frame. The one streaming decoder that works for models
    with no usable running state (``cnn``/``dnn``) and the offline ``bilstm``
    (TODO §12.2 mode B4). A stream shorter than ``win`` is one window."""
    T = len(x)
    starts = np.arange(0, max(T - win, 0) + 1, stride)
    ends = np.minimum(starts + win, T)
    xb = torch.stack([x[s:e] for s, e in zip(starts, ends)]).to(next(model.parameters()).device)
    model.eval()
    if arch == "dnn":
        p = torch.softmax(model(xb).float(), dim=-1).mean(1)
    else:
        p = torch.softmax(model.forward_full(xb).float(), dim=-1)
    return p.cpu().numpy(), ends - 1


def collapse_repeats(emissions: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """Drop an emission identical in class to the one just before it -- the
    usual CTC-style post-process. Suppresses a long sign being accepted over
    and over, at the cost of a genuinely repeated gloss (``WHO WHO``)."""
    out: list[tuple[int, int, float]] = []
    for e in emissions:
        if not out or out[-1][0] != e[0]:
            out.append(e)
    return out


def decode_windows(
    probs: np.ndarray, end_frames: np.ndarray, tau: float, min_run: int
) -> list[tuple[int, int, float]]:
    """Window predictions -> emitted glosses: a run of consecutive windows
    whose top class is the same and ``>= tau`` emits that class once, at the
    end frame of its ``min_run``-th window. Runs are broken by a window
    below ``tau`` or with a different top class, so a gloss repeated in the
    sentence can be emitted twice if a low-confidence gap separates them."""
    out: list[tuple[int, int, float]] = []
    top = probs.argmax(1)
    conf = probs[np.arange(len(probs)), top]
    run_cls, run_len = -1, 0
    for i, (c, p) in enumerate(zip(top, conf)):
        if p < tau:
            run_cls, run_len = -1, 0
            continue
        run_len = run_len + 1 if c == run_cls else 1
        run_cls = c
        if run_len == min_run:
            out.append((int(c), int(end_frames[i]), float(p)))
    return out


class AcceptTrigger:
    """Rule-based reset trigger: fires once the top class's confidence stays
    above `tau` for `hold_frames` consecutive frames.

    A minimal stand-in for TODO Plan 2 Phase D/E's reset decision-maker
    (rule-based here; an LLM later reads the same confidence stream and
    decides on its own criteria, e.g. matching a next-word suggestion). Kept
    as its own class so the reset MECHANISM (`RecurrentSession.reset`) can be
    validated independently of who decides to pull it.
    """

    def __init__(self, tau: float, hold_frames: int, exclude: int | None = None):
        self.tau = tau
        self.hold_frames = hold_frames
        self.exclude = exclude  # a class that can never be accepted (a continuous model's null)
        self._label: int | None = None
        self._streak = 0

    def update(self, probs: np.ndarray) -> int | None:
        """Feed one frame's probability vector; returns the accepted class
        index the moment its streak completes (and clears the streak), else
        None. The streak resets whenever the top class or the threshold
        crossing changes."""
        top = int(np.argmax(probs))
        held = probs[top] >= self.tau and top != self.exclude
        if held and top == self._label:
            self._streak += 1
        elif held:
            self._label, self._streak = top, 1
        else:
            self._label, self._streak = None, 0
        if self._streak >= self.hold_frames:
            accepted = self._label
            self._label, self._streak = None, 0
            return accepted
        return None
