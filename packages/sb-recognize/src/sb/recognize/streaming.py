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

RECURRENT_ARCHS = ("gru", "lstm")
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
        self._rnn = model.gru if arch == "gru" else model.lstm
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


class AcceptTrigger:
    """Rule-based reset trigger: fires once the top class's confidence stays
    above `tau` for `hold_frames` consecutive frames.

    A minimal stand-in for TODO Plan 2 Phase D/E's reset decision-maker
    (rule-based here; an LLM later reads the same confidence stream and
    decides on its own criteria, e.g. matching a next-word suggestion). Kept
    as its own class so the reset MECHANISM (`RecurrentSession.reset`) can be
    validated independently of who decides to pull it.
    """

    def __init__(self, tau: float, hold_frames: int):
        self.tau = tau
        self.hold_frames = hold_frames
        self._label: int | None = None
        self._streak = 0

    def update(self, probs: np.ndarray) -> int | None:
        """Feed one frame's probability vector; returns the accepted class
        index the moment its streak completes (and clears the streak), else
        None. The streak resets whenever the top class or the threshold
        crossing changes."""
        top = int(np.argmax(probs))
        held = probs[top] >= self.tau
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
