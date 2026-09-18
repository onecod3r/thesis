"""Custom DNN / LSTM / GRU for the landmark-importance notebook (TODO §3).

All three share one :class:`LandmarkAttention` gate as their first layer — a
learned scalar weight per landmark, applied identically regardless of what
comes after it (a per-frame MLP vs. a recurrent trunk). That is what makes the
landmark weight comparable *across* architectures: it is read directly off the
gate, not reverse-engineered from each architecture's own internal weight
shapes (a recurrent input-to-hidden matrix mixes all landmarks together and
has no per-landmark structure to read a weight off of otherwise).

- :class:`LandmarkDNN` — no recurrence: every frame is classified
  independently (the literal "one frame at a time" training the notebook's
  title cell describes), giving a memory-free baseline whose only source of
  temporal information is the finite-difference velocity/acceleration
  channels already baked into the input. At inference, per-frame predictions
  are combined into one video-level prediction by averaging softmax
  probabilities over frames (see the notebook's ``predict_dnn_video``).
- :class:`LandmarkRNN` — unidirectional (causal) LSTM or GRU, same
  pack/pad-and-read-last-valid-frame contract as
  ``sb.recognize.architectures.StreamingGRU``/``StreamingLSTM``, so it is
  streaming-viable by the same definition the rest of the repo uses.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sb.recognize.interp.features import (
    CHANNELS_PER_LANDMARK,
    FEATURE_DIM,
    PER_LANDMARK_DIM,
    RELATIONAL_DIM,
)

N_LANDMARKS = PER_LANDMARK_DIM // CHANNELS_PER_LANDMARK  # 543


class LandmarkAttention(nn.Module):
    """One learned scalar gate per landmark, ``sigmoid(logit)`` in (0, 1),
    multiplied onto that landmark's whole channel block.

    Initialized near-open (logit=2.0, sigmoid≈0.88) so training starts close
    to "every landmark contributes" rather than having to rediscover signal
    the gate suppressed at init. The learned ``logits`` (equivalently
    ``gate()``) is the primary landmark-importance signal this notebook
    reports — a direct, architecture-agnostic weight per landmark, unlike a
    post-hoc read of a recurrent weight matrix.
    """

    def __init__(self, n_landmarks: int = N_LANDMARKS, init_logit: float = 2.0):
        super().__init__()
        self.logits = nn.Parameter(torch.full((n_landmarks,), init_logit))

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def forward(self, per_landmark: torch.Tensor) -> torch.Tensor:
        """``per_landmark``: (..., n_landmarks, channels) -> same shape, gated."""
        g = self.gate().view(*([1] * (per_landmark.dim() - 2)), -1, 1)
        return per_landmark * g


def split_features(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., FEATURE_DIM) -> (per_landmark (..., 543, 10), relational (..., 12))."""
    per_landmark = flat[..., :PER_LANDMARK_DIM].reshape(
        *flat.shape[:-1], N_LANDMARKS, CHANNELS_PER_LANDMARK
    )
    relational = flat[..., PER_LANDMARK_DIM:]
    return per_landmark, relational


class LandmarkDNN(nn.Module):
    """Memory-free per-frame classifier: attention-gated landmarks -> flatten
    -> MLP -> per-frame logits. Trained on every frame independently
    (frame-level label = the video's label), so it never sees more than one
    frame's worth of state — the "one frame at a time" baseline."""

    def __init__(self, hidden_sizes: tuple[int, ...], num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.attn = LandmarkAttention()
        self.input_norm = nn.LayerNorm(FEATURE_DIM)
        layers: list[nn.Module] = []
        prev = FEATURE_DIM
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, T, FEATURE_DIM) -> per-frame logits (B, T, num_classes).
        Padded frames are classified too; the notebook masks them out of the
        loss/metrics using ``lengths`` (cheaper than packing for a model with
        no recurrent state to pack)."""
        per_landmark, relational = split_features(x)
        gated = self.attn(per_landmark).flatten(-2)  # (B, T, 5430)
        feats = self.input_norm(torch.cat([gated, relational], dim=-1))
        return self.head(self.trunk(feats))


class LandmarkRNN(nn.Module):
    """Unidirectional LSTM/GRU over attention-gated, projected landmark
    features — streaming-viable (causal, last-valid-frame readout), the same
    contract as ``sb.recognize.architectures.StreamingGRU``."""

    def __init__(
        self, cell: str, proj_size: int, hidden_size: int, num_layers: int,
        num_classes: int, dropout: float = 0.3,
    ):
        super().__init__()
        assert cell in ("gru", "lstm"), f"cell must be 'gru' or 'lstm', got {cell!r}"
        self.attn = LandmarkAttention()
        self.input_norm = nn.LayerNorm(FEATURE_DIM)
        self.proj = nn.Sequential(
            nn.Linear(FEATURE_DIM, proj_size), nn.LayerNorm(proj_size), nn.GELU()
        )
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            proj_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        per_landmark, relational = split_features(x)
        gated = self.attn(per_landmark).flatten(-2)
        feats = self.input_norm(torch.cat([gated, relational], dim=-1))
        proj = self.proj(feats)
        packed = pack_padded_sequence(proj, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1)).to(out.device)
        return self.head(out.gather(1, idx).squeeze(1))
