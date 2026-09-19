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
- :class:`LandmarkBiLSTM` — bidirectional LSTM, same **offline-only**
  constraint and fwd-last/bwd-first readout as
  ``sb.recognize.architectures.BiLSTM`` (TODO §3.6): the backward pass reads
  future frames, so this can never be a deployment candidate; it exists to
  price what bidirectionality (and, here, engineered features) buy at the
  accuracy ceiling.
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


def split_features(
    flat: torch.Tensor,
    per_landmark_dim: int = PER_LANDMARK_DIM,
    n_landmarks: int = N_LANDMARKS,
    channels_per_landmark: int = CHANNELS_PER_LANDMARK,
    angle_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(..., feature_dim) -> (per_landmark (..., n_landmarks, channels),
    angles (..., angle_dim), relational (..., remainder)).

    Column order ``[per_landmark, angles, relational]`` matches
    ``features_curated.build_frame_features``. Default args reproduce
    ``landmark_interp_v1``'s layout exactly (``angle_dim=0`` -> an empty
    angle slice, ``relational`` gets everything after ``per_landmark``) —
    every existing caller that doesn't pass these keeps working unchanged.
    """
    per_landmark = flat[..., :per_landmark_dim].reshape(
        *flat.shape[:-1], n_landmarks, channels_per_landmark
    )
    rest = flat[..., per_landmark_dim:]
    angles, relational = rest[..., :angle_dim], rest[..., angle_dim:]
    return per_landmark, angles, relational


class LandmarkDNN(nn.Module):
    """Memory-free per-frame classifier: attention-gated landmarks -> flatten
    -> MLP -> per-frame logits. Trained on every frame independently
    (frame-level label = the video's label), so it never sees more than one
    frame's worth of state — the "one frame at a time" baseline.

    Dimension args default to ``landmark_interp_v1``'s shape (full 543
    landmarks x 10 channels, no angle block, 12 relational) — pass explicit
    values for a different pipeline (e.g. ``features_curated``'s 126
    landmarks x 7 channels + 28 angles + 12 relational).
    """

    def __init__(
        self, hidden_sizes: tuple[int, ...], num_classes: int, dropout: float = 0.3,
        n_landmarks: int = N_LANDMARKS, channels_per_landmark: int = CHANNELS_PER_LANDMARK,
        angle_dim: int = 0, relational_dim: int = RELATIONAL_DIM,
    ):
        super().__init__()
        self.n_landmarks = n_landmarks
        self.channels_per_landmark = channels_per_landmark
        self.per_landmark_dim = n_landmarks * channels_per_landmark
        self.angle_dim = angle_dim
        feature_dim = self.per_landmark_dim + angle_dim + relational_dim
        self.attn = LandmarkAttention(n_landmarks=n_landmarks)
        self.input_norm = nn.LayerNorm(feature_dim)
        layers: list[nn.Module] = []
        prev = feature_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, T, feature_dim) -> per-frame logits (B, T, num_classes).
        Padded frames are classified too; callers mask them out of the
        loss/metrics using ``lengths`` (cheaper than packing for a model with
        no recurrent state to pack). This is already the "confidence score
        per frame, over every label" output a live/streaming reader would
        want — apply ``softmax(dim=-1)`` to get probabilities."""
        per_landmark, angles, relational = split_features(
            x, self.per_landmark_dim, self.n_landmarks, self.channels_per_landmark, self.angle_dim
        )
        gated = self.attn(per_landmark).flatten(-2)
        feats = self.input_norm(torch.cat([gated, angles, relational], dim=-1))
        return self.head(self.trunk(feats))


class LandmarkRNN(nn.Module):
    """Unidirectional LSTM/GRU over attention-gated, projected landmark
    features — streaming-viable (causal, last-valid-frame readout), the same
    contract as ``sb.recognize.architectures.StreamingGRU``.

    Dimension args default to ``landmark_interp_v1``'s shape, same as
    :class:`LandmarkDNN` — see its docstring.
    """

    def __init__(
        self, cell: str, proj_size: int, hidden_size: int, num_layers: int,
        num_classes: int, dropout: float = 0.3,
        n_landmarks: int = N_LANDMARKS, channels_per_landmark: int = CHANNELS_PER_LANDMARK,
        angle_dim: int = 0, relational_dim: int = RELATIONAL_DIM,
    ):
        super().__init__()
        assert cell in ("gru", "lstm"), f"cell must be 'gru' or 'lstm', got {cell!r}"
        self.n_landmarks = n_landmarks
        self.channels_per_landmark = channels_per_landmark
        self.per_landmark_dim = n_landmarks * channels_per_landmark
        self.angle_dim = angle_dim
        feature_dim = self.per_landmark_dim + angle_dim + relational_dim
        self.attn = LandmarkAttention(n_landmarks=n_landmarks)
        self.input_norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, proj_size), nn.LayerNorm(proj_size), nn.GELU()
        )
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            proj_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes)
        )

    def _encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """``x``: (B, T, feature_dim) -> unpacked RNN hidden states (B, T,
        hidden_size), shared by :meth:`forward` (reads the last valid frame)
        and :meth:`forward_all` (reads every frame)."""
        per_landmark, angles, relational = split_features(
            x, self.per_landmark_dim, self.n_landmarks, self.channels_per_landmark, self.angle_dim
        )
        gated = self.attn(per_landmark).flatten(-2)
        feats = self.input_norm(torch.cat([gated, angles, relational], dim=-1))
        proj = self.proj(feats)
        packed = pack_padded_sequence(proj, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return out

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Last-valid-frame logits (B, num_classes) — what training/the
        canonical streaming contract uses; unchanged behavior."""
        out = self._encode(x, lengths)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1)).to(out.device)
        return self.head(out.gather(1, idx).squeeze(1))

    def forward_all(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Per-frame logits (B, T, num_classes) — the same trained head
        applied to every timestep's hidden state, not just the last one.

        Purely a read-out choice: the model is still trained with a loss at
        the last valid frame only (:meth:`forward`), so frames before the
        sign is mostly complete were never supervised to have a meaningful
        confidence vector. Use this for inspecting how confidence evolves
        frame-by-frame, not as a claim that early-frame confidence is
        reliable — callers must mask past each sequence's own ``lengths``,
        same convention ``LandmarkDNN``/``run_epoch_dnn`` uses."""
        out = self._encode(x, lengths)
        return self.head(out)


class LandmarkBiLSTM(nn.Module):
    """Bidirectional LSTM over attention-gated, projected landmark features —
    **OFFLINE-ONLY**, same constraint as ``sb.recognize.architectures.BiLSTM``:
    the backward pass reads future frames, so this can never be a deployment
    candidate. Readout: forward direction at the last valid frame + backward
    direction at t=0 (which has seen the whole sequence), concatenated ->
    ``2*hidden_size`` head — identical convention to the production class,
    just with the attention gate + generalized dims every other model in this
    module has (`LandmarkDNN`/`LandmarkRNN`'s docstrings explain the pattern).
    """

    def __init__(
        self, proj_size: int, hidden_size: int, num_layers: int,
        num_classes: int, dropout: float = 0.3,
        n_landmarks: int = N_LANDMARKS, channels_per_landmark: int = CHANNELS_PER_LANDMARK,
        angle_dim: int = 0, relational_dim: int = RELATIONAL_DIM,
    ):
        super().__init__()
        self.n_landmarks = n_landmarks
        self.channels_per_landmark = channels_per_landmark
        self.per_landmark_dim = n_landmarks * channels_per_landmark
        self.angle_dim = angle_dim
        feature_dim = self.per_landmark_dim + angle_dim + relational_dim
        self.attn = LandmarkAttention(n_landmarks=n_landmarks)
        self.input_norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, proj_size), nn.LayerNorm(proj_size), nn.GELU()
        )
        self.lstm = nn.LSTM(
            proj_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden_size), nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        per_landmark, angles, relational = split_features(
            x, self.per_landmark_dim, self.n_landmarks, self.channels_per_landmark, self.angle_dim
        )
        gated = self.attn(per_landmark).flatten(-2)
        feats = self.input_norm(torch.cat([gated, angles, relational], dim=-1))
        proj = self.proj(feats)
        packed = pack_padded_sequence(proj, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, T, 2H)
        H = out.size(-1) // 2
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, H).to(out.device)
        fwd_last = out[..., :H].gather(1, idx).squeeze(1)  # fwd state at last valid frame
        bwd_first = out[:, 0, H:]  # bwd state at t=0 (saw everything)
        return self.head(torch.cat([fwd_last, bwd_first], dim=-1))
