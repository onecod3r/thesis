"""The GISLR benchmark architectures — the single definition of every model
class, loaded both by the training notebooks and by the canonical eval script
(modules/scripts/eval_gru.py), so state_dicts can never drift between the two.

All models share the same contract: ``forward(x, lengths)`` with
``x (B, T, F)`` zero-padded and ``lengths`` sorted descending (the collate in
``modules.model.data`` enforces this), returning ``(B, num_classes)`` logits
read out at the last valid frame.

Streaming viability is a per-architecture fact recorded in ``ARCHS`` — the
deployment path only ever uses ``streaming=True`` models; BiLSTM exists purely
to price the causality gap (project constraint, README §Constraints).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class StreamingGRU(nn.Module):
    """Unidirectional (causal) GRU — the deployment architecture."""

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, lengths):
        x = self.input_norm(x)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1)).to(out.device)
        return self.head(out.gather(1, idx).squeeze(1))

    def forward_full(self, x):
        out, _ = self.gru(self.input_norm(x))
        return self.head(out[:, -1])


class StreamingLSTM(nn.Module):
    """Unidirectional (causal) LSTM — streaming-viable. The direct LSTM-vs-GRU
    comparison (TODO §4): same hidden size, layers, readout and head as
    StreamingGRU, so the recurrent cell is the only variable."""

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, lengths):
        x = self.input_norm(x)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1)).to(out.device)
        return self.head(out.gather(1, idx).squeeze(1))

    def forward_full(self, x):
        out, _ = self.lstm(self.input_norm(x))
        return self.head(out[:, -1])


class BiLSTM(nn.Module):
    """Bidirectional LSTM — OFFLINE-ONLY accuracy reference: the backward pass
    reads future frames, so this can NEVER be a deployment candidate (project
    constraint). It exists to price how much accuracy streaming causality
    costs vs the unidirectional models.

    Readout: forward direction at the last valid frame + backward direction at
    t=0 (which has seen the whole sequence), concatenated -> 2*hidden head."""

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, num_classes),
        )

    def forward(self, x, lengths):
        x = self.input_norm(x)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, T, 2H)
        H = out.size(-1) // 2
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, H).to(out.device)
        fwd_last = (
            out[..., :H].gather(1, idx).squeeze(1)
        )  # fwd state at last valid frame
        bwd_first = out[:, 0, H:]  # bwd state at t=0 (saw everything)
        return self.head(torch.cat([fwd_last, bwd_first], dim=-1))

    def forward_full(self, x):
        out, _ = self.lstm(self.input_norm(x))
        H = out.size(-1) // 2
        return self.head(torch.cat([out[:, -1, :H], out[:, 0, H:]], dim=-1))


class CausalConv1D(nn.Module):
    """Dilated causal Conv1d stack — streaming-viable (TODO §4; first step
    toward the 1st-place 1D-CNN + Transformer port). Every conv is left-padded
    by (kernel_size-1)*dilation, so frame t never sees t+1. Normalization is a
    per-frame LayerNorm — deliberately NOT BatchNorm/GroupNorm, whose
    statistics would mix future frames (and padding) into past ones.
    Classified from the features at the last valid frame, like the recurrent
    models.

    Receptive field: 1 + (kernel_size-1) * sum(2**i for i in range(num_layers))
    frames = 125 at kernel 5 / 5 blocks (dilations 1,2,4,8,16) ≈ MAX_SEQ_LEN.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout=0.3,
        kernel_size=5,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        ch = input_size
        for i in range(num_layers):
            d = 2**i
            self.convs.append(
                nn.Sequential(
                    nn.ConstantPad1d(
                        ((kernel_size - 1) * d, 0), 0.0
                    ),  # causal left pad
                    nn.Conv1d(ch, hidden_size, kernel_size, dilation=d),
                )
            )
            self.norms.append(nn.LayerNorm(hidden_size))
            ch = hidden_size
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, lengths):
        x = self.input_norm(x).transpose(1, 2)  # (B, C, T)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x).transpose(1, 2)  # (B, T, H)
            x = self.drop(self.act(norm(x))).transpose(1, 2)
        out = x.transpose(1, 2)  # (B, T, H)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1)).to(out.device)
        return self.head(out.gather(1, idx).squeeze(1))

    def forward_full(self, x):
        x = self.input_norm(x).transpose(1, 2)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x).transpose(1, 2)
            x = self.drop(self.act(norm(x))).transpose(1, 2)
        return self.head(x.transpose(1, 2)[:, -1])


@dataclass(frozen=True)
class ArchSpec:
    cls: type[nn.Module]
    model_name: str
    streaming: bool
    description: str


ARCHS: dict[str, ArchSpec] = {
    "gru": ArchSpec(
        StreamingGRU,
        "StreamingGRU",
        True,
        "unidirectional/causal GRU, LayerNorm in/out",
    ),
    "lstm": ArchSpec(
        StreamingLSTM,
        "StreamingLSTM",
        True,
        "unidirectional/causal LSTM, LayerNorm in/out",
    ),
    "bilstm": ArchSpec(
        BiLSTM,
        "BiLSTM",
        False,
        "bidirectional LSTM, fwd-last + bwd-first readout, OFFLINE-ONLY reference",
    ),
    "cnn1d": ArchSpec(
        CausalConv1D,
        "CausalConv1D",
        True,
        "dilated causal Conv1d stack (kernel 5, dilations 1..16), per-frame LayerNorm",
    ),
}


def build_model(arch: str, feature_dim: int, num_classes: int, hyp: dict) -> nn.Module:
    """The ONLY model-constructor call in the training/eval stack."""
    return ARCHS[arch].cls(
        feature_dim, hyp["hidden_size"], hyp["num_layers"], num_classes, hyp["dropout"]
    )
