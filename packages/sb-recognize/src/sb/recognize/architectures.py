"""The GISLR benchmark architectures — the single definition of every model
class, loaded both by the training notebooks and by the canonical eval script
(modules/scripts/evaluate.py), so state_dicts can never drift between the two.

All models share the same contract: ``forward(x, lengths)`` with
``x (B, T, F)`` zero-padded and ``lengths`` sorted descending (the collate in
``sb.recognize.data`` enforces this), returning ``(B, num_classes)`` logits
read out at the last valid frame.

Streaming viability is a per-architecture fact recorded in ``ARCHS`` — the
deployment path only ever uses ``streaming=True`` models; BiLSTM exists purely
to price the causality gap (project constraint, README §Constraints).
"""

import inspect
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

    def forward_all(self, x):
        """Per-frame logits (B, T, num_classes) -- the same trained head
        applied to every timestep's hidden state, not just the last one.

        Purely a read-out choice: this model is still trained with a loss at
        the last valid frame only (`forward`), so frames before the sign is
        mostly complete were never supervised to have a meaningful confidence
        vector. Use this for inspecting how confidence evolves frame-by-frame
        (TODO's streaming-confidence-eval notebook), not as a claim that
        early-frame confidence is reliable -- same caveat as
        `sb.recognize.interp.models.LandmarkRNN.forward_all`. Unpadded input
        only (no `lengths`): this is an analysis readout, not the
        training/export contract."""
        out, _ = self.gru(self.input_norm(x))
        return self.head(out)


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

    def forward_all(self, x):
        """Per-frame logits (B, T, num_classes) -- see
        `StreamingGRU.forward_all`'s docstring for the same
        last-frame-only-supervision caveat and intended use."""
        out, _ = self.lstm(self.input_norm(x))
        return self.head(out)


class BiLSTM(nn.Module):
    """Bidirectional LSTM — OFFLINE-ONLY accuracy reference: the backward pass
    reads future frames, so this can NEVER be a deployment candidate (project
    constraint). It exists to price how much accuracy streaming causality
    costs vs the unidirectional models.

    Readout: forward direction at the last valid frame + backward direction at
    t=0 (which has seen the whole sequence), concatenated -> 2*hidden head.

    Deliberately has no `forward_all`: every per-frame position's backward
    half has already read the entire future of the sequence, so a "per-frame
    confidence" read off it would not describe what a live streaming reader
    could actually know at that frame -- there is no causal per-frame
    readout to expose."""

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

    def forward_all(self, x):
        """Per-frame logits (B, T, num_classes) -- see
        `StreamingGRU.forward_all`'s docstring for the same
        last-frame-only-supervision caveat. Unlike the recurrent models this
        IS bounded by a finite receptive field (125 frames at the default
        kernel/depth, see the class docstring), not unbounded history -- a
        frame's confidence here can only reflect its own trailing window."""
        x = self.input_norm(x).transpose(1, 2)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x).transpose(1, 2)
            x = self.drop(self.act(norm(x))).transpose(1, 2)
        return self.head(x.transpose(1, 2))


# ============================================================
# 1st-place port: 1D-CNN + Transformer (TODO §4.2)
# ============================================================
# Ported from gislr.0.competition.entry.1st.ipynb (hoyso48), recovered from git
# history at fd1c7aa. Paired with the feature pipeline in sb.recognize.features.firstplace_v1
# — the architecture alone is only part of that solution's edge.
#
# OFFLINE-ONLY (streaming=False), for two reasons that are properties of the
# design, not of this port:
#   * the readout is a global average over the whole sequence, and
#   * self-attention is unmasked in time, so frame t attends to frame t+k.
# The convolutions themselves ARE causal (left-padded depthwise), so a streaming
# variant is reachable: causal-mask the attention and swap the pooled readout for
# a last-frame readout. Until then this sits beside BiLSTM as an accuracy
# reference that prices the causality gap — see README §Constraints.


def _sample_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Dropout with the reference's ``noise_shape=(None, 1, 1)``: the mask is
    per-SAMPLE, so a whole residual branch is dropped at once (stochastic depth)
    rather than individual activations."""
    if not training or p <= 0.0:
        return x
    keep = 1.0 - p
    mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep).div_(keep)
    return x * mask


class MaskedBatchNorm1d(nn.Module):
    """BatchNorm over valid frames only.

    Padded frames must not enter the batch statistics — with padding to the
    batch max they are a large and variable fraction of the tensor, and letting
    them contribute makes the normalization depend on how a batch happened to be
    bucketed. Valid frames are gathered to (N, C), normalized, then scattered
    back; padded positions come out as exact zeros.
    """

    def __init__(self, num_features: int, momentum: float = 0.05, eps: float = 1e-3):
        super().__init__()
        # TF momentum=0.95 is the running-stat DECAY; PyTorch momentum is 1-decay
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum, eps=eps)

    def forward(self, x, mask):
        b, t, c = x.shape
        flat = x.reshape(-1, c)
        sel = mask.reshape(-1)
        out = torch.zeros_like(flat)
        out[sel] = self.bn(flat[sel])
        return out.reshape(b, t, c)


class ECA(nn.Module):
    """Efficient Channel Attention: masked global average pool over time, a
    single 1D conv along the CHANNEL axis, sigmoid, rescale."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x, mask):
        m = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)  # (B, C)
        w = torch.sigmoid(self.conv(pooled.unsqueeze(1)).squeeze(1))
        return x * w.unsqueeze(1)


class CausalDWConv1D(nn.Module):
    """Depthwise 1D conv, left-padded by (kernel_size-1)*dilation so frame t
    never sees t+1 (and so trailing padding cannot reach a valid frame)."""

    def __init__(self, channels: int, kernel_size: int = 17, dilation: int = 1):
        super().__init__()
        self.pad = nn.ConstantPad1d((dilation * (kernel_size - 1), 0), 0.0)
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels,
                              dilation=dilation, bias=False)

    def forward(self, x):  # (B, T, C)
        return self.conv(self.pad(x.transpose(1, 2))).transpose(1, 2)


class Conv1DBlock(nn.Module):
    """The reference's efficient conv block: expand -> causal depthwise ->
    masked BN -> ECA -> project, with a per-sample-dropped residual."""

    def __init__(self, dim: int, kernel_size: int = 17, drop_rate: float = 0.2,
                 expand: int = 2):
        super().__init__()
        hidden = dim * expand
        self.expand_fc = nn.Linear(dim, hidden)
        self.act = nn.SiLU()  # swish
        self.dw_conv = CausalDWConv1D(hidden, kernel_size)
        self.bn = MaskedBatchNorm1d(hidden)
        self.eca = ECA()
        self.project = nn.Linear(hidden, dim)
        self.drop_rate = drop_rate

    def forward(self, x, mask):
        skip = x
        h = self.act(self.expand_fc(x))
        h = self.dw_conv(h)
        h = self.bn(h, mask)
        h = self.eca(h, mask)
        h = self.project(h)
        return skip + _sample_dropout(h, self.drop_rate, self.training)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block using BatchNorm (not LayerNorm) — the
    reference's choice, kept deliberately."""

    def __init__(self, dim: int, num_heads: int = 4, expand: int = 2,
                 attn_dropout: float = 0.2, drop_rate: float = 0.2):
        super().__init__()
        self.bn1 = MaskedBatchNorm1d(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout,
                                          bias=False, batch_first=True)
        self.bn2 = MaskedBatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim * expand, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim * expand, dim, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x, mask):
        h = self.bn1(x, mask)
        attn, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        x = x + _sample_dropout(attn, self.drop_rate, self.training)
        h = self.bn2(x, mask)
        h = self.fc2(self.act(self.fc1(h)))
        return x + _sample_dropout(h, self.drop_rate, self.training)


class LateDropout(nn.Module):
    """Dropout that stays OFF for the first ``start_step`` optimizer steps.

    A 0.8 dropout from step 0 stalls early optimization; the reference switches
    it on after ~15 epochs, once the model has something worth regularizing. The
    step counter is a buffer so it survives checkpoint/resume.
    """

    step_count: torch.Tensor  # buffer; annotated so it types as a Tensor

    def __init__(self, p: float = 0.8, start_step: int = 0):
        super().__init__()
        self.p = p
        self.start_step = start_step
        self.register_buffer("step_count", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        if self.training:
            self.step_count.add_(1)
        if int(self.step_count.item()) < self.start_step:
            return x
        return nn.functional.dropout(x, self.p, self.training)


class Conv1DTransformerStage(nn.Module):
    """One stage of the 1st-place trunk: three causal Conv1DBlocks (local,
    causal temporal modelling) followed by one TransformerBlock (global, and
    the part that makes the whole model non-causal)."""

    def __init__(self, dim: int, kernel_size: int, num_heads: int, expand: int,
                 drop_rate: float):
        super().__init__()
        self.conv1 = Conv1DBlock(dim, kernel_size, drop_rate=drop_rate)
        self.conv2 = Conv1DBlock(dim, kernel_size, drop_rate=drop_rate)
        self.conv3 = Conv1DBlock(dim, kernel_size, drop_rate=drop_rate)
        self.attn = TransformerBlock(dim, num_heads=num_heads, expand=expand,
                                     drop_rate=drop_rate)

    def forward(self, x, mask):
        x = self.conv1(x, mask)
        x = self.conv2(x, mask)
        x = self.conv3(x, mask)
        return self.attn(x, mask)


class Conv1DTransformer(nn.Module):
    """1st-place GISLR architecture: stages of 3 causal Conv1DBlocks followed by
    one TransformerBlock, then a masked global-average readout.

    ``num_layers`` is the number of (3 conv + 1 transformer) STAGES — 2 for the
    reference base model (dim 192), 4 for its "4x sized" variant (dim 384).
    Input is expected to be the 6-channels-per-landmark tensor built by
    ``sb.recognize.features.firstplace_v1`` (positions + lag-1 + lag-2), not raw coordinates.
    """

    def __init__(self, input_size, hidden_size=192, num_layers=2, num_classes=250,
                 dropout=0.2, kernel_size=17, num_heads=4, expand=2,
                 late_dropout=0.8, late_dropout_start_step=0):
        super().__init__()
        dim = hidden_size
        self.stem = nn.Linear(input_size, dim, bias=False)
        self.stem_bn = MaskedBatchNorm1d(dim)
        self.stages = nn.ModuleList(
            Conv1DTransformerStage(dim, kernel_size, num_heads, expand, dropout)
            for _ in range(num_layers))
        self.top = nn.Linear(dim, dim * 2)
        self.late_drop = LateDropout(late_dropout, late_dropout_start_step)
        self.classifier = nn.Linear(dim * 2, num_classes)

    def _trunk(self, x, mask):
        x = self.stem_bn(self.stem(x), mask)
        for stage in self.stages:
            x = stage(x, mask)
        x = self.top(x)
        m = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)  # masked global avg
        return self.classifier(self.late_drop(pooled))

    def forward(self, x, lengths):
        mask = (torch.arange(x.size(1), device=x.device)[None, :]
                < lengths.to(x.device)[:, None])
        return self._trunk(x, mask)

    def forward_full(self, x):
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        return self._trunk(x, mask)


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
    "gru_deep": ArchSpec(
        StreamingGRU,
        "StreamingGRU",
        True,
        "same class as gru, deeper/wider via config overrides — depth-vs-plateau "
        "diagnostic (TODO §4.1) that stays streaming-viable, unlike BiLSTM",
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
    "conv1d_transformer": ArchSpec(
        Conv1DTransformer,
        "Conv1DTransformer",
        False,
        "1st-place port: (3x causal Conv1DBlock + TransformerBlock) x N stages, "
        "masked global-average readout, OFFLINE-ONLY reference",
    ),
}

# constructor parameters build_model always supplies positionally; anything else
# in the resolved HYP is forwarded only if the class actually declares it
_POSITIONAL = ("input_size", "hidden_size", "num_layers", "num_classes", "dropout")


def build_model(arch: str, feature_dim: int, num_classes: int, hyp: dict) -> nn.Module:
    """The ONLY model-constructor call in the training/eval stack.

    Architecture-specific hyperparameters (``kernel_size``, ``num_heads``,
    ``late_dropout``, …) are forwarded when — and only when — the model class
    declares them, so one flat HYP dict can serve every architecture without a
    per-arch construction branch here. A key no class accepts is ignored, which
    is safe because ``sb.recognize.config`` has already rejected any override
    that does not name a real ``shared`` parameter.
    """
    cls = ARCHS[arch].cls
    accepted = inspect.signature(cls.__init__).parameters
    extra = {k: v for k, v in hyp.items() if k in accepted and k not in _POSITIONAL}
    return cls(
        feature_dim, hyp["hidden_size"], hyp["num_layers"], num_classes,
        hyp["dropout"], **extra
    )
