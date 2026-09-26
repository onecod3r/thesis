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


class CosineGlossHead(nn.Module):
    """Per-frame gloss + null logits from a recurrent hidden state, as a
    **cosine classifier**: ``scale * <normalize(embed(h)), normalize(W_c)>``.

    Rows ``0..num_classes-1`` are the glosses in the dataset's own label
    order; row ``num_classes`` is **null** (non-signing). Cosine logits make a
    class's weight row interchangeable with a normalized mean embedding of
    its examples, which is what lets a new sign be added after training
    without retraining (weight imprinting, TODO §12.4): write its prototype
    into a row and unmask it.

    ``class_mask`` (buffer, glosses only) is False for glosses the model must
    not predict: held out of training for the open-vocabulary run, until
    :meth:`enroll` fills them. Masked rows get ``-inf`` logits, so the index
    space never changes and labels stay the dataset's.
    """

    class_mask: torch.Tensor  # registered buffer

    def __init__(self, hidden_size: int, num_classes: int, embed_dim: int, cos_scale: float):
        super().__init__()
        self.num_classes = num_classes
        self.embed = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, embed_dim))
        self.weight = nn.Parameter(torch.randn(num_classes + 1, embed_dim) * 0.02)
        self.scale = cos_scale
        self.register_buffer("class_mask", torch.ones(num_classes, dtype=torch.bool))

    def embeddings(self, h: torch.Tensor) -> torch.Tensor:
        """Unit-norm frame embeddings -- what enrollment averages."""
        return nn.functional.normalize(self.embed(h), dim=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.scale * self.embeddings(h) @ nn.functional.normalize(self.weight, dim=-1).T
        mask = torch.cat([self.class_mask, torch.ones(1, dtype=torch.bool, device=logits.device)])  # null always on
        return logits.masked_fill(~mask, float("-inf"))

    @torch.no_grad()
    def enroll(self, class_idx: int, prototype: torch.Tensor) -> None:
        """Add (or replace) one gloss from a prototype embedding -- the mean
        of :meth:`embeddings` over its example frames. No gradient step."""
        self.weight[class_idx] = nn.functional.normalize(prototype, dim=-1)
        self.class_mask[class_idx] = True


class ContinuousRNN(nn.Module):
    """Causal recurrent model trained on **continuous multi-sign streams**
    (TODO §12.3), not isolated clips.

    Per frame it emits (a) a confidence over every gloss plus null
    (:class:`CosineGlossHead`) and (b) a **sign-boundary** logit: "the sign in
    progress has just ended". The boundary head exists because a fixed
    "confident for N frames" rule cannot find sign ends across short and long
    signs (``docs/reports/sentence-baselines.md`` §3.2); the model has to
    signal them itself.

    Contract, compatible with the rest of the stack:
    - ``forward(x, lengths)`` -> ``(B, num_classes)`` gloss logits at the last
      valid frame, null column dropped -- the isolated-clip contract, so
      ``sb-evaluate`` scores it canonically like any other architecture.
    - ``forward_full(x)`` -> the same for an unpadded batch.
    - ``forward_all(x)`` -> ``(B, T, num_classes + 1)`` per-frame gloss + null
      logits, so ``sb.recognize.streaming``'s per-frame decoders and
      ``RecurrentSession`` (which calls ``input_norm``, the recurrent cell
      and ``head``) work unchanged.
    - ``forward_frames(x)`` -> ``(gloss_logits, boundary_logits)``.
    Padded frames trail every sequence and the model is causal, so no packing
    is needed; losses mask them out.
    """

    cell = "gru"
    head: CosineGlossHead

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3,
                 embed_dim=256, cos_scale=16.0):
        super().__init__()
        self.num_classes = num_classes
        self.input_norm = nn.LayerNorm(input_size)
        rnn_cls = nn.GRU if self.cell == "gru" else nn.LSTM
        rnn = rnn_cls(input_size, hidden_size, num_layers, batch_first=True,
                      dropout=dropout if num_layers > 1 else 0.0)
        setattr(self, self.cell, rnn)  # model.gru / model.lstm, as RecurrentSession expects
        self.dropout = nn.Dropout(dropout)
        self.head = CosineGlossHead(hidden_size, num_classes, embed_dim, cos_scale)
        self.boundary = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def hidden(self, x):
        out, _ = getattr(self, self.cell)(self.input_norm(x))
        return self.dropout(out)

    def forward_frames(self, x):
        h = self.hidden(x)
        return self.head(h), self.boundary(h).squeeze(-1)

    def forward_all(self, x):
        return self.head(self.hidden(x))

    def forward_full(self, x):
        return self.forward_all(x)[:, -1, : self.num_classes]

    def forward(self, x, lengths):
        logits = self.forward_all(x)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, logits.size(-1)).to(logits.device)
        return logits.gather(1, idx).squeeze(1)[:, : self.num_classes]


class ContinuousGRU(ContinuousRNN):
    """:class:`ContinuousRNN` on a GRU -- the §12.3 primary model."""

    cell = "gru"


class ContinuousLSTM(ContinuousRNN):
    """:class:`ContinuousRNN` on an LSTM -- the §12.3 architecture comparison."""

    cell = "lstm"


_POSE0, _LH0, _RH0 = 489, 468, 522
_FINGER_CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
_POSE_LR_PAIRS = ((11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24))


def _cos(a, v, b):
    u, w = a - v, b - v
    return (u * w).sum(-1) / (u.norm(dim=-1) * w.norm(dim=-1)).clamp_min(1e-6)


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)


class PhonologyFrontend(nn.Module):
    """Non-learned, **per-frame** (so causal and stream-safe) feature layer:
    raw landmarks of one subset, ``(..., n_rows * 3)`` xyz with 0 = missing,
    to the ASL phonological parameters found to carry sign identity in
    ``docs/reports/sign-patterns.md`` §7-§8 (TODO §3.9).

    Every frame is centred on the mid-shoulder point and divided by that
    frame's shoulder width (so framing, distance and aspect drop out). Per hand
    (41 each):

    - **handshape** (25): cosines of 15 finger joint angles, 5 fingertip-wrist
      distances / palm size, cosines of 4 finger spreads, thumb-index gap;
    - **orientation** (6): palm normal, wrist->middle-knuckle direction;
    - **location** (9): hand centroid (x, y) and its distance to nose, chin,
      forehead, mouth, same-side shoulder, chest and the other hand;
    - **present** (1).

    Plus both elbow angles (2). The left hand is computed in a mirrored frame
    (x negated), so a left-handed signer's dominant hand produces a
    right-handed signer's features. ``mode="phono+raw"`` appends every input
    row's shoulder-normalized (x, y).

    ``mirror_p`` (training only): the probability of mirroring a whole
    sample before the features (x -> 1 - x, left/right hands and pose pairs
    swapped), so the model is handedness-invariant without knowing which hand
    is dominant, which a stream cannot know in advance. Face rows get only
    the x flip.

    Accepts any leading shape (a ``(B, T, F)`` batch or a single ``(F,)``
    frame, as ``RecurrentSession`` feeds it) and always computes in float32.
    """

    MODES = ("phono", "phono+raw")
    PER_HAND = 41
    # index buffers (register_buffer), derived from the subset -- not in the state_dict
    i_lh: torch.Tensor
    i_rh: torch.Tensor
    i_sh: torch.Tensor
    i_el: torch.Tensor
    i_wr: torch.Tensor
    i_face: torch.Tensor
    perm: torch.Tensor
    mirror_x: torch.Tensor

    def __init__(self, landmark_subset: str, mode: str = "phono", mirror_p: float = 0.0):
        super().__init__()
        from sb.core.subsets import FACE_ANCHORS_5, get_subset

        if mode not in self.MODES:
            raise ValueError(f"frontend {mode!r} not in {self.MODES}")
        rows = [int(r) for r in get_subset(landmark_subset).indices]
        pos = {r: i for i, r in enumerate(rows)}
        need = (list(range(_LH0, _LH0 + 21)) + list(range(_RH0, _RH0 + 21))
                + [_POSE0 + k for k in (11, 12, 13, 14, 15, 16)] + FACE_ANCHORS_5)
        missing = [r for r in need if r not in pos]
        if missing:
            raise ValueError(f"subset {landmark_subset} lacks rows {missing} the phonology front-end needs "
                             f"(use PH_55 or ME_134)")
        self.mode, self.mirror_p, self.n_rows = mode, float(mirror_p), len(rows)
        idx = {"lh": [pos[_LH0 + k] for k in range(21)], "rh": [pos[_RH0 + k] for k in range(21)],
               "sh": [pos[_POSE0 + 11], pos[_POSE0 + 12]], "el": [pos[_POSE0 + 13], pos[_POSE0 + 14]],
               "wr": [pos[_POSE0 + 15], pos[_POSE0 + 16]],
               "face": [pos[r] for r in FACE_ANCHORS_5]}  # nose, forehead, upper lip, lower lip, chin
        for k, v in idx.items():
            self.register_buffer(f"i_{k}", torch.tensor(v), persistent=False)
        perm = list(range(len(rows)))
        for a, b in zip(idx["lh"], idx["rh"]):
            perm[a], perm[b] = b, a
        for a, b in _POSE_LR_PAIRS:
            ra, rb = _POSE0 + a, _POSE0 + b
            if ra in pos and rb in pos:
                perm[pos[ra]], perm[pos[rb]] = pos[rb], pos[ra]
        self.register_buffer("perm", torch.tensor(perm), persistent=False)
        self.register_buffer("mirror_x", torch.tensor([-1.0, 1.0, 1.0]), persistent=False)
        self.out_dim = 2 * self.PER_HAND + 2 + (2 * len(rows) if mode == "phono+raw" else 0)
        # output column groups, for feature-group importance (sb.recognize.phonology_eval)
        g: dict[str, list[int]] = {}
        c = 0
        for side in ("r", "l"):
            for name, n in (("handshape", 25), ("orientation", 6), ("location", 9), ("present", 1)):
                g[f"{side}_{name}"] = list(range(c, c + n))
                c += n
        g["elbows"] = [c, c + 1]
        c += 2
        if mode == "phono+raw":
            kind = {**{i: "raw_hands" for i in idx["lh"] + idx["rh"]}}
            for r, i in pos.items():
                kind.setdefault(i, "raw_pose" if _POSE0 <= r < _POSE0 + 33 else "raw_face")
            for i in range(len(rows)):
                g.setdefault(kind[i], []).extend([c + 2 * i, c + 2 * i + 1])
        self.groups = g

    def _hand(self, m, valid, own, other, shoulder):
        """41 features of one hand in frame ``m`` (already mirrored for the left)."""
        h = m[..., own, :]
        ok = valid[..., own[0]]  # wrist seen
        palm = (h[..., 9, :] - h[..., 0, :]).norm(dim=-1).clamp_min(1e-3)
        shape = []
        for chain in _FINGER_CHAINS:
            j = (0,) + chain
            shape += [_cos(h[..., j[k], :], h[..., j[k + 1], :], h[..., j[k + 2], :]) for k in range(3)]
        shape += [(h[..., ch[3], :] - h[..., 0, :]).norm(dim=-1) / palm for ch in _FINGER_CHAINS]
        dirs = [h[..., ch[3], :] - h[..., ch[0], :] for ch in _FINGER_CHAINS]
        shape += [_cos(dirs[k], torch.zeros_like(dirs[k]), dirs[k + 1]) for k in range(4)]
        shape.append((h[..., 4, :] - h[..., 8, :]).norm(dim=-1) / palm)
        normal = _unit(torch.linalg.cross(h[..., 5, :] - h[..., 0, :], h[..., 17, :] - h[..., 0, :]))
        point = _unit(h[..., 9, :] - h[..., 0, :])
        cen = h[..., :2].mean(-2)
        f = m[..., self.i_face, :2]
        anchors = [f[..., 0, :], f[..., 4, :], f[..., 1, :], (f[..., 2, :] + f[..., 3, :]) / 2,
                   m[..., shoulder, :2], torch.zeros_like(cen)]
        loc = [cen[..., 0], cen[..., 1]] + [(cen - a).norm(dim=-1) for a in anchors]
        o_ok = valid[..., other[0]]
        loc.append((cen - m[..., other, :2].mean(-2)).norm(dim=-1) * o_ok)
        out = torch.cat([torch.stack(shape, -1), normal, point, torch.stack(loc, -1), ok[..., None].float()], -1)
        return out * ok[..., None]

    def forward(self, x):
        lead = x.shape[:-1]
        with torch.autocast(device_type=x.device.type, enabled=False):
            p = x.float().reshape(*lead, self.n_rows, 3)
            present = (p != 0).any(-1)
            if self.training and self.mirror_p > 0 and p.dim() >= 3:
                flip = (torch.rand(p.shape[0], device=p.device) < self.mirror_p).view(-1, *[1] * (p.dim() - 2))
                q, qp = p[..., self.perm, :], present[..., self.perm]
                q = torch.cat([torch.where(qp, 1 - q[..., 0], q[..., 0])[..., None], q[..., 1:]], -1)
                p = torch.where(flip[..., None], q, p)
                present = torch.where(flip, qp, present)
            ls, rs = p[..., self.i_sh[0], :], p[..., self.i_sh[1], :]
            width = (ls[..., :2] - rs[..., :2]).norm(dim=-1)
            sh_ok = present[..., self.i_sh[0]] & present[..., self.i_sh[1]] & (width > 1e-3)
            n = (p - ((ls + rs) / 2)[..., None, :]) / width.clamp_min(1e-3)[..., None, None]
            valid = present & sh_ok[..., None]
            n = n * valid[..., None]
            feats = []
            for m, own, other, sh in ((n, self.i_rh, self.i_lh, self.i_sh[1]),
                                      (n * self.mirror_x, self.i_lh, self.i_rh, self.i_sh[0])):
                feats.append(self._hand(m, valid, own, other, sh))
            for s in (0, 1):
                ok = valid[..., self.i_sh[s]] & valid[..., self.i_el[s]] & valid[..., self.i_wr[s]]
                feats.append((_cos(n[..., self.i_sh[s], :2], n[..., self.i_el[s], :2], n[..., self.i_wr[s], :2])
                              * ok)[..., None])
            if self.mode == "phono+raw":
                feats.append(n[..., :2].reshape(*lead, -1))
            return torch.cat(feats, -1)


def _with_frontend(model: nn.Module, fe: PhonologyFrontend) -> None:
    """Put ``fe`` in front of ``model.input_norm``, so every path that feeds
    ``input_norm`` (training, ``forward_all``, ``RecurrentSession.step``)
    gets it."""
    norm = model.input_norm
    assert isinstance(norm, nn.Module)
    model.input_norm = nn.Sequential(fe, norm)


def frontend_of(model: nn.Module) -> PhonologyFrontend:
    """The :class:`PhonologyFrontend` of a ``*_phono`` model (``ValueError`` otherwise)."""
    norm = model.input_norm
    fe = norm[0] if isinstance(norm, nn.Sequential) else None
    if not isinstance(fe, PhonologyFrontend):
        raise ValueError(f"{type(model).__name__} has no phonology front-end")
    return fe


def _frontend(input_size, landmark_subset, frontend, mirror_p) -> PhonologyFrontend:
    fe = PhonologyFrontend(landmark_subset, frontend, mirror_p)
    if input_size != fe.n_rows * 3:
        raise ValueError(f"input_size {input_size} != 3 x {fe.n_rows} rows of {landmark_subset}: "
                         "phonology models need coords='xyz' of their subset")
    return fe


class PhonoGRU(StreamingGRU):
    """:class:`StreamingGRU` behind a :class:`PhonologyFrontend` (TODO §3.9).
    Causal and per-frame end to end, so it streams like ``gru``."""

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3,
                 landmark_subset="PH_55", frontend="phono", mirror_p=0.0):
        fe = _frontend(input_size, landmark_subset, frontend, mirror_p)
        super().__init__(fe.out_dim, hidden_size, num_layers, num_classes, dropout)
        _with_frontend(self, fe)


class PhonoBiLSTM(BiLSTM):
    """:class:`BiLSTM` behind a :class:`PhonologyFrontend`: the OFFLINE-ONLY
    accuracy reference for the phonology features (TODO §3.9)."""

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3,
                 landmark_subset="PH_55", frontend="phono", mirror_p=0.0):
        fe = _frontend(input_size, landmark_subset, frontend, mirror_p)
        super().__init__(fe.out_dim, hidden_size, num_layers, num_classes, dropout)
        _with_frontend(self, fe)


class ContinuousPhonoGRU(ContinuousGRU):
    """:class:`ContinuousGRU` behind a :class:`PhonologyFrontend`: the §12.3
    continuous recipe on phonology features, so an isolated ``gru_phono`` win
    ports to continuous signing by changing the arch key, not the pipeline."""

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3,
                 embed_dim=256, cos_scale=16.0, landmark_subset="PH_55", frontend="phono", mirror_p=0.0):
        fe = _frontend(input_size, landmark_subset, frontend, mirror_p)
        super().__init__(fe.out_dim, hidden_size, num_layers, num_classes, dropout, embed_dim, cos_scale)
        _with_frontend(self, fe)


@dataclass(frozen=True)
class ArchSpec:
    cls: type[nn.Module]
    model_name: str
    streaming: bool
    description: str
    pipeline: str = "base_v1"  # the feature pipeline (sb.recognize.features.<pipeline>) feeding it


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
    "gru_continuous": ArchSpec(
        ContinuousGRU,
        "ContinuousGRU",
        True,
        "causal GRU trained on continuous streams: per-frame cosine gloss+null head "
        "+ sign-boundary head (TODO §12.3)",
    ),
    "lstm_continuous": ArchSpec(
        ContinuousLSTM,
        "ContinuousLSTM",
        True,
        "causal LSTM trained on continuous streams: per-frame cosine gloss+null head "
        "+ sign-boundary head (TODO §12.3)",
    ),
    "gru_phono": ArchSpec(
        PhonoGRU,
        "PhonoGRU",
        True,
        "StreamingGRU behind the per-frame phonology front-end (handshape / orientation / "
        "location per hand, shoulder-normalized; TODO §3.9)",
    ),
    "gru_phono_raw": ArchSpec(
        PhonoGRU,
        "PhonoGRU",
        True,
        "same class as gru_phono, front-end mode phono+raw: phonology features + the "
        "subset's shoulder-normalized xy (TODO §3.9)",
    ),
    "bilstm_phono": ArchSpec(
        PhonoBiLSTM,
        "PhonoBiLSTM",
        False,
        "BiLSTM behind the phonology front-end, OFFLINE-ONLY accuracy reference (TODO §3.9)",
    ),
    "gru_phono130": ArchSpec(
        StreamingGRU,
        "StreamingGRU",
        True,
        "StreamingGRU on phonological features ONLY: the 130 per-frame features of "
        "sb.recognize.phonology from all 543 landmarks, in sequence (TODO §3.10)",
        pipeline="phono130_v1",
    ),
    "lstm_phono130": ArchSpec(
        StreamingLSTM,
        "StreamingLSTM",
        True,
        "StreamingLSTM on the 130 phonological features only (TODO §3.10)",
        pipeline="phono130_v1",
    ),
    "cnn1d_phono130": ArchSpec(
        CausalConv1D,
        "CausalConv1D",
        True,
        "dilated causal Conv1d stack on the 130 phonological features only (TODO §3.10)",
        pipeline="phono130_v1",
    ),
    "bilstm_phono130": ArchSpec(
        BiLSTM,
        "BiLSTM",
        False,
        "BiLSTM on the same 130 phonological features, OFFLINE-ONLY ceiling for gru_phono130 (TODO §3.10)",
        pipeline="phono130_v1",
    ),
    "gru_continuous_phono130": ArchSpec(
        ContinuousGRU,
        "ContinuousGRU",
        True,
        "ContinuousGRU (per-frame gloss+null + boundary heads) on the 130 phonological features only; "
        "streams composed in feature space (sb.recognize.continuous.phono; TODO §3.10 -> §12.3)",
        pipeline="phono130_v1",
    ),
    "gru_continuous_phono": ArchSpec(
        ContinuousPhonoGRU,
        "ContinuousPhonoGRU",
        True,
        "ContinuousGRU behind the phonology front-end: the port of gru_phono to "
        "continuous streams (TODO §3.9 -> §12.3)",
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
