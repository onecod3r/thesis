"""Rebuild a trained PyTorch run as a native Keras model, for TFLite export.

**Why this exists.** The obvious route — PyTorch → ONNX → `onnx2tf` → TFLite —
does not work for these models (measured 2026-07-19, TODO §6.2): `onnx2tf`
declares no dependencies of its own, cannot convert the `IsInf` op that
`torch.nan_to_num` emits, permutes 3D input layouts, fails outright on the GRU /
BiLSTM / CNN graphs, and produces a malformed `Squeeze` for the one architecture
it does convert. Keras' own recurrent layers sit on TFLite's well-supported
(and for LSTM, fused) conversion path, so rebuilding the graph natively and
transferring the trained weights removes the whole failure surface.

**The correctness risk this creates** is weight transfer: PyTorch and Keras
order their recurrent gates differently and parameterize biases differently, so
a silent mismatch would produce a model that converts cleanly and predicts
garbage. Every builder here is therefore paired with `check_parity`, which runs
the same random input through the PyTorch model and the Keras rebuild and
asserts they agree — export refuses to write anything until they do.

Gate-order and bias conventions handled below:

- **GRU** — PyTorch packs gates as ``[r, z, n]``, Keras as ``[z, r, h]``.
  Keras `reset_after=True` (its default) matches PyTorch's formulation exactly:
  the reset gate is applied *after* the recurrent matmul, ``r * (h @ Uh + bh_r)``,
  which is why the recurrent bias must be kept as a separate row rather than
  folded into the input bias.
- **LSTM** — both use ``[i, f, g/c, o]``, so the kernels transpose directly; but
  PyTorch carries two bias vectors (``b_ih``, ``b_hh``) where Keras carries one,
  so they are **summed**.
- **LayerNorm** — PyTorch defaults to ``eps=1e-5``, Keras to ``1e-3``. Left at
  the default this alone shifts the logits, so epsilon is set explicitly.
- **Conv1d** — PyTorch ``(out, in, k)`` vs Keras ``(k, in, out)``.

Dropout is omitted entirely: it is an inference no-op, and leaving it out keeps
the exported graph smaller.
"""

import numpy as np
import torch

from modules.model.data import ROWS_PER_FRAME

LAYER_NORM_EPS = 1e-5  # PyTorch nn.LayerNorm default (Keras defaults to 1e-3)
N_CLASSES = 250


# ============================================================
# Weight conversion — PyTorch tensor layouts -> Keras layouts
# ============================================================

def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy()


def gru_layer_weights(torch_gru, layer: int) -> list[np.ndarray]:
    """PyTorch nn.GRU layer -> Keras GRU weights [kernel, recurrent, bias].

    Gate reorder [r, z, n] -> [z, r, h]; bias kept as two rows for
    ``reset_after=True``.
    """
    w_ih = _np(getattr(torch_gru, f"weight_ih_l{layer}"))  # (3H, in)
    w_hh = _np(getattr(torch_gru, f"weight_hh_l{layer}"))  # (3H, H)
    b_ih = _np(getattr(torch_gru, f"bias_ih_l{layer}"))    # (3H,)
    b_hh = _np(getattr(torch_gru, f"bias_hh_l{layer}"))
    h = w_hh.shape[1]
    r, z, n = slice(0, h), slice(h, 2 * h), slice(2 * h, 3 * h)
    kernel = np.concatenate([w_ih[z], w_ih[r], w_ih[n]], axis=0).T
    recurrent = np.concatenate([w_hh[z], w_hh[r], w_hh[n]], axis=0).T
    bias = np.stack([
        np.concatenate([b_ih[z], b_ih[r], b_ih[n]]),
        np.concatenate([b_hh[z], b_hh[r], b_hh[n]]),
    ])
    return [kernel, recurrent, bias]


def lstm_layer_weights(torch_lstm, layer: int, reverse: bool = False) -> list[np.ndarray]:
    """PyTorch nn.LSTM layer -> Keras LSTM weights [kernel, recurrent, bias].

    Gate order already matches ([i, f, g, o]); the two PyTorch biases are summed
    into Keras' single bias.
    """
    suffix = f"_reverse" if reverse else ""
    w_ih = _np(getattr(torch_lstm, f"weight_ih_l{layer}{suffix}"))
    w_hh = _np(getattr(torch_lstm, f"weight_hh_l{layer}{suffix}"))
    b_ih = _np(getattr(torch_lstm, f"bias_ih_l{layer}{suffix}"))
    b_hh = _np(getattr(torch_lstm, f"bias_hh_l{layer}{suffix}"))
    return [w_ih.T, w_hh.T, b_ih + b_hh]


def layer_norm_weights(torch_ln) -> list[np.ndarray]:
    return [_np(torch_ln.weight), _np(torch_ln.bias)]


def dense_weights(torch_linear) -> list[np.ndarray]:
    return [_np(torch_linear.weight).T, _np(torch_linear.bias)]


def conv1d_weights(torch_conv) -> list[np.ndarray]:
    """PyTorch Conv1d (out, in, k) -> Keras Conv1D (k, in, out)."""
    return [_np(torch_conv.weight).transpose(2, 1, 0), _np(torch_conv.bias)]


# ============================================================
# Per-architecture rebuilds
# ============================================================

def _keras():
    import tensorflow as tf

    return tf.keras


def _head(x, torch_head, k):
    """Shared classifier head: LayerNorm -> Dense (Dropout dropped, inference)."""
    ln = k.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
    x = ln(x)
    ln.set_weights(layer_norm_weights(torch_head[0]))
    dense = k.layers.Dense(torch_head[2].out_features)
    x = dense(x)
    dense.set_weights(dense_weights(torch_head[2]))
    return x


def build_recurrent(torch_model, feature_dim: int, kind: str):
    """StreamingGRU / StreamingLSTM -> Keras functional model, weights transferred.

    Readout is the last timestep, matching ``forward_full``; with a single
    unpadded sequence that is the last *valid* frame, which is what the training
    forward reads out via packing.
    """
    k = _keras()
    inp = k.Input(shape=(None, feature_dim), dtype="float32")

    ln_in = k.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
    x = ln_in(inp)
    ln_in.set_weights(layer_norm_weights(torch_model.input_norm))

    rnn = torch_model.gru if kind == "gru" else torch_model.lstm
    for layer in range(rnn.num_layers):
        last = layer == rnn.num_layers - 1
        if kind == "gru":
            cell = k.layers.GRU(rnn.hidden_size, return_sequences=not last,
                                reset_after=True)
            x = cell(x)
            cell.set_weights(gru_layer_weights(rnn, layer))
        else:
            cell = k.layers.LSTM(rnn.hidden_size, return_sequences=not last)
            x = cell(x)
            cell.set_weights(lstm_layer_weights(rnn, layer))

    return k.Model(inp, _head(x, torch_model.head, k))


def build_bilstm(torch_model, feature_dim: int):
    """BiLSTM -> Keras. Readout: forward at the last frame + backward at t=0.

    Keras' Bidirectional re-reverses the backward output so it is time-aligned,
    so index 0 of the backward half is the state that has seen the whole
    sequence — exactly the PyTorch readout.
    """
    k = _keras()
    import tensorflow as tf

    inp = k.Input(shape=(None, feature_dim), dtype="float32")
    ln_in = k.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
    x = ln_in(inp)
    ln_in.set_weights(layer_norm_weights(torch_model.input_norm))

    lstm = torch_model.lstm
    for layer in range(lstm.num_layers):
        bi = k.layers.Bidirectional(
            k.layers.LSTM(lstm.hidden_size, return_sequences=True), merge_mode="concat")
        x = bi(x)
        bi.set_weights(lstm_layer_weights(lstm, layer)
                       + lstm_layer_weights(lstm, layer, reverse=True))

    h = lstm.hidden_size
    readout = k.layers.Concatenate()([x[:, -1, :h], x[:, 0, h:]])
    return k.Model(inp, _head(readout, torch_model.head, k))


def build_cnn1d(torch_model, feature_dim: int):
    """CausalConv1D -> Keras. `padding="causal"` reproduces the explicit
    left-pad + Conv1d of the PyTorch stack, so frame t never sees t+1."""
    k = _keras()
    inp = k.Input(shape=(None, feature_dim), dtype="float32")
    ln_in = k.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
    x = ln_in(inp)
    ln_in.set_weights(layer_norm_weights(torch_model.input_norm))

    for i, (block, norm) in enumerate(zip(torch_model.convs, torch_model.norms)):
        torch_conv = block[1]  # block = (ConstantPad1d, Conv1d)
        conv = k.layers.Conv1D(
            torch_conv.out_channels, torch_conv.kernel_size[0],
            dilation_rate=torch_conv.dilation[0], padding="causal")
        x = conv(x)
        conv.set_weights(conv1d_weights(torch_conv))
        ln = k.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
        x = ln(x)
        ln.set_weights(layer_norm_weights(norm))
        x = k.layers.Activation("gelu")(x)

    return k.Model(inp, _head(x[:, -1], torch_model.head, k))


BUILDERS = {
    "gru": lambda m, f: build_recurrent(m, f, "gru"),
    "lstm": lambda m, f: build_recurrent(m, f, "lstm"),
    "bilstm": build_bilstm,
    "cnn1d": build_cnn1d,
}


def build_keras_model(torch_model, arch: str, feature_dim: int):
    """The single entry point: trained PyTorch model -> equivalent Keras model."""
    assert arch in BUILDERS, f"no Keras rebuild for architecture {arch!r}"
    return BUILDERS[arch](torch_model, feature_dim)


# ============================================================
# Serving module — the grader's calling convention
# ============================================================

def serving_module(keras_model, landmark_rows, coord_cols):
    """Wrap a rebuilt model so it takes raw ``(T, 543, 3)`` frames with NaNs.

    Preprocessing lives inside the graph, so the exported model owns it and the
    grader's raw frames work unchanged whatever subset/coords the run trained on.
    """
    import tensorflow as tf

    rows = tf.constant(np.asarray(landmark_rows), dtype=tf.int32)
    cols = tf.constant(np.asarray(coord_cols), dtype=tf.int32)

    class ServingModule(tf.Module):
        def __init__(self):
            super().__init__()
            self.model = keras_model

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, ROWS_PER_FRAME, 3], dtype=tf.float32,
                          name="inputs")
        ])
        def serving_default(self, inputs):
            # NaN -> 0 via a NaN-only test: no IsInf/IsFinite op, and GISLR
            # parquet contains no infinities (0 in 19M sampled values)
            x = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
            x = tf.gather(x, rows, axis=1)          # landmark subset
            x = tf.gather(x, cols, axis=2)          # drops z for xy-trained runs
            n = tf.shape(x)[0]
            x = tf.reshape(x, (1, n, tf.shape(x)[1] * tf.shape(x)[2]))
            return {"outputs": tf.squeeze(self.model(x, training=False), axis=0)}

    return ServingModule()


# ============================================================
# Parity — the gate that makes weight transfer trustworthy
# ============================================================

def check_parity(torch_model, keras_model, feature_dim: int, n_frames: int = 48,
                 atol: float = 2e-4) -> float:
    """Max |difference| between the PyTorch and Keras forwards on one input.

    Tolerance is looser than the PyTorch-internal parity check because the two
    frameworks accumulate float32 reductions in different orders; anything
    beyond this is a transfer bug, not arithmetic noise.
    """
    torch.manual_seed(0)
    x = torch.randn(1, n_frames, feature_dim)
    with torch.no_grad():
        ref = torch_model.forward_full(x).numpy()
    got = np.asarray(keras_model(x.numpy(), training=False))
    diff = float(np.abs(ref - got).max())
    assert diff <= atol, (
        f"Keras rebuild disagrees with PyTorch by {diff:.2e} (> {atol:.0e}) — "
        "weight transfer is wrong; exporting would ship a different model"
    )
    return diff
