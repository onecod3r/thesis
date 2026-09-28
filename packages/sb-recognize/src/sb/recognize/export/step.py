"""Browser export for the continuous model (TODO §12.5): one frame per call.

The Kaggle path (``tflite``) exports a *whole-clip* graph, so the recurrence is
a TFLite ``WHILE`` loop that needs ``SELECT_TF_OPS`` (Flex). LiteRT.js, the
browser runtime, has no Flex kernels. Live recognition also never has the whole
clip. So this export is a **single step** instead:

    inputs   frame  (543, 3) float32  raw holistic frame, NaN where undetected
             state  (L, H)   float32  recurrent state (zeros = fresh session)
    outputs  embedding (E,)  float32  unit-norm cosine-head embedding
             boundary  (1,)  float32  sign-boundary logit (sigmoid -> prob)
             state_out (L, H) float32 state to feed back on the next frame

The client owns the loop, the state and the reset. Nothing in the graph loops,
so it converts with ``TFLITE_BUILTINS`` only.

**The cosine head's class matrix stays out of the graph.** The client computes
``logits = cos_scale * W @ embedding``, with ``W`` = the normalized class rows
(glosses + null, shipped as ``classes.f32``) and masked rows set to ``-inf``.
Custom signs (§12.4/§12.7) are then a row the client appends to ``W``, the
same thing :meth:`CosineGlossHead.enroll` does in PyTorch, with no re-export.

The GRU is written out as matmuls from the PyTorch weights in PyTorch's own
gate order ``[r, z, n]``. No Keras layer is involved, so there is no gate
reordering to get wrong (compare ``keras.gru_layer_weights``). Parity is still
gated: the exported file is stepped frame by frame against the PyTorch batch
``forward_frames`` and must agree before anything is written.

Measured 2026-09-24 on C1 (``1790143122``): 3.46 MB, 17 builtin ops, max prob
diff 2e-6 vs PyTorch, 0.20 ms/step median in headless Chrome on LiteRT.js WASM
(``docs/reports/deployment-research.md`` §3).

**Other bodies (2026-09-26, TODO §12.8):**

- ``lstm_continuous`` (C2): the LSTM in PyTorch's gate order ``[i, f, g, o]``. Its state is
  ``(2L, H)``: the L hidden rows, then the L cell rows.
- ``gru_continuous_norm`` (C4): the :class:`~sb.recognize.architectures.StreamNormFrontend`
  (hands re-slotted by pose wrist, shoulder-centred, shoulder-width scaled) written in TF ops
  before the input LayerNorm, so the browser feeds the same raw frame to every model.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from sb.mlops import registry as R
from sb.recognize.data import ROWS_PER_FRAME
from sb.recognize.export import keras as KE
from sb.recognize.export.tflite import load_run_model

LAYER_NORM_EPS = 1e-5  # PyTorch nn.LayerNorm default
STEP_ARCHS = ("gru_continuous", "lstm_continuous", "gru_continuous_norm")

# Isolated (whole-sign) classifiers: plain StreamingGRU/-LSTM head (softmax over a Linear, no
# cosine head, no null class, no boundary signal) -- TODO §12.9, "individual sign recognition"
# mode, whose stop condition is external (hand leaves frame / a stable repeat), not a learned
# boundary head. `gru_phono_raw` behind a `PhonologyFrontend` is the only one wired below; plain
# `gru`/`gru_phono` would need the same treatment (skip `_phono_tf` for `gru`).
ISOLATED_STEP_ARCHS = ("gru_phono_raw",)

# Whole-clip (bidirectional) classifiers -- TODO §16.2, "record then recognize" mode. A
# bidirectional model cannot step frame-by-frame (it needs the backward pass over the whole
# sequence), so this is the opposite contract from the step exports above: the recorded clip
# goes in whole, one softmax distribution over the 250 glosses comes out. Only `bilstm` is
# wired (the accuracy leader, TODO §4.3/§3.7, 0.7569 canonical -- run `1784447175`). Verified
# 2026-09-28: `Bidirectional(LSTM)` converts through the fused `UnidirectionalSequenceLSTM`
# builtin for *both* directions, with the time dimension left dynamic, no Flex needed -- unlike
# the Kaggle grader export (`export/tflite.py`), which does need `SELECT_TF_OPS` for its
# combined GRU/BiLSTM/CNN graph (TFLite has no fused GRU builtin at all, which is why the step
# exports above hand-roll the GRU as matmuls instead of using a Keras layer).
WHOLECLIP_ARCHS = ("bilstm",)


def state_shape(model) -> tuple[int, int]:
    """``(rows, hidden)`` of the step graph's recurrent state: ``(L, H)`` for a GRU, ``(2L, H)``
    (hidden rows, then cell rows) for an LSTM."""
    rnn = getattr(model, model.cell)
    return (rnn.num_layers * (2 if model.cell == "lstm" else 1), rnn.hidden_size)


def _stream_norm_tf(x, fe):
    """:class:`StreamNormFrontend` on one ``(1, 2 * rows)`` xy frame (0 = missing), in TF ops."""
    import tensorflow as tf

    n = fe.n_rows
    p = tf.reshape(x, [n, 2])
    valid = tf.reduce_any(tf.not_equal(p, 0.0), -1)
    i_lh, i_rh = fe.i_lh.cpu().numpy(), fe.i_rh.cpu().numpy()
    s0, s1 = (int(v) for v in fe.i_sh.cpu().numpy())
    w0, w1 = (int(v) for v in fe.i_wr.cpu().numpy())
    lh, rh = tf.gather(p, i_lh), tf.gather(p, i_rh)
    lok, rok = valid[int(i_lh[0])], valid[int(i_rh[0])]
    pl, pr = p[w0], p[w1]
    wok = tf.logical_and(valid[w0], valid[w1])

    def d(a, b):
        return tf.sqrt(tf.reduce_sum(tf.square(a - b)))

    lw, rw = lh[0], rh[0]
    both = tf.logical_and(tf.logical_and(lok, rok), d(lw, pr) + d(rw, pl) < d(lw, pl) + d(rw, pr))
    only_l = tf.logical_and(tf.logical_and(lok, tf.logical_not(rok)), d(lw, pr) < d(lw, pl))
    only_r = tf.logical_and(tf.logical_and(rok, tf.logical_not(lok)), d(rw, pl) < d(rw, pr))
    swap = tf.logical_and(wok, tf.logical_or(both, tf.logical_or(only_l, only_r)))
    new_l, new_r = tf.where(swap, rh, lh), tf.where(swap, lh, rh)
    # put the (maybe swapped) hands back in place: concatenate, then one gather restores row order
    keep = np.setdiff1d(np.arange(n), np.r_[i_lh, i_rh])
    order = np.argsort(np.r_[keep, i_lh, i_rh])
    p = tf.gather(tf.concat([tf.gather(p, keep), new_l, new_r], 0), order)
    valid = tf.reduce_any(tf.not_equal(p, 0.0), -1)
    a, b = p[s0], p[s1]
    sok = tf.logical_and(valid[s0], valid[s1])
    centre = (a + b) / 2.0
    width = tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(a - b))), 1e-3)
    keep_pt = tf.cast(tf.logical_and(valid, sok), tf.float32)[:, None]
    out = (p - centre) / width * keep_pt
    flags = tf.cast(tf.stack([valid[int(i_lh[0])], valid[int(i_rh[0])]]), tf.float32) * tf.cast(sok, tf.float32)
    return tf.reshape(tf.concat([tf.reshape(out, [-1]), flags], 0), [1, -1])


def _phono_tf(x, fe):
    """:class:`~sb.recognize.architectures.PhonologyFrontend` on one ``(1, n_rows*3)`` xyz frame
    (0 = missing), in TF ops -- port of its ``forward``, ``mode="phono+raw"`` only (mirroring is
    training-only, so it never runs here). Faithful to the PyTorch order: handshape (25) +
    orientation (6) + location (9) + present (1), per hand (right then left, left mirrored),
    then both elbow angles (2), then every row's shoulder-normalized (x, y) if the frontend was
    built with ``mode="phono+raw"``. All row indices are resolved to plain Python ints here (as in
    :func:`_stream_norm_tf`), never traced, so they are safe inside a ``@tf.function``."""
    import tensorflow as tf

    from sb.recognize.architectures import _FINGER_CHAINS

    def idx(t):
        return [int(v) for v in t.cpu().numpy()]

    n = fe.n_rows
    i_lh, i_rh, i_el, i_wr, i_face = idx(fe.i_lh), idx(fe.i_rh), idx(fe.i_el), idx(fe.i_wr), idx(fe.i_face)
    s0, s1 = idx(fe.i_sh)
    p = tf.reshape(x, [n, 3])
    present = tf.reduce_any(tf.not_equal(p, 0.0), -1)
    ls, rs = p[s0], p[s1]
    width = tf.sqrt(tf.reduce_sum(tf.square(ls[:2] - rs[:2])))
    sh_ok = tf.logical_and(tf.logical_and(present[s0], present[s1]), width > 1e-3)
    centre = (ls + rs) / 2.0
    valid = tf.logical_and(present, sh_ok)
    n_p = (p - centre) / tf.maximum(width, 1e-3) * tf.cast(valid, tf.float32)[:, None]

    def cos(a, v, b):
        u, w = a - v, b - v
        return tf.reduce_sum(u * w) / tf.maximum(tf.norm(u) * tf.norm(w), 1e-6)

    def unit(v):
        return v / tf.maximum(tf.norm(v), 1e-6)

    def cross(a, b):
        # tf.linalg.cross has no TFLite builtin kernel; the 3-vector formula does.
        return tf.stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])

    def hand_feats(m, own: list[int], other: list[int], sh: int):
        h = tf.gather(m, own)  # (21, 3)
        ok = valid[own[0]]
        palm = tf.maximum(tf.norm(h[9] - h[0]), 1e-3)
        shape = []
        for a, b, c, d in _FINGER_CHAINS:
            j = (0, a, b, c, d)
            shape += [cos(h[j[k]], h[j[k + 1]], h[j[k + 2]]) for k in range(3)]
        for a, b, c, d in _FINGER_CHAINS:
            shape.append(tf.norm(h[d] - h[0]) / palm)
        dirs = [h[d] - h[a] for a, b, c, d in _FINGER_CHAINS]
        shape += [cos(dirs[k], tf.zeros_like(dirs[k]), dirs[k + 1]) for k in range(4)]
        shape.append(tf.norm(h[4] - h[8]) / palm)
        normal = unit(cross(h[5] - h[0], h[17] - h[0]))
        point = unit(h[9] - h[0])
        cen = tf.reduce_mean(h[:, :2], axis=0)
        f = tf.gather(m, i_face)[:, :2]
        anchors = [f[0], f[4], f[1], (f[2] + f[3]) / 2.0, m[sh, :2], tf.zeros_like(cen)]
        loc = [cen[0], cen[1]] + [tf.norm(cen - a) for a in anchors]
        oc = tf.reduce_mean(tf.gather(m, other)[:, :2], axis=0)
        o_ok = tf.cast(valid[other[0]], tf.float32)
        loc.append(tf.norm(cen - oc) * o_ok)
        out = tf.stack(shape + [normal[0], normal[1], normal[2], point[0], point[1], point[2]]
                       + loc + [tf.cast(ok, tf.float32)])
        return out * tf.cast(ok, tf.float32)

    mirror_x = tf.constant([-1.0, 1.0, 1.0], tf.float32)
    right_feats = hand_feats(n_p, i_rh, i_lh, s1)
    left_feats = hand_feats(n_p * mirror_x, i_lh, i_rh, s0)
    elbows = []
    for s0_, el, wr in ((s0, i_el[0], i_wr[0]), (s1, i_el[1], i_wr[1])):
        ok = tf.logical_and(tf.logical_and(valid[s0_], valid[el]), valid[wr])
        elbows.append(cos(n_p[s0_, :2], n_p[el, :2], n_p[wr, :2]) * tf.cast(ok, tf.float32))
    out = [right_feats, left_feats, tf.stack(elbows)]
    if fe.mode == "phono+raw":
        out.append(tf.reshape(n_p[:, :2], [-1]))
    return tf.reshape(tf.concat(out, 0), [1, -1])


def _build_step_module(model, rows: np.ndarray, cols: np.ndarray, feature_dim: int):
    """PyTorch continuous model (GRU, LSTM, or GRU behind the stream norm) -> tf.Module with one
    ``step`` signature."""
    import tensorflow as tf

    from sb.recognize.architectures import StreamNormFrontend

    def const(t):
        return tf.constant(t.detach().cpu().numpy().astype(np.float32))

    def layer_norm(x, ln):
        mu = tf.reduce_mean(x, -1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mu), -1, keepdims=True)
        return (x - mu) * tf.math.rsqrt(var + LAYER_NORM_EPS) * const(ln.weight) + const(ln.bias)

    rnn = getattr(model, model.cell)
    lstm = model.cell == "lstm"
    n_layers, hidden = rnn.num_layers, rnn.hidden_size
    weights = [tuple(const(getattr(rnn, f"{w}_l{i}"))
                     for w in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))
               for i in range(n_layers)]
    norm = model.input_norm
    fe, ln_in = (norm[0], norm[1]) if isinstance(norm, torch.nn.Sequential) else (None, norm)
    assert fe is None or isinstance(fe, StreamNormFrontend), f"no step port for front-end {type(fe).__name__}"
    rows_state = n_layers * (2 if lstm else 1)
    emb_ln, emb_lin = model.head.embed[0], model.head.embed[1]
    b_ln, b_lin = model.boundary[0], model.boundary[1]
    rows_c = tf.constant(rows, tf.int32)
    cols_c = tf.constant(cols, tf.int32)

    class Step(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec([ROWS_PER_FRAME, 3], tf.float32, name="frame"),
            tf.TensorSpec([rows_state, hidden], tf.float32, name="state"),
        ])
        def step(self, frame, state):
            # NaN -> 0 with a NaN-only test (no IsInf op), as in keras.serving_module
            x = tf.where(tf.math.is_nan(frame), tf.zeros_like(frame), frame)
            x = tf.gather(tf.gather(x, rows_c, axis=0), cols_c, axis=1)
            x = tf.reshape(x, [1, feature_dim])
            if fe is not None:
                x = _stream_norm_tf(x, fe)
            x = layer_norm(x, ln_in)
            new_h, new_c = [], []
            for i, (w_ih, w_hh, b_ih, b_hh) in enumerate(weights):
                h = state[i:i + 1]
                if lstm:  # PyTorch gate order [i, f, g, o]; cell rows follow the hidden rows
                    c = state[n_layers + i:n_layers + i + 1]
                    g_i, g_f, g_g, g_o = tf.split(tf.matmul(x, w_ih, transpose_b=True) + b_ih
                                                  + tf.matmul(h, w_hh, transpose_b=True) + b_hh, 4, axis=1)
                    c = tf.sigmoid(g_f) * c + tf.sigmoid(g_i) * tf.tanh(g_g)
                    h = tf.sigmoid(g_o) * tf.tanh(c)
                    new_c.append(c)
                else:
                    i_r, i_z, i_n = tf.split(tf.matmul(x, w_ih, transpose_b=True) + b_ih, 3, axis=1)
                    h_r, h_z, h_n = tf.split(tf.matmul(h, w_hh, transpose_b=True) + b_hh, 3, axis=1)
                    r = tf.sigmoid(i_r + h_r)
                    z = tf.sigmoid(i_z + h_z)
                    h = (1.0 - z) * tf.tanh(i_n + r * h_n) + z * h
                new_h.append(h)
                x = h
            new_state = new_h + new_c
            e = tf.matmul(layer_norm(x, emb_ln), const(emb_lin.weight), transpose_b=True) \
                + const(emb_lin.bias)
            e = e * tf.math.rsqrt(tf.reduce_sum(tf.square(e), -1, keepdims=True) + 1e-24)
            b = tf.matmul(layer_norm(x, b_ln), const(b_lin.weight), transpose_b=True) \
                + const(b_lin.bias)
            return {"embedding": tf.reshape(e, [-1]), "boundary": tf.reshape(b, [1]),
                    "state_out": tf.concat(new_state, 0)}

    return Step()


def convert_step(module) -> tuple[bytes, list[str]]:
    """tf.Module -> (TFLite bytes, sorted op names). Builtins only: LiteRT.js
    has no Flex kernels, so a Flex op would fail at load time in the browser."""
    import tensorflow as tf

    conv = tf.lite.TFLiteConverter.from_concrete_functions(
        [module.step.get_concrete_function()], module)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    blob = conv.convert()
    interp = tf.lite.Interpreter(model_content=blob)
    ops = sorted({d["op_name"] for d in interp._get_ops_details()})
    assert not any("Flex" in o for o in ops), f"Flex op in step graph: {ops}"
    return blob, ops


def class_matrix(model) -> np.ndarray:
    """Normalized cosine-head rows, ``(num_classes + 1, E)``, null last."""
    w = torch.nn.functional.normalize(model.head.weight.detach().float(), dim=-1)
    return w.cpu().numpy().astype(np.float32)


def check_step_parity(model, blob: bytes, rows: np.ndarray, cols: np.ndarray,
                      n_frames: int = 200, atol: float = 1e-4) -> dict:
    """Step the TFLite file frame by frame (state fed back, NaNs in the input)
    and compare per-frame probabilities and boundary logits with PyTorch's
    batch ``forward_frames`` on the same stream. Raises if they disagree."""
    import tensorflow as tf

    rng = np.random.default_rng(0)
    frames = (rng.standard_normal((n_frames, ROWS_PER_FRAME, 3)) * 0.3 + 0.5).astype(np.float32)
    frames[rng.random(frames.shape) < 0.06] = np.nan
    x = np.nan_to_num(frames)[:, rows][:, :, cols].reshape(1, n_frames, -1)
    model.eval()
    with torch.no_grad():
        gl, bl = model.forward_frames(torch.from_numpy(x))
    ref_p, ref_b = torch.softmax(gl, -1)[0].numpy(), bl[0].numpy()

    run = tf.lite.Interpreter(model_content=blob).get_signature_runner()
    w, scale = class_matrix(model), float(model.head.scale)
    mask = np.r_[model.head.class_mask.cpu().numpy(), True]
    state = np.zeros(state_shape(model), np.float32)
    got_p, got_b = [], []
    for t in range(n_frames):
        o = run(frame=frames[t], state=state)
        state = o["state_out"]
        logits = scale * (w @ o["embedding"])
        logits[~mask] = -np.inf
        p = np.exp(logits - logits.max())
        got_p.append(p / p.sum())
        got_b.append(float(o["boundary"][0]))
    dp = float(np.abs(np.array(got_p) - ref_p).max())
    db = float(np.abs(np.array(got_b) - ref_b).max())
    assert dp <= atol and db <= atol * 10, (
        f"step export disagrees with PyTorch: prob {dp:.2e}, boundary {db:.2e} "
        "-- the exported model is not the evaluated model")
    return {"max_prob_diff": dp, "max_boundary_logit_diff": db, "n_frames": n_frames}


def export_web(run_dir: Path, checkpoint: str = R.CKPT_BEST, decoder: dict | None = None,
               register: bool = True) -> dict:
    """Continuous run -> ``<run_dir>/export/web/``: ``model.tflite`` (step
    graph), ``classes.f32`` (class matrix, row-major float32) and
    ``manifest.json`` (everything the client needs to use them).

    ``decoder`` is recorded verbatim in the manifest (the client's D3/D1
    settings). Pass the ones chosen on the selection signers, e.g.
    ``{"name": "D3", "collapsed": True, "nu": 0.7, "min_len": 4}`` for C1.
    """
    model, ck = load_run_model(run_dir, checkpoint)
    arch = ck.get("arch")
    assert arch in STEP_ARCHS, f"step export supports {STEP_ARCHS}, not {arch!r}"
    rows = np.asarray(ck["landmarks"], dtype=np.int32)
    cols = np.asarray(["xyz".index(c) for c in ck.get("coords", "xyz")], dtype=np.int32)

    blob, ops = convert_step(_build_step_module(model, rows, cols, ck["feature_dim"]))
    parity = check_step_parity(model, blob, rows, cols)
    w = class_matrix(model)

    out = run_dir / "export" / "web"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "model.tflite").write_bytes(blob)
    (out / "classes.f32").write_bytes(w.tobytes())

    idx2sign = {i: s for s, i in ck["sign2idx"].items()}
    manifest = {
        "format": "signbridge-web-step/1",
        "run_id": int(run_dir.name),
        "architecture": arch,
        "checkpoint": checkpoint,
        "landmarks": rows.tolist(),
        "coords": ck.get("coords", "xyz"),
        "state_shape": list(state_shape(model)),
        "embed_dim": int(w.shape[1]),
        "cos_scale": float(model.head.scale),
        "glosses": [idx2sign[i] for i in range(len(idx2sign))],
        "null_index": len(idx2sign),
        "class_mask": model.head.class_mask.cpu().numpy().astype(bool).tolist(),
        "decoder": decoder,
        "tflite_ops": ops,
        "parity": parity,
        "sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                   for name in ("model.tflite", "classes.f32")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if register:
        R.register_assets(run_dir, web_step_tflite="export/web/model.tflite",
                          web_manifest="export/web/manifest.json")
    return {"out_dir": str(out), "tflite_mb": round(len(blob) / 1e6, 2), "ops": ops, **parity}


# ============================================================
# Isolated (whole-sign) classifiers -- TODO §12.9, "individual sign recognition" mode.
#
# StreamingGRU's head is a plain Sequential(LayerNorm, Dropout, Linear): a softmax classifier,
# no cosine head, no null class, no boundary signal. Simpler to export than the continuous
# family -- the classifier weights go straight into the graph, so there is no separate
# classes.f32/class_matrix step, and the manifest carries no null_index/class_mask/cos_scale.
# The stop condition (when a sign is "done") is external and lives in the browser, not the
# model: hand-out-of-frame or a stable repeated top-1 guess (sb.recognize's own analogue is the
# rule-based reset trigger already used for the isolated-classifier streaming eval, §4.3.2.6).
# ============================================================

def _build_isolated_step_module(model, rows: np.ndarray, cols: np.ndarray, feature_dim: int):
    """PyTorch isolated GRU classifier (plain, or behind a :class:`PhonologyFrontend`) -> tf.Module
    with one ``step`` signature returning per-frame class probabilities directly (no external
    class matrix needed -- the Linear head's weights are baked into the graph)."""
    import tensorflow as tf

    from sb.recognize.architectures import PhonologyFrontend

    def const(t):
        return tf.constant(t.detach().cpu().numpy().astype(np.float32))

    def layer_norm(x, ln):
        mu = tf.reduce_mean(x, -1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mu), -1, keepdims=True)
        return (x - mu) * tf.math.rsqrt(var + LAYER_NORM_EPS) * const(ln.weight) + const(ln.bias)

    rnn = model.gru
    n_layers, hidden = rnn.num_layers, rnn.hidden_size
    weights = [tuple(const(getattr(rnn, f"{w}_l{i}"))
                     for w in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))
               for i in range(n_layers)]
    norm = model.input_norm
    fe, ln_in = (norm[0], norm[1]) if isinstance(norm, torch.nn.Sequential) else (None, norm)
    assert fe is None or isinstance(fe, PhonologyFrontend), f"no isolated step port for front-end {type(fe).__name__}"
    head_ln, head_lin = model.head[0], model.head[2]  # head[1] is Dropout, a no-op at eval
    rows_c = tf.constant(rows, tf.int32)
    cols_c = tf.constant(cols, tf.int32)

    class Step(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec([ROWS_PER_FRAME, 3], tf.float32, name="frame"),
            tf.TensorSpec([n_layers, hidden], tf.float32, name="state"),
        ])
        def step(self, frame, state):
            x = tf.where(tf.math.is_nan(frame), tf.zeros_like(frame), frame)
            x = tf.gather(tf.gather(x, rows_c, axis=0), cols_c, axis=1)
            x = tf.reshape(x, [1, feature_dim])
            if fe is not None:
                x = _phono_tf(x, fe)
            x = layer_norm(x, ln_in)
            new_h = []
            for i, (w_ih, w_hh, b_ih, b_hh) in enumerate(weights):
                h = state[i:i + 1]
                i_r, i_z, i_n = tf.split(tf.matmul(x, w_ih, transpose_b=True) + b_ih, 3, axis=1)
                h_r, h_z, h_n = tf.split(tf.matmul(h, w_hh, transpose_b=True) + b_hh, 3, axis=1)
                r = tf.sigmoid(i_r + h_r)
                z = tf.sigmoid(i_z + h_z)
                h = (1.0 - z) * tf.tanh(i_n + r * h_n) + z * h
                new_h.append(h)
                x = h
            logits = tf.matmul(layer_norm(x, head_ln), const(head_lin.weight), transpose_b=True) \
                + const(head_lin.bias)
            probs = tf.nn.softmax(logits, axis=-1)
            return {"probs": tf.reshape(probs, [-1]), "state_out": tf.concat(new_h, 0)}

    return Step()


def check_isolated_step_parity(model, blob: bytes, rows: np.ndarray, cols: np.ndarray,
                               n_frames: int = 200, atol: float = 1e-4) -> dict:
    """Step the TFLite file frame by frame and compare with PyTorch's batch ``forward_all`` (the
    per-frame readout of the same trained head, not just the whole-clip last frame) on the same
    stream. Raises if they disagree."""
    import tensorflow as tf

    rng = np.random.default_rng(0)
    frames = (rng.standard_normal((n_frames, ROWS_PER_FRAME, 3)) * 0.3 + 0.5).astype(np.float32)
    frames[rng.random(frames.shape) < 0.06] = np.nan
    x = np.nan_to_num(frames)[:, rows][:, :, cols].reshape(1, n_frames, -1)
    model.eval()
    with torch.no_grad():
        logits = model.forward_all(torch.from_numpy(x))
    ref_p = torch.softmax(logits, -1)[0].numpy()

    run = tf.lite.Interpreter(model_content=blob).get_signature_runner()
    state = np.zeros((model.gru.num_layers, model.gru.hidden_size), np.float32)
    got_p = []
    for t in range(n_frames):
        o = run(frame=frames[t], state=state)
        state = o["state_out"]
        got_p.append(o["probs"])
    dp = float(np.abs(np.array(got_p) - ref_p).max())
    assert dp <= atol, (f"isolated step export disagrees with PyTorch: prob {dp:.2e} "
                        "-- the exported model is not the evaluated model")
    return {"max_prob_diff": dp, "n_frames": n_frames}


def export_web_isolated(run_dir: Path, checkpoint: str = R.CKPT_BEST, register: bool = True) -> dict:
    """Isolated run -> ``<run_dir>/export/web/``: ``model.tflite`` (step graph, outputs class
    probabilities directly) and ``manifest.json``. No ``classes.f32`` -- the classifier weights
    are baked into the graph, unlike the continuous family's external cosine-head matrix."""
    model, ck = load_run_model(run_dir, checkpoint)
    arch = ck.get("arch")
    assert arch in ISOLATED_STEP_ARCHS, f"isolated step export supports {ISOLATED_STEP_ARCHS}, not {arch!r}"
    rows = np.asarray(ck["landmarks"], dtype=np.int32)
    cols = np.asarray(["xyz".index(c) for c in ck.get("coords", "xyz")], dtype=np.int32)

    blob, ops = convert_step(_build_isolated_step_module(model, rows, cols, ck["feature_dim"]))
    parity = check_isolated_step_parity(model, blob, rows, cols)

    out = run_dir / "export" / "web"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "model.tflite").write_bytes(blob)

    idx2sign = {i: s for s, i in ck["sign2idx"].items()}
    manifest = {
        "format": "signbridge-web-isolated/1",
        "run_id": int(run_dir.name),
        "architecture": arch,
        "checkpoint": checkpoint,
        "landmarks": rows.tolist(),
        "coords": ck.get("coords", "xyz"),
        "state_shape": [model.gru.num_layers, model.gru.hidden_size],
        "glosses": [idx2sign[i] for i in range(len(idx2sign))],
        "tflite_ops": ops,
        "parity": parity,
        "sha256": {"model.tflite": hashlib.sha256((out / "model.tflite").read_bytes()).hexdigest()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if register:
        R.register_assets(run_dir, web_step_tflite="export/web/model.tflite",
                          web_manifest="export/web/manifest.json")
    return {"out_dir": str(out), "tflite_mb": round(len(blob) / 1e6, 2), "ops": ops, **parity}


# ============================================================
# Whole-clip (bidirectional) -- TODO §16.2, "record then recognize" mode.
#
# `bilstm` cannot step frame-by-frame; it reads the sequence forward *and*
# backward before it says anything. So this export takes the opposite shape
# from every step export above: one call, the whole clip, one softmax
# distribution out. It reuses `export/keras.py`'s already-parity-checked
# `build_bilstm` Keras rebuild (the same one the Kaggle grader path uses) --
# only the TFLite conversion step differs, because the Kaggle path allows
# `SELECT_TF_OPS` (Flex) and LiteRT.js in the browser does not.
# ============================================================

def _build_wholeclip_module(keras_model, rows: np.ndarray, cols: np.ndarray, feature_dim: int):
    """Keras rebuild -> tf.Module with one ``recognize(frames)`` signature: the
    whole clip's raw ``(T, 543, 3)`` frames in (NaN where undetected, dynamic
    T), a softmax distribution over the training glosses out. Same NaN
    handling / landmark-subset gather as the Kaggle `serving_module`, but
    reshaped for a single (unbatched) clip rather than the grader's batch-of-1
    convention, and wrapped so the dynamic time dimension converts without
    Flex (measured 2026-09-28: `Bidirectional(LSTM)` needs no fused-GRU-style
    fallback -- see `WHOLECLIP_ARCHS`'s note)."""
    import tensorflow as tf

    rows_c = tf.constant(rows, tf.int32)
    cols_c = tf.constant(cols, tf.int32)

    class WholeClip(tf.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        @tf.function(input_signature=[
            tf.TensorSpec([None, ROWS_PER_FRAME, 3], tf.float32, name="frames"),
        ])
        def recognize(self, frames):
            x = tf.where(tf.math.is_nan(frames), tf.zeros_like(frames), frames)
            x = tf.gather(tf.gather(x, rows_c, axis=1), cols_c, axis=2)
            t = tf.shape(x)[0]
            x = tf.reshape(x, [1, t, feature_dim])
            logits = self.model(x, training=False)
            return {"probs": tf.reshape(tf.nn.softmax(logits, axis=-1), [-1])}

    return WholeClip(keras_model)


def convert_wholeclip(module) -> tuple[bytes, list[str]]:
    """tf.Module -> (TFLite bytes, sorted op names), builtins only. The Keras
    RNN layers hold resource variables; freezing them to constants first is
    required (`export/tflite.py`'s `export_saved_model` hit
    ``READ_VARIABLE ... variable != nullptr`` inside the RNN without this, in
    the dynamic-loop grader path -- the same fix applies here). Freezing also
    loses the concrete function's output *name* (same doc note there), so the
    frozen function is re-wrapped in a module that re-declares the exact
    ``recognize(frames) -> {"probs": ...}`` signature before conversion."""
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2)

    frozen = convert_variables_to_constants_v2(module.recognize.get_concrete_function())

    class Frozen(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec([None, ROWS_PER_FRAME, 3], tf.float32, name="frames")])
        def recognize(self, frames):
            return {"probs": tf.identity(frozen(frames)[0], name="probs")}

    rewrapped = Frozen()
    conv = tf.lite.TFLiteConverter.from_concrete_functions(
        [rewrapped.recognize.get_concrete_function()], rewrapped)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    blob = conv.convert()
    interp = tf.lite.Interpreter(model_content=blob)
    ops = sorted({d["op_name"] for d in interp._get_ops_details()})
    assert not any("Flex" in o for o in ops), f"Flex op in whole-clip graph: {ops}"
    return blob, ops


def check_wholeclip_parity(torch_model, blob: bytes, rows: np.ndarray, cols: np.ndarray,
                           lengths: tuple[int, ...] = (1, 5, 22, 64, 131, 200),
                           atol: float = 1e-3) -> dict:
    """Run the exported TFLite file on several clip lengths (median/p95/max
    from §12.1's GISLR-Sentences facts, plus edge cases) and compare with
    PyTorch's own ``forward_full`` on the same frames. Raises if any disagree
    by more than `atol`."""
    import tensorflow as tf

    run = tf.lite.Interpreter(model_content=blob).get_signature_runner()
    torch_model.eval()
    worst = 0.0
    for i, n in enumerate(lengths):
        rng = np.random.default_rng(i)
        frames = (rng.standard_normal((n, ROWS_PER_FRAME, 3)) * 0.3 + 0.5).astype(np.float32)
        frames[rng.random(frames.shape) < 0.06] = np.nan
        x = np.nan_to_num(frames)[:, rows][:, :, cols].reshape(1, n, -1)
        with torch.no_grad():
            ref = torch.softmax(torch_model.forward_full(torch.from_numpy(x)), -1)[0].numpy()
        got = run(frames=frames)["probs"]
        worst = max(worst, float(np.abs(got - ref).max()))
    assert worst <= atol, (
        f"whole-clip export disagrees with PyTorch by {worst:.2e} (> {atol:.0e}) over "
        f"lengths {lengths} -- the exported model is not the evaluated model")
    return {"max_prob_diff": worst, "lengths": list(lengths)}


def export_web_wholeclip(run_dir: Path, checkpoint: str = R.CKPT_BEST,
                         register: bool = True) -> dict:
    """Bidirectional run -> ``<run_dir>/export/web/``: ``model.tflite`` (whole-
    clip graph, one call per recorded segment) and ``manifest.json``. No
    external class matrix -- the classifier weights are baked in, same as
    `export_web_isolated`; the difference is the input contract (whole clip,
    not one frame + state)."""
    torch_model, ck = load_run_model(run_dir, checkpoint)
    arch = ck.get("arch")
    assert arch in WHOLECLIP_ARCHS, f"whole-clip export supports {WHOLECLIP_ARCHS}, not {arch!r}"
    rows = np.asarray(ck["landmarks"], dtype=np.int32)
    cols = np.asarray(["xyz".index(c) for c in ck.get("coords", "xyz")], dtype=np.int32)

    keras_model = KE.build_keras_model(torch_model, arch, ck["feature_dim"])
    keras_parity = KE.check_parity(torch_model, keras_model, ck["feature_dim"])
    blob, ops = convert_wholeclip(_build_wholeclip_module(keras_model, rows, cols, ck["feature_dim"]))
    parity = check_wholeclip_parity(torch_model, blob, rows, cols)

    out = run_dir / "export" / "web"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "model.tflite").write_bytes(blob)

    idx2sign = {i: s for s, i in ck["sign2idx"].items()}
    manifest = {
        "format": "signbridge-web-wholeclip/1",
        "run_id": int(run_dir.name),
        "architecture": arch,
        "checkpoint": checkpoint,
        "landmarks": rows.tolist(),
        "coords": ck.get("coords", "xyz"),
        "glosses": [idx2sign[i] for i in range(len(idx2sign))],
        "tflite_ops": ops,
        "keras_parity": keras_parity,
        "parity": parity,
        "sha256": {"model.tflite": hashlib.sha256((out / "model.tflite").read_bytes()).hexdigest()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if register:
        R.register_assets(run_dir, web_step_tflite="export/web/model.tflite",
                          web_manifest="export/web/manifest.json")
    return {"out_dir": str(out), "tflite_mb": round(len(blob) / 1e6, 2), "ops": ops,
            "keras_parity": keras_parity, **parity}
