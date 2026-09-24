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
from sb.recognize.export.tflite import load_run_model

LAYER_NORM_EPS = 1e-5  # PyTorch nn.LayerNorm default
STEP_ARCHS = ("gru_continuous",)


def _build_step_module(model, rows: np.ndarray, cols: np.ndarray, feature_dim: int):
    """PyTorch ContinuousGRU -> tf.Module with one ``step`` signature."""
    import tensorflow as tf

    def const(t):
        return tf.constant(t.detach().cpu().numpy().astype(np.float32))

    def layer_norm(x, ln):
        mu = tf.reduce_mean(x, -1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mu), -1, keepdims=True)
        return (x - mu) * tf.math.rsqrt(var + LAYER_NORM_EPS) * const(ln.weight) + const(ln.bias)

    gru = model.gru
    n_layers, hidden = gru.num_layers, gru.hidden_size
    weights = [tuple(const(getattr(gru, f"{w}_l{i}"))
                     for w in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))
               for i in range(n_layers)]
    emb_ln, emb_lin = model.head.embed[0], model.head.embed[1]
    b_ln, b_lin = model.boundary[0], model.boundary[1]
    rows_c = tf.constant(rows, tf.int32)
    cols_c = tf.constant(cols, tf.int32)

    class Step(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec([ROWS_PER_FRAME, 3], tf.float32, name="frame"),
            tf.TensorSpec([n_layers, hidden], tf.float32, name="state"),
        ])
        def step(self, frame, state):
            # NaN -> 0 with a NaN-only test (no IsInf op), as in keras.serving_module
            x = tf.where(tf.math.is_nan(frame), tf.zeros_like(frame), frame)
            x = tf.gather(tf.gather(x, rows_c, axis=0), cols_c, axis=1)
            x = layer_norm(tf.reshape(x, [1, feature_dim]), model.input_norm)
            new_state = []
            for i, (w_ih, w_hh, b_ih, b_hh) in enumerate(weights):
                h = state[i:i + 1]
                i_r, i_z, i_n = tf.split(tf.matmul(x, w_ih, transpose_b=True) + b_ih, 3, axis=1)
                h_r, h_z, h_n = tf.split(tf.matmul(h, w_hh, transpose_b=True) + b_hh, 3, axis=1)
                r = tf.sigmoid(i_r + h_r)
                z = tf.sigmoid(i_z + h_z)
                h = (1.0 - z) * tf.tanh(i_n + r * h_n) + z * h
                new_state.append(h)
                x = h
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
    state = np.zeros((model.gru.num_layers, model.gru.hidden_size), np.float32)
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
        "state_shape": [model.gru.num_layers, model.gru.hidden_size],
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
