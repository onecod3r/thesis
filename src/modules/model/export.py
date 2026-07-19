"""Deployment export: registry run -> Keras rebuild -> TFLite -> submission.zip.

This is the Kaggle-submission path, moved out of the training notebooks (which
are training drivers only) so every architecture exports through one code path.
Driven by `src/gislr.2.models.evaluation.ipynb`.

The grader's contract (asl-signs competition):

- input  ``inputs``  — raw ``(T, 543, 3)`` float32 frames, **NaNs included**,
  one video, variable T
- output ``outputs`` — ``(250,)`` float scores
- called through the TFLite ``serving_default`` signature runner
- < 100 ms/video average, < 40 MB model

NaN cleanup and the landmark-subset gather live **inside** the exported graph,
so the exported model owns its preprocessing and the grader's raw frames work
unchanged — including for xy-trained runs, whose z column is dropped internally.

**Route: native Keras rebuild, not ONNX.** PyTorch -> ONNX -> ``onnx2tf`` was
tried first and abandoned (TODO §6.2): onnx2tf declares no dependencies, cannot
convert the ``IsInf`` op ``torch.nan_to_num`` emits, permutes 3D input layouts,
fails outright on the GRU/BiLSTM/CNN graphs, and emits a malformed ``Squeeze``
for the one architecture it does convert. ``keras_export`` instead rebuilds each
architecture with native Keras layers and transfers the trained weights; Keras'
recurrent layers are on TFLite's well-supported conversion path.

That trades a conversion problem for a **weight-transfer** problem, so every
export is gated on numerical parity: the PyTorch model and its Keras rebuild are
run on the same input and must agree, and nothing is written if they don't
(``keras_export.check_parity``, measured ~4e-6 for all four architectures).

Remaining caveat: the exported model consumes the **full** sequence, while
training/eval uniformly subsample past ``MAX_SEQ_LEN = 128`` frames. Verify
behaviour on long videos before trusting a submission.
"""

import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch

from modules.model import keras_export as KE
from modules.model import registry as R
from modules.model.architectures import build_model
from modules.model.data import ROWS_PER_FRAME

N_CLASSES = 250
TFLITE_SIZE_CAP_MB = 40.0  # competition storage cap


def load_run_model(run_dir: Path, checkpoint: str = R.CKPT_BEST):
    """Rebuild a run's trained PyTorch model + its export metadata."""
    ck = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=False)
    model = build_model(
        ck.get("arch", "gru"), ck["feature_dim"], len(ck["sign2idx"]), ck["hyp"]
    )
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def build_serving_model(run_dir: Path, checkpoint: str = R.CKPT_BEST):
    """Parity-checked Keras serving module for a run, ready for TFLite.

    Returns ``(serving_module, parity_diff, checkpoint_dict)``.
    """
    torch_model, ck = load_run_model(run_dir, checkpoint)
    arch = ck.get("arch", "gru")
    coords = ck.get("coords", "xyz")
    coord_cols = ["xyz".index(c) for c in coords]

    keras_model = KE.build_keras_model(torch_model, arch, ck["feature_dim"])
    diff = KE.check_parity(torch_model, keras_model, ck["feature_dim"])
    return KE.serving_module(keras_model, ck["landmarks"], coord_cols), diff, ck


def export_saved_model(serving, export_dir: Path) -> Path:
    """Freeze the model's weights into graph constants, then save it.

    Saving the Keras-backed module directly keeps its weights as **resource
    variables**, and the resulting TFLite model dies at invoke time with
    ``READ_VARIABLE ... variable != nullptr`` inside the RNN's WHILE loop (the
    variables are never initialized in the TFLite runtime). Disabling
    ``experimental_enable_resource_variables`` on the converter does not help.

    Freezing to constants fixes it — but ``from_concrete_functions`` on the
    frozen function loses the output *name*, and the grader calls
    ``prediction_fn(inputs=...)["outputs"]`` by name. So the frozen function is
    re-wrapped in a module that re-declares the exact signature, and conversion
    goes through the SavedModel again.
    """
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2)

    frozen = convert_variables_to_constants_v2(
        serving.serving_default.get_concrete_function())

    class FrozenServing(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec([None, ROWS_PER_FRAME, 3], tf.float32, name="inputs")])
        def serving_default(self, inputs):
            return {"outputs": tf.identity(frozen(inputs)[0], name="outputs")}

    module = FrozenServing()
    path = export_dir / "saved_model"
    tf.saved_model.save(module, str(path),
                        signatures={"serving_default": module.serving_default})
    return path


def export_tflite(export_dir: Path, quantize: bool = False) -> Path:
    """Convert the frozen SavedModel to TFLite.

    ``quantize=False`` by default. ``tf.lite.Optimize.DEFAULT`` applies int8
    dynamic-range quantization to the weights, which shifted logits by ~1e-1 vs
    PyTorch (caught by ``validate_against_torch``) — and these models are ~1-3 MB
    against a 40 MB cap, so it buys nothing and costs accuracy. Only enable it if
    a future model actually approaches the size limit, and re-check parity if so.
    """
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir / "saved_model"))
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,  # recurrent dynamic-loop ops need the fallback
    ]
    path = export_dir / "model.tflite"
    path.write_bytes(converter.convert())
    return path


def validate_tflite(tflite_path: Path, n_frames: int = 38,
                    nan_rate: float = 0.1) -> dict:
    """Run the exported model exactly the way the grader does, NaNs included."""
    import tensorflow as tf

    size_mb = tflite_path.stat().st_size / 1e6
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    sigs = list(interpreter.get_signature_list())
    assert "serving_default" in sigs, f"missing serving_default signature; found {sigs}"
    runner = interpreter.get_signature_runner("serving_default")

    rng = np.random.default_rng(0)
    dummy = rng.standard_normal((n_frames, ROWS_PER_FRAME, 3)).astype(np.float32)
    dummy[rng.random(dummy.shape) < nan_rate] = np.nan  # grader data contains NaN
    out = runner(inputs=dummy)["outputs"]
    assert out.shape[-1] == N_CLASSES, f"expected {N_CLASSES} classes, got {out.shape}"
    assert np.isfinite(out).all(), "exported model produced non-finite scores"
    return {
        "size_mb": round(size_mb, 2),
        "within_size_cap": size_mb < TFLITE_SIZE_CAP_MB,
        "output_shape": list(out.shape),
        "argmax": int(np.argmax(out)),
    }


def validate_against_torch(run_dir: Path, tflite_path: Path,
                           checkpoint: str = R.CKPT_BEST, n_frames: int = 64,
                           atol: float = 1e-3) -> float:
    """End-to-end check: the TFLite file vs the PyTorch model, raw frames in.

    This is the one that matters — it exercises the whole deployed path
    (NaN handling, landmark gather, coordinate selection, quantized graph)
    rather than just the Keras rebuild, so it catches anything the conversion
    itself changed.
    """
    import tensorflow as tf

    torch_model, ck = load_run_model(run_dir, checkpoint)
    coord_cols = ["xyz".index(c) for c in ck.get("coords", "xyz")]
    rows = np.asarray(ck["landmarks"])

    rng = np.random.default_rng(1)
    frames = rng.standard_normal((n_frames, ROWS_PER_FRAME, 3)).astype(np.float32)
    frames[rng.random(frames.shape) < 0.06] = np.nan  # realistic NaN rate

    clean = np.nan_to_num(frames, nan=0.0)
    x = clean[:, rows, :][:, :, coord_cols].reshape(1, n_frames, -1)
    with torch.no_grad():
        ref = torch_model.forward_full(torch.from_numpy(x)).numpy().squeeze(0)

    runner = tf.lite.Interpreter(model_path=str(tflite_path)).get_signature_runner(
        "serving_default")
    got = runner(inputs=frames)["outputs"]

    diff = float(np.abs(ref - got).max())
    assert diff <= atol, (
        f"{run_dir.name}: TFLite output differs from PyTorch by {diff:.2e} "
        f"(> {atol:.0e}) — the exported model is not the evaluated model")
    return diff


def package_submission(export_dir: Path) -> Path:
    """model.tflite -> submission.zip, the artifact the competition wants."""
    zip_path = export_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(export_dir / "model.tflite", arcname="model.tflite")
    return zip_path


def export_run(run_dir: Path, checkpoint: str = R.CKPT_BEST, clean: bool = True,
               register: bool = True) -> dict:
    """Full chain for one run: Keras rebuild -> parity -> SavedModel -> TFLite ->
    validation (shape + vs PyTorch) -> submission.zip. Returns a summary dict.

    Everything lands in ``<run_dir>/export/`` (gitignored). ``clean=True``
    removes a previous export first, so a re-run can never mix artifacts from
    two different checkpoints.
    """
    export_dir = run_dir / "export"
    if clean and export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    serving, parity, ck = build_serving_model(run_dir, checkpoint)
    export_saved_model(serving, export_dir)
    tflite_path = export_tflite(export_dir)
    validation = validate_tflite(tflite_path)
    tflite_parity = validate_against_torch(run_dir, tflite_path, checkpoint)
    zip_path = package_submission(export_dir)

    if register:
        R.register_assets(run_dir, submission_zip="export/submission.zip",
                          tflite="export/model.tflite")
    return {
        "run_id": int(run_dir.name),
        "arch": ck.get("arch", "gru"),
        "keras_parity": parity,
        "tflite_parity": tflite_parity,
        "submission_zip": str(zip_path),
        **validation,
    }
