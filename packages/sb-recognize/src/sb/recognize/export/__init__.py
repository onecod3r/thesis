"""Deployment export.

TFLite is reached by rebuilding the trained model in native Keras and
transferring weights (``keras``), gated on numerical parity with the PyTorch
model — the ONNX/onnx2tf route failed on 3 of 4 architectures (TODO §6.2).
``tflite`` drives that end to end and packages submission.zip.

``step`` is the browser export for continuous models (TODO §12.5): a
single-frame, stateful, Flex-free graph for LiteRT.js. The client owns the
loop, the state and the cosine class matrix.
"""
