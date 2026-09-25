"""Browser export of the T5 gloss refiner (TODO §13 Phase 4, user decision 2026-09-25:
speech -> gloss runs in the browser, on the Workers Free plan).

Two ONNX graphs, no KV cache:

    encoder  input_ids (B, S) int64, attention_mask (B, S) int64
             -> hidden (B, S, d_model) float32
    decoder  decoder_input_ids (B, T) int64, encoder_hidden (B, S, d_model),
             encoder_attention_mask (B, S) int64 -> logits (B, T, vocab) float32

The client (``apps/web/src/speech/t5.ts``) owns the beam search. The decoder
re-reads the whole prefix each step, which is cheap for glosses of 10–20 tokens,
and it keeps the graph one plain forward with no past-key-value plumbing.

**Parity gate before anything is written:** both graphs are run in onnxruntime
against the PyTorch model on a few inputs (max |logit| diff). **int8:**
``quantize_dynamic`` on MatMul + Gather (the 32k × 768 embedding sits in both
graphs, 99 MB each in fp32). What quantization costs is measured on real
sentences by ``apps/web/tools/export_speech.py`` (int8 vs torch agreement),
not assumed.

**Workers static assets cap each file at 25 MiB**, so every ``.onnx`` is cut
into ``PART_BYTES`` chunks. The client fetches and concatenates the parts,
checks the SHA-256 of the whole, and keeps it in Cache Storage.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

PART_BYTES = 24 * 1024 * 1024  # Workers static assets: 25 MiB per file
OPSET = 17


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _wrappers(model):
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model.get_encoder()

        def forward(self, input_ids, attention_mask):
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    class Decoder(torch.nn.Module):
        """``model(...)``'s logits given the encoder output: runs the full model
        forward with ``encoder_outputs`` supplied, so the tied-embedding output
        scaling is whatever this transformers version does."""

        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, decoder_input_ids, encoder_hidden, encoder_attention_mask):
            return self.model(encoder_outputs=(encoder_hidden,), attention_mask=encoder_attention_mask,
                              decoder_input_ids=decoder_input_ids, use_cache=False).logits

    return Encoder().eval(), Decoder().eval()


def _export_fp32(model, tok, work: Path) -> dict[str, Path]:
    import torch

    enc, dec = _wrappers(model)
    ids = tok(["English: hello there Rule gloss: HELLO Produce ASL gloss:"] * 2, return_tensors="pt")
    with torch.no_grad():
        hidden = enc(ids.input_ids, ids.attention_mask)
    dids = torch.tensor([[0, 37, 5], [0, 12, 9]], dtype=torch.long)
    paths = {"encoder": work / "encoder.onnx", "decoder": work / "decoder.onnx"}
    with torch.no_grad():
        torch.onnx.export(enc, (ids.input_ids, ids.attention_mask), str(paths["encoder"]), dynamo=False,
                          input_names=["input_ids", "attention_mask"], output_names=["hidden"],
                          dynamic_axes={"input_ids": {0: "B", 1: "S"}, "attention_mask": {0: "B", 1: "S"},
                                        "hidden": {0: "B", 1: "S"}}, opset_version=OPSET)
        torch.onnx.export(dec, (dids, hidden, ids.attention_mask), str(paths["decoder"]), dynamo=False,
                          input_names=["decoder_input_ids", "encoder_hidden", "encoder_attention_mask"],
                          output_names=["logits"],
                          dynamic_axes={"decoder_input_ids": {0: "B", 1: "T"}, "encoder_hidden": {0: "B", 1: "S"},
                                        "encoder_attention_mask": {0: "B", 1: "S"}, "logits": {0: "B", 1: "T"}},
                          opset_version=OPSET)
    return paths


def check_parity(model, tok, paths: dict[str, Path], sentences: list[str]) -> dict[str, float]:
    """Max |diff| of encoder hidden and decoder logits, ONNX (CPU) vs PyTorch (CPU)."""
    import onnxruntime as ort
    import torch

    enc, dec = _wrappers(model)
    se = ort.InferenceSession(str(paths["encoder"]), providers=["CPUExecutionProvider"])
    sd = ort.InferenceSession(str(paths["decoder"]), providers=["CPUExecutionProvider"])
    b = tok(sentences, return_tensors="pt", padding=True)
    with torch.no_grad():
        h_t = enc(b.input_ids, b.attention_mask)
    h_o = se.run(None, {"input_ids": b.input_ids.numpy(), "attention_mask": b.attention_mask.numpy()})[0]
    dec_diff = 0.0
    # The tracer freezes Python shape branches (causal mask, padding), so check
    # the lengths beam search uses, the first step (T=1) above all.
    for t in (1, 2, 7):
        dids = torch.randint(3, 3000, (len(sentences), t))
        dids[:, 0] = model.config.decoder_start_token_id
        with torch.no_grad():
            l_t = dec(dids, h_t, b.attention_mask)
        l_o = sd.run(None, {"decoder_input_ids": dids.numpy(), "encoder_hidden": h_o,
                            "encoder_attention_mask": b.attention_mask.numpy()})[0]
        dec_diff = max(dec_diff, float(np.abs(l_o - l_t.numpy()).max()))
    return {"encoder_max_diff": float(np.abs(h_o - h_t.numpy()).max()), "decoder_max_diff": dec_diff}


def quantize(paths: dict[str, Path], work: Path) -> dict[str, Path]:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = {}
    for name, p in paths.items():
        q = work / f"{name}.int8.onnx"
        quantize_dynamic(str(p), str(q), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"])
        out[name] = q
    return out


def split_parts(src: Path, dest_dir: Path, stem: str) -> dict:
    """``src`` -> ``dest_dir/<stem>.partNN`` (each <= PART_BYTES), plus its manifest entry."""
    parts = []
    with open(src, "rb") as f:
        i = 0
        while block := f.read(PART_BYTES):
            name = f"{stem}.part{i:02d}"
            (dest_dir / name).write_bytes(block)
            parts.append(name)
            i += 1
    return {"parts": parts, "bytes": src.stat().st_size, "sha256": sha256_file(src)}


def export(model_dir: str | Path, work_dir: str | Path, out_dir: str | Path, *, t5_cfg: dict,
           preset: str = "guarded", parity_tol: float = 1e-3) -> dict:
    """Export, gate, quantize and chunk. ``work_dir`` keeps the fp32 and int8
    ``.onnx`` files (for the Node parity tests); ``out_dir`` gets what the
    browser loads: the parts, ``tokenizer.json`` and ``manifest.json``."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_dir, work, out = Path(model_dir), Path(work_dir), Path(out_dir)
    work.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    fp32 = _export_fp32(model, tok, work)
    probe = ["English: I will go to the store tomorrow. Rule gloss: I FUTURE GO STORE TOMORROW Produce ASL gloss:",
             "English: Can you help me? Rule gloss: YOU HELP I Produce ASL gloss:"]
    parity = check_parity(model, tok, fp32, probe)
    worst = max(parity.values())
    if worst > parity_tol:
        raise RuntimeError(f"ONNX vs PyTorch parity failed: {parity} (tol {parity_tol})")
    int8 = quantize(fp32, work)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files = {name: split_parts(p, out, f"{name}.int8.onnx") for name, p in int8.items()}
    shutil.copy2(model_dir / "tokenizer.json", out / "tokenizer.json")
    cfg = model.config
    manifest = {
        "_comment": "T5 gloss refiner for the browser (TODO §13). Written by sb.synthesize.gloss.export_web.",
        "source_dir": str(model_dir).replace("\\", "/"),
        "weights_sha256": sha256_file(next(model_dir.glob("*.safetensors"))) if any(model_dir.glob("*.safetensors")) else None,
        "files": files,
        "tokenizer": {"file": "tokenizer.json", "sha256": sha256_file(out / "tokenizer.json")},
        "decoder_start_token_id": cfg.decoder_start_token_id,
        "eos_token_id": cfg.eos_token_id,
        "pad_token_id": cfg.pad_token_id,
        "vocab_size": cfg.vocab_size,
        "max_input_len": t5_cfg["max_input_len"],
        "preset": preset,
        "generate": dict(t5_cfg["generate"][preset]),
        "guard": dict(t5_cfg.get("guard", {})),
        "parity_fp32": parity,
        "fp32_bytes": {k: p.stat().st_size for k, p in fp32.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (work / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest
