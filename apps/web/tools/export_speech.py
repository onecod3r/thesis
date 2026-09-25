"""Speech -> gloss: build what the browser page loads, and its parity fixtures (TODO §13 Phase 4).

    .venv/Scripts/python.exe apps/web/tools/export_speech.py assets     # -> apps/web/public/assets/t5/
    .venv/Scripts/python.exe apps/web/tools/export_speech.py fixtures   # -> apps/web/test/fixtures/speech.json

Both outputs are generated and gitignored. ``--model-dir`` overrides the checkpoint
(default: ``t5.model_dir`` of ``experiments/synthesis/configs/aslg.text2gloss.json``),
e.g. the public ``t5-base`` for a dry run of the plumbing.

``assets``
    ``sb.synthesize.gloss.export_web.export``: encoder + decoder ONNX, parity-gated
    against PyTorch, int8, cut into < 25 MiB parts. The fp32/int8 ``.onnx`` files stay in
    ``data/cache/synthesis/t5-web/`` for the Node tests.

``fixtures``
    Python's answers on team30 + a fixed ASLG-PC12 test sample: ``spell_numbers``,
    ``rules_v1``, ``rules_v2``, T5's input ids, T5's ``generate()`` output (the deployed
    ``guarded`` preset, one sentence at a time, as the browser runs it) and the guard's
    decision on it. The TS tests compare against these.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sb.core.paths import CACHE_DIR, ROOT_DIR as REPO_ROOT

APP = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "experiments" / "synthesis" / "configs" / "aslg.text2gloss.json"
T5_OUT = APP / "public" / "assets" / "t5"
WORK = CACHE_DIR / "synthesis" / "t5-web"
FIXTURE = APP / "test" / "fixtures" / "speech.json"
N_ASLG = 300  # ASLG-PC12 test sentences in the fixture (seeded sample)
NUMBER_CASES = [
    "I will meet you at 3 o'clock.", "We leave at 10:30 tomorrow.", "She is 25 years old.",
    "It costs 2,000 dollars.", "The meeting is at 9:00.", "Pi is about 3.14 and 1.5 is a decimal.",
    "He has 101 dogs and 7 cats.", "Room 12 has 1000000 books.", "Call me at 5", "In 1999 we had 45,678 votes.",
]


def load_cfg() -> dict:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def model_dir(arg: str | None, cfg: dict) -> Path:
    p = Path(arg or cfg["t5"]["model_dir"])
    return p if p.is_absolute() else REPO_ROOT / p


def assets(args) -> None:
    from sb.synthesize.gloss.export_web import export

    cfg = load_cfg()
    md = model_dir(args.model_dir, cfg)
    if not (md / "config.json").is_file() or not any(md.glob("*.safetensors")):
        sys.exit(f"no checkpoint at {md} (needs config.json + model.safetensors)")
    t0 = time.time()
    m = export(md, WORK, T5_OUT, t5_cfg=cfg["t5"])
    sizes = {k: f"{v['bytes'] / 2**20:.0f} MiB in {len(v['parts'])} parts" for k, v in m["files"].items()}
    print(f"T5 -> {T5_OUT}: {sizes}; fp32 parity {m['parity_fp32']}; {time.time() - t0:.0f}s")


def sentences(cfg: dict) -> list[dict]:
    from sb.synthesize import evalsets
    from sb.synthesize.data import aslg_splits

    rows = [{"set": "team30", "english": e} for e in evalsets.load("team30", cfg["eval"]["team30_version"]).english]
    test = aslg_splits(cfg["splits"]["fractions"], cfg["splits"]["seed"], cfg["splits"]["version"])["test"]
    rows += [{"set": "aslg_test", "english": e} for e in test.sample(N_ASLG, random_state=0).text]
    return rows


def fixtures(args) -> None:
    import torch

    from sb.synthesize.asr import spell_numbers
    from sb.synthesize.gloss import hybrid, rules_v1, rules_v2, split_sentences
    from sb.synthesize.gloss.t5 import T5Refiner, build_input

    cfg = load_cfg()
    t5_cfg = cfg["t5"]
    md = model_dir(args.model_dir, cfg)
    rows = sentences(cfg)
    for r in rows:
        r["rules_v1"] = rules_v1.convert(r["english"])
        r["rules_v2"] = rules_v2.convert(r["english"])
    ref = T5Refiner.from_config(t5_cfg, preset="guarded", model_dir=md)
    tok, model = ref.tokenizer, ref.model
    gen = dict(t5_cfg["generate"]["guarded"])
    t0 = time.time()
    for r in rows:
        enc = tok([build_input(r["english"], r["rules_v1"])], return_tensors="pt",
                  max_length=t5_cfg["max_input_len"], truncation=True).to(ref.device)
        r["input_ids"] = enc.input_ids[0].tolist()
        with torch.no_grad():
            ids = model.generate(**enc, **gen)[0].tolist()
        r["t5_ids"] = ids
        dec = tok.decode(ids, skip_special_tokens=True).strip()
        r["t5"] = dec.upper() if dec else r["rules_v1"]
        o = hybrid.combine(r["english"], r["rules_v1"], r["t5"], r["rules_v2"],
                           min_content_recall=t5_cfg["guard"]["min_content_recall"])
        r["hybrid"] = {"gloss": o.gloss, "source": o.source, "reasons": o.reasons}
    t5_s = time.time() - t0
    out = {
        "_comment": "Written by apps/web/tools/export_speech.py fixtures. Python's reference answers.",
        "model_dir": str(md.relative_to(REPO_ROOT) if md.is_relative_to(REPO_ROOT) else md).replace("\\", "/"),
        "generate": gen,
        "max_input_len": t5_cfg["max_input_len"],
        "numbers": [{"text": s, "spelled": spell_numbers(s)} for s in NUMBER_CASES],
        "split": [{"text": t, "sentences": split_sentences(t)} for t in
                  ["Hello there. How are you? I am fine!", "No punctuation", "Wait... what?! Yes."]],
        "sentences": rows,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} sentences -> {FIXTURE} (T5 {t5_s:.0f}s on {ref.device})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["assets", "fixtures"])
    ap.add_argument("--model-dir")
    args = ap.parse_args()
    {"assets": assets, "fixtures": fixtures}[args.what](args)


if __name__ == "__main__":
    main()
