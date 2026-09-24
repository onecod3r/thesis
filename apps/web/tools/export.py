"""Build what the browser app loads, and the fixtures its parity tests compare against.

    .venv/Scripts/python.exe apps/web/tools/export.py assets     # -> apps/web/public/assets/
    .venv/Scripts/python.exe apps/web/tools/export.py fixtures   # -> apps/web/test/fixtures/

Both outputs are generated and gitignored. Python is the reference for every
number the app uses (TODO §12.5/§12.6), so this tool writes the reference
outputs next to the inputs:

``assets``
    - ``model/``: the C1 step export (``sb.recognize.export.step``). It is built
      if the registry has none.
    - ``prior.json``: the deployed trigram (``sb.rescore.prior.NgramLM.to_dict``).
    - ``lexicon.json``: the corpus POS lexicon that gloss -> English reads.
    - ``pipeline.json``: ``pipeline.config.json`` plus D3's settings and hashes.
    - ``replay/``: held-out evaluation-signer streams, the model's 132 landmarks
      in xy, with the glosses Python's TFLite + ``OnlineDecoder`` accept on them
      under the deployed rule. The app's replay check compares against these.

``fixtures`` (for ``npm test``, no browser)
    - ``decoder_probs.f32`` + ``decoder.json``: cached C1 per-frame outputs for
      clean and noisy streams, with ``OnlineDecoder``'s emissions for several rules.
    - ``ngram.json``: next-gloss distributions for many histories.
    - ``gloss2en.json``: English for every corpus sentence and every recognized
      (often wrong) gloss sequence from the demo analysis.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sb.core.paths import CACHE_DIR, gislr_sentences_dir
from sb.recognize.continuous import fuse as FU
from sb.recognize.continuous import select as SL
from sb.recognize.continuous.cache import iter_chunks
from sb.recognize.continuous.online import OnlineDecoder
from sb.recognize.continuous.train import run_dir_for
from sb.recognize.export.step import export_web
from sb.recognize.sequences import baselines as B
from sb.recognize.sequences.compose import read_sequence
from sb.recognize.sequences.corpus import load_lexicon, load_sentences
from sb.rescore import gloss2en as G
from sb.rescore import prior as P

APP = Path(__file__).resolve().parents[1]
CFG = json.loads((APP / "pipeline.config.json").read_text(encoding="utf-8"))
ASSETS, FIXTURES = APP / "public" / "assets", APP / "test" / "fixtures"
SEED, N_SELECT = 42, 5  # the signer split every §12 notebook uses


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=float), encoding="utf-8")


def web_export() -> tuple[Path, dict]:
    run_dir = run_dir_for(CFG["run"])
    assert run_dir is not None, f"no registry run for {CFG['run']}"
    web = run_dir / "export" / "web"
    if not (web / "manifest.json").exists():
        export_web(run_dir, register=False)
    return web, json.loads((web / "manifest.json").read_text())


def fitted_prior(manifest: dict):
    """The deployed trigram as ``(NgramLM, prior(history ids) -> label-order dist)``."""
    sents = load_sentences(CFG["corpus_version"])
    glosses = sorted({g for s in sents for g in s.glosses})
    lm = P.NgramLM(glosses, CFG["prior"]["order"], CFG["prior"]["discount"]).fit([s.glosses for s in sents])
    perm = np.array([glosses.index(g) for g in manifest["glosses"]])
    memo: dict = {}

    def prior(hist):
        if hist not in memo:
            memo[hist] = lm.gloss_dist([manifest["glosses"][c] for c in hist])[perm]
        return memo[hist]
    return lm, prior


def deployed_rule():
    r = CFG["rule"]
    if r["kind"] == "lattice":
        return SL.Lattice(theta=r["theta"], lam=r["lam"], k=r["k"], lag=r["lag"], max_len=r["max_len"])
    return FU.Rule(mode=r["mode"], lam=r["lam"], theta=r["theta"], max_len=r["max_len"])


def run_decoder(gp: np.ndarray, rule, prior, manifest: dict) -> list[list]:
    d = manifest["decoder"]
    dec = OnlineDecoder(manifest["null_index"], d["nu"], d["min_len"], rule, prior, collapse=CFG["collapse"])
    out = []
    for t, p in enumerate(gp):
        e = dec.step(p, t)
        if e:
            out.append([int(e[0]), int(e[1]), float(e[2]), t])
    out += [[int(c), int(f), float(conf), len(gp)] for c, f, conf in dec.flush_all(len(gp))]
    return out


def eval_rows() -> tuple[Path, pd.DataFrame]:
    root = gislr_sentences_dir("v1")
    seq = pd.read_csv(root / "sequences.csv", keep_default_na=False)
    _, ev = B.signer_split(seq["participant_id"], N_SELECT, SEED)
    return root, seq[(seq.kind == "sentence") & seq.participant_id.isin(ev)]


def assets() -> None:
    web, manifest = web_export()
    model_dir = ASSETS / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.tflite", "classes.f32", "manifest.json"):
        shutil.copyfile(web / name, model_dir / name)
    lm, prior = fitted_prior(manifest)
    write_json(ASSETS / "prior.json", lm.to_dict())
    lex = load_lexicon(CFG["corpus_version"])
    write_json(ASSETS / "lexicon.json", lex)

    # replay: seeded held-out streams + what Python's TFLite + OnlineDecoder accept on them
    import tensorflow as tf  # the same interpreter the demo notebook uses

    step = tf.lite.Interpreter(model_path=str(web / "model.tflite")).get_signature_runner()
    W = np.fromfile(web / "classes.f32", np.float32).reshape(-1, manifest["embed_dim"])
    lms_idx = np.asarray(manifest["landmarks"])
    root, ev = eval_rows()
    rng = np.random.default_rng(CFG["replay"]["seed"])
    rows = ev.iloc[sorted(rng.choice(len(ev), CFG["replay"]["n_streams"], replace=False))]
    rule = deployed_rule()
    index = []
    for _, row in rows.iterrows():
        raw = read_sequence(root / row["npz_relpath"])["landmarks"].astype(np.float32)
        xy = raw[:, lms_idx, :2]  # what the app stores and rebuilds
        frames = np.full((len(raw), 543, 3), np.nan, np.float32)
        frames[:, lms_idx, :2] = xy
        state = np.zeros(manifest["state_shape"], np.float32)
        gp = np.zeros((len(frames), W.shape[0]), np.float32)
        for t, f in enumerate(frames):
            o = step(frame=f, state=state)
            state = o["state_out"]
            logits = manifest["cos_scale"] * (W @ o["embedding"])
            e = np.exp(logits - logits.max())
            gp[t] = e / e.sum()
        em = run_decoder(gp, rule, prior, manifest)
        glosses = str(row["glosses"]).split()
        rec = [manifest["glosses"][c] for c, _, _, _ in em]
        name = f"{row['seq_id']}.f32"
        (ASSETS / "replay").mkdir(parents=True, exist_ok=True)
        xy.astype(np.float32).tofile(ASSETS / "replay" / name)
        index.append({"seq_id": row["seq_id"], "file": name, "frames": len(raw), "signer": int(row["participant_id"]),
                      "sentence_id": row["sentence_id"], "signed": glosses, "english_signed": G.convert(glosses, lex),
                      "expected": [{"gloss": manifest["glosses"][c], "frame": f, "conf": conf, "decided_at": at}
                                   for c, f, conf, at in em],
                      "expected_english": G.convert(rec, lex) if rec else ""})
    write_json(ASSETS / "replay" / "index.json", {"landmarks": manifest["landmarks"], "coords": "xy", "streams": index})

    pipe = {k: v for k, v in CFG.items() if not k.startswith("_")}
    pipe["decoder"] = manifest["decoder"]
    pipe["model"] = {"run_id": manifest["run_id"], "sha256": {n: sha256(model_dir / n) for n in ("model.tflite", "classes.f32")}}
    pipe["prior"]["sha256"] = sha256(ASSETS / "prior.json")
    write_json(ASSETS / "pipeline.json", pipe)
    print(f"assets -> {ASSETS}: model ({(model_dir / 'model.tflite').stat().st_size / 1e6:.2f} MB), prior, lexicon, "
          f"{len(index)} replay streams")


def fixtures() -> None:
    _, manifest = web_export()
    lm, prior = fitted_prior(manifest)
    FIXTURES.mkdir(parents=True, exist_ok=True)

    # decoder: cached C1 outputs (the offline twin of the browser path), several rules
    fwd = CACHE_DIR / "gislr" / "downstream" / "forward"
    streams = []
    for variant in ("sentence-clean", "sentence-noisy"):
        for i, st in enumerate(iter_chunks(fwd / variant / "eval")):
            if i >= 30:
                break
            streams.append((variant, st))
    rules = {
        "d3": FU.Rule(),
        "floor": FU.Rule(theta=0.3),
        "rescore_floor": FU.Rule(mode="rescore", lam=0.3, theta=0.3),
        "agree": FU.Rule(mode="agree", lam=1.0, k=10, theta_lo=0.1, theta_hi=0.3),
        "lattice_lag1": SL.Lattice(theta=0.2627, lam=0.3, k=5, lag=1),
        "lattice_lag2": SL.Lattice(theta=0.2687, lam=0.2, k=5, lag=2),
    }
    blobs, meta, off = [], [], 0
    for variant, st in streams:
        gp = st["gp"].astype(np.float32)
        blobs.append(gp)
        meta.append({"variant": variant, "row": st["row"], "offset": off, "frames": len(gp),
                     "expected": {name: run_decoder(gp, r, prior, manifest) for name, r in rules.items()}})
        off += len(gp)
    np.concatenate(blobs).astype(np.float32).tofile(FIXTURES / "decoder_probs.f32")
    write_json(FIXTURES / "decoder.json", {"n_classes": int(blobs[0].shape[1]), "null_index": manifest["null_index"],
                                           "nu": manifest["decoder"]["nu"], "min_len": manifest["decoder"]["min_len"],
                                           "rules": {n: r.as_dict() | {"kind": "lattice" if isinstance(r, SL.Lattice) else "rule"}
                                                     for n, r in rules.items()},
                                           "streams": meta})

    # n-gram: every prefix of 300 corpus sentences, plus random and out-of-corpus histories
    sents = load_sentences(CFG["corpus_version"])
    rng = np.random.default_rng(0)
    hists = {()}
    for s in [sents[i] for i in rng.choice(len(sents), 300, replace=False)]:
        for n in range(len(s.glosses)):
            hists.add(tuple(manifest["glosses"].index(g) for g in s.glosses[:n]))
    for _ in range(100):
        hists.add(tuple(int(x) for x in rng.integers(0, len(manifest["glosses"]), rng.integers(1, 5))))
    write_json(FIXTURES / "ngram.json", [{"history": list(h), "dist": prior(h).tolist()} for h in sorted(hists)])

    # gloss -> English: corpus sentences + recognized sequences (the hard, ungrammatical inputs)
    lex = load_lexicon(CFG["corpus_version"])
    seqs = {tuple(s.glosses) for s in sents}
    analysis = CACHE_DIR / "gislr" / "pipeline_demo" / "analysis" / "sentences.parquet"
    if analysis.exists():
        seqs |= {tuple(h.split()) for h in pd.read_parquet(analysis)["hyp"] if h}
    write_json(FIXTURES / "gloss2en.json", [{"glosses": list(s), "english": G.convert(list(s), lex)} for s in sorted(seqs)])
    print(f"fixtures -> {FIXTURES}: {len(streams)} decoder streams x {len(rules)} rules, {len(hists)} histories, "
          f"{len(seqs)} gloss sequences")


if __name__ == "__main__":
    what = sys.argv[1:] or ["assets", "fixtures"]
    for w in what:
        {"assets": assets, "fixtures": fixtures}[w]()
