"""Builder for gislr.3.streaming.bilstm-wholeclip-eval.ipynb -- run once to (re)generate
the notebook file, then execute it with nbclient/jupyter. Not itself part of the pipeline;
a scratch generator, kept only so the notebook's cells can be regenerated if this changes."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "gislr.3.streaming.bilstm-wholeclip-eval.ipynb"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


cells = []

cells.append(md("""\
# GISLR — BiLSTM whole-clip evaluation: "record then recognize" mode (TODO §16.2)

**What this notebook does.** Diagnostic, not a pipeline stage. Measures the record-then-
recognize design before any browser UI is built: segment a recorded stream by
**stillness** (the offline sibling of `apps/web/src/pipeline/movement.ts`'s live gate,
ported in `sb.recognize.sequences.windows.movement_segments` — same constants, so this
tests the actual heuristic the browser will run, not a stand-in for it), classify each
segment whole with **`bilstm`** (the accuracy leader, §4.1/§4.3, 0.7569 canonical, run
`1784447175`, exported Flex-free in `export/step.py::export_web_wholeclip`), and re-rank
with a **sentence-context prior** (the same Kneser-Ney trigram already deployed for the
continuous pipeline, §12.6 — `pipeline.config.json`'s `prior`). No training: `bilstm` is
already trained, and `NgramLM.fit` is a closed-form count fit, not a gradient step.

**Data.** GISLR-Sentences v1 (§12.1), `sb.core.paths.gislr_sentences_dir`.

**Arms, in order of what each one isolates:**

| arm | segmentation | prior | isolates |
|---|---|---|---|
| B0 | oracle (true sign boundaries) | none | `bilstm`'s own classification error, no segmentation noise — the ceiling this pipeline could reach |
| B-mv | `movement_segments` | none | what stillness-only segmentation costs on top of B0 |
| B0+prior / B-mv+prior | same two | trigram, shallow fusion (`rescore`) | how much of either arm's error the sentence-context downstream model recovers |

Compare the headline GER against the existing references: oracle B1 `gru_reg` **0.221**,
best streaming baseline (`gru` sliding window) **0.507**, best continuous model (C1 D3
collapsed) **0.293** (`docs/reports/sentence-baselines.md`, `continuous-models.md`) —
this pipeline is a *different* mode (record-then-process, not live-streaming), so it is
not a drop-in replacement for those; the comparison says whether it is competitive enough
to be worth shipping as a second mode.

**Artifacts:** `data/cache/gislr/bilstm_wholeclip_eval/results/` — `sweep_*.json`,
`final.json`, `final_table.csv`."""))

cells.append(md("## Setup"))

cells.append(code("""\
# ============================================================
# Imports, tunables, dataset, signer split, model + features
# ============================================================
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from sb.core.paths import CACHE_DIR, MODELS_DIR, gislr_sentences_dir
from sb.recognize.continuous import fuse as F
from sb.recognize.sequences import baselines as B
from sb.recognize.sequences import metrics as M
from sb.recognize.sequences.compose import FRAME_KINDS
from sb.recognize.sequences.corpus import load_sentences
from sb.recognize.sequences.windows import movement_segments
from sb.rescore.prior import NgramLM, viterbi_rescore

RUN_ID = 1784447175          # bilstm, ME_126 xy, 0.7569 canonical -- TODO §4.1/§4.3 leader
CORPUS_VERSION = "v1"
N_SELECTION_SIGNERS = 5       # same split convention as sentence-baselines.ipynb
SEED = 42
MV_MIN_LEN = 3                # drop movement_segments spans shorter than this (noise, not a sign)
LAM = 0.2                     # prior weight -- pipeline.config.json's deployed value, starting point
THETA_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]  # rescore acceptance floor, swept on selection signers
LIMIT_PER_GROUP = None        # e.g. 20 for a quick smoke run

DATASET_ROOT = gislr_sentences_dir(CORPUS_VERSION)
OUT = CACHE_DIR / "gislr" / "bilstm_wholeclip_eval" / ("results" if LIMIT_PER_GROUP is None else "results_smoke")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

LABEL_MAP = json.loads((DATASET_ROOT / "sign_to_prediction_index_map.json").read_text(encoding="utf-8"))
GLOSSES = [g for g, _ in sorted(LABEL_MAP.items(), key=lambda kv: kv[1])]
N_CLASSES = len(LABEL_MAP)
SEQ = pd.read_csv(DATASET_ROOT / "sequences.csv", keep_default_na=False)
SEQ["row"] = np.arange(len(SEQ))
SELECT, EVAL = B.signer_split(SEQ["participant_id"], N_SELECTION_SIGNERS, SEED)
SEQ["group"] = np.where(SEQ["participant_id"].isin(SELECT), "select", "eval")
if LIMIT_PER_GROUP is not None:
    SEQ = SEQ.groupby(["kind", "group"], group_keys=False).head(LIMIT_PER_GROUP)

MODEL = B.load_registry_model(MODELS_DIR / str(RUN_ID), N_CLASSES, DEVICE)
assert MODEL.arch == "bilstm", MODEL.arch
FEATS = B.StreamFeatures.build(DATASET_ROOT, pd.read_csv(DATASET_ROOT / "sequences.csv", keep_default_na=False),
                                MODEL.rows, MODEL.coords,
                                progress=lambda d, t: None)

NGRAM = NgramLM(GLOSSES, order=3, discount=0.75).fit([s.glosses for s in load_sentences(CORPUS_VERSION)])


def prior_fn(history: tuple) -> np.ndarray:
    \"\"\"class-index history -> a distribution over GLOSSES, same order as `bilstm`'s
    softmax. Fit on the full corpus, same choice as the deployed continuous prior
    (`pipeline.config.json`) -- the circularity caveat there applies here too (TODO
    §12.1): the corpus is Claude-written, so a held-out-sentence protocol would be the
    stricter test; not built here, flagged as a follow-up below.\"\"\"
    return NGRAM.gloss_dist(tuple(GLOSSES[c] for c in history))


def seq_arrays(r) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array(r["labels"].split(), dtype=np.int64)
    seg = np.stack([np.array(r["starts"].split(), int), np.array(r["ends"].split(), int)], 1)
    return labels, seg


def frame_totals(rows) -> dict:
    kinds = np.concatenate([FEATS.kind(i) for i in rows]) if len(rows) else np.array([], np.uint8)
    return {name: int((kinds == k).sum()) for k, name in FRAME_KINDS.items()}


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


print(f"dataset: {DATASET_ROOT}")
print(f"model: {MODEL.name} ({MODEL.arch}, {len(MODEL.rows)} rows/{MODEL.coords})")
print(f"results: {OUT}   device: {DEVICE}")
print(f"selection signers {SELECT}\\nevaluation signers ({len(EVAL)}) {EVAL}")
print(SEQ.groupby(["kind", "group"]).size().unstack())"""))

cells.append(md("""\
## 1. Segmentation quality (diagnostic, before any scoring)

How well does stillness alone carve a recorded stream into sign-shaped spans? Not a
score by itself -- §2/§3 below measure end to end -- but a sanity check that the
constants ported from `movement.ts` behave reasonably on real streams before trusting
what comes out of them."""))

cells.append(code("""\
# ============================================================
# movement_segments vs the true sign count/coverage (selection signers, sentence kind)
# ============================================================
sel_sentence = SEQ[(SEQ.group == "select") & (SEQ.kind == "sentence")]
n_true, n_seg, cover = [], [], []
for _, r in tqdm(sel_sentence.iterrows(), total=len(sel_sentence), desc="segmentation check"):
    x = FEATS.x(r["row"])
    labels, seg = seq_arrays(r)
    segs = movement_segments(x, min_len=MV_MIN_LEN)
    n_true.append(len(labels))
    n_seg.append(len(segs))
    hit = 0
    for a, b in seg:
        best = max((max(0, min(b, e) - max(a, s)) for s, e in segs), default=0)
        hit += best >= 0.5 * (b - a)
    cover.append(hit / max(len(labels), 1))
print(f"{len(sel_sentence)} sequences | true signs/seq {np.mean(n_true):.2f} | "
      f"candidate segments/seq {np.mean(n_seg):.2f} | mean 50%-overlap coverage {np.mean(cover):.3f}")"""))

cells.append(md("## 2. B0: oracle segmentation + `bilstm` (the ceiling)"))

cells.append(code("""\
# ============================================================
# B0 -- true sign spans, bilstm.forward_full per span, no prior. Also the per-span
# top-5 hit rate: how often the true gloss is among bilstm's 5 highest-probability
# classes on its own span, the headroom a downstream re-ranker has to work with.
# ============================================================
@torch.no_grad()
def span_probs(x: np.ndarray, spans: list[tuple[int, int]]) -> list[np.ndarray]:
    out = []
    for a, b in spans:
        t = torch.from_numpy(np.ascontiguousarray(x[a:b])).unsqueeze(0).to(DEVICE)
        out.append(torch.softmax(MODEL.model.forward_full(t), -1).squeeze(0).float().cpu().numpy())
    return out


def oracle_segments(r) -> tuple[list[F.Segment], np.ndarray]:
    x = FEATS.x(r["row"])
    labels, seg = seq_arrays(r)
    spans = [(int(a), int(b)) for a, b in seg]
    qs = span_probs(x, spans)
    return [F.Segment(a, b, q, float(b - a), q) for (a, b), q in zip(spans, qs)], labels


path = OUT / "b0_top5.json"
if not path.exists():
    top5_hits, n_spans = 0, 0
    for _, r in tqdm(SEQ[(SEQ.group == "select") & (SEQ.kind == "sentence")].iterrows(),
                     total=(SEQ.group.eq("select") & SEQ.kind.eq("sentence")).sum(), desc="B0 top-5"):
        segs, labels = oracle_segments(r)
        for seg, lab in zip(segs, labels):
            n_spans += 1
            top5_hits += int(lab in np.argsort(-seg.q)[:5])
    write_json(path, {"top5_hit_rate": top5_hits / max(n_spans, 1), "n_spans": n_spans})
print(json.loads(path.read_text()))"""))

cells.append(md("## 3. B-mv: `movement_segments` + `bilstm` (no downstream context yet)"))

cells.append(code("""\
# ============================================================
# Shared scorer: one arm (segmentation fn, optional prior+rule) over one subset of SEQ
# ============================================================
def score_arm(subset: pd.DataFrame, segment_fn, rule: F.Rule, prior) -> list[dict]:
    out = []
    for _, r in tqdm(subset.iterrows(), total=len(subset), desc="scoring", leave=False):
        x = FEATS.x(r["row"])
        labels, seg = seq_arrays(r)
        spans = segment_fn(x, seg)
        qs = span_probs(x, spans)
        segs = [F.Segment(a, b, q, float(b - a), q) for (a, b), q in zip(spans, qs)]
        emissions = F.decode_fused(segs, prior, rule)
        out.append(M.score_sequence(emissions, labels, seg, FEATS.kind(r["row"])))
    return out


def oracle_spans(x, seg):
    return [(int(a), int(b)) for a, b in seg]


def movement_spans(x, seg):
    return movement_segments(x, min_len=MV_MIN_LEN)


NO_PRIOR = F.Rule(mode="none", theta=0.0)"""))

cells.append(code("""\
# ============================================================
# B0 / B-mv, no prior, selection signers (both segmentations, sentence kind)
# ============================================================
sel = SEQ[(SEQ.group == "select") & (SEQ.kind == "sentence")]
for name, fn in (("b0", oracle_spans), ("bmv", movement_spans)):
    path = OUT / f"select_{name}_noprior.json"
    if not path.exists():
        sc = score_arm(sel, fn, NO_PRIOR, None)
        write_json(path, {"aggregate": M.aggregate(sc, frame_totals(sel["row"])), "n": len(sc)})
    print(name, json.loads(path.read_text())["aggregate"]["ger"])"""))

cells.append(md("""\
## 4. Downstream: sentence-context re-ranking

Shallow fusion (`rescore`: accept `argmax(q · p^lam)` if its normalized score clears
`theta`), the simplest of `fuse.Rule`'s modes and the one that needs no per-segment
lattice look-ahead -- appropriate here since a recorded clip's segments are all in hand
at once anyway (no live-decoding latency constraint to trade off, unlike §12.6's lag-2
lattice)."""))

cells.append(code("""\
# ============================================================
# Sweep theta (fixed lam) on selection signers, oracle and movement segmentation
# ============================================================
for name, fn in (("b0", oracle_spans), ("bmv", movement_spans)):
    path = OUT / f"sweep_{name}_prior.json"
    if path.exists():
        continue
    rows = []
    for theta in THETA_GRID:
        rule = F.Rule(mode="rescore", lam=LAM, theta=theta)
        sc = score_arm(sel, fn, rule, prior_fn)
        rows.append({"theta": theta, "lam": LAM, **M.aggregate(sc, frame_totals(sel["row"]))})
    write_json(path, rows)
    best = min(rows, key=lambda d: d["ger"])
    print(name, "best theta", best["theta"], "selection GER", round(best["ger"], 4))"""))

cells.append(md("## 5. Final: evaluation signers, every arm"))

cells.append(code("""\
# ============================================================
# Pick each prior arm's theta on selection GER, then score every arm on the
# evaluation signers, both dataset kinds (sentence / control).
# ============================================================
chosen_theta = {}
for name in ("b0", "bmv"):
    sweep = json.loads((OUT / f"sweep_{name}_prior.json").read_text())
    chosen_theta[name] = min(sweep, key=lambda d: d["ger"])["theta"]
print("chosen theta:", chosen_theta)

final = []
for kind in ("sentence", "control"):
    ev = SEQ[(SEQ.group == "eval") & (SEQ.kind == kind)]
    ft = frame_totals(ev["row"])
    for name, fn in (("b0", oracle_spans), ("bmv", movement_spans)):
        sc = score_arm(ev, fn, NO_PRIOR, None)
        final.append({"arm": name, "prior": False, "kind": kind, **M.aggregate(sc, ft)})
        rule = F.Rule(mode="rescore", lam=LAM, theta=chosen_theta[name])
        sc = score_arm(ev, fn, rule, prior_fn)
        final.append({"arm": name, "prior": True, "kind": kind, "theta": chosen_theta[name],
                      **M.aggregate(sc, ft)})
write_json(OUT / "final.json", {"results": final, "chosen_theta": chosen_theta})
print(f"scored {len(final)} (arm x prior x kind) rows on {len(EVAL)} evaluation signers")"""))

cells.append(md("""\
## 6. Hybrid: C4's segmentation + `bilstm`'s classification

B0/B-mv above isolate the two things a working pipeline needs -- `bilstm` alone has
excellent classification (B0), but stillness alone can't segment well (B-mv). C4 (the
**deployed** continuous model, GER 0.278 with its full lattice+prior, TODO §12.8) is
already good at the segmentation half: its D3 decoder (null-gated runs, the same
mechanism as `sb.recognize.continuous.decode.decode_null`) needs no new training and no
new export -- it's the live model. This section asks: swap C4's own per-segment vote
for `bilstm`'s whole-clip vote on the *same* C4-produced segments, does it beat C4
alone? `bilstm` gets frames C4 never sees during training (transition/rest-contaminated
spans, the same content that broke B-mv) via C4's segments, so this is not guaranteed
to work -- it's an experiment, not a foregone conclusion."""))

cells.append(code("""\
# ============================================================
# C4 setup (loaded here, not the main setup cell, since only this section needs it)
# ============================================================
from sb.recognize.continuous import decode as CDEC
from sb.recognize.continuous.train import load_run, run_dir_for

C4_NU, C4_MIN_LEN = 0.5, 4  # C4's deployed D3 settings -- apps/web/pipeline.config.json
C4_RUN_NAME = "C4"

c4_dir = run_dir_for(C4_RUN_NAME)
C4_MODEL, C4_CK = load_run(c4_dir, DEVICE)
assert C4_CK["arch"] == "gru_continuous_norm", C4_CK["arch"]
C4_FEATS = B.StreamFeatures.build(DATASET_ROOT, pd.read_csv(DATASET_ROOT / "sequences.csv", keep_default_na=False),
                                  np.asarray(C4_CK["landmarks"]), C4_CK["coords"])
print(f"C4: run {c4_dir.name}, null_index {C4_CK['null_index']}")


def c4_segments(row) -> list[F.Segment]:
    \"\"\"C4's own D3 segments for one sequences.csv row -- frame indices are on the
    same time axis as `FEATS` (bilstm's own feature cache), just a different landmark
    subset/coords per frame, so a span from here indexes `FEATS.x(row)` unchanged.\"\"\"
    x = torch.from_numpy(np.ascontiguousarray(C4_FEATS.x(row["row"])))
    gp, _ = CDEC.frame_outputs(C4_MODEL, x)
    return F.segments_d3(gp, C4_NU, C4_MIN_LEN, C4_CK["null_index"])


def score_hybrid(subset: pd.DataFrame, classify: bool, rule: F.Rule, prior) -> list[dict]:
    \"\"\"C4's segmentation, either C4's own vote (`classify=False`, the reference arm)
    or bilstm's whole-clip vote on the same spans (`classify=True`, the hybrid).\"\"\"
    out = []
    for _, r in tqdm(subset.iterrows(), total=len(subset), desc="scoring hybrid", leave=False):
        labels, seg = seq_arrays(r)
        segs = c4_segments(r)
        if classify:
            x = FEATS.x(r["row"])
            qs = span_probs(x, [(s.start, s.end) for s in segs])
            segs = [F.Segment(s.start, s.end, q, s.mass, q) for s, q in zip(segs, qs)]
        emissions = F.decode_fused(segs, prior, rule)
        out.append(M.score_sequence(emissions, labels, seg, FEATS.kind(r["row"])))
    return out"""))

cells.append(code("""\
# ============================================================
# Sweep theta for the hybrid+prior arm on selection signers, sentence kind
# ============================================================
path = OUT / "sweep_hybrid_prior.json"
if not path.exists():
    rows = []
    for theta in THETA_GRID:
        rule = F.Rule(mode="rescore", lam=LAM, theta=theta)
        sc = score_hybrid(sel, True, rule, prior_fn)
        rows.append({"theta": theta, "lam": LAM, **M.aggregate(sc, frame_totals(sel["row"]))})
    write_json(path, rows)
best = min(json.loads(path.read_text()), key=lambda d: d["ger"])
chosen_theta["hybrid"] = best["theta"]
print("hybrid best theta", best["theta"], "selection GER", round(best["ger"], 4))"""))

cells.append(code("""\
# ============================================================
# Final: C4-alone (reference) vs hybrid vs hybrid+prior, evaluation signers, both kinds
# ============================================================
hybrid_final = []
for kind in ("sentence", "control"):
    ev = SEQ[(SEQ.group == "eval") & (SEQ.kind == kind)]
    ft = frame_totals(ev["row"])
    sc = score_hybrid(ev, False, NO_PRIOR, None)
    hybrid_final.append({"arm": "c4_alone", "prior": False, "kind": kind, **M.aggregate(sc, ft)})
    sc = score_hybrid(ev, True, NO_PRIOR, None)
    hybrid_final.append({"arm": "hybrid", "prior": False, "kind": kind, **M.aggregate(sc, ft)})
    rule = F.Rule(mode="rescore", lam=LAM, theta=chosen_theta["hybrid"])
    sc = score_hybrid(ev, True, rule, prior_fn)
    hybrid_final.append({"arm": "hybrid", "prior": True, "kind": kind, "theta": chosen_theta["hybrid"],
                         **M.aggregate(sc, ft)})
write_json(OUT / "final_hybrid.json", {"results": hybrid_final, "chosen_theta": chosen_theta["hybrid"]})
print(f"scored {len(hybrid_final)} rows on {len(EVAL)} evaluation signers")"""))

cells.append(md("""\
## 7. Global sentence decoding: top-5 per segment, picked by "best sentence" (not greedy)

The user's ask (2026-09-28): take the hybrid's top-5 per segment and have a downstream
model "correctly pick the sign based on the best possible sentence construction" --
i.e. decode the *whole sentence* at once, not one segment at a time. `fuse.decide`
(§6 above) is greedy: it commits each segment before it has seen the next one, and only
ever compares `argmax(q)` against the prior. `sb.rescore.prior.viterbi_rescore` instead
searches every combination of each segment's top-`k` candidates for the one whose
*whole sequence* scores best under the n-gram, by dynamic programming (exact for this
`k` and n-gram order, not a heuristic). No training: `NgramLM` is already fit (§Setup);
this is search over its output, and `bilstm`'s top-5 are already computed.

**Model choice: n-gram, not an LLM.** This repo already investigated a hosted LLM for
exactly this role (§12.6, `deployment-research.md` §5) and found Workers AI gives no
per-token logprobs -- without those, an LLM prior means either one generation per
candidate (slow) or asking it to "rank these," which isn't a calibrated probability a
search like this needs. The n-gram is what actually implements "best sentence
construction" today; an LLM is the documented fallback if this still leaves a real
gap, not the default."""))

cells.append(code("""\
# ============================================================
# Score: hybrid top-5 (K=5, per the user's ask) + exact Viterbi sentence decode
# over the n-gram, vs the same hybrid's greedy per-segment rescore (§6)
# ============================================================
K_CANDIDATES = 5
BEAM_WIDTH = 64
LAM_GRID_V = [0.1, 0.2, 0.3, 0.4, 0.5]


def score_viterbi(subset: pd.DataFrame, lam: float) -> list[dict]:
    out = []
    for _, r in tqdm(subset.iterrows(), total=len(subset), desc="scoring viterbi", leave=False):
        labels, seg = seq_arrays(r)
        segs = c4_segments(r)
        x = FEATS.x(r["row"])
        qs = span_probs(x, [(s.start, s.end) for s in segs])
        choice = viterbi_rescore(qs, NGRAM, GLOSSES, lam, k=K_CANDIDATES, beam_width=BEAM_WIDTH)
        emissions = [(c, s.end - 1, 1.0) for c, s in zip(choice, segs)]
        out.append(M.score_sequence(emissions, labels, seg, FEATS.kind(r["row"])))
    return out


path = OUT / "sweep_viterbi_lam.json"
if not path.exists():
    rows = [{"lam": lam, **M.aggregate(score_viterbi(sel, lam), frame_totals(sel["row"]))}
            for lam in LAM_GRID_V]
    write_json(path, rows)
best_v = min(json.loads(path.read_text()), key=lambda d: d["ger"])
LAM_V = best_v["lam"]
print("viterbi best lam", LAM_V, "selection GER", round(best_v["ger"], 4))

viterbi_final = []
for kind in ("sentence", "control"):
    ev = SEQ[(SEQ.group == "eval") & (SEQ.kind == kind)]
    ft = frame_totals(ev["row"])
    sc = score_viterbi(ev, LAM_V)
    viterbi_final.append({"arm": "hybrid_viterbi", "prior": True, "kind": kind, "lam": LAM_V,
                          **M.aggregate(sc, ft)})
write_json(OUT / "final_viterbi.json", {"results": viterbi_final, "lam": LAM_V})
print(f"scored {len(viterbi_final)} rows on {len(EVAL)} evaluation signers")"""))

cells.append(md("## Results"))

cells.append(code("""\
# ============================================================
# Headline table + comparison against the existing references
# ============================================================
res = pd.json_normalize(json.loads((OUT / "final.json").read_text())["results"])
res_h = pd.json_normalize(json.loads((OUT / "final_hybrid.json").read_text())["results"])
res_v = pd.json_normalize(json.loads((OUT / "final_viterbi.json").read_text())["results"])
res = pd.concat([res, res_h, res_v], ignore_index=True)
table = res[res.kind == "sentence"].set_index(["arm", "prior"])[
    ["ger", "correct_rate", "sub_rate", "del_rate", "ins_rate", "sentence_acc", "emissions_per_sign"]
].sort_values("ger")
table.to_csv(OUT / "final_table.csv")
with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
    print(table)

REFERENCES = {
    "oracle B1 (gru_reg, isolated-model upper bound)": 0.221,
    "best streaming baseline (gru sliding window, B4)": 0.507,
    "best continuous model (C1 D3 collapsed)": 0.293,
    "deployed (C4 D3 collapsed)": 0.278,
}
print("\\nReferences (different modes -- context, not a leaderboard; "
      "docs/reports/sentence-baselines.md, continuous-models.md):")
for k, v in REFERENCES.items():
    print(f"  {k}: {v:.3f}")"""))

with OUT.open("w", encoding="utf-8") as f:
    json.dump({
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }, f, indent=1)
print(f"wrote {OUT} ({len(cells)} cells)")
