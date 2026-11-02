"""Audit statistics for docs/reports/research-state-assessment.md.

Reads only committed files (registry/runs/<id>/assets/val_predictions.npz, history.json,
meta.json) -- no GPU, no dataset download, numpy only.  Run from anywhere:

    python docs/reports/assets/research-state/audit_stats.py

Prints (and writes audit_stats.json beside this file):
  1. per-run accuracy, 95% bootstrap CI (2000 resamples of the 18,896 val clips, seed 0), top-5
  2. paired comparison of the champion against each other run: discordant counts, accuracy
     difference with a paired-bootstrap CI, exact/normal McNemar p
  3. checkpoint-selection optimism proxy: best val accuracy minus the median of the last 10
     epochs (the val set is also the set used to pick the checkpoint -- see the report, Q2)
  4. seed audit: distinct training seeds over the whole registry

The bootstrap and McNemar treat the 18,896 clips as independent.  They are NOT: 21 signers
contribute many clips each, so these intervals are lower bounds on the real uncertainty.
They also say nothing about training-seed variance (every run uses seed 42).
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / "registry" / "runs"

NAMED = {
    "gru_phono_raw ME_134 (champion)": "1790355555",
    "gru ME_132 xy": "1789559734",
    "gru ME_132 xy (rerun)": "1790327942",
    "bilstm ME_126 xy (offline)": "1790347646",
    "gru_phono130 FULL_543": "1790413039",
    "gru ME_126 xy": "1789558839",
    "gru_phono_raw PH_55": "1790354810",
    "lstm ME_126 xy": "1790338279",
    "gru_deep ME_126 xy": "1790333188",
    "gru_phono PH_55 (phonology only)": "1790352947",
}
CHAMP = "gru_phono_raw ME_134 (champion)"


def load(run_id: str):
    z = np.load(RUNS / run_id / "assets" / "val_predictions.npz")
    return z["labels"], z["preds"], z["topk_idx"]


def main() -> None:
    data = {k: load(v) for k, v in NAMED.items()}
    y = data[CHAMP][0]
    n = len(y)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, (2000, n))
    correct = {k: (v[1] == v[0]) for k, v in data.items()}
    out: dict = {"n_val": int(n), "n_classes": int(len(set(y.tolist()))), "runs": {}, "paired": {}}

    print(f"n_val={n}  classes={out['n_classes']}  chance={1 / out['n_classes']:.4f}\n")
    for k, c in correct.items():
        b = c[idx].mean(1)
        top5 = float(np.mean([y[i] in data[k][2][i] for i in range(n)]))
        out["runs"][k] = dict(acc=float(c.mean()), ci_lo=float(np.percentile(b, 2.5)),
                              ci_hi=float(np.percentile(b, 97.5)), top5=top5)
        print(f"{k:36s} acc {c.mean():.4f}  95% CI [{np.percentile(b, 2.5):.4f}, "
              f"{np.percentile(b, 97.5):.4f}]  top-5 {top5:.4f}")

    def paired(ka: str, kb: str) -> dict:
        a, b = correct[ka], correct[kb]
        only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
        m, kk = only_a + only_b, min(only_a, only_b)
        p = (min(1.0, 2 * sum(comb(m, i) for i in range(kk + 1)) / 2**m) if m < 1000
             else math.erfc((abs(only_a - only_b) - 1) / math.sqrt(m) / math.sqrt(2)))
        bd = a[idx].mean(1) - b[idx].mean(1)
        return dict(only_a=only_a, only_b=only_b, diff=float(a.mean() - b.mean()),
                    ci_lo=float(np.percentile(bd, 2.5)), ci_hi=float(np.percentile(bd, 97.5)), p=float(p))

    def show(ka: str, kb: str) -> None:
        r = paired(ka, kb)
        out["paired"][f"{ka} | {kb}"] = r
        print(f"  {ka[:30]:30s} vs {kb:34s} only-A {r['only_a']:5d} only-B {r['only_b']:5d} "
              f"diff {r['diff']:+.4f} CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p={r['p']:.2g}")

    print("\npaired: champion (A) vs each (B)")
    for k in correct:
        if k != CHAMP:
            show(CHAMP, k)
    print("\npaired: the claims that are NOT significant, plus seed-free reruns")
    base = "gru ME_132 xy"
    for kb in ("bilstm ME_126 xy (offline)", "gru_phono130 FULL_543", "gru ME_126 xy", "gru ME_132 xy (rerun)"):
        show(base, kb)

    print("\ncheckpoint-selection optimism proxy (best val acc - median of last 10 epochs)")
    out["optimism"] = {}
    for k, rid in NAMED.items():
        h = json.loads((RUNS / rid / "assets" / "history.json").read_text())
        va, ta = np.array(h["val_acc"]), np.array(h["train_acc"])
        bi = int(va.argmax())
        gap = float(va.max() - np.median(va[-10:]))
        out["optimism"][k] = dict(best=float(va.max()), best_epoch=bi + 1, last10_median=float(np.median(va[-10:])),
                                  optimism=gap, train_at_best=float(ta[bi]), train_val_gap=float(ta[bi] - va[bi]))
        print(f"  {k:36s} best {va.max():.4f}@{bi + 1:<3d} last10-median {np.median(va[-10:]):.4f} "
              f"optimism {gap * 100:+.2f}pp  train@best {ta[bi]:.4f}  gap {(ta[bi] - va[bi]) * 100:.1f}pp")

    rows = list(csv.DictReader(open(ROOT / "registry" / "index.csv")))
    seeds = Counter(r["hyp_seed"] for r in rows)
    out["seeds"] = dict(seeds)
    cur = [r for r in rows if r["legacy_split"] != "True"]
    out["registry"] = dict(total=len(rows), current_split=len(cur),
                           current_canonical=sum(r["eval_status"] == "canonical" for r in cur),
                           legacy=len(rows) - len(cur))
    print(f"\nseeds used across {len(rows)} runs: {dict(seeds)}")
    print(f"registry: {out['registry']}")
    (Path(__file__).parent / "audit_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
