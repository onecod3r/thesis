"""Scoring a decoded gloss stream against a GISLR-Sentences sequence (TODO §12).

A decoder emits ``[(class, frame, confidence), ...]``. Scoring has two
halves, kept separate because they answer different questions:

- **What was said** -- the emitted class sequence against the reference by
  Levenshtein alignment: substitutions, deletions, insertions, and the gloss
  error rate ``(S + D + I) / N`` (WER over glosses). Order-only; frames are
  ignored.
- **When and where** -- each emission's frame against the sequence's
  segments: commit latency for correctly-aligned signs (``accept frame -
  last frame of the true segment``, negative = committed before the sign
  ended), and which kind of frame every emission and every insertion landed
  on (sign / transition / rest). Rest frames in GISLR-Sentences v1 have NaN
  hands, which makes them easier than real rest, so they are always
  reported apart from transitions (``docs/logs/daily/2026-09-23.md``).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from sb.recognize.sequences.compose import FRAME_KINDS

MATCH, SUB, DEL, INS = "match", "sub", "del", "ins"


def align(ref: list[int], hyp: list[int]) -> list[tuple[str, int | None, int | None]]:
    """Minimum-edit alignment as ``[(op, ref_index, hyp_index), ...]`` in
    order. Ties prefer match/substitution over insertion over deletion, so
    the result is deterministic."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]), d[i, j - 1] + 1, d[i - 1, j] + 1)
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append((MATCH if ref[i - 1] == hyp[j - 1] else SUB, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif j > 0 and d[i, j] == d[i, j - 1] + 1:
            ops.append((INS, None, j - 1))
            j -= 1
        else:
            ops.append((DEL, i - 1, None))
            i -= 1
    return ops[::-1]


def score_sequence(
    emissions: list[tuple[int, int, float]],
    labels: np.ndarray,
    segments: np.ndarray,
    frame_kind: np.ndarray,
) -> dict:
    """One sequence's scores. ``labels``/``segments``/``frame_kind`` are the
    sequence's own arrays (``segments`` end-exclusive, in the same frame
    coordinates the emissions use)."""
    ref = [int(x) for x in labels]
    hyp = [c for c, _, _ in emissions]
    ops = align(ref, hyp)
    counts = Counter(op for op, _, _ in ops)
    latencies = [emissions[j][1] - (int(segments[i][1]) - 1)
                 for op, i, j in ops if op == MATCH and i is not None and j is not None]
    ref_correct = [False] * len(ref)
    for op, i, _ in ops:
        if op == MATCH and i is not None:
            ref_correct[i] = True

    def kind_of(frame: int) -> str:
        return FRAME_KINDS[int(frame_kind[min(frame, len(frame_kind) - 1)])]

    return {
        "n_ref": len(ref),
        "n_hyp": len(hyp),
        "match": counts[MATCH], "sub": counts[SUB], "del": counts[DEL], "ins": counts[INS],
        "exact": hyp == ref,
        "latencies": latencies,
        "ref_correct": ref_correct,  # per reference sign: recognized (aligned as a match)?
        "emit_kinds": Counter(kind_of(f) for _, f, _ in emissions),
        "ins_kinds": Counter(kind_of(emissions[j][1]) for op, _, j in ops if op == INS and j is not None),
    }


def aggregate(scores: list[dict], frame_totals: dict[str, int] | None = None) -> dict:
    """Corpus-level numbers over many :func:`score_sequence` results.

    ``frame_totals`` (``{"sign": n, "transition": n, "rest": n}`` over the
    same sequences) turns insertion counts into rates per 1,000 frames of
    each kind, so rest and transition are comparable despite their very
    different sizes.
    """
    n = sum(s["n_ref"] for s in scores)
    tot = {k: sum(s[k] for s in scores) for k in ("match", "sub", "del", "ins", "n_hyp")}
    lat = np.concatenate([s["latencies"] for s in scores]) if scores else np.array([])
    emit = sum((s["emit_kinds"] for s in scores), Counter())
    ins = sum((s["ins_kinds"] for s in scores), Counter())
    out = {
        "n_sequences": len(scores),
        "n_ref": n,
        "ger": (tot["sub"] + tot["del"] + tot["ins"]) / max(n, 1),
        "correct_rate": tot["match"] / max(n, 1),
        "sub_rate": tot["sub"] / max(n, 1),
        "del_rate": tot["del"] / max(n, 1),
        "ins_rate": tot["ins"] / max(n, 1),
        "sentence_acc": float(np.mean([s["exact"] for s in scores])) if scores else float("nan"),
        "emissions_per_sign": tot["n_hyp"] / max(n, 1),
        "latency_median": float(np.median(lat)) if len(lat) else float("nan"),
        "latency_p90": float(np.percentile(lat, 90)) if len(lat) else float("nan"),
        "emit_share": {k: emit[k] / max(sum(emit.values()), 1) for k in FRAME_KINDS.values()},
    }
    if frame_totals:
        out["ins_per_1k_frames"] = {k: 1000 * ins[k] / max(frame_totals.get(k, 0), 1)
                                    for k in FRAME_KINDS.values()}
    return out
