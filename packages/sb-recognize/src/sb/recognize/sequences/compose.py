"""Sentences -> continuous landmark sequences with per-frame labels.

Two phases, deliberately split so the expensive one is resumable and the
cheap one is reviewable before anything is written:

1. **Plan** (:func:`plan_sentences`, :func:`plan_control`) -- pure
   bookkeeping over ``test.csv``: which clip fills which slot of which
   sequence, plus every synthesized gap/rest length. A plan row fully
   determines its sequence, so materialization is deterministic and can be
   interrupted and resumed per sequence.
2. **Materialize** (:func:`materialize`) -- read the clips, synthesize the
   non-sign frames, return the arrays for one sequence.

**Clip usage.** ``plan_sentences`` is coverage-driven: it keeps drawing
sentences until every source clip has been placed at least once, preferring
sentences whose glosses the chosen signer still has *unused* clips for. A
slot is only filled with an already-used clip when no sentence can be
completed from unused ones -- the ``reused`` flag records each such slot so
the reuse rate is measurable, never hidden. ``plan_control`` uses every clip
exactly once, in random order (no sentence structure): the "no language
prior" control.

**One signer per sequence.** Every slot of a sequence comes from the same
``participant_id`` -- a real continuous signer does not change identity (and
body position) at every sign boundary. A sentence is only eligible for a
signer who recorded every gloss in it.

**Synthesized frames** (GISLR clips are trimmed to the sign; nothing in the
dataset shows a hand at rest or moving between signs):

- *transition* between consecutive signs: ``g`` frames linearly interpolated,
  per landmark and coordinate, from the last frame of one clip to the first
  frame of the next. NaN on either end stays NaN (a hand absent at either
  end is not invented).
- *rest* before the first and after the last sign: the adjacent clip's edge
  frame held still with small Gaussian jitter, **both hands NaN** -- what
  MediaPipe emits while the hands are down and out of frame. The body and
  face are present.

Both are labeled null (``NULL_LABEL``) and tagged in ``frame_kind`` so an
evaluation can drop them (``frame_kind == SIGN`` recovers the hard-cut,
back-to-back concatenation exactly) or score them separately.

Landmarks are written in the repo's canonical gislr-holistic row order
(``sb.core.schema.GROUPS``), NOT the source files' stored order -- the
permutation is undone once here, so readers of this dataset never need
``CANONICAL_TO_STORED``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sb.core.schema import GROUPS, N_LANDMARKS, validate_tensor
from sb.recognize.features.gislr_stratified import load_npz
from sb.recognize.sequences.corpus import Sentence

NULL_LABEL = -1
SIGN, TRANSITION, REST = 0, 1, 2  # frame_kind values
FRAME_KINDS = {SIGN: "sign", TRANSITION: "transition", REST: "rest"}

_ALL_ROWS = np.arange(N_LANDMARKS)
HAND_ROWS = np.concatenate([
    np.arange(g.offset, g.offset + g.size) for g in GROUPS if g.name in ("left_hand", "right_hand")
])


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def _synth_lengths(n_signs: int, rng: np.random.Generator, gap_range, rest_range) -> dict:
    lo, hi = gap_range
    rlo, rhi = rest_range
    return {
        "gaps": [int(g) for g in rng.integers(lo, hi + 1, size=max(n_signs - 1, 0))],
        "rest_in": int(rng.integers(rlo, rhi + 1)),
        "rest_out": int(rng.integers(rlo, rhi + 1)),
    }


def plan_sentences(
    df: pd.DataFrame,
    sentences: list[Sentence],
    rng: np.random.Generator,
    *,
    gap_range: tuple[int, int],
    rest_range: tuple[int, int],
    reuse_penalty: float = 2.0,
    repeat_penalty: float = 0.15,
    length_bonus: float = 0.0,
    pressure_weight: float = 0.0,
    progress=None,
) -> tuple[list[dict], dict]:
    """Coverage-driven sentence instantiation over every clip in ``df``.

    Loop: take the signer with the most unused clips left; score every
    sentence eligible for that signer as

        fresh_frac - reuse_penalty * reuse_frac - repeat_penalty * times_used_by_signer
        + length_bonus * n_fresh / max_len + pressure_weight * pressure

    where ``fresh_frac`` is the fraction of the sentence's slots fillable with
    that signer's unused clips and ``pressure`` is the mean number of unused
    clips its glosses still have, relative to the signer's average; take the
    best (random tiebreak), fill it. Stops when every clip is placed. Returns
    ``(plan, stats)``.

    Scoring by *fraction* keeps any fully-fresh sentence ahead of a partly
    reused one; ``length_bonus`` then breaks the tie toward longer sentences
    (without it, 2-gloss sentences win because they are the easiest to fill
    fresh). ``pressure`` drains the glosses a signer has most of first, so the
    last clips left are not a scatter no sentence can use without reuse.
    ``repeat_penalty`` spreads a signer across many sentences instead of
    repeating a favourite.
    """
    glosses = sorted(df["sign"].unique())
    gidx = {g: i for i, g in enumerate(glosses)}
    G, N = len(glosses), len(sentences)
    S = np.zeros((N, G), dtype=np.int32)
    for i, s in enumerate(sentences):
        for g in s.glosses:
            S[i, gidx[g]] += 1
    lens = S.sum(1).astype(np.float64)
    max_len = lens.max()

    unused: dict[int, list[list[str]]] = {}
    used: dict[int, list[list[str]]] = {}
    for pid in sorted(int(p) for p in df["participant_id"].unique()):
        sub = df[df["participant_id"] == pid]
        pool = [[] for _ in range(G)]
        for uid, g in zip(sub["uid"], sub["sign"]):
            pool[gidx[g]].append(uid)
        for lst in pool:
            rng.shuffle(lst)
        unused[pid] = pool
        used[pid] = [[] for _ in range(G)]
    avail = {p: np.array([len(x) for x in pool]) for p, pool in unused.items()}
    has = {p: a > 0 for p, a in avail.items()}
    eligible = {p: (S > 0).astype(np.int32) @ (~h).astype(np.int32) == 0 for p, h in has.items()}
    uses = {p: np.zeros(N) for p in unused}

    plan, orphans = [], 0
    total = remaining = len(df)
    while remaining > 0:
        p = max(avail, key=lambda q: avail[q].sum())
        a = avail[p]
        fresh = np.minimum(S, a[None, :]).sum(1)
        score = (fresh - reuse_penalty * (lens - fresh)) / lens - repeat_penalty * uses[p]
        score += length_bonus * fresh / max_len
        if pressure_weight:
            pressure = (S @ a) / lens / max(a[a > 0].mean(), 1e-9)
            score += pressure_weight * pressure
        score += rng.random(N) * 1e-6
        score[~eligible[p] | (fresh == 0)] = -np.inf
        if not np.isfinite(score).any():
            # this signer's leftover clips are of glosses no eligible sentence
            # contains -- they cannot be placed in a sentence at all
            orphans += int(a.sum())
            remaining -= int(a.sum())
            avail[p] = np.zeros_like(a)
            continue
        k = int(np.argmax(score))
        uses[p][k] += 1
        uids, reused = [], []
        for g in sentences[k].glosses:
            j = gidx[g]
            if unused[p][j]:
                uid = unused[p][j].pop()
                used[p][j].append(uid)
                a[j] -= 1
                remaining -= 1
                uids.append(uid)
                reused.append(False)
            else:
                uids.append(used[p][j][int(rng.integers(len(used[p][j])))])
                reused.append(True)
        plan.append({
            "kind": "sentence",
            "sentence_id": sentences[k].id,
            "participant_id": int(p),
            "glosses": list(sentences[k].glosses),
            "source_uids": uids,
            "reused": reused,
            **_synth_lengths(len(uids), rng, gap_range, rest_range),
        })
        if progress is not None:
            progress(total - remaining, total)

    n_slots = sum(len(r["source_uids"]) for r in plan)
    n_reused = sum(sum(r["reused"]) for r in plan)
    stats = {
        "n_sequences": len(plan),
        "n_slots": n_slots,
        "n_reused_slots": n_reused,
        "reuse_rate": n_reused / max(n_slots, 1),
        "n_clips": total,
        "n_orphan_clips": orphans,
        "n_distinct_sentences": len({r["sentence_id"] for r in plan}),
    }
    return plan, stats


def plan_control(
    df: pd.DataFrame,
    lengths: np.ndarray,
    rng: np.random.Generator,
    *,
    gap_range: tuple[int, int],
    rest_range: tuple[int, int],
) -> list[dict]:
    """Every clip exactly once, random gloss order, one signer per sequence;
    sequence lengths drawn from ``lengths`` (pass the sentence split's
    lengths so the two splits differ only in word order/meaning)."""
    plan = []
    for pid in sorted(int(p) for p in df["participant_id"].unique()):
        sub = df[df["participant_id"] == pid]
        idx = rng.permutation(len(sub))
        uids = sub["uid"].to_numpy()[idx]
        signs = sub["sign"].to_numpy()[idx]
        i = 0
        while i < len(uids):
            n = int(rng.choice(lengths))
            if len(uids) - (i + n) == 1:  # never leave a 1-sign tail
                n += 1
            j = min(i + n, len(uids))
            plan.append({
                "kind": "control",
                "sentence_id": "",
                "participant_id": int(pid),
                "glosses": [str(s) for s in signs[i:j]],
                "source_uids": [str(u) for u in uids[i:j]],
                "reused": [False] * (j - i),
                **_synth_lengths(j - i, rng, gap_range, rest_range),
            })
            i = j
    order = rng.permutation(len(plan))  # interleave signers
    return [plan[i] for i in order]


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------

def load_clip(data_dir: Path, relpath: str) -> np.ndarray:
    """One GISLR_Stratified clip in canonical row order, NaN preserved."""
    return load_npz(Path(data_dir) / relpath, _ALL_ROWS, "xyz")


def transition(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """``n`` frames strictly between frame ``a`` and frame ``b`` (endpoints
    excluded); NaN at either end propagates."""
    t = (np.arange(1, n + 1, dtype=np.float32) / (n + 1))[:, None, None]
    return a[None] + t * (b - a)[None]


def rest(edge: np.ndarray, n: int, rng: np.random.Generator, jitter: float) -> np.ndarray:
    """``n`` frames of ``edge`` held still with Gaussian jitter, hands NaN."""
    out = np.repeat(edge[None], n, axis=0).astype(np.float32)
    out += rng.normal(0.0, jitter, size=out.shape).astype(np.float32)
    out[:, HAND_ROWS, :] = np.nan
    return out


def materialize(
    row: dict,
    data_dir: Path,
    relpath_of: dict[str, str],
    label_of: dict[str, int],
    rng: np.random.Generator,
    *,
    rest_jitter: float,
) -> dict:
    """One plan row -> ``{landmarks, frame_labels, frame_kind, segments, labels}``.

    ``segments[i] = (start, end)``, end exclusive, in output frames: slicing
    ``landmarks[start:end]`` returns source clip ``i`` bit-for-bit.
    """
    clips = [load_clip(data_dir, relpath_of[u]) for u in row["source_uids"]]
    labels = [label_of[g] for g in row["glosses"]]
    parts, kinds, flabels, segments = [], [], [], []
    t = 0

    def add(arr, kind, label):
        nonlocal t
        parts.append(arr)
        kinds.append(np.full(len(arr), kind, np.uint8))
        flabels.append(np.full(len(arr), label, np.int16))
        t += len(arr)

    add(rest(clips[0][0], row["rest_in"], rng, rest_jitter), REST, NULL_LABEL)
    for i, (clip, lab) in enumerate(zip(clips, labels)):
        if i > 0:
            add(transition(clips[i - 1][-1], clip[0], row["gaps"][i - 1]), TRANSITION, NULL_LABEL)
        segments.append((t, t + len(clip)))
        add(clip, SIGN, lab)
    add(rest(clips[-1][-1], row["rest_out"], rng, rest_jitter), REST, NULL_LABEL)

    landmarks = np.concatenate(parts).astype(np.float32)
    validate_tensor(landmarks, dtype=np.float32, check_values=True, where="sequence")
    return {
        "landmarks": landmarks,
        "frame_labels": np.concatenate(flabels),
        "frame_kind": np.concatenate(kinds),
        "segments": np.asarray(segments, dtype=np.int32).reshape(-1, 2),
        "labels": np.asarray(labels, dtype=np.int16),
    }


def write_sequence(path: Path, arrays: dict) -> Path:
    """Atomic compressed npz (temp file + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def read_sequence(path: Path) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def check_roundtrip(arrays: dict, row: dict, data_dir: Path, relpath_of: dict[str, str]) -> None:
    """Every segment slices back to its exact source clip; labels and kinds
    agree with the segments. Raises on the first mismatch."""
    fk, fl = arrays["frame_kind"], arrays["frame_labels"]
    covered = np.zeros(len(fk), bool)
    for (s, e), uid, lab in zip(arrays["segments"], row["source_uids"], arrays["labels"]):
        src = load_clip(data_dir, relpath_of[uid])
        if not np.array_equal(arrays["landmarks"][s:e], src, equal_nan=True):
            raise AssertionError(f"segment {s}:{e} != source clip {uid}")
        if not ((fk[s:e] == SIGN).all() and (fl[s:e] == lab).all()):
            raise AssertionError(f"labels/kinds wrong inside segment {s}:{e}")
        covered[s:e] = True
    if (fk[~covered] == SIGN).any() or (fl[~covered] != NULL_LABEL).any():
        raise AssertionError("a frame outside every segment is labelled as a sign")


def summarize_row(row: dict, arrays: dict, seq_id: str, relpath: str) -> dict:
    """The ``sequences.csv`` row for one materialized sequence."""
    seg = arrays["segments"]
    return {
        "seq_id": seq_id,
        "kind": row["kind"],
        "sentence_id": row["sentence_id"],
        "participant_id": row["participant_id"],
        "n_signs": len(row["glosses"]),
        "glosses": " ".join(row["glosses"]),
        "labels": " ".join(str(int(x)) for x in arrays["labels"]),
        "starts": " ".join(str(int(s)) for s in seg[:, 0]),
        "ends": " ".join(str(int(e)) for e in seg[:, 1]),
        "source_uids": " ".join(row["source_uids"]),
        "reused": " ".join(str(int(r)) for r in row["reused"]),
        "gaps": " ".join(str(g) for g in row["gaps"]),
        "rest_in": row["rest_in"],
        "rest_out": row["rest_out"],
        "n_frames": int(len(arrays["frame_kind"])),
        "npz_relpath": relpath,
    }
