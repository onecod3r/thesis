"""Synthetic non-sign activity inserted into GISLR-Sentences streams (TODO
§12.6): what the recognizer must *not* accept.

GISLR-Sentences v1 has only clean signs, interpolated transitions and
hands-absent rest. A live user also fidgets, holds their hands up while
thinking, or makes sign-like movements that are not signs. Every noise
block is built in the model's feature space (``StreamFeatures`` rows: the
subset's xy values, NaN already 0) from **other streams' sign frames**, so it
has real hand shapes and real motion statistics but no gloss:

- ``fidget``: 3-8-frame fragments of random signs, spliced together with a
  short interpolation. Long (``length``), continuously moving, no one sign
  dominates. The main "noise spanning a while" case.
- ``hold``: one frame from the middle of a random sign, held with small
  jitter. The hands are up and visible but static.
- ``reverse``: one whole random sign played backwards. Sign-length and
  sign-like; the hardest case for a length gate, which it is meant to show.

A block goes into a gap (before the first sign, between two signs, or after
the last) and is blended in with ``interp`` frames at each edge. Its frames
are marked ``TRANSITION`` so the standard scorer counts any emission there
as an insertion; ``blocks`` records where each one is, for the noise
false-accept rate (``fuse.noise_false_accepts``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sb.recognize.sequences.compose import SIGN, TRANSITION

NOISE_KINDS = ("fidget", "hold", "reverse")


@dataclass(frozen=True)
class NoiseSpec:
    kinds: tuple[str, ...] = NOISE_KINDS
    blocks_per_stream: tuple[int, int] = (1, 2)  # inclusive range
    length: tuple[int, int] = (40, 150)  # fidget / hold frames, inclusive
    frag_len: tuple[int, int] = (3, 8)
    hold_jitter: float = 0.002
    interp: int = 3


class DonorPool:
    """Sign segments ``(x, start, end)`` noise is cut from. Build one per
    signer group so noise never mixes selection and evaluation signers."""

    def __init__(self, segments: list[np.ndarray]):
        assert segments, "empty donor pool"
        self.segments = segments

    @classmethod
    def from_streams(cls, xs: list[np.ndarray], segs: list[np.ndarray]) -> "DonorPool":
        return cls([x[a:b] for x, sg in zip(xs, segs) for a, b in sg if b - a >= 4])

    def pick(self, rng: np.random.Generator) -> np.ndarray:
        return self.segments[int(rng.integers(len(self.segments)))]


def _blend(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """``n`` frames linearly from ``a`` to ``b`` (both excluded)."""
    if n <= 0:
        return np.zeros((0, a.shape[-1]), a.dtype)
    w = np.linspace(0, 1, n + 2, dtype=np.float32)[1:-1, None]
    return (1 - w) * a[None] + w * b[None]


def make_block(kind: str, pool: DonorPool, rng: np.random.Generator, spec: NoiseSpec) -> np.ndarray:
    if kind == "fidget":
        n = int(rng.integers(spec.length[0], spec.length[1] + 1))
        parts: list[np.ndarray] = []
        total = 0
        while total < n:
            seg = pool.pick(rng)
            L = int(rng.integers(spec.frag_len[0], spec.frag_len[1] + 1))
            L = min(L, len(seg))
            s = int(rng.integers(0, len(seg) - L + 1))
            frag = seg[s:s + L]
            if parts:
                bridge = _blend(parts[-1][-1], frag[0], 2)
                parts.append(bridge)
                total += len(bridge)
            parts.append(frag)
            total += len(frag)
        return np.concatenate(parts)[:n].astype(np.float32)
    if kind == "hold":
        n = int(rng.integers(spec.length[0], spec.length[1] + 1))
        seg = pool.pick(rng)
        f = seg[len(seg) // 2]
        noise = rng.normal(0, spec.hold_jitter, (n, len(f))).astype(np.float32)
        return (f[None] + noise * (f[None] != 0)).astype(np.float32)  # missing (0) stays missing
    if kind == "reverse":
        return pool.pick(rng)[::-1].astype(np.float32).copy()
    raise ValueError(f"unknown noise kind {kind!r}")


def inject(x: np.ndarray, kinds: np.ndarray, seg: np.ndarray, pool: DonorPool,
           rng: np.random.Generator, spec: NoiseSpec = NoiseSpec()):
    """One stream -> ``(x, kinds, seg, blocks)`` with noise inserted.

    ``blocks`` is ``[(start, end, kind), ...]`` in the new frame coordinates;
    ``seg`` is shifted to match. Labels are unchanged: noise adds no gloss.
    """
    n_signs = len(seg)
    n_blocks = int(rng.integers(spec.blocks_per_stream[0], spec.blocks_per_stream[1] + 1))
    gaps = sorted(rng.choice(n_signs + 1, size=min(n_blocks, n_signs + 1), replace=False).tolist())
    # insertion point for gap g: the middle of the null run between sign g-1 and sign g
    cuts = []
    for g in gaps:
        lo = 0 if g == 0 else int(seg[g - 1][1])
        hi = len(x) if g == n_signs else int(seg[g][0])
        cuts.append((lo + hi) // 2)
    pieces_x, pieces_k, blocks = [], [], []
    shift = np.zeros(len(x) + 1, np.int64)  # frames inserted before original frame i
    prev, added = 0, 0
    for cut in cuts:
        kind = spec.kinds[int(rng.integers(len(spec.kinds)))]
        blk = make_block(kind, pool, rng, spec)
        left = x[cut - 1] if cut > 0 else blk[0]
        right = x[cut] if cut < len(x) else blk[-1]
        lead, tail = _blend(left, blk[0], spec.interp), _blend(blk[-1], right, spec.interp)
        pieces_x += [x[prev:cut], lead, blk, tail]
        pieces_k += [kinds[prev:cut], np.full(len(lead) + len(blk) + len(tail), TRANSITION, kinds.dtype)]
        start = cut + added + len(lead)
        blocks.append((int(start), int(start + len(blk)), kind))
        added += len(lead) + len(blk) + len(tail)
        shift[cut:] = added
        prev = cut
    pieces_x.append(x[prev:])
    pieces_k.append(kinds[prev:])
    x2 = np.concatenate(pieces_x).astype(np.float32)
    k2 = np.concatenate(pieces_k)
    seg2 = np.stack([seg[:, 0] + shift[seg[:, 0]], seg[:, 1] + shift[seg[:, 1] - 1]], 1) if n_signs else seg
    assert (k2[seg2[:, 0]] == SIGN).all() if n_signs else True, "a sign segment moved off its frames"
    return x2, k2, seg2, blocks
