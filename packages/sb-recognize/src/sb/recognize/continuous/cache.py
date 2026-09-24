"""Per-frame model outputs for many streams, cached in resumable chunks
(TODO §12.6).

The fusion sweeps decode the same streams thousands of times with different
priors and rules, so the model runs once per stream and its outputs are
stored: gloss+null probabilities ``gp`` and boundary probabilities ``bp``
(float16), with each stream's own ``frame_kind``, segments, labels and any
injected noise blocks. The noisy variants change the frames, so they have to
travel with the outputs.

One ``chunk_<i>.npz`` holds up to N streams, concatenated with offsets. A
chunk is written to a temp file and renamed, so a chunk that exists is
complete, and an interrupted run redoes at most one chunk.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from sb.recognize.sequences.noise import NOISE_KINDS


def _flat(arrays: list[np.ndarray], dtype, width: int | None = None):
    off = np.cumsum([0] + [len(a) for a in arrays]).astype(np.int64)
    if not arrays or off[-1] == 0:
        shape = (0,) if width is None else (0, width)
        return np.zeros(shape, dtype), off
    return np.concatenate([np.asarray(a, dtype).reshape(-1, width) if width else np.asarray(a, dtype)
                           for a in arrays]), off


def write_chunk(path: Path, items: list[dict]) -> None:
    """``items``: dicts with ``row``, ``gp`` (T, C+1), ``bp`` (T,), ``kinds``
    (T,), ``seg`` (n, 2), ``labels`` (n,), ``blocks`` [(start, end, kind)]."""
    gp, g_off = _flat([it["gp"] for it in items], np.float16, items[0]["gp"].shape[1])
    bp, _ = _flat([it["bp"] for it in items], np.float16)
    kinds, _ = _flat([it["kinds"] for it in items], np.uint8)
    seg, s_off = _flat([np.asarray(it["seg"]).reshape(-1, 2) for it in items], np.int64, 2)
    lab, _ = _flat([it["labels"] for it in items], np.int64)
    blk, b_off = _flat([np.asarray([(a, b) for a, b, _ in it["blocks"]], np.int64).reshape(-1, 2)
                        for it in items], np.int64, 2)
    bkind, _ = _flat([np.asarray([NOISE_KINDS.index(k) for _, _, k in it["blocks"]], np.int8)
                      for it in items], np.int8)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, rows=np.asarray([it["row"] for it in items], np.int64), gp=gp, bp=bp, kinds=kinds,
             frame_off=g_off, seg=seg, seg_off=s_off, labels=lab, blocks=blk, block_off=b_off,
             block_kind=bkind)
    os.replace(tmp, path)


def iter_chunks(directory: Path) -> Iterator[dict]:
    """Every cached stream in ``directory``, in chunk order, one dict each:
    ``row, gp (float32), bp, kinds, seg, labels, blocks``."""
    for path in sorted(directory.glob("chunk_*.npz"), key=lambda p: int(p.stem.split("_")[1])):
        if ".tmp" in path.name:
            continue
        z = np.load(path)
        fo, so, bo = z["frame_off"], z["seg_off"], z["block_off"]
        for i, row in enumerate(z["rows"]):
            a, b = fo[i], fo[i + 1]
            blocks = [(int(s), int(e), NOISE_KINDS[int(k)])
                      for (s, e), k in zip(z["blocks"][bo[i]:bo[i + 1]], z["block_kind"][bo[i]:bo[i + 1]])]
            yield {"row": int(row), "gp": z["gp"][a:b].astype(np.float32), "bp": z["bp"][a:b].astype(np.float32),
                   "kinds": z["kinds"][a:b], "seg": z["seg"][so[i]:so[i + 1]],
                   "labels": z["labels"][so[i]:so[i + 1]], "blocks": blocks}


def chunk_path(directory: Path, index: int) -> Path:
    return directory / f"chunk_{index}.npz"
