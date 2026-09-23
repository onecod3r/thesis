"""Existing isolated-sign models on GISLR-Sentences (TODO §12.2).

Three pieces the baselines notebook needs, none of them notebook-specific:

- :class:`StreamFeatures` -- every sequence of GISLR-Sentences turned into
  the model input once: row subset, coordinates, NaN -> 0, the same per-frame
  transform ``base_v1`` applies to isolated clips (it has no cross-frame
  step, so it applies to a continuous stream unchanged). One flat in-RAM
  array per landmark set, cached under a content key that includes the
  dataset build's config and corpus hashes.
- :func:`load_registry_model` / :func:`load_five_arch_model` -- the
  checkpoints, with the landmark rows and coordinates each was trained on.
- :func:`signer_split` -- the seeded signer partition used to choose decoder
  thresholds on some signers and report on the others.

Clips are fed at native length. Isolated training/eval uniformly subsamples
clips longer than ``MAX_SEQ_LEN = 128`` (~8% of clips); a live stream is
never subsampled, so the baselines are not either (the same open mismatch as
TODO §11.1, reported rather than hidden).
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sb.core.paths import CACHE_DIR
from sb.core.subsets import SUBSETS
from sb.recognize.architectures import BiLSTM, CausalConv1D, StreamingGRU, StreamingLSTM, build_model
from sb.recognize.sequences.compose import read_sequence

_COORD_INDEX = {"x": 0, "y": 1, "z": 2}
FEATURES_ROOT = CACHE_DIR / "gislr" / "sentence_baselines" / "features"


@dataclass
class LoadedModel:
    name: str
    arch: str  # decoding kind: gru / lstm / cnn / dnn / bilstm
    model: torch.nn.Module
    rows: np.ndarray
    coords: str
    streaming: bool
    note: str


def seq_arrays(row) -> tuple[np.ndarray, np.ndarray]:
    """``(labels, segments)`` of one ``sequences.csv`` row."""
    labels = np.array(row["labels"].split(), dtype=np.int64)
    seg = np.stack([np.array(row["starts"].split(), int), np.array(row["ends"].split(), int)], 1)
    return labels, seg


def hard_cut(x: np.ndarray, kinds: np.ndarray, seg: np.ndarray):
    """Drop every null frame: the back-to-back concatenation of the clips,
    with segments remapped."""
    from sb.recognize.sequences.compose import SIGN

    keep = kinds == SIGN
    newpos = np.cumsum(keep) - 1
    seg2 = np.stack([newpos[seg[:, 0]], newpos[seg[:, 1] - 1] + 1], 1)
    return x[keep], kinds[keep], seg2


def frame_totals(kinds_list) -> dict:
    """``{"sign": n, "transition": n, "rest": n}`` over a list of
    ``frame_kind`` arrays -- the denominators of the per-kind insertion rates."""
    from sb.recognize.sequences.compose import FRAME_KINDS

    k = np.concatenate(list(kinds_list)) if len(kinds_list) else np.array([], np.uint8)
    return {name: int((k == v).sum()) for v, name in FRAME_KINDS.items()}


def signer_split(participants, n_select: int, seed: int) -> tuple[list[int], list[int]]:
    """``(selection, evaluation)`` signer ids: ``n_select`` drawn with
    ``seed`` from the sorted ids, the rest for reporting."""
    ids = sorted(int(p) for p in set(participants))
    sel = sorted(int(p) for p in np.random.default_rng(seed).choice(ids, n_select, replace=False))
    return sel, [p for p in ids if p not in sel]


class StreamFeatures:
    """Model inputs for every sequence, one flat ``(frames, feature_dim)``
    float32 array + offsets, plus the flat ``frame_kind``. Row ``i`` is the
    ``i``-th row of ``sequences.csv``."""

    def __init__(self, data: np.ndarray, offsets: np.ndarray, kinds: np.ndarray):
        self.data, self.offsets, self.kinds = data, offsets, kinds

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def x(self, i: int) -> np.ndarray:
        return self.data[self.offsets[i]:self.offsets[i + 1]]

    def kind(self, i: int) -> np.ndarray:
        return self.kinds[self.offsets[i]:self.offsets[i + 1]]

    @staticmethod
    def key(root: Path, rows: np.ndarray, coords: str) -> str:
        built = json.loads((root / "build_info.json").read_text(encoding="utf-8"))
        ident = {"pipeline": "base_v1-stream", "rows": [int(r) for r in rows], "coords": coords,
                 "dataset_version": built["dataset_version"],
                 "config_sha256": built["config_sha256"], "corpus_sha256": built["corpus_sha256"]}
        return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def build(cls, root: Path, seq_df: pd.DataFrame, rows: np.ndarray, coords: str,
              progress=None) -> "StreamFeatures":
        """Load from the content-addressed cache, or build it (atomic,
        skip-if-exists)."""
        out = FEATURES_ROOT / cls.key(root, rows, coords)
        paths = {n: out / f"{n}.npy" for n in ("data", "offsets", "kinds")}
        if all(p.is_file() for p in paths.values()):
            return cls(*(np.load(paths[n], mmap_mode=None) for n in ("data", "offsets", "kinds")))
        coord_idx = [_COORD_INDEX[c] for c in coords]

        def one(rel):
            a = read_sequence(root / rel)
            x = a["landmarks"][:, rows][:, :, coord_idx].astype(np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).reshape(len(x), -1)
            return x, a["frame_kind"]

        xs, ks, offsets = [], [], [0]
        with ThreadPoolExecutor(8) as ex:
            for i, (x, k) in enumerate(ex.map(one, seq_df["npz_relpath"])):
                xs.append(x)
                ks.append(k)
                offsets.append(offsets[-1] + len(x))
                if progress is not None:
                    progress(i + 1, len(seq_df))
        arrays = {"data": np.concatenate(xs), "offsets": np.asarray(offsets, np.int64),
                  "kinds": np.concatenate(ks)}
        out.mkdir(parents=True, exist_ok=True)
        for n, arr in arrays.items():
            tmp = out / f"{n}.tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, paths[n])
        (out / "inputs.json").write_text(json.dumps(
            {"rows": [int(r) for r in rows], "coords": coords, "root": str(root)}, indent=1))
        return cls(arrays["data"], arrays["offsets"], arrays["kinds"])


def load_registry_model(run_dir: Path, n_classes: int, device) -> LoadedModel:
    """A registry run's ``best.pt`` (fetched from the artifact remote if it
    is not on disk), with its own landmark rows and coords."""
    from sb.mlops.artifacts import ensure_local

    ckpt = torch.load(ensure_local(run_dir, "best.pt"), map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], ckpt["feature_dim"], n_classes, ckpt["hyp"]).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    return LoadedModel(
        name=f"{ckpt['arch']}_reg{Path(run_dir).name}", arch=ckpt["arch"], model=model,
        rows=np.asarray(ckpt["landmarks"]), coords=ckpt["coords"], streaming=ckpt["arch"] != "bilstm",
        note="registry run; checkpoint selected on test.csv (the canonical val set)")


def load_five_arch_model(arm: str, root: Path, config: dict, n_classes: int, device) -> LoadedModel:
    """One ``gislr.1.models.five-arch-benchmark.ipynb`` arm (ME-126 / xy),
    rebuilt exactly as that notebook built it, ``best_state`` loaded."""
    from sb.recognize.interp.models import LandmarkDNN

    rows = SUBSETS["ME_126"].array
    fdim = len(rows) * 2
    hyp = {**config["shared"], **config["architectures"][arm].get("overrides", {})}
    args = (fdim, hyp["hidden_size"], hyp["num_layers"], n_classes, hyp["dropout"])
    if arm == "dnn":
        model = LandmarkDNN(hidden_sizes=tuple(config["architectures"]["dnn"]["hidden_sizes"]),
                            num_classes=n_classes, dropout=hyp["dropout"], n_landmarks=len(rows),
                            channels_per_landmark=2, angle_dim=0, relational_dim=0)
    else:
        cls = {"gru": StreamingGRU, "lstm": StreamingLSTM, "cnn": CausalConv1D, "bilstm": BiLSTM}[arm]
        model = cls(*args)
    ckpt = torch.load(Path(root) / arm / "final.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["best_state"])
    return LoadedModel(
        name=arm, arch=arm, model=model.to(device).eval(), rows=np.asarray(rows), coords="xy",
        streaming=arm != "bilstm",
        note="five-arch benchmark; selected on an internal val carved from train.csv")
