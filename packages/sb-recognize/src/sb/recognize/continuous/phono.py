"""Continuous training streams in **phonological-feature space** (feature pipeline ``phono130_v1``,
TODO §3.10 → §12.3).

:mod:`sb.recognize.continuous.data` composes streams from raw landmarks and synthesizes rest by
moving landmark rows. A phonology model's inputs are the 130 features of
:mod:`sb.recognize.phonology` (model units, missing = 0), and re-extracting them per composed stream
would cost minutes per epoch. So the same stream structure is composed directly in feature space,
with the same config (``composer`` block) and the same outputs as :func:`~sb.recognize.continuous.data.compose`:

- **sign frames**: each clip's cached features;
- **gaps** (0-15 frames): linear interpolation between the neighbouring clips' edge frames; a hand
  missing at either end stays missing (its features 0, presence 0);
- **rest** (5-30 frames at both ends): both hands' landmark-derived features absent (as GISLR shows a
  hand below the frame: not detected), the face and head features held from the edge frame, and the
  pose-derived arm features (wrist position, elbow, wrist speed) either held (``hands-absent`` rest)
  or ramped over ``rest_ramp_frames`` to a resting arm (``lowered`` rest, probability
  ``p_lowered_rest``). The resting arm was measured on the phono130 train cache (2026-09-26, the
  non-dominant arm when not raised, 84% of its frames): wrist 1.035 shoulder widths below the
  shoulders, 0.78 out to its side, elbow 155°.

Evaluation streams come from the raw GISLR-Sentences landmarks through the real extractor
(:class:`PhonoStreamFeatures`), so the test does not share this composer's approximations.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from sb.recognize import phonology as PH

REST_WRIST_Y, REST_WRIST_X, REST_ELBOW = 1.035, 0.78, 155.0 / 90.0  # model units (see module doc)
POSE_FEATURES = ("_wrist_x", "_wrist_y", "_wrist_speed", "_elbow")


def _cols(prefix: str) -> np.ndarray:
    return np.array([i for i, n in enumerate(PH.NAMES) if n.startswith(prefix)])


H1_HAND = np.array([i for i in _cols("h1_") if not PH.NAMES[i].endswith(POSE_FEATURES)])
H2_HAND = np.array([i for i in _cols("h2_") if not PH.NAMES[i].endswith(POSE_FEATURES)])
TWO_HAND = np.array([PH.IDX[n] for n in ("both_present", "touch_hands", "h2_rel_x", "h2_rel_y", "sym_vel", "shape_sim")])
ARM = {h: {k: PH.IDX[f"{h}_{k}"] for k in ("wrist_x", "wrist_y", "wrist_speed", "elbow")} for h in ("h1", "h2")}
POSE_COLS = np.array(sorted(i for a in ARM.values() for i in a.values()))
VEL_SCALE = float(PH.UNIT_SCALE[PH.IDX["h1_wrist_speed"]])


def _no_hands(frames: np.ndarray) -> np.ndarray:
    frames[:, H1_HAND] = 0
    frames[:, H2_HAND] = 0
    frames[:, TWO_HAND] = 0
    return frames


def rest(edge: np.ndarray, n: int, lowered: bool, entering: bool, cfg: dict,
         rng: np.random.Generator) -> np.ndarray:
    """``n`` rest frames next to ``edge`` (a clip's first frame when ``entering``, else its last)."""
    frames = _no_hands(np.repeat(edge[None], n, axis=0).astype(np.float32))
    if lowered:
        k = min(cfg["rest_ramp_frames"], n)
        t = np.clip(np.arange(1, n + 1) / (k + 1), 0, 1)  # 0 -> 1: edge -> rest (leaving)
        for h, side in (("h1", 1.0), ("h2", -1.0)):
            a = ARM[h]
            for col, target in ((a["wrist_y"], max(REST_WRIST_Y, float(edge[a["wrist_y"]]))),
                                (a["wrist_x"], side * REST_WRIST_X), (a["elbow"], REST_ELBOW)):
                frames[:, col] = edge[col] + t * (target - edge[col])
            dy = np.abs(np.diff(np.r_[edge[a["wrist_y"]], frames[:, a["wrist_y"]]]))
            dx = np.abs(np.diff(np.r_[edge[a["wrist_x"]], frames[:, a["wrist_x"]]]))
            frames[:, a["wrist_speed"]] = np.hypot(dx, dy) * VEL_SCALE
        if entering:
            frames = frames[::-1].copy()
    else:
        for a in ARM.values():
            frames[:, a["wrist_speed"]] = 0
    frames[:, POSE_COLS] += rng.normal(0, cfg["rest_jitter"] * 2, (n, len(POSE_COLS))).astype(np.float32)
    return np.clip(frames, -PH.CLIP, PH.CLIP)


def gap(a: np.ndarray, b: np.ndarray, g: int) -> np.ndarray:
    """``g`` frames interpolated from ``a`` to ``b``; a hand missing at either end stays missing."""
    w = (np.arange(1, g + 1, dtype=np.float32) / (g + 1))[:, None]
    out = a[None] + w * (b - a)[None]
    for pres, cols in ((PH.IDX["h1_present"], H1_HAND), (PH.IDX["h2_present"], H2_HAND)):
        if a[pres] < 0.5 or b[pres] < 0.5:
            out[:, cols] = 0
    if a[PH.IDX["both_present"]] < 0.5 or b[PH.IDX["both_present"]] < 0.5:
        out[:, TWO_HAND] = 0
    return out.astype(np.float32)


def compose(clips: list[np.ndarray], labels: list[int], null_index: int, lay, cfg: dict,
            rng: np.random.Generator) -> dict:
    """Same contract as :func:`sb.recognize.continuous.data.compose` (``lay`` is unused)."""
    g0, g1 = cfg["gap_frames"]
    r0, r1 = cfg["rest_frames"]
    parts, ys = [], []

    def add(arr, label):
        parts.append(arr)
        ys.append(np.full(len(arr), label, np.int64))

    lowered = rng.random() < cfg["p_lowered_rest"]
    add(rest(clips[0][0], int(rng.integers(r0, r1 + 1)), lowered, True, cfg, rng), null_index)
    segments, t = [], len(parts[0])
    for i, (clip, lab) in enumerate(zip(clips, labels)):
        if i > 0:
            g = int(rng.integers(g0, g1 + 1))
            if g:
                add(gap(clips[i - 1][-1], clip[0], g), null_index)
                t += g
        segments.append((t, t + len(clip)))
        add(clip, lab)
        t += len(clip)
    add(rest(clips[-1][-1], int(rng.integers(r0, r1 + 1)), lowered, False, cfg, rng), null_index)
    y = np.concatenate(ys)
    b = np.zeros(len(y), np.float32)
    for s, e in segments:
        b[max(s, e - cfg["boundary_before"]):min(len(y), e + cfg["boundary_after"])] = 1.0
    return {"x": np.concatenate(parts).astype(np.float32), "y": y, "b": b,
            "segments": np.asarray(segments, np.int64)}


# ---------------------------------------------------------------------------
# evaluation streams: GISLR-Sentences through the real extractor
# ---------------------------------------------------------------------------

def _stream(path: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    x = PH.model_features(z["landmarks"])
    assert len(x) == len(z["frame_kind"]), "every sentence frame has shoulders"
    return x, z["frame_kind"]


class PhonoStreamFeatures:
    """Like :class:`sb.recognize.sequences.baselines.StreamFeatures`, but every stream is the
    phono130 features of its raw 543 landmarks (one stream = one signer = one session: the dominant
    hand and shoulder width are the stream's own). Cached, content-addressed, atomic."""

    def __init__(self, data: np.ndarray, offsets: np.ndarray, kinds: np.ndarray):
        self.data, self.offsets, self.kinds = data, offsets, kinds

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def x(self, i: int) -> np.ndarray:
        return self.data[self.offsets[i]:self.offsets[i + 1]]

    def kind(self, i: int) -> np.ndarray:
        return self.kinds[self.offsets[i]:self.offsets[i + 1]]

    @staticmethod
    def key(root: Path) -> str:
        from sb.recognize.features import phono130_v1

        built = json.loads((root / "build_info.json").read_text(encoding="utf-8"))
        ident = {"pipeline": f"{phono130_v1.PIPELINE}-stream", "version": phono130_v1.PIPELINE_VERSION,
                 "dataset_version": built["dataset_version"], "config_sha256": built["config_sha256"],
                 "corpus_sha256": built["corpus_sha256"]}
        return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def build(cls, root: Path, seq_df: pd.DataFrame, progress=None) -> "PhonoStreamFeatures":
        from sb.recognize.sequences.baselines import FEATURES_ROOT

        out = FEATURES_ROOT / cls.key(root)
        paths = {n: out / f"{n}.npy" for n in ("data", "offsets", "kinds")}
        if all(p.is_file() for p in paths.values()):
            return cls(*(np.load(paths[n]) for n in ("data", "offsets", "kinds")))
        xs, ks, offsets = [], [], [0]
        with ProcessPoolExecutor(12) as ex:
            for i, (x, k) in enumerate(ex.map(_stream, [str(root / p) for p in seq_df["npz_relpath"]], chunksize=16)):
                xs.append(x)
                ks.append(k)
                offsets.append(offsets[-1] + len(x))
                if progress is not None:
                    progress(i + 1, len(seq_df))
        out.mkdir(parents=True, exist_ok=True)
        arrays = {"data": np.concatenate(xs), "offsets": np.asarray(offsets, np.int64), "kinds": np.concatenate(ks)}
        for n, arr in arrays.items():
            tmp = out / f"{n}.tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, paths[n])
        return cls(arrays["data"], arrays["offsets"], arrays["kinds"])
