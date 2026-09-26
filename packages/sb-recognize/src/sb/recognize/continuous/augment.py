"""Training-stream augmentation for continuous models, v2 (TODO §12.8 Fix 3).

C1 was trained on clean composed streams and breaks on a live camera
(``docs/reports/live-streaming-gap.md``): low fps makes it miss signs, jitter
and long sessions make it add signs, framing and mirroring break it outright,
and real signing runs signs together. Each augmentation here targets one of
those measured failures. Every one is off unless its composer key turns it on,
so v1 runs are unchanged.

Per clip, before composition (co-articulation, TODO §12.3 hard-cut):

- **trim** (``p_trim``, ``trim_max``): cut up to ``trim_max`` of a clip's frames
  from its start and from its end. Signs inside sentences are shorter than
  citation clips; linear-interpolation stitching makes sentences ~1.6x too
  long (BRAID, arXiv 2605.14705).
- **speed** (``p_speed``, ``speed``): resample a clip in time by a factor
  (> 1 = faster), nearest frame.

Per stream, the composer's own options:

- **hard cut** (``p_hardcut``): no gap frames between signs at all.
- **noise as null** (``p_noise``, ``noise_len``): one block of non-sign
  movement (fidget / hold / reverse, :mod:`sb.recognize.sequences.noise`) cut
  from other clips of the batch and spliced into a gap, labelled null (§12.6).

Per stream, after composition (``augment`` dict; the camera and the person):

- **geometry**, one draw per stream as for one session: scale, x-only aspect,
  rotation, shift (``p_geom``, ``scale``, ``aspect``, ``rotate_deg``, ``shift``);
- **mirror** (``p_mirror``): x flipped, left/right hands and pose joints
  swapped, face points swapped with their mirror partner (paired from the
  clip bank's mean face);
- **jitter** (``p_jitter``, ``jitter``): Gaussian noise on present points;
- **hand dropout** (``p_hand_drop``, ``hand_drop``): a hand missing on random
  frames;
- **low fps** (``p_fps``, ``fps_factor``): every k-th frame kept and the others
  either repeated or interpolated (the web app does either).

Coordinates are MediaPipe image xy with NaN = missing (the clip bank's
convention); NaN stays NaN throughout.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sb.recognize.sequences import noise as N

_LH0, _RH0, _POSE0 = 468, 522, 489
_POSE_LR = ((11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24))

DEFAULTS: dict[str, Any] = {  # composer keys this module reads, with their "off" values
    "p_trim": 0.0, "trim_max": 0.0, "p_speed": 0.0, "speed": [1.0, 1.0],
    "p_hardcut": 0.0, "p_noise": 0.0, "noise_len": [20, 90], "augment": None,
}


def active(ccfg: dict) -> bool:
    """True when any v2 augmentation is switched on."""
    c = {**DEFAULTS, **ccfg}
    return bool(c["p_trim"] or c["p_speed"] or c["p_hardcut"] or c["p_noise"] or c["augment"])


def mirror_permutation(rows: np.ndarray, sample: np.ndarray | None) -> np.ndarray:
    """Row permutation of a mirror image: left/right hands and pose pairs
    swapped; each face row paired with the face row nearest its reflection in
    the mean face of ``sample`` ((frames, 2 * len(rows)) xy, NaN allowed).
    Face rows stay in place when ``sample`` is None."""
    rows = np.asarray(rows)
    pos = {int(r): i for i, r in enumerate(rows)}
    perm = np.arange(len(rows))
    for k in range(21):
        a, b = _LH0 + k, _RH0 + k
        if a in pos and b in pos:
            perm[pos[a]], perm[pos[b]] = pos[b], pos[a]
    for a, b in _POSE_LR:
        a, b = _POSE0 + a, _POSE0 + b
        if a in pos and b in pos:
            perm[pos[a]], perm[pos[b]] = pos[b], pos[a]
    face = np.array([i for i, r in enumerate(rows) if r < _LH0])
    if sample is not None and len(face):
        from scipy.optimize import linear_sum_assignment

        xy = sample.reshape(len(sample), -1, 2)[:, face]
        xy = xy[np.isfinite(xy).all((1, 2))]  # frames with the whole face
        mean = (xy - xy.mean(axis=1, keepdims=True)).mean(axis=0)  # face-centred mean shape
        refl = mean * np.array([-1.0, 1.0])
        d = np.linalg.norm(refl[:, None] - mean[None], axis=-1)
        _, match = linear_sum_assignment(d + d.T)  # symmetric cost: one-to-one, pairs agree both ways
        perm[face] = face[match]
    return perm


class Augmenter:
    """All v2 augmentations for one (rows, xy) feature layout."""

    def __init__(self, ccfg: dict, rows: np.ndarray, coords: str, face_sample: np.ndarray | None = None):
        if coords != "xy":
            raise ValueError("v2 augmentation is written for coords='xy'")
        self.c: dict[str, Any] = {**DEFAULTS, **ccfg}
        self.a: dict[str, Any] = self.c["augment"] or {}
        self.rows = np.asarray(rows)
        self.perm = mirror_permutation(self.rows, face_sample)
        pos = {int(r): i for i, r in enumerate(self.rows)}
        self.hands = [np.array([pos[h + k] for k in range(21) if h + k in pos]) for h in (_LH0, _RH0)]

    # -- per clip -----------------------------------------------------------
    def clip(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c, T = self.c, len(x)
        if T > 8 and rng.random() < c["p_trim"]:
            h = int(rng.uniform(0, c["trim_max"]) * T)
            t = int(rng.uniform(0, c["trim_max"]) * T)
            if T - h - t >= max(4, T // 2):
                x = x[h:T - t]
        if rng.random() < c["p_speed"]:
            f = rng.uniform(*c["speed"])
            idx = np.unique(np.round(np.arange(0, len(x), f)).astype(int).clip(0, len(x) - 1))
            if len(idx) >= 4:
                x = x[idx]
        return x

    def stream_cfg(self, ccfg: dict, rng: np.random.Generator) -> dict:
        if rng.random() < self.c["p_hardcut"]:
            return {**ccfg, "gap_frames": [0, 0]}
        return ccfg

    # -- per stream ---------------------------------------------------------
    def stream(self, s: dict, donors: list[np.ndarray], null_index: int, rng: np.random.Generator) -> dict:
        if donors and rng.random() < self.c["p_noise"]:
            s = self._noise(s, donors, null_index, rng)
        s["x"] = self._camera(s["x"], rng)
        return s

    def _noise(self, s: dict, donors: list[np.ndarray], null_index: int, rng: np.random.Generator) -> dict:
        lo, hi = self.c["noise_len"]
        spec = N.NoiseSpec(length=(lo, hi))
        kind = N.NOISE_KINDS[int(rng.integers(len(N.NOISE_KINDS)))]
        blk = N.make_block(kind, N.DonorPool([d for d in donors if len(d) >= 4] or donors), rng, spec)
        seg = s["segments"]
        g = int(rng.integers(len(seg) + 1))  # which gap: before sign g
        a = 0 if g == 0 else int(seg[g - 1][1])
        b = len(s["y"]) if g == len(seg) else int(seg[g][0])
        cut = (a + b) // 2
        x = s["x"]
        left = x[cut - 1] if cut > 0 else blk[0]
        right = x[cut] if cut < len(x) else blk[-1]
        ins = np.concatenate([N._blend(left, blk[0], 3), blk, N._blend(blk[-1], right, 3)]).astype(np.float32)
        n = len(ins)
        segs = seg.copy()
        segs[g:] += n
        return {"x": np.concatenate([x[:cut], ins, x[cut:]]),
                "y": np.concatenate([s["y"][:cut], np.full(n, null_index, s["y"].dtype), s["y"][cut:]]),
                "b": np.concatenate([s["b"][:cut], np.zeros(n, s["b"].dtype), s["b"][cut:]]),
                "segments": segs}

    def _camera(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        a = self.a
        if not a:
            return x
        T = len(x)
        P = x.reshape(T, -1, 2).copy()
        if rng.random() < a.get("p_mirror", 0.0):
            P = P[:, self.perm]
            P[..., 0] = 1.0 - P[..., 0]
        if rng.random() < a.get("p_geom", 0.0):
            s = np.exp(rng.uniform(*np.log(a["scale"])))
            asp = rng.uniform(*a["aspect"])
            th = np.deg2rad(rng.uniform(-a["rotate_deg"], a["rotate_deg"]))
            sh = rng.uniform(-a["shift"], a["shift"], 2)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * s
            R[0] *= asp
            P = (P - 0.5) @ R.T + 0.5 + sh
        if rng.random() < a.get("p_jitter", 0.0):
            P = P + rng.normal(0, rng.uniform(0, a["jitter"]), P.shape)
        if rng.random() < a.get("p_hand_drop", 0.0):
            p = rng.uniform(0, a["hand_drop"])
            for h in self.hands:
                P[np.ix_(rng.random(T) < p, h)] = np.nan
        if T > 2 and rng.random() < a.get("p_fps", 0.0):
            k = int(rng.choice(a["fps_factor"]))
            lo = (np.arange(T) // k) * k
            if rng.random() < 0.5:  # the app repeats the last real frame
                P = P[lo]
            else:                   # the app interpolates toward the next real frame
                hi = np.minimum(lo + k, T - 1)
                w = ((np.arange(T) - lo) / k)[:, None, None]
                mix = P[lo] * (1 - w) + P[hi] * w
                P = np.where(np.isnan(mix), P[lo], mix)
        return P.reshape(T, -1).astype(np.float32)
