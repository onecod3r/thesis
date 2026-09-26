"""ASL phonological features from full MediaPipe Holistic frames (TODO §3.10).

Input: one clip as ``(T, 543, 3)`` (face 0-467, left hand 468-488, pose 489-521, right hand
522-542; NaN = not detected). Output: ``(T, F)`` per-frame features, each one a measurable
correlate of a phonological parameter of ASL (:data:`CATALOG`), plus clip-level summaries and
discrete codes (:func:`clip_record`). No model is fitted anywhere here.

**Normalization.**

1. *Anchor*: every coordinate minus that frame's mid-shoulder point (pose 11/12), the centre of
   the signing space. A frame without shoulders uses the clip's median anchor.
2. *Dominance*: the dominant hand (H1) is the hand detected in more frames (ties: the longer
   wrist path). x is negated for right-dominant clips (the signer's right is image-left), so in
   every clip +x points to the dominant side, left- and right-handed signers read alike, and
   "ipsilateral" means the dominant side.
3. *Scale* (optional, the experiment's two variants): lengths and velocities divided by the clip's
   median shoulder width (xy). Unscaled, they stay in image-width units. Hand-internal and face
   features are ratios or angles, so scaling does not touch them (:data:`SCALED`).

Hand geometry (handshape, orientation) uses the hand's own xyz: MediaPipe's hand z is relative to
the wrist on roughly the x scale, so angles inside one hand are meaningful. Location, contact and
movement use xy only: pose, hand and face z are measured against different origins.
"""

from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np

from sb.recognize.features.gislr_stratified import load_npz

LH0, P0, RH0 = 468, 489, 522
N_ROWS = 543
# Face mesh rows (the signer's own left/right), checked on GISLR clips 2026-09-26: 33/105/234
# sit on the signer's right (next to pose 5 right eye / pose 8 right ear), forehead 10 above
# brow 105 above lids 159/145 above nose 1 above lips 13/14 above chin 152.
FACE = {
    "forehead": 10, "nose": 1, "upper_lip": 13, "lower_lip": 14, "chin": 152,
    "eye_r_out": 33, "eye_r_in": 133, "lid_r_up": 159, "lid_r_lo": 145, "brow_r": 105,
    "eye_l_out": 263, "eye_l_in": 362, "lid_l_up": 386, "lid_l_lo": 374, "brow_l": 334,
    "mouth_r": 61, "mouth_l": 291, "cheek_r": 234, "cheek_l": 454,
}
POSE = {"ear_l": 7, "ear_r": 8, "sh_l": 11, "sh_r": 12, "el_l": 13, "el_r": 14, "wr_l": 15, "wr_r": 16,
        "hip_l": 23, "hip_r": 24}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CHAIN = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
         "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
SITES = ("forehead", "eyes", "nose", "mouth", "chin", "cheek", "ear", "shoulder_ipsi",
         "shoulder_contra", "torso", "other_hand")
FACE_SITES = SITES[:7]  # measured from the closest hand point (where contact happens), the rest from the palm centre
PARAMETERS = ("handshape", "orientation", "location", "contact", "movement", "hand_arrangement",
              "non_manual")


@dataclass(frozen=True)
class Feature:
    name: str
    parameter: str  # one of PARAMETERS, or "presence"
    unit: str       # "ratio", "deg", "unit_vector", "length", "velocity", "flag"
    meaning: str


def _hand_catalog(h: str) -> list[Feature]:
    who = "dominant" if h == "h1" else "non-dominant"
    f = [Feature(f"{h}_present", "presence", "flag", f"{who} hand detected")]
    f += [Feature(f"{h}_ext_{n}", "handshape", "ratio",
                  f"{n} extension: base-to-tip distance / finger length (1 = straight); selected vs unselected fingers")
          for n in FINGERS]
    f += [Feature(f"{h}_base_flex_{n}", "handshape", "deg", f"{n} base-joint (MCP) flexion; 'flat'/'bent' handshapes")
          for n in FINGERS]
    f += [Feature(f"{h}_nonbase_flex_{n}", "handshape", "deg",
                  f"{n} non-base (PIP+DIP, thumb IP) flexion; 'curved'/'closed' handshapes") for n in FINGERS]
    f += [Feature(f"{h}_spread_{p}", "handshape", "deg", f"abduction angle {p}; spread vs unspread")
          for p in ("index_middle", "middle_ring", "ring_pinky", "thumb_index")]
    f += [Feature(f"{h}_aperture_{n}", "handshape", "ratio", f"thumb tip to {n} tip / palm size; aperture, thumb contact")
          for n in FINGERS[1:]]
    f += [Feature(f"{h}_thumb_to_index_base", "handshape", "ratio", "thumb tip to index MCP / palm; thumb open vs closed"),
          Feature(f"{h}_thumb_to_pinky_base", "handshape", "ratio", "thumb tip to pinky MCP / palm; thumb across the palm")]
    f += [Feature(f"{h}_palm_{a}", "orientation", "unit_vector", f"palm normal, {a} component (x toward the dominant side, y down, z away from the camera)")
          for a in "xyz"]
    f += [Feature(f"{h}_finger_dir_{a}", "orientation", "unit_vector", f"wrist -> middle knuckle direction, {a} component")
          for a in "xyz"]
    f += [Feature(f"{h}_pos_{a}", "location", "length", f"palm centre {a} relative to the mid-shoulder anchor")
          for a in "xy"]
    f += [Feature(f"{h}_dist_{s}", "location", "length",
                  f"{'closest hand point' if s in FACE_SITES else 'palm centre'} to {s} (xy); place of articulation")
          for s in SITES]
    grp = "location" if h == "h1" else "hand_arrangement"
    f += [Feature(f"{h}_wrist_{a}", grp, "length", f"{who} pose wrist {a} (pose is detected in every frame; hands are not)")
          for a in "xy"]
    f += [Feature(f"{h}_wrist_speed", "movement" if h == "h1" else "hand_arrangement", "velocity",
                  f"{who} pose wrist speed; path movement from the arm"),
          Feature(f"{h}_elbow", grp, "deg", f"{who} elbow angle (shoulder-elbow-wrist)")]
    f += [Feature(f"{h}_touch_face", "contact", "length", "closest hand point to any face-mesh point (xy); contact proxy")]
    f += [Feature(f"{h}_vel_{a}", "movement", "velocity", f"palm-centre velocity {a} per frame; path movement")
          for a in "xy"]
    f += [Feature(f"{h}_speed", "movement", "velocity", "palm-centre speed; path movement"),
          Feature(f"{h}_accel", "movement", "velocity", "palm-centre acceleration magnitude; movement onset/stop"),
          Feature(f"{h}_d_aperture", "movement", "ratio", "change of mean aperture per frame; local movement (opening/closing)"),
          Feature(f"{h}_d_flex", "movement", "deg", "change of mean non-base flexion per frame; handshape change"),
          Feature(f"{h}_d_orient", "movement", "deg", "palm-normal rotation per frame; orientation change / wrist twist")]
    return f


CATALOG: list[Feature] = (
    _hand_catalog("h1") + _hand_catalog("h2")
    + [Feature("both_present", "hand_arrangement", "flag", "both hands detected; one- vs two-handed"),
       Feature("touch_hands", "hand_arrangement", "length", "closest point between the hands (xy); hand-hand contact"),
       Feature("h2_rel_x", "hand_arrangement", "length", "non-dominant minus dominant palm centre, x"),
       Feature("h2_rel_y", "hand_arrangement", "length", "non-dominant minus dominant palm centre, y"),
       Feature("sym_vel", "hand_arrangement", "ratio", "cosine of dominant velocity and mirrored non-dominant velocity; symmetric (+1) / alternating (-1)"),
       Feature("shape_sim", "hand_arrangement", "ratio", "cosine of the two hands' flexion profiles; same vs different handshape (Battison)")]
    + [Feature("brow_raise_r", "non_manual", "ratio", "right brow to upper lid / eye distance"),
       Feature("brow_raise_l", "non_manual", "ratio", "left brow to upper lid / eye distance"),
       Feature("eye_open_r", "non_manual", "ratio", "right lid gap / eye width"),
       Feature("eye_open_l", "non_manual", "ratio", "left lid gap / eye width"),
       Feature("mouth_open", "non_manual", "ratio", "lip gap / eye distance"),
       Feature("mouth_width", "non_manual", "ratio", "mouth corner distance / eye distance"),
       Feature("head_yaw", "non_manual", "ratio", "nose offset from mid-eyes, x / eye distance"),
       Feature("head_pitch", "non_manual", "ratio", "nose drop below mid-eyes / eye distance"),
       Feature("head_roll", "non_manual", "deg", "eye-line angle"),
       Feature("shoulder_tilt", "non_manual", "deg", "shoulder-line angle")]
)
NAMES = [f.name for f in CATALOG]
F = len(CATALOG)
IDX = {n: i for i, n in enumerate(NAMES)}
SCALED = np.array([f.unit in ("length", "velocity") for f in CATALOG])  # divided by shoulder width when scaled
GROUP = np.array([f.parameter for f in CATALOG])
RAW_DIMS = N_ROWS * 3
POSE_SWAP = np.arange(33)
for _a, _b in ((1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
               (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)):
    POSE_SWAP[_a], POSE_SWAP[_b] = _b, _a
PALM_SIGN = -1.0  # checked on GISLR: BYE (palm to the viewer) gives z < 0 with it


def load(path) -> np.ndarray:
    """One GISLR_Stratified npz -> ``(T, 543, 3)`` float32, NaN kept."""
    return load_npz(path, np.arange(N_ROWS), "xyz")


# ---------------------------------------------------------------------------
# geometry helpers (all NaN-propagating)
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _angle(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Degrees between vectors ``u`` and ``w`` (last axis)."""
    return np.degrees(np.arccos(np.clip((_unit(u) * _unit(w)).sum(-1), -1.0, 1.0)))


def _d2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)


def _min_pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(T, n, 2)`` x ``(T, m, 2)`` -> ``(T,)`` closest xy distance, NaN if either side is empty."""
    d = np.linalg.norm(a[:, :, None, :2] - b[:, None, :, :2], axis=-1).reshape(len(a), -1)
    out = np.full(len(a), np.nan, np.float32)
    ok = ~np.isnan(d).all(axis=1)
    out[ok] = np.nanmin(d[ok], axis=1)
    return out


def _handshape(h: np.ndarray) -> np.ndarray:
    """``(T, 21, 3)`` hand -> ``(T, 25)``: extension 5, base flex 5, non-base flex 5, spread 4,
    aperture 4, thumb position 2 (the order of :func:`_hand_catalog`)."""
    wrist = h[:, 0]
    palm = np.linalg.norm(h[:, 9] - wrist, axis=-1)[:, None] + 1e-9
    ext, base, nonbase = [], [], []
    for n in FINGERS:
        a, b, c, d = (h[:, k] for k in CHAIN[n])
        length = (np.linalg.norm(b - a, axis=-1) + np.linalg.norm(c - b, axis=-1)
                  + np.linalg.norm(d - c, axis=-1)) + 1e-9
        ext.append(np.linalg.norm(d - a, axis=-1) / length)
        if n == "thumb":  # CMC(1) MCP(2) IP(3) TIP(4): base = MCP bend, non-base = IP bend
            base.append(_angle(b - a, c - b))
            nonbase.append(_angle(c - b, d - c))
        else:             # MCP(a) PIP(b) DIP(c) TIP(d): base = palm-to-finger bend at MCP
            base.append(_angle(a - wrist, b - a))
            nonbase.append(_angle(b - a, c - b) + _angle(c - b, d - c))
    dirs = {n: h[:, CHAIN[n][3]] - h[:, CHAIN[n][0]] for n in FINGERS}
    spread = [_angle(dirs["index"], dirs["middle"]), _angle(dirs["middle"], dirs["ring"]),
              _angle(dirs["ring"], dirs["pinky"]), _angle(dirs["thumb"], dirs["index"])]
    tip = h[:, 4]
    aperture = [np.linalg.norm(tip - h[:, CHAIN[n][3]], axis=-1) / palm[:, 0] for n in FINGERS[1:]]
    thumb = [np.linalg.norm(tip - h[:, 5], axis=-1) / palm[:, 0], np.linalg.norm(tip - h[:, 17], axis=-1) / palm[:, 0]]
    return np.stack(ext + base + nonbase + spread + aperture + thumb, axis=1)


def _orientation(h: np.ndarray, flip: bool) -> np.ndarray:
    """Palm normal (3) and finger direction (3). After :func:`extract`'s mirror the two hands
    have opposite chirality, so the non-dominant hand's normal is flipped (``flip``) to point out
    of its palm too; PALM_SIGN makes z < 0 mean "palm toward the camera"."""
    n = PALM_SIGN * _unit(np.cross(h[:, 5] - h[:, 0], h[:, 17] - h[:, 0]))
    if flip:
        n = -n
    return np.concatenate([n, _unit(h[:, 9] - h[:, 0])], axis=1)


def dominance(clip: np.ndarray) -> str:
    """``"right"`` or ``"left"``: the hand detected in more frames (ties: longer wrist path)."""
    pres = [np.sum(~np.isnan(clip[:, h0, 0])) for h0 in (RH0, LH0)]
    if pres[0] != pres[1]:
        return "right" if pres[0] > pres[1] else "left"

    def path(h0):
        w = clip[:, h0, :2]
        w = w[~np.isnan(w[:, 0])]
        return float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()) if len(w) > 1 else 0.0

    return "right" if path(RH0) >= path(LH0) else "left"


# ---------------------------------------------------------------------------
# per-frame features
# ---------------------------------------------------------------------------

def extract(clip: np.ndarray) -> tuple[np.ndarray, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames (a hand or face missing)
        return _extract(clip)


def _extract(clip: np.ndarray) -> tuple[np.ndarray, dict]:
    """``(T, 543, 3)`` -> ``(T', F)`` **unscaled** features (image-width units for lengths), and
    info: ``width`` (median shoulder width), ``dominant``, ``frames`` (T), ``kept`` (T').

    Frames without shoulders on a clip that never has them are dropped; otherwise every frame is
    kept, and a missing hand leaves its features NaN. :func:`scale` gives the scaled variant."""
    sh = clip[:, [P0 + POSE["sh_l"], P0 + POSE["sh_r"]], :2]
    has_sh = ~np.isnan(sh).any(axis=(1, 2))
    info = {"frames": len(clip), "dominant": dominance(clip)}
    if not has_sh.any():
        return np.zeros((0, F), np.float32), {**info, "width": np.nan, "kept": 0}
    anchor = sh.mean(axis=1)
    anchor[~has_sh] = np.nanmedian(anchor[has_sh], axis=0)
    width = float(np.nanmedian(np.linalg.norm(sh[has_sh, 0] - sh[has_sh, 1], axis=-1)))
    c = clip.astype(np.float32).copy()
    c[..., :2] -= anchor[:, None, :]
    right = info["dominant"] == "right"
    if right:
        c[..., 0] *= -1  # +x = the dominant side in every clip
    side = {"ipsi": "r", "contra": "l"} if right else {"ipsi": "l", "contra": "r"}
    h1 = c[:, RH0:RH0 + 21] if right else c[:, LH0:LH0 + 21]
    h2 = c[:, LH0:LH0 + 21] if right else c[:, RH0:RH0 + 21]
    face, pose = c[:, :468], c[:, P0:P0 + 33]
    fp = {k: face[:, i] for k, i in FACE.items()}
    pp = {k: pose[:, i] for k, i in POSE.items()}
    eyes = (fp["eye_r_out"] + fp["eye_l_out"]) / 2
    torso = (pp["sh_l"] + pp["sh_r"] + pp["hip_l"] + pp["hip_r"]) / 4
    T = len(c)
    out = np.full((T, F), np.nan, np.float32)

    def put(prefix: str, values) -> None:
        i = IDX[prefix]
        v = np.asarray(values, np.float32)
        v = v[:, None] if v.ndim == 1 else v
        out[:, i:i + v.shape[1]] = v

    centres = {}
    for tag, h, other, ipsi, contra in (("h1", h1, h2, side["ipsi"], side["contra"]),
                                        ("h2", h2, h1, side["contra"], side["ipsi"])):
        present = ~np.isnan(h[:, 0, 0])
        put(f"{tag}_present", present.astype(np.float32))
        put(f"{tag}_ext_thumb", _handshape(h))
        # the non-dominant hand has the opposite chirality once the clip reads right-dominant
        put(f"{tag}_palm_x", _orientation(h, flip=(tag == "h2")))
        centre = h[:, [0, 5, 9, 13, 17]].mean(axis=1)
        centres[tag] = centre
        put(f"{tag}_pos_x", centre[:, :2])
        oc = other[:, [0, 5, 9, 13, 17]].mean(axis=1)
        sites = {"forehead": fp["forehead"], "eyes": eyes, "nose": fp["nose"],
                 "mouth": (fp["upper_lip"] + fp["lower_lip"]) / 2, "chin": fp["chin"],
                 "cheek": fp[f"cheek_{ipsi}"], "ear": pp[f"ear_{ipsi}"],
                 "shoulder_ipsi": pp[f"sh_{ipsi}"], "shoulder_contra": pp[f"sh_{contra}"],
                 "torso": torso, "other_hand": oc}
        put(f"{tag}_dist_forehead", np.stack(
            [_min_pairwise(h, sites[s][:, None]) if s in FACE_SITES else _d2(centre, sites[s]) for s in SITES], axis=1))
        s_ = ipsi
        wr, el, shd = pp[f"wr_{s_}"], pp[f"el_{s_}"], pp[f"sh_{s_}"]
        put(f"{tag}_wrist_x", wr[:, :2])
        put(f"{tag}_wrist_speed", np.r_[np.nan, np.linalg.norm(np.diff(wr[:, :2], axis=0), axis=1)])
        put(f"{tag}_elbow", _angle(shd[:, :2] - el[:, :2], wr[:, :2] - el[:, :2]))
        put(f"{tag}_touch_face", _min_pairwise(h, face))
        vel = np.vstack([np.full((1, 2), np.nan, np.float32), np.diff(centre[:, :2], axis=0)])
        acc = np.vstack([np.full((1, 2), np.nan, np.float32), np.diff(vel, axis=0)])
        hs = out[:, IDX[f"{tag}_ext_thumb"]:IDX[f"{tag}_ext_thumb"] + 25]
        apert = np.nanmean(hs[:, 19:23], axis=1) if present.any() else np.full(T, np.nan)
        flex = np.nanmean(hs[:, 11:15], axis=1) if present.any() else np.full(T, np.nan)
        normal = out[:, IDX[f"{tag}_palm_x"]:IDX[f"{tag}_palm_x"] + 3]
        rot = np.concatenate([[np.nan], _angle(normal[1:], normal[:-1])])
        put(f"{tag}_vel_x", vel)
        put(f"{tag}_speed", np.linalg.norm(vel, axis=1))
        put(f"{tag}_accel", np.linalg.norm(acc, axis=1))
        put(f"{tag}_d_aperture", np.concatenate([[np.nan], np.diff(apert)]))
        put(f"{tag}_d_flex", np.concatenate([[np.nan], np.diff(flex)]))
        put(f"{tag}_d_orient", rot)

    both = ~np.isnan(h1[:, 0, 0]) & ~np.isnan(h2[:, 0, 0])
    put("both_present", both.astype(np.float32))
    put("touch_hands", _min_pairwise(h1, h2))
    put("h2_rel_x", (centres["h2"] - centres["h1"])[:, :2])
    v1 = out[:, IDX["h1_vel_x"]:IDX["h1_vel_x"] + 2]
    v2 = out[:, IDX["h2_vel_x"]:IDX["h2_vel_x"] + 2] * np.array([-1, 1], np.float32)
    put("sym_vel", (_unit(v1) * _unit(v2)).sum(-1))
    f1 = out[:, IDX["h1_base_flex_thumb"]:IDX["h1_base_flex_thumb"] + 10]
    f2 = out[:, IDX["h2_base_flex_thumb"]:IDX["h2_base_flex_thumb"] + 10]
    put("shape_sim", (_unit(f1 - 45) * _unit(f2 - 45)).sum(-1))  # centred so open vs closed differ in sign

    eye_d = np.linalg.norm(fp["eye_r_out"][:, :2] - fp["eye_l_out"][:, :2], axis=-1) + 1e-9
    put("brow_raise_r", _d2(fp["brow_r"], fp["lid_r_up"]) / eye_d)
    put("brow_raise_l", _d2(fp["brow_l"], fp["lid_l_up"]) / eye_d)
    put("eye_open_r", _d2(fp["lid_r_up"], fp["lid_r_lo"]) / (_d2(fp["eye_r_out"], fp["eye_r_in"]) + 1e-9))
    put("eye_open_l", _d2(fp["lid_l_up"], fp["lid_l_lo"]) / (_d2(fp["eye_l_out"], fp["eye_l_in"]) + 1e-9))
    put("mouth_open", _d2(fp["upper_lip"], fp["lower_lip"]) / eye_d)
    put("mouth_width", _d2(fp["mouth_r"], fp["mouth_l"]) / eye_d)
    mid_eye = (fp["eye_r_out"] + fp["eye_l_out"]) / 2
    put("head_yaw", (fp["nose"][:, 0] - mid_eye[:, 0]) / eye_d)
    put("head_pitch", (fp["nose"][:, 1] - mid_eye[:, 1]) / eye_d)
    er, el = fp["eye_r_out"], fp["eye_l_out"]
    put("head_roll", np.degrees(np.arctan2(el[:, 1] - er[:, 1], np.abs(el[:, 0] - er[:, 0]) + 1e-9)))
    sr, sl = pp["sh_r"], pp["sh_l"]
    put("shoulder_tilt", np.degrees(np.arctan2(sl[:, 1] - sr[:, 1], np.abs(sl[:, 0] - sr[:, 0]) + 1e-9)))
    return out, {**info, "width": width, "kept": T}


def scale(feats: np.ndarray, width: float) -> np.ndarray:
    """The scaled variant: length and velocity features divided by the shoulder width."""
    out = feats.copy()
    out[:, SCALED] /= max(width, 1e-6)
    return out


def raw_normalized(clip: np.ndarray, width: float | None) -> np.ndarray:
    """The raw-landmark baseline with the same anchor, dominance mirror and (optional) scale:
    ``(T, 1629)``. z is left as MediaPipe gives it."""
    sh = clip[:, [P0 + POSE["sh_l"], P0 + POSE["sh_r"]], :2]
    has_sh = ~np.isnan(sh).any(axis=(1, 2))
    if not has_sh.any():
        return np.zeros((0, RAW_DIMS), np.float32)
    anchor = sh.mean(axis=1)
    anchor[~has_sh] = np.nanmedian(anchor[has_sh], axis=0)
    c = clip.astype(np.float32).copy()
    c[..., :2] -= anchor[:, None, :]
    if dominance(clip) == "right":
        c[..., 0] *= -1
    else:  # dominant hand into the right-hand slot, left/right pose pairs swapped (face mesh is not)
        pose = c[:, P0:RH0][:, POSE_SWAP]
        c = np.concatenate([c[:, :LH0], c[:, RH0:RH0 + 21], pose, c[:, LH0:P0]], axis=1)
    if width:
        c[..., :2] /= max(width, 1e-6)
    return c.reshape(len(c), -1)


# ---------------------------------------------------------------------------
# clip level: temporal summary + discrete phonological codes
# ---------------------------------------------------------------------------

SEGMENTS = ("onset", "medial", "final")  # thirds of the clip, after Liddell & Johnson's hold-movement-hold
SEQ_LEN = 32
SUMMARY = [f"{s}:{n}" for s in SEGMENTS for n in NAMES] + [f"std:{n}" for n in NAMES]
GLOBAL_WIDTH = 0.56  # median GISLR shoulder width (image-x units, 1,000 clips, 2026-09-26): unscaled thresholds

# code thresholds; lengths in shoulder widths (x GLOBAL_WIDTH when unscaled), the rest scale-free
EXT_T = 0.90          # finger extension ratio: extended
THUMB_OPEN_T = 0.60   # thumb tip to index MCP / palm: thumb open
APERTURE_T = 0.30     # thumb tip to a fingertip / palm: thumb contact
SPREAD_T = 12.0       # mean adjacent-finger angle (deg): spread
HEAD_T = 0.15         # closest hand point within this of a head site: head location
HAND_T = 0.35         # palm centre within this of the other hand: hand location
CONTACT_T = 0.03      # hand point within this of the face or other hand: contact
STILL_T = 0.15        # path length below this: no path movement
AMP_T = 0.03          # a direction reversal must travel at least this: repetition
DIR_T = 0.10          # net displacement below this: no path direction
RAISED_T = 0.7        # non-dominant wrist less than this below the shoulder line: raised (rest is ~1.0)
FLEX_CHANGE_T = 40.0  # onset-to-final change of mean non-base flexion (deg)
APERT_CHANGE_T = 0.50  # onset-to-final change of mean aperture (palm units)
ORI_CHANGE_T = 60.0   # onset-to-final palm-normal rotation (deg)
HEAD_SITES = ("forehead", "eyes", "nose", "mouth", "chin", "cheek", "ear")
CODES = ("selected_fingers", "flexion", "spread", "thumb", "thumb_contact", "palm_dir", "finger_dir",
         "major_location", "minor_location", "contact", "movement", "path_dir", "repeated",
         "handshape_change", "orientation_change", "sign_type")
DIRS = ("ipsi", "contra", "down", "up", "signer", "camera")  # +x, -x, +y, -y, +z, -z


def summarize(feats: np.ndarray) -> np.ndarray:
    """``(T, F)`` -> ``(4F,)``: the mean of each third of the clip, then the std over the clip."""
    return _summarize(feats, F).astype(np.float32)


def summarize_raw(x: np.ndarray) -> np.ndarray:
    """The raw baseline's summary, same recipe over the 1,629 coordinates, float16."""
    return _summarize(x, RAW_DIMS).astype(np.float16)


def _summarize(x: np.ndarray, width: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if len(x) == 0:
            return np.full(4 * width, np.nan)
        thirds = [np.nanmean(s, axis=0) if len(s) else np.full(width, np.nan) for s in np.array_split(x, 3)]
        return np.concatenate(thirds + [np.nanstd(x, axis=0)])


def resample(x: np.ndarray, n: int = SEQ_LEN) -> np.ndarray:
    """Nearest-frame resampling to ``n`` steps (NaN kept)."""
    if len(x) == 0:
        return np.full((n, x.shape[1]), np.nan, x.dtype)
    return x[np.round(np.linspace(0, len(x) - 1, n)).astype(int)]


def _block(f: np.ndarray, first: str, n: int) -> np.ndarray:
    return f[:, IDX[first]:IDX[first] + n]


def _direction(v: np.ndarray, labels: tuple[str, ...] = DIRS) -> str:
    """Largest component of a vector -> its label (``labels`` = +x, -x, +y, -y[, +z, -z])."""
    if np.isnan(v).any():
        return "na"
    k = int(np.argmax(np.abs(v)))
    return labels[2 * k + (0 if v[k] > 0 else 1)]


def reversals(s: np.ndarray, amp: float) -> int:
    """Direction reversals of a 1-D signal (3-frame smoothed), each swing at least ``amp``."""
    if len(s) < 4:
        return 0
    s = np.convolve(s, np.ones(3) / 3, mode="valid")
    turns, direction, pivot = 0, 0, s[0]
    for v in s[1:]:
        if direction == 0:
            if abs(v - pivot) >= amp:
                direction, pivot = int(np.sign(v - pivot)), v
        elif (v - pivot) * direction > 0:
            pivot = v  # still going the same way: extend the swing
        elif abs(v - pivot) >= amp:
            turns, direction, pivot = turns + 1, -direction, v
    return turns


def _smooth(p: np.ndarray, k: int = 3) -> np.ndarray:
    """Drop NaN rows, then a ``k``-frame moving average."""
    p = p[~np.isnan(p).any(axis=1)]
    if len(p) <= k:
        return p
    w = np.ones(k) / k
    return np.stack([np.convolve(p[:, i], w, mode="valid") for i in range(p.shape[1])], axis=1)


def _turning(p: np.ndarray, step: float) -> float:
    """Total heading change (deg) along a path resampled every ``step`` of arc length, so
    jitter below ``step`` adds nothing."""
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < 3 * step:
        return 0.0
    grid = np.arange(0, s[-1], step)
    q = np.stack([np.interp(grid, s, p[:, i]) for i in range(2)], axis=1)
    v = np.diff(q, axis=0)
    h = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    return float(np.degrees(np.abs(np.diff(h)).sum()))


def _path_reversals(p: np.ndarray, amp: float) -> int:
    """Reversals of a 2-D path along its principal axis."""
    if len(p) < 4:
        return 0
    c = p - p.mean(axis=0)
    return reversals(c @ np.linalg.svd(c, full_matrices=False)[2][0], amp)


NUCLEUS_TRIM = 0.2  # share of the dominant hand's detected span dropped at each end


def nucleus(f: np.ndarray) -> slice:
    """Frames of the sign nucleus: the dominant hand's first-to-last detected frame, with
    :data:`NUCLEUS_TRIM` of that span dropped at each end."""
    idx = np.flatnonzero(f[:, IDX["h1_present"]] == 1) if len(f) else np.zeros(0, int)
    if len(idx) == 0:
        return slice(0, 0)
    a, b = int(idx[0]), int(idx[-1]) + 1
    cut = int(round((b - a) * NUCLEUS_TRIM))
    return slice(a + cut, max(b - cut, a + cut + 1))


def codes(feats: np.ndarray, unit: float = 1.0) -> dict[str, str]:
    """Discrete phonological codes of one clip (:data:`CODES`), for the dominant hand over its
    detected frames split in thirds (onset / medial / final). ``feats``: the scaled variant with
    ``unit=1``, or the unscaled one with ``unit=GLOBAL_WIDTH`` (length thresholds x unit)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _codes(feats, unit)


def _codes(f: np.ndarray, unit: float) -> dict[str, str]:
    out = dict.fromkeys(CODES, "na")
    a = f[f[:, IDX["h1_present"]] == 1] if len(f) else f
    if len(a) == 0:
        out["sign_type"] = "no_hand"
        return out
    onset, medial, final = np.array_split(a, 3) if len(a) >= 3 else (a, a, a)

    def med(seg: np.ndarray, name: str) -> float:
        return float(np.nanmedian(seg[:, IDX[name]]))

    # handshape (medial third)
    ext = [med(medial, f"h1_ext_{n}") for n in FINGERS[1:]]
    sel = "".join(ch for ch, e in zip("imrp", ext) if e > EXT_T)
    thumb_open = med(medial, "h1_thumb_to_index_base") > THUMB_OPEN_T
    out["selected_fingers"] = sel or ("t" if thumb_open else "closed")
    base = float(np.nanmean([med(medial, f"h1_base_flex_{n}") for n in FINGERS[1:]]))
    nonbase = float(np.nanmean([med(medial, f"h1_nonbase_flex_{n}") for n in FINGERS[1:]]))
    out["flexion"] = ("closed" if nonbase >= 150 or np.nanmean(ext) < 0.75 else "curved" if nonbase >= 60
                      else "bent" if base >= 45 else "open")
    pairs = [("i", "m", "index_middle"), ("m", "r", "middle_ring"), ("r", "p", "ring_pinky")]
    sp = [med(medial, f"h1_spread_{n}") for x, y, n in pairs if x in sel and y in sel]
    if sp:
        out["spread"] = str(int(np.nanmean(sp) > SPREAD_T))
    out["thumb"] = "open" if thumb_open else "closed"
    apert = float(np.nanmin(_block(medial, "h1_aperture_index", 4)))
    out["thumb_contact"] = "na" if np.isnan(apert) else str(int(apert < APERTURE_T))
    # orientation (medial third)
    out["palm_dir"] = _direction(np.nanmedian(_block(medial, "h1_palm_x", 3), axis=0))
    out["finger_dir"] = _direction(np.nanmedian(_block(medial, "h1_finger_dir_x", 3), axis=0))
    # location (medial third) and contact (whole clip)
    head = {s: med(medial, f"h1_dist_{s}") for s in HEAD_SITES}
    head = {s: d for s, d in head.items() if not np.isnan(d)}
    if med(medial, "h1_dist_other_hand") < HAND_T * unit:
        out["major_location"] = "hand"
    elif head and min(head.values()) < HEAD_T * unit:
        out["major_location"] = "head"
        out["minor_location"] = min(head, key=lambda s: head[s])
    else:  # 2-D landmarks cannot tell the torso from the space in front of it
        out["major_location"] = "body_neutral"
    touch = np.r_[a[:, IDX["h1_touch_face"]], a[:, IDX["touch_hands"]]]
    if not np.isnan(touch).all():
        out["contact"] = str(int(np.nanmin(touch) < CONTACT_T * unit))
    # movement: the dominant pose wrist (detected in every frame, unlike the hand) over the sign
    # nucleus, the middle of the span where the hand is detected, which leaves out the transition
    # from rest into the sign and back (Liddell & Johnson's movement epenthesis)
    pos = _smooth(_block(f[nucleus(f)], "h1_wrist_x", 2))
    length = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) if len(pos) > 1 else 0.0
    disp = pos[-1] - pos[0] if len(pos) > 1 else np.zeros(2)
    rev = _path_reversals(pos, AMP_T * unit)
    if length < STILL_T * unit:
        out["movement"] = "none"
    else:
        turning = _turning(pos, 0.02 * unit)
        straight = float(np.linalg.norm(disp)) / max(length, 1e-9)
        out["movement"] = ("circular" if turning > 300 and straight < 0.4 else "back_and_forth" if rev >= 2
                           else "curved" if straight < 0.75 or turning > 90 else "straight")
    out["path_dir"] = "none" if np.linalg.norm(disp) < DIR_T * unit else _direction(disp, DIRS[:4])
    apt = np.nanmean(_block(a, "h1_aperture_index", 4), axis=1)
    apt = apt[~np.isnan(apt)]
    out["repeated"] = str(int(rev >= 2 or reversals(apt, 0.3) >= 2))
    fl0, fl1 = (float(np.nanmean([med(s, f"h1_nonbase_flex_{n}") for n in FINGERS[1:]])) for s in (onset, final))
    ap0, ap1 = (float(np.nanmean([med(s, f"h1_aperture_{n}") for n in FINGERS[1:]])) for s in (onset, final))
    out["handshape_change"] = str(int(abs(fl1 - fl0) > FLEX_CHANGE_T or abs(ap1 - ap0) > APERT_CHANGE_T))
    n0 = np.nanmedian(_block(onset, "h1_palm_x", 3), axis=0)
    n1 = np.nanmedian(_block(final, "h1_palm_x", 3), axis=0)
    out["orientation_change"] = str(int(float(_angle(n0, n1)) > ORI_CHANGE_T))
    # hand arrangement (Battison's sign types)
    # (GISLR's hand landmarks almost never show both hands in one frame, so this reads the arms)
    w2 = _smooth(_block(f, "h2_wrist_x", 2))
    raised = float(np.mean(w2[:, 1] < RAISED_T * unit)) if len(w2) else 0.0
    l2 = float(np.linalg.norm(np.diff(w2, axis=0), axis=1).sum()) if len(w2) > 1 else 0.0
    out["sign_type"] = ("one" if raised < 0.5 else "two_moving" if l2 > max(0.5 * length, STILL_T * unit)
                        else "base")
    return out


def clip_record(path) -> dict:
    """Everything the experiment keeps for one clip (process-pool friendly): info, both
    variants' summaries and codes, the raw-landmark baseline summaries, the unscaled sequence."""
    clip = load(path)
    feats, info = extract(clip)
    width = info["width"] if np.isfinite(info["width"]) else GLOBAL_WIDTH
    rec: dict = {"info": info}
    for variant, x, unit, w in (("unscaled", feats, GLOBAL_WIDTH, None),
                                ("scaled", scale(feats, width), 1.0, width)):
        rec[variant] = {"summary": summarize(x), "codes": codes(x, unit),
                        "raw": summarize_raw(raw_normalized(clip, w))}
    rec["seq"] = resample(feats).astype(np.float16)
    for h in ("h1", "h2"):
        rec[f"{h}_share"] = float(np.mean(feats[:, IDX[f"{h}_present"]])) if len(feats) else 0.0
    return rec
