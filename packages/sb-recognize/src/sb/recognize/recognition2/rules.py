"""Recognition v2 (TODO §14): a rule-based, **pose-free** implementation of ASL-LEX 2.0's 18
phonological parameters, for the GISLR glosses that map into ASL-LEX.

Isolated from `sb.recognize.phonology`/`aslex` by request (2026-09-27) -- neither is edited. This
module imports their pose-independent math helpers only (generic vector geometry: `_handshape`,
`_orientation`, `_angle`, `_unit`, `_d2`, `_min_pairwise`, `reversals`, `_turning`,
`_path_reversals`, `_smooth`, `dominance`) and writes fresh per-parameter rules that target
ASL-LEX's own category vocabulary directly (checked against `signdata.csv` for the 233 GISLR
glosses that map into ASL-LEX, this conversation, 2026-09-26/27), so output is comparable to
`aslex.gloss_codes()` with no translation layer.

**No pose by default** (user ask, 2026-09-27, "test without pose if that is possible"): every rule
reads only face (rows 0-467) and hand (468-488, 522-542) landmarks -- never rows 489-521, unless
``use_pose=True`` is passed to :func:`extract` (added 2026-09-27 for the "does pose add a significant
benefit" ablation the user asked for -- run both and compare, not a permanent design change).
Consequences of the pose-free default:

- **Anchor**: mid-shoulder is unavailable, so translation invariance uses a face centroid (mean of
  both eye-outer corners and both mouth corners) instead. This is weaker under head rotation than
  the shoulder anchor (a nod/turn moves this centroid without the hands translating) -- a deliberate
  ablation, not a claim of parity; see `docs/reports/phonology-pipeline-proposal.md` §2.
- **Scale**: inter-eye distance substitutes for shoulder width.
- **Movement/RepeatedMovement**: read the hand's own palm-centre trajectory directly, not the pose
  wrist `phonology.py` falls back to when the hand briefly drops out -- fine for GISLR's short,
  mostly-continuously-tracked clips, degrades if the dominant hand is undetected for a stretch.
- **MajorLocation** can only reach `Hand`/`Head`/`Neutral` -- ASL-LEX's `Body`/`Arm` categories name
  torso/forearm contact this module cannot see without pose, and are reported `NOT_COMPUTABLE`
  rather than guessed.
- **MinorLocation**: only the face-adjacent categories in :data:`MINOR_LOCATION_MAP` are attempted;
  the rest (torso/arm sites, and hand-part-of-contact categories like `FingerTip`/`Palm`/`WristBack`)
  are `NOT_COMPUTABLE`.
- **SecondMinorLocation** (added 2026-09-28, was previously 0% coverage): ASL-LEX's own value set
  here is dominated by `Neutral`/`HeadAway`/`HandAway`/`BodyAway` (80% of the non-null ground truth
  among the 233 mapped glosses) -- not a second face site, but whether the hand moves *away* from
  wherever `MajorLocation` found contact by the end of the sign. Implemented as a path-departure
  check: find the hand's closest approach to that site, then see if it has moved more than
  ``away_t`` (x eye distance) away from that point by the sign's last tracked frame. `Neutral` when
  `MajorLocation` is `Neutral`/`Arm` (nothing to leave); `NOT_COMPUTABLE` when `MajorLocation` is
  `Body` without pose, or the departure can't be measured (too few tracked frames). Finer sub-site
  categories (`TorsoMid`, `ElbowBack`, `Other`, ...) are not attempted -- still `NOT_COMPUTABLE`.

**With ``use_pose=True``** (hands and face stay the primary signal for every hand-internal parameter
-- pose only fills the specific gaps above, per the user's "prioritize face and hands" instruction):
anchor becomes mid-shoulder and scale becomes shoulder width (matching `phonology.py`); `major_location`
additionally attempts `Body` (torso proximity) and `Arm` (ipsilateral upper-arm segment proximity); the
dominant hand's own trajectory is still used for `movement`/`repeated_movement` unless it is absent for
more than half the nucleus, in which case the pose wrist's trajectory substitutes (`phonology.py`'s
stated reason for preferring pose there). `minor_location`/`second_minor_location`'s torso/arm sub-sites
and any hand-part-of-contact category are still `NOT_COMPUTABLE` either way -- out of scope for this
pass, not claimed to need pose.

Every threshold below is a default; the notebook overrides them from
`experiments/recognition2/configs/stage1-rules.json` via :func:`load_thresholds`.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from sb.recognize.phonology import (
    FACE,
    LH0,
    P0,
    POSE,
    RH0,
    _angle,
    _handshape,
    _min_pairwise,
    _orientation,
    _path_reversals,
    _smooth,
    _turning,
    _unit,
    dominance,
    reversals,
)
from sb.recognize.phonology import load as load_clip  # noqa: F401  (re-exported for notebooks)

NOT_COMPUTABLE = "NOT_COMPUTABLE"  # this module cannot see the needed geometry without pose

PARAMETERS = (
    "sign_type", "handshape", "marked_handshape", "selected_fingers", "flexion", "flexion_change",
    "spread", "spread_change", "thumb_position", "thumb_contact", "ulnar_rotation", "movement",
    "repeated_movement", "major_location", "minor_location", "second_minor_location", "contact",
    "non_dominant_handshape",
)

# ASL-LEX face-adjacent MinorLocation/SecondMinorLocation categories this module can reach without
# pose (checked against signdata.csv's actual value sets, this conversation, 2026-09-27). Everything
# else (torso/arm sites, and hand-part contact categories like FingerTip/Palm/WristBack) is
# NOT_COMPUTABLE -- reported, not guessed.
MINOR_LOCATION_MAP = {
    "forehead": "Forehead", "eyes": "Eye", "nose": "CheekNose", "cheek": "CheekNose",
    "mouth": "Mouth", "chin": "Chin",
}
HEAD_SITES = tuple(MINOR_LOCATION_MAP)


@dataclass(frozen=True)
class Thresholds:
    ext_t: float = 0.90            # finger extension ratio: extended
    thumb_open_t: float = 0.60     # thumb tip to index MCP / palm: thumb open
    aperture_t: float = 0.30       # thumb tip to a fingertip / palm: thumb contact
    spread_t: float = 20.0         # mean adjacent-finger angle (deg): spread
    head_t: float = 0.35           # closest hand point within this of a head site (x eye distance)
    hand_t: float = 0.9            # palm centre within this of the other hand (x eye distance)
    body_t: float = 0.5            # closest hand point within this of the torso centre (x eye distance, use_pose only)
    arm_t: float = 0.35            # closest hand point within this of the ipsi upper-arm segment (x eye distance, use_pose only)
    hand_absent_frac: float = 0.5  # movement falls back to the pose wrist above this share of nucleus frames missing (use_pose only)
    contact_t: float = 0.09        # hand point within this of the face/other hand (x eye distance)
    still_t: float = 0.40          # path length below this: no path movement (x eye distance)
    amp_t: float = 0.04            # a direction reversal must travel at least this (x eye distance)
    apt_amp_t: float = 0.15        # an aperture (finger-opening) reversal must swing at least this (palm units)
    flex_change_t: float = 40.0    # onset-to-final change of mean non-base flexion (deg)
    apert_change_t: float = 0.50   # onset-to-final change of mean aperture (palm units)
    ori_change_t: float = 60.0     # onset-to-final palm-normal rotation (deg)
    spread_change_t: float = 12.0  # onset-to-final change of mean adjacent-finger angle (deg)
    fully_open_t: float = 20.0     # non-base flexion below this: fingers straight
    flat_base_t: float = 25.0      # base flexion at/above this with straight non-base: "Flat"
    curved_t: float = 90.0         # non-base flexion at/above this: "Curved"
    closed_t: float = 150.0        # non-base flexion at/above this (or low extension): "FullyClosed"
    nucleus_trim: float = 0.2      # share of a hand's own detected span dropped at each end
    away_t: float = 0.5            # hand's displacement from its closest approach to a site, by the
                                    # sign's end, needed to call SecondMinorLocation "...Away" (x eye distance)


def load_thresholds(path: str | Path | None = None) -> Thresholds:
    if path is None:
        return Thresholds()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(Thresholds)}
    return Thresholds(**{k: v for k, v in raw.items() if k in known})


# ---------------------------------------------------------------------------
# ASL-LEX-derived lookup tables (data-driven, not per-test-gloss -- fixed mappings true across the
# whole lexicon, the same way a linguist's handshape chart would be, not read off the glosses this
# module is scored against)
# ---------------------------------------------------------------------------

def _norm_cell(v) -> str | None:
    """One ASL-LEX cell -> a canonical string: pandas gives string columns (SelectedFingers,
    Flexion, ThumbPosition) as ``str`` already, but binary-flag columns (Spread, ThumbContact,
    MarkedHandshape, ...) come back as ``float64``/``int64`` depending on the column (checked this
    conversation, 2026-09-27) -- normalized here to ``"0"``/``"1"`` either way, so a lookup key never
    has to care which. ``NaN`` -> ``None``."""
    if isinstance(v, str):
        return v
    try:
        if v != v:  # NaN
            return None
    except TypeError:
        pass
    return str(int(v))


def build_handshape_lookup() -> tuple[dict[tuple, str], dict[str, bool]]:
    """``{(selected_fingers, flexion, spread, thumb_position, thumb_contact): modal Handshape}`` and
    ``{Handshape: is_marked}``, built once from the *entire* ASL-LEX lexicon (not just the 233 mapped
    glosses) -- Handshape is defined by signdataKEY.csv as exactly this combination, so this is a
    lookup of ASL-LEX's own coding scheme, not a fit to the test set."""
    from collections import Counter

    from sb.recognize.aslex import signdata

    d = signdata()
    combo_votes: dict[tuple, Counter] = {}
    marked: dict[str, set[str | None]] = {}
    for _, row in d.iterrows():
        hs = _norm_cell(row.get("Handshape.2.0"))
        if hs is None:
            continue
        key = tuple(_norm_cell(row.get(c)) for c in
                    ("SelectedFingers.2.0", "Flexion.2.0", "Spread.2.0", "ThumbPosition.2.0", "ThumbContact.2.0"))
        if any(v is None for v in key):
            continue
        combo_votes.setdefault(key, Counter())[hs] += 1
        marked.setdefault(hs, set()).add(_norm_cell(row.get("MarkedHandshape.2.0")))
    lookup = {k: v.most_common(1)[0][0] for k, v in combo_votes.items()}
    is_marked = {hs: vals != {"0"} for hs, vals in marked.items()}
    return lookup, is_marked


# ---------------------------------------------------------------------------
# geometry (pose-free)
# ---------------------------------------------------------------------------

def _face_anchor(face: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(T, 468, 3)`` face mesh -> ``(anchor xy (T, 2), eye_distance (T,))``. Anchor: mean of both
    eye-outer corners and both mouth corners (robust to a hand briefly covering the chin/mouth on
    one side); scale: inter-eye distance, the same unit used for `phonology.py`'s non-manual
    features."""
    pts = face[:, [FACE["eye_r_out"], FACE["eye_l_out"], FACE["mouth_r"], FACE["mouth_l"]], :2]
    anchor = np.nanmean(pts, axis=1)
    bad = np.isnan(anchor).any(axis=1)
    if bad.any() and (~bad).any():
        anchor[bad] = np.nanmedian(anchor[~bad], axis=0)
    eye_d = np.linalg.norm(face[:, FACE["eye_r_out"], :2] - face[:, FACE["eye_l_out"], :2], axis=-1)
    eye_d = np.where(np.isnan(eye_d) | (eye_d < 1e-6), np.nanmedian(eye_d[eye_d > 1e-6]) if (eye_d > 1e-6).any() else 1.0, eye_d)
    return anchor, eye_d.astype(np.float32)


def _pose_anchor(clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(T, 543, 3)`` -> ``(mid-shoulder anchor xy (T, 2), shoulder width (T,))``, matching
    `phonology.py`'s anchor exactly (median substitution where shoulders are undetected)."""
    sh = clip[:, [P0 + POSE["sh_l"], P0 + POSE["sh_r"]], :2]
    has_sh = ~np.isnan(sh).any(axis=(1, 2))
    anchor = np.nanmean(sh, axis=1)
    if has_sh.any() and (~has_sh).any():
        anchor[~has_sh] = np.nanmedian(anchor[has_sh], axis=0)
    width = np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1)
    width = np.where(np.isnan(width) | (width < 1e-6),
                      np.nanmedian(width[width > 1e-6]) if (width > 1e-6).any() else 1.0, width)
    return anchor.astype(np.float32), width.astype(np.float32)


def extract(clip: np.ndarray, use_pose: bool = False) -> dict:
    """``(T, 543, 3)`` -> a dict of per-frame arrays for both hands. ``use_pose`` (added 2026-09-27,
    the "does pose add a significant benefit" ablation): mid-shoulder anchor + shoulder-width scale
    and the extra `Body`/`Arm`/pose-wrist-fallback fields described in the module docstring; hands
    and face stay the primary signal either way. Wraps :func:`_extract` to silence the all-NaN-slice
    warnings a hand-absent frame triggers (same pattern as `phonology.extract`)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _extract(clip, use_pose)


def _extract(clip: np.ndarray, use_pose: bool = False) -> dict:
    """See :func:`extract`. Keyed the same way for ``h1``/``h2`` (dominant/non-dominant, mirrored so
    +x is always the dominant side)."""
    face = clip[:, :468].astype(np.float32)
    if use_pose:
        anchor, scale = _pose_anchor(clip)
    else:
        anchor, scale = _face_anchor(face)
    c = clip.astype(np.float32).copy()
    c[..., :2] -= anchor[:, None, :]
    dom = dominance(clip)
    right = dom == "right"
    if right:
        c[..., 0] *= -1
    h1 = c[:, RH0:RH0 + 21] if right else c[:, LH0:LH0 + 21]
    h2 = c[:, LH0:LH0 + 21] if right else c[:, RH0:RH0 + 21]
    face_c = c[:, :468]
    sites = {k: face_c[:, i] for k, i in FACE.items() if k in ("forehead", "nose", "upper_lip",
                                                                "lower_lip", "chin", "eye_r_out",
                                                                "eye_l_out", "cheek_r", "cheek_l")}
    eyes_mid = (sites["eye_r_out"] + sites["eye_l_out"]) / 2
    mouth_mid = (sites["upper_lip"] + sites["lower_lip"]) / 2
    T = len(c)
    # eye_d is inter-point distance, unaffected by which anchor was subtracted or by mirroring, so
    # it is always the *true* inter-eye distance -- kept fixed across both variants (not `scale`,
    # which is shoulder-width under use_pose) so every threshold means the same physical distance in
    # both, and a with_pose vs no_pose delta reflects the pose ablation, not a silently rescaled
    # threshold (found 2026-09-27: shoulder width is several times inter-eye distance, so reusing
    # `scale` as the threshold unit made every distance-based rule less sensitive under use_pose).
    eye_d = np.linalg.norm(sites["eye_r_out"][:, :2] - sites["eye_l_out"][:, :2], axis=-1)
    eye_d = np.where(np.isnan(eye_d) | (eye_d < 1e-6),
                      np.nanmedian(eye_d[eye_d > 1e-6]) if (eye_d > 1e-6).any() else 1.0, eye_d).astype(np.float32)
    out: dict = {"dominant": dom, "eye_d": eye_d, "scale": scale, "use_pose": use_pose, "frames": T}
    pose_c = c[:, P0:P0 + 33] if use_pose else None
    side = ({"ipsi": "r", "contra": "l"} if right else {"ipsi": "l", "contra": "r"}) if use_pose else None
    torso_centre = None
    if use_pose:
        assert pose_c is not None
        torso_centre = (pose_c[:, POSE["sh_l"]] + pose_c[:, POSE["sh_r"]]
                         + pose_c[:, POSE["hip_l"]] + pose_c[:, POSE["hip_r"]]) / 4
    for tag, h, other in (("h1", h1, h2), ("h2", h2, h1)):
        present = ~np.isnan(h[:, 0, 0])
        centre = h[:, [0, 5, 9, 13, 17]].mean(axis=1)
        oc = other[:, [0, 5, 9, 13, 17]].mean(axis=1)
        hs = _handshape(h) if present.any() else np.full((T, 25), np.nan, np.float32)
        normal_dir = _orientation(h, flip=(tag == "h2"))
        head = {"forehead": sites["forehead"], "eyes": eyes_mid, "nose": sites["nose"],
                "mouth": mouth_mid, "chin": sites["chin"],
                "cheek": (sites["cheek_r"] + sites["cheek_l"]) / 2}
        dist_head = {s: _min_pairwise(h, head[s][:, None]) for s in HEAD_SITES}
        vel = np.vstack([np.full((1, 2), np.nan, np.float32), np.diff(centre[:, :2], axis=0)])
        acc = np.vstack([np.full((1, 2), np.nan, np.float32), np.diff(vel, axis=0)])
        # _handshape's 25 columns, in order: ext(thumb,i,m,r,p)=0-4, base=5-9, nonbase=10-14,
        # spread(im,mr,rp,thumb_i)=15-18, aperture(i,m,r,p)=19-22, thumb(to_index,to_pinky)=23-24.
        # base/nonbase/aperture below exclude the thumb column, matching FINGERS[1:] in phonology.py.
        apert = np.nanmean(hs[:, 19:23], axis=1) if present.any() else np.full(T, np.nan)
        base = np.nanmean(hs[:, 6:10], axis=1) if present.any() else np.full(T, np.nan)
        nonbase = np.nanmean(hs[:, 11:15], axis=1) if present.any() else np.full(T, np.nan)
        spread = np.nanmean(hs[:, 15:19], axis=1) if present.any() else np.full(T, np.nan)
        rot = np.concatenate([[np.nan], _angle(normal_dir[1:, :3], normal_dir[:-1, :3])])
        entry = {
            "present": present, "centre": centre[:, :2], "hs": hs, "normal": normal_dir[:, :3],
            "finger_dir": normal_dir[:, 3:], "dist_head": dist_head,
            "touch_face": _min_pairwise(h, face_c),
            "touch_other": _min_pairwise(h, other),
            "vel": vel, "speed": np.linalg.norm(vel, axis=1), "accel": np.linalg.norm(acc, axis=1),
            "aperture": apert, "base_flex": base, "nonbase_flex": nonbase, "spread": spread,
            "thumb_open": hs[:, 23] if hs.shape[1] > 23 else np.full(T, np.nan),
            "thumb_contact": np.nanmin(hs[:, 19:23], axis=1) if present.any() else np.full(T, np.nan),
            "d_aperture": np.concatenate([[np.nan], np.diff(apert)]),
            "d_flex": np.concatenate([[np.nan], np.diff(nonbase)]),
            "d_orient": rot, "centre_other": oc[:, :2],
        }
        if use_pose:
            assert pose_c is not None and side is not None and torso_centre is not None
            # ASL-LEX's "Arm" location means this hand touching the *other* (passive) arm -- the
            # contralateral elbow, never this hand's own (ipsilateral) side: a hand's own wrist
            # landmark nearly coincides with its own pose wrist by anatomy (same joint), so using
            # ipsi elbow/wrist made "Arm" fire on almost every clip (729/1414 predicted vs ~24 true,
            # found 2026-09-27). Elbow only, not wrist, for the same reason at one remove.
            arm_side = side["contra"] if tag == "h1" else side["ipsi"]
            own_side = side["ipsi"] if tag == "h1" else side["contra"]
            elbow = pose_c[:, POSE[f"el_{arm_side}"]]
            own_wrist = pose_c[:, POSE[f"wr_{own_side}"]]  # this hand's own wrist -- for the movement fallback
            entry["dist_torso"] = _min_pairwise(h, torso_centre[:, None])
            entry["dist_arm"] = _min_pairwise(h, elbow[:, None])
            entry["pose_wrist"] = own_wrist[:, :2]
        out[tag] = entry
    return out


# ---------------------------------------------------------------------------
# per-parameter rules, targeting ASL-LEX's own category strings
# ---------------------------------------------------------------------------

def _med(a: np.ndarray) -> float:
    with np.errstate(all="ignore"):
        return float(np.nanmedian(a)) if len(a) else float("nan")


def _away(dist: np.ndarray, centre: np.ndarray, thresh: float) -> bool | None:
    """For `second_minor_location`: does the hand end up more than ``thresh`` from the point where
    it was closest to ``dist``'s site, by the last tracked frame? ``None`` if there isn't enough
    valid data to judge (never guessed as True/False)."""
    valid = ~np.isnan(dist)
    if not valid.any():
        return None
    close_pt = centre[int(np.nanargmin(dist))]
    tracked = np.flatnonzero(~np.isnan(centre[:, 0]))
    if len(tracked) == 0 or np.isnan(close_pt).any():
        return None
    return float(np.linalg.norm(centre[tracked[-1]] - close_pt)) > thresh


def _selected_fingers(medial: dict, t: Thresholds) -> str:
    ext = medial["hs"][:, 0:5]  # thumb, index, middle, ring, pinky extension (see _handshape order)
    ext_med = np.nanmedian(ext, axis=0)
    sel = "".join(ch for ch, e in zip("imrp", ext_med[1:]) if e > t.ext_t)
    thumb_open = _med(medial["thumb_open"]) > t.thumb_open_t if not np.isnan(_med(medial["thumb_open"])) else False
    if sel:
        return sel
    return "t" if thumb_open else "closed"


def _flexion(medial: dict, t: Thresholds) -> str:
    base, nonbase = _med(medial["base_flex"]), _med(medial["nonbase_flex"])
    if np.isnan(base) or np.isnan(nonbase):
        return NOT_COMPUTABLE
    if nonbase >= t.closed_t:
        return "FullyClosed"
    if nonbase >= t.curved_t:
        return "Curved"
    if nonbase >= t.fully_open_t:
        return "Bent" if base >= t.flat_base_t else "Curved"
    return "Flat" if base >= t.flat_base_t else "FullyOpen"
    # "Stacked" (layered finger curl) needs finger-to-finger occlusion this rule set can't see from
    # 21 sparse points -- never predicted, a documented limitation (report §, TODO §14).


def _sign_type(h1: dict, h2: dict, unmarked: set[str], h1_hs: str, h2_hs: str) -> str:
    if not h2["present"].any() or np.mean(h2["present"]) < 0.05:
        return "OneHanded"
    same_shape = h1_hs == h2_hs and h1_hs != NOT_COMPUTABLE
    v1, v2 = h1["vel"], h2["vel"] * np.array([-1, 1], np.float32)
    both_move = np.nanmean(np.linalg.norm(v1, axis=1)) > 0 and np.nanmean(np.linalg.norm(v2, axis=1)) > 0
    sym = float(np.nanmean((_unit(v1) * _unit(v2)).sum(-1))) if both_move else 1.0
    if same_shape:
        return "SymmetricalOrAlternating" if not np.isnan(sym) and sym > -0.3 else "AsymmetricalSameHandshape"
    if h2_hs in unmarked or h2_hs == NOT_COMPUTABLE:
        return "AsymmetricalDifferentHandshape"
    return "DominanceViolation"


def codes(feats: dict, t: Thresholds, lookup: dict[tuple, str], is_marked: dict[str, bool]) -> dict[str, str]:
    """Pose-free discrete ASL-LEX-style codes for one clip, all 18 :data:`PARAMETERS`. Unreachable
    categories are :data:`NOT_COMPUTABLE`, never guessed. Wraps :func:`_codes` to silence
    all-NaN-slice warnings."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _codes(feats, t, lookup, is_marked)


def _codes(feats: dict, t: Thresholds, lookup: dict[tuple, str], is_marked: dict[str, bool]) -> dict[str, str]:
    """See :func:`codes`."""
    h1, h2, eye_d = feats["h1"], feats["h2"], feats["eye_d"]
    unit = float(np.nanmedian(eye_d)) if np.isfinite(np.nanmedian(eye_d)) else 1.0
    present_idx = np.flatnonzero(h1["present"]) if h1["present"].any() else np.zeros(0, int)
    out = dict.fromkeys(PARAMETERS, NOT_COMPUTABLE)
    if len(present_idx) == 0:
        return out
    a, b = int(present_idx[0]), int(present_idx[-1]) + 1
    cut = int(round((b - a) * t.nucleus_trim))
    nucleus = slice(a + cut, max(b - cut, a + cut + 1))

    def seg(d: dict, key: str, sl):
        return d[key][sl]

    def own_nucleus(present: np.ndarray) -> slice | None:
        """Same trim-the-ends logic as `nucleus`, but on a hand's *own* detected span -- added
        2026-09-28 for `non_dominant_handshape`: reusing `nucleus` (h1's span) for hand 2 meant its
        features were sliced at whatever window h1 happened to be active in, which is frequently a
        window where h2 itself is absent even on two-handed signs where h2 IS tracked elsewhere in
        the clip (found analyzing the near-zero non_dominant_handshape coverage, 2026-09-28)."""
        idx = np.flatnonzero(present)
        if len(idx) == 0:
            return None
        pa, pb = int(idx[0]), int(idx[-1]) + 1
        pcut = int(round((pb - pa) * t.nucleus_trim))
        return slice(pa + pcut, max(pb - pcut, pa + pcut + 1))

    def medial_of(d: dict, nuc: slice | None) -> dict:
        keys = ("hs", "thumb_open", "base_flex", "nonbase_flex", "spread", "aperture",
                "thumb_contact", "normal", "finger_dir")
        if nuc is None:
            out: dict = {k: np.full((0, *np.shape(d[k])[1:]), np.nan, np.float32) for k in keys}
            out["dist_head"] = {s: np.full(0, np.nan, np.float32) for s in d["dist_head"]}
            return out
        n = nuc.stop - nuc.start
        third = slice(nuc.start + n // 3, nuc.stop - n // 3) if n >= 3 else nuc
        out2: dict = {k: d[k][third] for k in keys}
        out2["dist_head"] = {s: v[third] for s, v in d["dist_head"].items()}
        return out2

    def _bin(value: bool | None) -> str:
        """``value`` -> a canonical ``"0"``/``"1"``, matching :func:`_norm_cell`'s normalization of
        ASL-LEX's own binary-flag columns (some are ``float64``, some ``int64`` -- never ``str`` --
        checked against signdata.csv, this conversation, 2026-09-27), so a rule's output compares
        directly to ground truth with no per-parameter format lookup."""
        return NOT_COMPUTABLE if value is None else str(int(value))

    has_h2 = h2["present"].any() and np.mean(h2["present"]) >= 0.05  # matches _sign_type's own threshold
    m1 = medial_of(h1, nucleus)
    m2 = medial_of(h2, own_nucleus(h2["present"]) if has_h2 else None)
    sel1 = _selected_fingers(m1, t)
    flex1 = _flexion(m1, t)
    spread1_v = None if np.isnan(_med(m1["spread"])) else _med(m1["spread"]) > t.spread_t
    thumb_open1 = None if np.isnan(_med(m1["thumb_open"])) else _med(m1["thumb_open"]) > t.thumb_open_t
    thumb_pos1 = NOT_COMPUTABLE if thumb_open1 is None else ("Open" if thumb_open1 else "Closed")
    thumb_contact1_v = None if np.isnan(_med(m1["thumb_contact"])) else _med(m1["thumb_contact"]) < t.aperture_t
    key1 = (sel1, flex1, _bin(spread1_v), thumb_pos1, _bin(thumb_contact1_v))
    hs1 = lookup.get(key1, NOT_COMPUTABLE)
    out["handshape"] = hs1
    out["marked_handshape"] = _bin(is_marked.get(hs1))
    out["selected_fingers"] = sel1
    out["flexion"] = flex1
    out["spread"] = _bin(spread1_v)
    out["thumb_position"] = thumb_pos1
    out["thumb_contact"] = _bin(thumb_contact1_v)

    sel2 = _selected_fingers(m2, t) if has_h2 else NOT_COMPUTABLE
    flex2 = _flexion(m2, t) if has_h2 else NOT_COMPUTABLE
    spread2_v = None if not has_h2 or np.isnan(_med(m2["spread"])) else _med(m2["spread"]) > t.spread_t
    thumb_open2 = None if not has_h2 or np.isnan(_med(m2["thumb_open"])) else _med(m2["thumb_open"]) > t.thumb_open_t
    thumb_pos2 = NOT_COMPUTABLE if thumb_open2 is None else ("Open" if thumb_open2 else "Closed")
    thumb_contact2_v = None if not has_h2 or np.isnan(_med(m2["thumb_contact"])) else _med(m2["thumb_contact"]) < t.aperture_t
    key2 = (sel2, flex2, _bin(spread2_v), thumb_pos2, _bin(thumb_contact2_v))
    hs2 = lookup.get(key2, NOT_COMPUTABLE)
    out["non_dominant_handshape"] = hs2 if has_h2 else NOT_COMPUTABLE

    n0, n1 = m1["normal"][:1], m1["normal"][-1:]
    if len(m1["normal"]) >= 2 and not np.isnan(n0).all() and not np.isnan(n1).all():
        out["ulnar_rotation"] = str(int(float(_angle(np.nanmedian(m1["normal"][:max(1, len(m1['normal'])//3)], axis=0),
                                                       np.nanmedian(m1["normal"][-max(1, len(m1['normal'])//3):], axis=0))) > t.ori_change_t))

    touch = np.r_[h1["touch_face"][nucleus], h1["touch_other"][nucleus]]
    if not np.isnan(touch).all():
        out["contact"] = str(int(np.nanmin(touch) < t.contact_t * unit))

    # min, not median: a location touch (own hand or the other hand) can be brief within the
    # nucleus -- a median washes it out almost entirely (found 2026-09-27 analyzing a full run:
    # "Hand" predicted 5/1308 times against 108 true cases, once this used the same _med as the
    # rest of this function; "Head" recall was also depressed the same way).
    head = {s: np.nanmin(m1["dist_head"][s]) if not np.all(np.isnan(m1["dist_head"][s])) else np.nan
            for s in HEAD_SITES}
    head = {s: d for s, d in head.items() if not np.isnan(d)}
    other_dist = np.nanmin(h1["touch_other"][nucleus]) if not np.all(np.isnan(h1["touch_other"][nucleus])) else np.nan
    if not np.isnan(other_dist) and other_dist < t.hand_t * unit:
        out["major_location"] = "Hand"
    elif head and min(head.values()) < t.head_t * unit:
        out["major_location"] = "Head"
        out["minor_location"] = MINOR_LOCATION_MAP[min(head, key=lambda s: head[s])]
    elif "dist_torso" in h1 and not np.all(np.isnan(h1["dist_torso"][nucleus])) and \
            np.nanmin(h1["dist_torso"][nucleus]) < t.body_t * unit:
        out["major_location"] = "Body"  # use_pose only; minor sub-site (Clavicle/Hips/... ) not attempted
    elif "dist_arm" in h1 and not np.all(np.isnan(h1["dist_arm"][nucleus])) and \
            np.nanmin(h1["dist_arm"][nucleus]) < t.arm_t * unit:
        out["major_location"] = "Arm"  # use_pose only; minor sub-site (UpperArm/ForearmBack/...) not attempted
    else:
        out["major_location"] = "Neutral"

    # SecondMinorLocation (added 2026-09-28): ASL-LEX's own value set here is mostly "does the hand
    # move away from wherever MajorLocation found contact, by the sign's end" -- see the module
    # docstring. Reuses the site MajorLocation already picked; Body needs pose (matching
    # MajorLocation's own gating), Neutral/Arm have no site to leave.
    if out["major_location"] == "Hand":
        away = _away(h1["touch_other"][a:b], h1["centre"][a:b], t.away_t * unit)
        out["second_minor_location"] = NOT_COMPUTABLE if away is None else ("HandAway" if away else "Neutral")
    elif out["major_location"] == "Head":
        site = min(head, key=lambda s: head[s])
        away = _away(h1["dist_head"][site][a:b], h1["centre"][a:b], t.away_t * unit)
        out["second_minor_location"] = NOT_COMPUTABLE if away is None else ("HeadAway" if away else "Neutral")
    elif out["major_location"] == "Body" and "dist_torso" in h1:
        away = _away(h1["dist_torso"][a:b], h1["centre"][a:b], t.away_t * unit)
        out["second_minor_location"] = NOT_COMPUTABLE if away is None else ("BodyAway" if away else "Neutral")
    elif out["major_location"] in ("Neutral", "Arm"):
        out["second_minor_location"] = "Neutral"

    # movement: the hand's own trajectory, unless it is absent for more than `hand_absent_frac` of
    # the nucleus and a pose wrist is available to fall back to (use_pose only) -- phonology.py's
    # stated reason for preferring the pose wrist there.
    pos_src = h1
    if "pose_wrist" in h1:
        cov = np.mean(~np.isnan(h1["centre"][nucleus, 0]))
        if cov < (1.0 - t.hand_absent_frac) and not np.all(np.isnan(h1["pose_wrist"][nucleus, 0])):
            pos_src = {"centre": h1["pose_wrist"]}
    pos = _smooth(seg(pos_src, "centre", nucleus))
    length = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) if len(pos) > 1 else 0.0
    disp = pos[-1] - pos[0] if len(pos) > 1 else np.zeros(2)
    rev = _path_reversals(pos, t.amp_t * unit) if len(pos) >= 4 else 0
    if length < t.still_t * unit:
        out["movement"] = "Straight"  # effectively no path movement; ASL-LEX has no "none" bucket
    else:
        turning = _turning(pos, 0.02 * unit)
        straight = float(np.linalg.norm(disp)) / max(length, 1e-9)
        out["movement"] = ("Circular" if turning > 300 and straight < 0.4
                            else "BackAndForth" if rev >= 2
                            else "Curved" if straight < 0.75 or turning > 90
                            else "Straight")
    apt = seg(h1, "aperture", slice(a, b))
    apt = apt[~np.isnan(apt)]
    out["repeated_movement"] = str(int(rev >= 2 or (len(apt) >= 4 and reversals(apt, t.apt_amp_t) >= 2)))

    fl0 = _med(seg(h1, "nonbase_flex", slice(a, a + max(1, (b - a) // 3))))
    fl1 = _med(seg(h1, "nonbase_flex", slice(b - max(1, (b - a) // 3), b)))
    if not (np.isnan(fl0) or np.isnan(fl1)):
        out["flexion_change"] = _bin(abs(fl1 - fl0) > t.flex_change_t)
    sp0 = _med(seg(h1, "spread", slice(a, a + max(1, (b - a) // 3))))
    sp1 = _med(seg(h1, "spread", slice(b - max(1, (b - a) // 3), b)))
    if not (np.isnan(sp0) or np.isnan(sp1)):
        out["spread_change"] = _bin(abs(sp1 - sp0) > t.spread_change_t)

    lookup_unmarked = {hs for hs, marked in is_marked.items() if not marked}
    out["sign_type"] = _sign_type(h1, h2, lookup_unmarked, hs1, hs2)
    return out


# ---------------------------------------------------------------------------
# continuous (pre-threshold) features, for training a classifier per parameter instead of
# hand-picking a threshold (added 2026-09-28, TODO §14, user ask: "train a model ... from these
# features"). Mirrors the scalars `_codes` computes internally, but returns the float, never the
# category -- e.g. `spread_angle` (degrees) instead of `spread` ("0"/"1"). NaN wherever `_codes`
# would have returned NOT_COMPUTABLE. A separate, additive function -- does not touch `_codes`.
# ---------------------------------------------------------------------------

CONTINUOUS_FEATURES = (
    "ext_thumb", "ext_index", "ext_middle", "ext_ring", "ext_pinky",
    "base_flex", "nonbase_flex", "spread_angle", "thumb_open", "thumb_contact_dist",
    "ulnar_rotation_deg", "movement_length", "movement_straightness", "movement_turning",
    "movement_reversals", "aperture_reversals", "flexion_change_deg", "spread_change_deg",
    "contact_min_dist", "hand_min_dist", "head_min_dist", "has_h2", "sign_type_sym",
)


def continuous_features(feats: dict, t: Thresholds) -> dict[str, float]:
    """Pre-threshold continuous measurements for the dominant hand (one float per :data:`_codes`
    decision), for training a classifier per ASL-LEX parameter directly on the measurement instead
    of the rule engine's binarized code. Wraps :func:`_continuous_features` to silence all-NaN-slice
    warnings, matching :func:`codes`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _continuous_features(feats, t)


def _continuous_features(feats: dict, t: Thresholds) -> dict[str, float]:
    """See :func:`continuous_features`. Units match the thresholds each value would otherwise be
    compared against: angles in degrees, distances as multiples of ``eye_d`` (or shoulder width
    under ``use_pose``), ratios dimensionless (fraction of palm length, from `_handshape`)."""
    h1, h2, eye_d = feats["h1"], feats["h2"], feats["eye_d"]
    unit = float(np.nanmedian(eye_d)) if np.isfinite(np.nanmedian(eye_d)) else 1.0
    out: dict[str, float] = dict.fromkeys(CONTINUOUS_FEATURES, float("nan"))
    present_idx = np.flatnonzero(h1["present"]) if h1["present"].any() else np.zeros(0, int)
    out["has_h2"] = float(h2["present"].any() and np.mean(h2["present"]) >= 0.05)
    if len(present_idx) == 0:
        return out
    a, b = int(present_idx[0]), int(present_idx[-1]) + 1
    cut = int(round((b - a) * t.nucleus_trim))
    nucleus = slice(a + cut, max(b - cut, a + cut + 1))
    n = nucleus.stop - nucleus.start
    third = slice(nucleus.start + n // 3, nucleus.stop - n // 3) if n >= 3 else nucleus

    hs = h1["hs"][third]
    if hs.shape[0] and not np.all(np.isnan(hs[:, 0:5])):
        ext = np.nanmedian(hs[:, 0:5], axis=0)
        (out["ext_thumb"], out["ext_index"], out["ext_middle"],
         out["ext_ring"], out["ext_pinky"]) = (float(v) for v in ext)
    out["base_flex"] = _med(h1["base_flex"][third])
    out["nonbase_flex"] = _med(h1["nonbase_flex"][third])
    out["spread_angle"] = _med(h1["spread"][third])
    out["thumb_open"] = _med(h1["thumb_open"][third])
    out["thumb_contact_dist"] = _med(h1["thumb_contact"][third])

    normal = h1["normal"][third]
    if len(normal) >= 2 and not np.isnan(normal[:1]).all() and not np.isnan(normal[-1:]).all():
        n0 = np.nanmedian(normal[:max(1, len(normal) // 3)], axis=0)
        n1 = np.nanmedian(normal[-max(1, len(normal) // 3):], axis=0)
        out["ulnar_rotation_deg"] = float(_angle(n0, n1))

    pos = _smooth(h1["centre"][nucleus])
    if len(pos) > 1:
        length = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
        disp = pos[-1] - pos[0]
        out["movement_length"] = length / unit
        out["movement_straightness"] = float(np.linalg.norm(disp)) / max(length, 1e-9)
        if len(pos) >= 4:
            out["movement_turning"] = _turning(pos, 0.02 * unit)
            out["movement_reversals"] = float(_path_reversals(pos, t.amp_t * unit))

    apt = h1["aperture"][a:b]
    apt = apt[~np.isnan(apt)]
    if len(apt) >= 4:
        out["aperture_reversals"] = float(reversals(apt, t.apt_amp_t))

    third_len = max(1, (b - a) // 3)
    fl0, fl1 = _med(h1["nonbase_flex"][a:a + third_len]), _med(h1["nonbase_flex"][b - third_len:b])
    if not (np.isnan(fl0) or np.isnan(fl1)):
        out["flexion_change_deg"] = fl1 - fl0
    sp0, sp1 = _med(h1["spread"][a:a + third_len]), _med(h1["spread"][b - third_len:b])
    if not (np.isnan(sp0) or np.isnan(sp1)):
        out["spread_change_deg"] = sp1 - sp0

    touch = np.r_[h1["touch_face"][nucleus], h1["touch_other"][nucleus]]
    if not np.isnan(touch).all():
        out["contact_min_dist"] = float(np.nanmin(touch)) / unit
    other_dist = h1["touch_other"][nucleus]
    if not np.all(np.isnan(other_dist)):
        out["hand_min_dist"] = float(np.nanmin(other_dist)) / unit
    head_dists = [np.nanmin(h1["dist_head"][s][third]) for s in HEAD_SITES
                  if not np.all(np.isnan(h1["dist_head"][s][third]))]
    if head_dists:
        out["head_min_dist"] = float(min(head_dists)) / unit

    if out["has_h2"]:
        v1, v2 = h1["vel"], h2["vel"] * np.array([-1, 1], np.float32)
        both_move = (np.nanmean(np.linalg.norm(v1, axis=1)) > 0
                     and np.nanmean(np.linalg.norm(v2, axis=1)) > 0)
        if both_move:
            out["sign_type_sym"] = float(np.nanmean((_unit(v1) * _unit(v2)).sum(-1)))
    return out
