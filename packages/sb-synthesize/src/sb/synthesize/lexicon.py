"""Gloss -> sign lexicon: coverage metric and WLASL clip lookup (TODO §13).

Two lexicons:
- **GISLR** (250 signs): what our recognizer knows. A few GISLR labels are
  named differently from gloss convention (``minemy``, ``hesheit``, ``mom``,
  ``dad``, ``callonphone``); :data:`GISLR_ALIASES` bridges them (report §8).
- **WLASL v0.3** (2,000 glosses, C-UDA: academic use only): the index JSON is
  public; the videos are not bundled. Put any you have under
  ``data/raw/wlasl/videos/<video_id>.mp4`` and :meth:`Lexicon.clip` finds them.

A gloss sequence is segmented greedily, **longest match first** (WLASL has
multi-word entries such as ``NOT YET`` and ``WAKE UP``). Each unit is
``sign`` (in WLASL or GISLR) or ``fingerspell`` (in neither: names, places
and words with no lexical sign). This is the audit's coverage metric (§8)
and replaces the team's exact-match lookup. The signasl.org scraping
fallback is **not** carried over (licence and ToS, report §3).

Clips come back with their **trim window** (``frame_start``/``frame_end``,
1-indexed at 25 fps, ``-1`` = end of video), which the team notebook ignored.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sb.core.paths import RAW_DIR

WLASL_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
WLASL_DIR = RAW_DIR / "wlasl"
WLASL_JSON = WLASL_DIR / "WLASL_v0.3.json"
WLASL_VIDEOS = WLASL_DIR / "videos"

GISLR_ALIASES = {"ME": "minemy", "MY": "minemy", "MINE": "minemy", "HE": "hesheit", "SHE": "hesheit",
                 "IT": "hesheit", "HIM": "hesheit", "HER": "hesheit", "MOTHER": "mom", "FATHER": "dad",
                 "CALL": "callonphone"}
# Gloss spellings WLASL files under another label (ASL uses one pointing sign for he/she/it).
WLASL_ALIASES = {"HE": "she", "HIM": "her", "CAN'T": "cannot", "IT": "she"}
_SUFFIXES = ("EST", "ER", "ING", "ED", "S")
_NUMBER_WORDS = {"ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"}


def ensure_wlasl_index() -> Path:
    """Download the WLASL index JSON (~12 MB) once."""
    if not WLASL_JSON.is_file():
        WLASL_DIR.mkdir(parents=True, exist_ok=True)
        tmp = WLASL_JSON.with_suffix(".tmp")
        urllib.request.urlretrieve(WLASL_URL, tmp)
        tmp.replace(WLASL_JSON)
    return WLASL_JSON


def gislr_signs() -> set[str]:
    """GISLR's 250 sign labels (lowercase), from the dataset's label map."""
    from sb.core.paths import gislr_dir
    from sb.core.vocab import load_label_map

    return {s.lower() for s in load_label_map(gislr_dir())}


@dataclass
class Unit:
    gloss: str  # the (possibly multi-word) gloss unit, uppercase
    kind: str  # "sign" | "fingerspell"
    wlasl: str | None  # matched WLASL label
    gislr: str | None  # matched GISLR label


@dataclass
class Clip:
    gloss: str
    video_id: str
    path: Path | None  # None if the video isn't on disk
    start_s: float  # trim window in seconds
    end_s: float | None  # None = to the end
    signer_id: int


class Lexicon:
    def __init__(self, wlasl_json: Path | None = None, gislr: set[str] | None = None,
                 videos_dir: Path = WLASL_VIDEOS):
        data = json.loads(Path(wlasl_json or ensure_wlasl_index()).read_text(encoding="utf-8"))
        self.wlasl = {e["gloss"].lower(): e["instances"] for e in data}
        self.gislr = gislr if gislr is not None else gislr_signs()
        self.videos_dir = Path(videos_dir)
        self.max_span = max(len(g.split()) for g in self.wlasl)

    # ---- lookup ---------------------------------------------------------
    def _wlasl_label(self, g: str) -> str | None:
        low = g.lower()
        for cand in (low, WLASL_ALIASES.get(g, "").lower(), low.replace("-", " ")):
            if cand and cand in self.wlasl:
                return cand
        for suf in _SUFFIXES:  # NEAREST -> near, BOOKS -> book
            if g.endswith(suf) and len(g) - len(suf) >= 3 and g[: -len(suf)].lower() in self.wlasl:
                return g[: -len(suf)].lower()
        return None

    def _gislr_label(self, g: str) -> str | None:
        low = g.lower().replace(" ", "")
        if low in self.gislr:
            return low
        a = GISLR_ALIASES.get(g)
        return a if a in self.gislr else None

    def segment(self, gloss: str) -> list[Unit]:
        """Gloss string -> units, longest WLASL match first. ``|`` separators
        and punctuation are ignored."""
        toks = [t for t in re.split(r"\s+", gloss.upper()) if t and t != "|" and re.search(r"\w", t)]
        out: list[Unit] = []
        i = 0
        while i < len(toks):
            for n in range(min(self.max_span, len(toks) - i), 0, -1):
                g = " ".join(toks[i:i + n])
                w = self._wlasl_label(g)
                gi = self._gislr_label(g) if n == 1 else None
                if w or gi or n == 1:
                    if not (w or gi) and (toks[i].isdigit() or toks[i] in _NUMBER_WORDS):
                        w = None  # a number outside WLASL: still a sign (ASL numbers are productive)
                        out.append(Unit(g, "sign", None, None))
                    else:
                        out.append(Unit(g, "sign" if (w or gi) else "fingerspell", w, gi))
                    i += n
                    break
        return out

    def coverage(self, glosses: list[str]) -> dict:
        """Corpus coverage over units: share that are signs (any lexicon), in
        WLASL, in GISLR, and fingerspelled; plus sentences fully covered."""
        units = [self.segment(g) for g in glosses]
        flat = [u for us in units for u in us]
        n = max(len(flat), 1)
        return {"units": len(flat),
                "sign": sum(u.kind == "sign" for u in flat) / n,
                "wlasl": sum(u.wlasl is not None for u in flat) / n,
                "gislr": sum(u.gislr is not None for u in flat) / n,
                "fingerspell": sum(u.kind == "fingerspell" for u in flat) / n,
                "sentences_all_signs": sum(all(u.kind == "sign" for u in us) for us in units) / max(len(units), 1),
                "sentences_all_gislr": sum(all(u.gislr is not None for u in us) for us in units) / max(len(units), 1),
                "top_fingerspelled": Counter(u.gloss for u in flat if u.kind == "fingerspell").most_common(15)}

    # ---- clips (renderer A; rendering is deferred, TODO §13) ------------
    def clip(self, label: str, prefer_signer: int | None = None, require_file: bool = True) -> Clip | None:
        """Best WLASL instance for a label: a video on disk if ``require_file``;
        the preferred signer first (one signer per sentence reads better),
        then the lowest instance id."""
        inst = self.wlasl.get(label.lower(), [])
        rows = []
        for it in inst:
            p = self.videos_dir / f"{it['video_id']}.mp4"
            if require_file and not p.is_file():
                continue
            rows.append((it.get("signer_id") != prefer_signer, it.get("instance_id", 0), it, p if p.is_file() else None))
        if not rows:
            return None
        _, _, it, p = min(rows, key=lambda r: (r[0], r[1]))
        fps = it.get("fps", 25) or 25
        start = max(it.get("frame_start", 1) - 1, 0) / fps
        end = None if it.get("frame_end", -1) == -1 else it["frame_end"] / fps
        return Clip(label.upper(), it["video_id"], p, start, end, it.get("signer_id", -1))

    def playlist(self, gloss: str, require_file: bool = True) -> list[tuple[Unit, list[Clip]]]:
        """Units with their clips: one clip per sign; a fingerspelled unit gets
        one clip per letter (WLASL has only some letters -- missing ones are
        skipped, and the caller can show the letters as text)."""
        units = self.segment(gloss)
        signers = Counter(c.signer_id for u in units if u.wlasl
                          for c in [self.clip(u.wlasl, require_file=require_file)] if c)
        pref = signers.most_common(1)[0][0] if signers else None
        out = []
        for u in units:
            if u.wlasl:
                c = self.clip(u.wlasl, prefer_signer=pref, require_file=require_file)
                out.append((u, [c] if c else []))
            else:
                letters = [self.clip(ch, prefer_signer=pref, require_file=require_file) for ch in u.gloss if ch.isalpha()]
                out.append((u, [c for c in letters if c]))
        return out
