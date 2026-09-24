"""HTML for notebooks: a gloss as lexicon-coloured chips, and a WLASL clip
strip (TODO §13; the team notebook's STEP 6 display, without scraping or
base64-inlined videos).

Clips are referenced by path relative to the notebook, trimmed with a media
fragment (``#t=start,end``), so Jupyter serves them from disk and nothing
heavy lands in the ``.ipynb``.
"""

from __future__ import annotations

import html
import os
from pathlib import Path

from sb.synthesize.lexicon import Clip, Lexicon, Unit

_COLOR = {"gislr": "#2e7d32", "wlasl": "#1565c0", "fingerspell": "#e65100"}


def _unit_kind(u: Unit) -> str:
    return "gislr" if u.gislr else ("wlasl" if u.wlasl else "fingerspell")


def gloss_html(units: list[Unit]) -> str:
    chips = []
    for u in units:
        k = _unit_kind(u)
        tip = f"GISLR: {u.gislr}" if u.gislr else (f"WLASL: {u.wlasl}" if u.wlasl else "no sign -> fingerspell")
        chips.append(f'<span title="{html.escape(tip)}" style="display:inline-block;margin:2px 4px;padding:3px 8px;'
                     f'border-radius:6px;border:2px solid {_COLOR[k]};font-family:monospace;font-weight:bold">'
                     f"{html.escape(u.gloss)}</span>")
    legend = " ".join(f'<span style="color:{c}">&#9632; {k}</span>' for k, c in _COLOR.items())
    return f'<div>{"".join(chips)}</div><div style="font-size:11px;margin-top:4px">{legend} (GISLR = our recognizer knows it)</div>'


def _src(c: Clip, notebook_dir: Path) -> str:
    assert c.path is not None, "clip has no file on disk"
    rel = os.path.relpath(c.path, notebook_dir).replace(os.sep, "/")
    frag = f"#t={c.start_s:.2f}" + (f",{c.end_s:.2f}" if c.end_s else "")
    return html.escape(rel + frag)


def clips_html(playlist: list[tuple[Unit, list[Clip]]], notebook_dir: Path) -> str:
    cards = []
    for u, clips in playlist:
        clips = [c for c in clips if c.path is not None]
        if not clips:
            cards.append(f'<div style="display:inline-block;margin:4px;text-align:center;font-family:monospace">'
                         f'<div style="width:140px;height:110px;border:1px dashed #999;line-height:110px">'
                         f"{html.escape(u.gloss)}</div><small>no clip on disk</small></div>")
            continue
        for c in clips:
            cards.append(f'<div style="display:inline-block;margin:4px;text-align:center;font-family:monospace">'
                         f'<video src="{_src(c, notebook_dir)}" width="140" autoplay muted loop playsinline></video>'
                         f"<br><b>{html.escape(c.gloss)}</b> <small>{'spell' if u.kind == 'fingerspell' else 'sign'}"
                         f" · signer {c.signer_id}</small></div>")
    return "<div>" + "".join(cards) + "</div>"


def lexicon_view(lex: Lexicon, gloss: str, notebook_dir: Path, clips: bool = True) -> str:
    out = gloss_html(lex.segment(gloss))
    if clips and lex.videos_dir.is_dir() and any(lex.videos_dir.glob("*.mp4")):
        out += clips_html(lex.playlist(gloss), notebook_dir)
    return out
