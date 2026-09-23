"""The sentence corpus: ASL-gloss-order sentences over the 250 GISLR glosses.

Committed and versioned beside this module (``corpus_data/``), like
``sb.rescore``'s prompts: a dataset built from ``sentences.v1`` records
:func:`corpus_sha256` so an edited corpus can never masquerade as the one a
dataset was built from. A new wording is a new version directory, never an
in-place edit of a published one.

Layout of one version::

    corpus_data/lexicon.v1.json         {gloss: [part-of-speech tags]}
    corpus_data/sentences.v1/<theme>.txt  one sentence per line, glosses separated
                                     by spaces, '#' starts a comment

Sentence ids are ``<theme>-<line index within theme>`` so adding a line to one
theme file never renumbers another theme's sentences.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus_data"

POS_TAGS = (
    "noun", "verb", "adj", "time", "wh", "pronoun", "neg", "modal",
    "prep", "conj", "quant", "greeting", "response",
)


@dataclass(frozen=True)
class Sentence:
    id: str
    theme: str
    glosses: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.glosses)


def lexicon_path(version: str = "v1") -> Path:
    return CORPUS_DIR / f"lexicon.{version}.json"


def sentences_dir(version: str = "v1") -> Path:
    return CORPUS_DIR / f"sentences.{version}"


def load_lexicon(version: str = "v1") -> dict[str, list[str]]:
    """``{gloss: [pos, ...]}`` -- a gloss can carry several tags (``drink``
    is noun and verb)."""
    lex = json.loads(lexicon_path(version).read_text(encoding="utf-8"))
    bad = {g: t for g, t in lex.items() if set(t) - set(POS_TAGS)}
    if bad:
        raise ValueError(f"unknown POS tags in lexicon.{version}: {bad}")
    return lex


def load_sentences(version: str = "v1") -> list[Sentence]:
    """Every sentence of one corpus version, themes in sorted-filename order."""
    out = []
    for path in sorted(sentences_dir(version).glob("*.txt")):
        theme = path.stem
        i = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            out.append(Sentence(f"{theme}-{i:03d}", theme, tuple(line.split())))
            i += 1
    return out


def corpus_sha256(version: str = "v1") -> str:
    """Content hash over the lexicon and every sentence file (name + bytes),
    recorded by anything built from the corpus."""
    h = hashlib.sha256()
    files = [lexicon_path(version), *sorted(sentences_dir(version).glob("*.txt"))]
    for path in files:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def validate(
    sentences: list[Sentence],
    vocab,
    *,
    min_len: int = 2,
    max_len: int = 8,
    min_per_gloss: int = 12,
) -> dict:
    """Hard failures raise; soft coverage gaps are returned for the notebook
    to print. Hard: a token outside the vocabulary, a sentence outside
    ``[min_len, max_len]``, or an exact duplicate sentence (it would silently
    double that sentence's sampling weight). Soft: glosses appearing in fewer
    than ``min_per_gloss`` sentences -- the composer can still use every clip
    of such a gloss, but only by repeating its few sentences."""
    vocab = set(vocab)
    unknown = [(s.id, g) for s in sentences for g in s.glosses if g not in vocab]
    if unknown:
        raise ValueError(f"{len(unknown)} tokens outside the vocabulary, e.g. {unknown[:10]}")
    bad_len = [s.id for s in sentences if not min_len <= len(s) <= max_len]
    if bad_len:
        raise ValueError(f"{len(bad_len)} sentences outside [{min_len}, {max_len}]: {bad_len[:10]}")
    seen: dict[tuple[str, ...], str] = {}
    dups = []
    for s in sentences:
        if s.glosses in seen:
            dups.append((seen[s.glosses], s.id))
        seen.setdefault(s.glosses, s.id)
    if dups:
        raise ValueError(f"{len(dups)} duplicate sentences: {dups[:10]}")

    # coverage = number of SENTENCES containing each gloss (not token count)
    coverage = Counter(g for s in sentences for g in set(s.glosses))
    missing = sorted(vocab - set(coverage))
    thin = sorted((g, coverage[g]) for g in vocab if 0 < coverage[g] < min_per_gloss)
    lengths = Counter(len(s) for s in sentences)
    return {
        "n_sentences": len(sentences),
        "n_themes": len({s.theme for s in sentences}),
        "coverage": dict(coverage),
        "missing": missing,
        "thin": thin,
        "length_hist": dict(sorted(lengths.items())),
        "mean_len": sum(len(s) for s in sentences) / max(len(sentences), 1),
    }
