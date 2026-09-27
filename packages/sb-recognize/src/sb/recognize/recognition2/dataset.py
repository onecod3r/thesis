"""Recognition v2 (TODO §14): an ASL-LEX-restricted variant of `GISLR-Sentences`.

Reuses, read-only: `sb.recognize.sequences.corpus` (the committed sentence corpus) and
`sb.recognize.sequences.compose` (the landmark-space composer GISLR-Sentences v1 already uses).
Neither is edited -- this module only filters the corpus's input and the clip pool before handing
both to the existing composer, so the resulting dataset is built in landmark space from the start
(the P1 continuous model's feature-space composer mismatch, `phonology-models.md` §9, never applies
here since a different composer produced the mismatch, not this one).
"""

from __future__ import annotations

import pandas as pd

from sb.recognize.aslex import gloss_entries
from sb.recognize.sequences.corpus import Sentence, load_sentences


def mapped_glosses(glosses: list[str]) -> set[str]:
    """The subset of ``glosses`` that maps into ASL-LEX (`aslex.gloss_entries`, reused)."""
    return set(gloss_entries(glosses))


def filter_sentences(sentences: list[Sentence], allowed: set[str]) -> tuple[list[Sentence], list[Sentence]]:
    """``(kept, dropped)``: a sentence is kept only if every one of its glosses is in ``allowed``."""
    kept, dropped = [], []
    for s in sentences:
        (kept if set(s.glosses) <= allowed else dropped).append(s)
    return kept, dropped


def filter_clip_pool(df: pd.DataFrame, allowed: set[str]) -> pd.DataFrame:
    """``df`` (``train.csv``/``test.csv``) restricted to clips of an allowed gloss -- so the
    composer's clip pool never even offers an unmapped-gloss clip, and nothing is left as an
    unplaceable "orphan" for a reason this dataset already knows about."""
    return df[df["sign"].isin(allowed)].reset_index(drop=True)


def load_filtered_corpus(glosses: list[str], version: str = "v1") -> dict:
    """Everything a build notebook needs in one call: the ASL-LEX-mapped gloss set, the corpus split
    into kept/dropped sentences, and a coverage summary (how many sentences and how much of the
    250-gloss vocabulary survive the filter)."""
    allowed = mapped_glosses(glosses)
    sentences = load_sentences(version)
    kept, dropped = filter_sentences(sentences, allowed)
    covered = {g for s in kept for g in s.glosses}
    return {
        "allowed_glosses": allowed,
        "sentences_kept": kept,
        "sentences_dropped": dropped,
        "glosses_covered": covered,
        "glosses_lost": allowed - covered,
        "n_sentences_total": len(sentences),
        "n_sentences_kept": len(kept),
    }
