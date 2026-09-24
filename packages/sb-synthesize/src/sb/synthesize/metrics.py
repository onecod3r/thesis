"""Text -> gloss and ASR metrics (TODO §13).

The team notebook reported METEOR, ROUGE-1/2 and MiniLM cosine only, on a
hand-set reference, and dropped BLEU. The audit (report §3) asked for the
standard set, so results compare with the text-to-gloss literature
(ASLG-PC12 papers report BLEU-4 and ROUGE-L):

- ``bleu4``: corpus BLEU-4 (sacrebleu, gloss tokens as-is: ``tokenize="none"``);
- ``chrf``: corpus chrF (sacrebleu);
- ``rouge_l``: mean ROUGE-L F1 (rouge-score);
- ``meteor``: mean METEOR (nltk; WordNet is fetched to ``data/external/nltk_data``);
- ``gloss_wer``: corpus word error rate over gloss tokens, with its S/D/I split;
- ``exact``: share of sentences whose gloss matches the reference exactly.

MiniLM cosine is not used: it is an English sentence encoder, and gloss is
not English (report §3).

Scores are 0-100 for BLEU/chrF/ROUGE-L/METEOR (the literature's scale) and
0-1 for WER/exact.
"""

from __future__ import annotations

import re
import string

import numpy as np

_PUNCT_TOK = re.compile(r"^[\W_]+$")
_ASLG_PREFIX = re.compile(r"\b(DESC|X)-")


def normalize_gloss(g: str, style: str = "plain") -> str:
    """Uppercase, drop punctuation-only tokens and the ``|`` sentence joiner.
    ``style="aslg"`` also strips ASLG-PC12's ``DESC-``/``X-`` markers
    (``X-I`` -> ``I``, ``DESC-NEW`` -> ``NEW``) so its references can be
    compared with plain-label glosses."""
    g = str(g).upper()
    if style == "aslg":
        g = _ASLG_PREFIX.sub("", g)
    return " ".join(t for t in g.split() if not _PUNCT_TOK.match(t))


def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Levenshtein alignment -> (substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    s = dl = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dl += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return int(s), int(dl), int(ins)


def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    s, d, i = edit_ops(r, hyp.split())
    return (s + d + i) / max(len(r), 1)


def _meteor_ready() -> bool:
    import nltk

    from sb.core.paths import EXTERNAL_DIR

    root = EXTERNAL_DIR / "nltk_data"
    if str(root) not in nltk.data.path:
        nltk.data.path.append(str(root))
    try:
        nltk.data.find("corpora/wordnet")
        return True
    except LookupError:
        try:
            root.mkdir(parents=True, exist_ok=True)
            return bool(nltk.download("wordnet", download_dir=str(root), quiet=True)
                        and nltk.download("omw-1.4", download_dir=str(root), quiet=True))
        except Exception:
            return False


def sentence_scores(ref: str, hyp: str) -> dict:
    """Per-sentence scores for error tables (normalized inputs expected)."""
    import sacrebleu

    s, d, i = edit_ops(ref.split(), hyp.split())
    return {"wer": (s + d + i) / max(len(ref.split()), 1), "sub": s, "del": d, "ins": i,
            "bleu": sacrebleu.sentence_bleu(hyp, [ref], tokenize="none").score,
            "chrf": sacrebleu.sentence_chrf(hyp, [ref]).score, "exact": ref == hyp}


def corpus_scores(refs: list[str], hyps: list[str], *, meteor: bool = True) -> dict:
    """Corpus-level scores; ``refs``/``hyps`` already normalized (see
    :func:`normalize_gloss`)."""
    import sacrebleu
    from rouge_score import rouge_scorer

    assert len(refs) == len(hyps)
    ops = np.array([edit_ops(r.split(), h.split()) for r, h in zip(refs, hyps)]).reshape(-1, 3)
    n_ref = sum(len(r.split()) for r in refs)
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    out = {
        "n": len(refs),
        "bleu4": sacrebleu.corpus_bleu(hyps, [refs], tokenize="none").score,
        "chrf": sacrebleu.corpus_chrf(hyps, [refs]).score,
        "rouge_l": 100 * float(np.mean([sc.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)])),
        "gloss_wer": float(ops.sum() / max(n_ref, 1)),
        "sub": float(ops[:, 0].sum() / max(n_ref, 1)),
        "del": float(ops[:, 1].sum() / max(n_ref, 1)),
        "ins": float(ops[:, 2].sum() / max(n_ref, 1)),
        "exact": float(np.mean([r == h for r, h in zip(refs, hyps)])),
    }
    if meteor and _meteor_ready():
        from nltk.translate.meteor_score import meteor_score

        out["meteor"] = 100 * float(np.mean([meteor_score([r.lower().split()], h.lower().split())
                                             for r, h in zip(refs, hyps)]))
    else:
        out["meteor"] = float("nan")
    return out


def normalize_text(t: str) -> str:
    """For ASR WER: lowercase, punctuation stripped, whitespace collapsed
    (the team notebook's ``normalize_for_wer``)."""
    t = str(t).lower().strip().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", t)


def asr_wer(refs: list[str], hyps: list[str]) -> float:
    ops = [edit_ops(normalize_text(r).split(), normalize_text(h).split()) for r, h in zip(refs, hyps)]
    return sum(map(sum, ops)) / max(sum(len(normalize_text(r).split()) for r in refs), 1)
