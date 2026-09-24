"""The **guarded hybrid** (TODO §13): accept T5's refinement only when it
provably keeps the sentence's meaning, else fall back to rules_v2.

The team's hybrid trusts T5 whenever it returns anything. The audit found
its worst errors are fluent glosses with a different meaning
(``docs/reports/speech-to-sign-audit.md`` §2): #19 you -> HE, #25 her -> ME,
#20 ``HE HE HE HE HE HE HE``, subjects dropped (#2, #4, #12). Each check below
targets one of those, and none needs a model, so the guard also runs on
T5 outputs recorded earlier (the team's batch), before the checkpoint arrives.

Checks, against the rules_v1 gloss T5 was given and the English source:
1. ``empty``: T5 returned nothing.
2. ``repetition``: some token occurs more often in T5's output than in both
   the rule gloss and the English (``WHY … WHY``, ``HE HE HE``).
3. ``person``: the set of grammatical persons differs (1st ``I ME MY WE …``,
   2nd ``YOU YOUR``, 3rd ``HE SHE THEY …``; dummy ``IT`` ignored). Catches
   you -> HE and dropped subjects.
4. ``polarity``: negation (``NOT NEVER CAN'T NO``) present in one and not the
   other.
5. ``content``: T5 kept under ``min_content_recall`` of the rule gloss's
   content words (prefix match, so ``IMMEDIATE`` matches ``IMMEDIATELY``).
6. ``invented``: T5 emitted a word found in neither the English nor the rule
   gloss (``GIFT`` for "give"), outside a small set of gloss markers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

PERSON = {
    1: {"I", "ME", "MY", "MINE", "MYSELF", "WE", "US", "OUR", "OURS", "OURSELVES"},
    2: {"YOU", "YOUR", "YOURS", "YOURSELF", "YOURSELVES"},
    3: {"HE", "HIM", "HIS", "HIMSELF", "SHE", "HER", "HERS", "HERSELF", "THEY", "THEM", "THEIR", "THEIRS",
        "THEMSELVES"},
}
NEGATION = {"NOT", "NEVER", "CAN'T", "NO", "NONE", "NOTHING", "NOBODY"}
FUNCTION = set().union(*PERSON.values()) | NEGATION | {"FUTURE", "IT", "BE", "DO", "TO", "THE", "A", "AN"}
# Gloss markers T5 may add without them being "invented".
MARKERS = {"ME", "FUTURE", "NOT", "PLEASE", "IF", "BUT", "BEFORE", "AFTER", "BECAUSE", "WHY", "WHAT",
           "WHERE", "WHEN", "WHO", "HOW", "WHICH", "TIME"}


def _toks(g: str) -> list[str]:
    return [t for t in g.upper().replace(",", " ").split() if t]


def _persons(toks) -> set[int]:
    return {p for t in toks for p, words in PERSON.items() if t in words}


def _match(tok: str, pool: set[str], n: int = 4) -> bool:
    return tok in pool or any(len(tok) >= n and len(p) >= n and tok[:n] == p[:n] for p in pool)


@dataclass
class GuardResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)


def _source_words(english: str) -> list[str]:
    """English words, uppercased, with contractions split the way a gloss
    spells them (``cannot`` -> CAN NOT, ``don't`` -> DO NOT, ``I`` -> ME)."""
    out: list[str] = []
    for w in english.split():
        w = w.strip(".,!?;:\"'").upper()
        if w == "CANNOT":
            out += ["CAN", "NOT"]
        elif w.endswith("N'T"):
            out += [{"WON": "WILL", "CA": "CAN", "SHAN": "SHALL"}.get(w[:-3], w[:-3]), "NOT"]
        elif w:
            out.append(w)
    return ["ME" if w == "I" else w for w in out]


def guard(english: str, rule_gloss: str, t5_gloss: str, *, min_content_recall: float = 0.6) -> GuardResult:
    rule = ["ME" if t == "I" else t for t in _toks(rule_gloss)]  # v1 spells ME as I
    t5 = _toks(t5_gloss)
    words = _source_words(english)
    src = set(words) | {w.replace("'", "") for w in words}
    reasons: list[str] = []
    if not t5:
        return GuardResult(False, ["empty"])
    rc, tc, sc = Counter(rule), Counter(t5), Counter(words)
    if any(n > max(rc[t], sc[t], 1) for t, n in tc.items()):
        reasons.append("repetition")
    if _persons(rule) != _persons(t5):
        reasons.append("person")
    if bool(set(rule) & NEGATION) != bool(set(t5) & NEGATION):
        reasons.append("polarity")
    content = [t for t in rule if t not in FUNCTION]
    if content:
        kept = sum(_match(t, set(t5)) for t in content) / len(content)
        if kept < min_content_recall:
            reasons.append("content")
    invented = [t for t in t5 if t not in MARKERS and not _match(t, src | set(rule))]
    if invented:
        reasons.append("invented:" + "+".join(invented))
    return GuardResult(not reasons, reasons)


@dataclass
class HybridOutput:
    gloss: str
    source: str  # "t5" | "rules_v2" | "rules_v1"
    rule_v1: str
    t5: str | None
    reasons: list[str]


def combine(english: str, rule_v1: str, t5_gloss: str | None, fallback: str, *,
            min_content_recall: float = 0.6) -> HybridOutput:
    """Guarded choice between an (already computed) T5 output and the
    fallback gloss (rules_v2). ``t5_gloss=None`` means no checkpoint."""
    if t5_gloss is None:
        return HybridOutput(fallback, "rules_v2", rule_v1, None, ["no_t5"])
    g = guard(english, rule_v1, t5_gloss, min_content_recall=min_content_recall)
    return HybridOutput(t5_gloss if g.accepted else fallback, "t5" if g.accepted else "rules_v2",
                        rule_v1, t5_gloss, g.reasons)
