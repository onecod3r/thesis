"""English -> ASL gloss, rule engine **v2: the fixed one** (TODO §13, Phase 1).

Fixes every rules_v1 defect listed in the audit
(``docs/reports/speech-to-sign-audit.md`` §2), from **one** spaCy parse:

- ``I``/``me`` -> ``ME`` (possessives stay ``MY``/``YOUR``/...).
- Keeps discourse words v1 drops: ``IF BUT BECAUSE BEFORE AFTER WHEN ...``
  and ``PLEASE``.
- Keeps phrasal-verb particles (``TURN OFF``, ``WAKE UP``).
- **Time fronting**: time words (``YESTERDAY``, ``TOMORROW``, ``NOW``, day
  names, ``THIS MORNING``, ``AT NIGHT`` -> ``NIGHT``, ``three o'clock`` ->
  ``TIME THREE``) move to the front of their clause. ``FUTURE`` is dropped
  when the sentence already carries a time word (ASL marks tense once).
- Negation: ``not``/``n't`` -> ``NOT``, ``never`` -> ``NEVER``, ``can't``/
  ``cannot`` -> ``CAN'T``, ``not ... yet`` -> ``NOT YET``.
- Questions: the wh-phrase goes to the end (``what time`` -> ``TIME WHAT``);
  a yes/no question's leading modal goes to the end (``Can you help me?`` ->
  ``YOU HELP ME CAN``); ``do``-support is dropped.
- Deletes copula ``be``, auxiliary ``do``/``have``, infinitive ``to``,
  articles, most prepositions, the complementizer ``that``, and the dummy
  ``it`` of weather verbs (``it was raining`` -> ``RAIN``).
- Punctuation never sticks to a gloss (spaCy tokens, not ``str.split``).

The target conventions are the ones the draft references in
``sb.synthesize.evalsets`` (team30) were written to. Both were written by the
same author (Claude), so v2's score on team30 is a *development* number, not a
test result (TODO §13).
"""

from __future__ import annotations

from functools import lru_cache

import spacy

WH_WORDS = {"who", "what", "where", "when", "why", "how", "which"}
# Front of clause. Durations ("two days", "five minutes") are not here: they stay in place.
TIME_FRONT = {
    "yesterday", "today", "tomorrow", "tonight", "now",
    "morning", "afternoon", "evening", "night", "weekend",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
# Fronted only when made relative ("last week", "next year"); "two weeks" is a duration.
TIME_RELATIVE = {"week", "month", "year"}
RELATIVE_MARKERS = {"this", "last", "next", "every"}
# Carry tense, so FUTURE is redundant next to them, but they stay where they are.
TIME_IN_PLACE = {"later", "soon", "recently", "already"}
KEEP_CONJ = {"if", "but", "because", "before", "after", "when", "while", "so", "or", "and",
             "until", "although", "though", "since"}
KEEP_INTJ = {"please", "yes", "no", "hello", "hi", "thanks", "sorry", "ok", "okay"}
WEATHER_VERBS = {"rain", "snow", "storm", "hail", "drizzle", "pour", "thunder"}
MODALS = {"can": "CAN", "ca": "CAN", "could": "COULD", "should": "SHOULD", "must": "MUST",
          "might": "MIGHT", "may": "MAY", "would": "WOULD"}
FUTURE_FORMS = {"will", "'ll", "wo", "shall"}
PRONOUN_MAP = {"i": "ME", "me": "ME", "myself": "MYSELF"}


@lru_cache(maxsize=1)
def nlp():
    return spacy.load("en_core_web_sm")


def _is_weather(tok) -> bool:
    low = tok.text.lower()
    return tok.lemma_.lower() in WEATHER_VERBS or any(low.startswith(w) for w in WEATHER_VERBS)


def _relative_time(tok) -> bool:
    return tok.lemma_.lower() in TIME_RELATIVE and any(c.text.lower() in RELATIVE_MARKERS for c in tok.children)


def _items(doc) -> list[list[dict]]:
    """Tokens -> kept items, split into clauses at conjunctions. Each item:
    ``{"g": gloss, "k": kind}``; kinds: word, time, wh, modal, future, conj,
    neg, yet."""
    clauses: list[list[dict]] = [[]]
    toks = list(doc)
    for i, t in enumerate(toks):
        low = t.text.lower()
        lemma = t.lemma_.lower()
        cur = clauses[-1]
        if t.is_punct or t.is_space:
            continue
        # clause boundaries: subordinating / coordinating conjunctions we keep
        if low in KEEP_CONJ and (t.pos_ == "CCONJ" or t.dep_ in {"mark", "cc"}):
            if cur:
                clauses.append([])
            clauses[-1].append({"g": low.upper(), "k": "conj", "i": t.i})
            continue
        if low in KEEP_INTJ and t.pos_ in {"INTJ", "ADV", "VERB", "DET"} and t.dep_ in {"intj", "advmod", "ROOT", "det"}:
            cur.append({"g": low.upper(), "k": "word", "i": t.i})
            continue
        # negation
        if low in {"not", "n't"} or (lemma == "not" and t.dep_ == "neg"):
            if cur and cur[-1]["k"] == "modal" and cur[-1]["g"] == "CAN":
                cur[-1] = {"g": "CAN'T", "k": "word", "i": t.i}
            else:
                cur.append({"g": "NOT", "k": "neg", "i": t.i})
            continue
        if low == "never":
            cur.append({"g": "NEVER", "k": "neg", "i": t.i})
            continue
        if low == "yet":
            cur.append({"g": "YET", "k": "yet", "i": t.i})
            continue
        # auxiliaries, modals, copula
        if low in FUTURE_FORMS and t.pos_ in {"AUX", "VERB"}:
            cur.append({"g": "FUTURE", "k": "future", "i": t.i})
            continue
        if low in MODALS and t.pos_ == "AUX":
            cur.append({"g": MODALS[low], "k": "modal", "i": t.i})
            continue
        if lemma == "be" and t.pos_ in {"AUX", "VERB"}:
            continue
        if lemma in {"do", "have"} and t.pos_ == "AUX":
            continue
        if t.tag_ == "TO" or (low == "to" and t.pos_ == "PART"):
            continue
        # "three o'clock" -> TIME THREE (time phrase)
        if low in {"o'clock", "oclock"}:
            if cur and cur[-1]["k"] == "word" and toks[i - 1].pos_ == "NUM":
                num = cur.pop()["g"]
                cur.extend([{"g": "TIME", "k": "time", "i": t.i}, {"g": num, "k": "time", "i": t.i}])
            continue
        # wh-words
        if low in WH_WORDS and t.pos_ in {"PRON", "ADV", "DET", "SCONJ"} and t.dep_ != "mark":
            cur.append({"g": low.upper(), "k": "wh", "i": t.i, "head": t.head.i if t.dep_ == "det" else None})
            continue
        # time words (and "this"/"that" determining them)
        if lemma in TIME_FRONT or low in TIME_FRONT or _relative_time(t):
            cur.append({"g": (lemma if lemma in TIME_RELATIVE else low).upper(), "k": "time", "i": t.i})
            continue
        if low in RELATIVE_MARKERS | {"that"} and t.dep_ in {"det", "amod"} and (
                t.head.lemma_.lower() in TIME_FRONT or _relative_time(t.head)):
            cur.append({"g": low.upper(), "k": "time", "i": t.i})
            continue
        if t.pos_ == "DET":
            if low == "no":
                cur.append({"g": "NO", "k": "neg", "i": t.i})
            elif low in {"this", "that", "these", "those", "all", "some", "many", "every", "each", "other"}:
                cur.append({"g": low.upper(), "k": "word", "i": t.i})
            continue
        if t.pos_ == "ADP":
            if t.dep_ == "prt":
                cur.append({"g": low.upper(), "k": "word", "i": t.i})
            continue
        if t.pos_ == "SCONJ" or t.pos_ == "CCONJ":  # "that" complementizer, other connectives
            continue
        if t.pos_ == "PART":
            if t.dep_ == "prt":
                cur.append({"g": low.upper(), "k": "word", "i": t.i})
            continue
        if t.pos_ == "PRON":
            # the parser sometimes hangs "it" on the auxiliary ("it was raining"), so look one up
            if low == "it" and (_is_weather(t.head) or (t.head.pos_ == "AUX" and _is_weather(t.head.head))
                                or t.dep_ == "expl"):
                continue
            cur.append({"g": PRONOUN_MAP.get(low, low.upper()), "k": "word", "i": t.i})
            continue
        if t.pos_ == "NUM":
            cur.append({"g": t.text.upper(), "k": "word", "i": t.i})
            continue
        if t.pos_ == "ADJ" and t.tag_ in {"JJS", "JJR"}:
            cur.append({"g": t.text.upper(), "k": "word", "i": t.i})
            continue
        if t.pos_ == "NOUN" and t.tag_ == "NNS" and not low.endswith("s"):
            cur.append({"g": t.text.upper(), "k": "word", "i": t.i})  # lexical plurals: CHILDREN, PEOPLE
            continue
        if t.pos_ == "PROPN":
            cur.append({"g": t.text.upper(), "k": "word", "i": t.i})
            continue
        if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "INTJ", "AUX"}:
            if lemma in TIME_IN_PLACE:
                cur.append({"g": lemma.upper(), "k": "time_in_place", "i": t.i})
            else:
                cur.append({"g": lemma.upper(), "k": "word", "i": t.i})
            continue
        # X, SYM, anything else: dropped
    return [c for c in clauses if c]


def _order_clause(items: list[dict], sentence_type: str, first: bool) -> list[str]:
    conj = [it for it in items if it["k"] == "conj"]
    time = [it for it in items if it["k"] == "time"]
    rest = [it for it in items if it["k"] not in {"conj", "time"}]
    # not ... yet -> NOT YET
    if any(it["k"] == "yet" for it in rest) and any(it["g"] == "NOT" for it in rest):
        rest = [it for it in rest if it["k"] != "yet"]
        rest = [{"g": "NOT YET", "k": "neg"} if it["g"] == "NOT" else it for it in rest]
    tail: list[dict] = []
    if sentence_type == "wh":
        whs = [it for it in rest if it["k"] == "wh"]
        if whs:
            wh = whs[0]
            rest.remove(wh)
            noun = next((it for it in rest if it.get("i") == wh.get("head")), None)
            if noun is not None:
                rest.remove(noun)
                tail.append(noun)
            tail.append(wh)
    if sentence_type == "yn" and first and rest and rest[0]["k"] == "modal":
        tail.append(rest.pop(0))
    return [it["g"] for it in time + conj + rest + tail]


def _sentence_type(doc) -> str:
    words = [t for t in doc if not (t.is_punct or t.is_space)]
    if not words:
        return "declarative"
    first = words[0].text.lower()
    is_q = doc.text.strip().endswith("?")
    if first in WH_WORDS:
        return "wh"
    if first in KEEP_INTJ or first in KEEP_CONJ:
        return "wh" if is_q and any(w.text.lower() in WH_WORDS for w in words) else "declarative"
    if words[0].pos_ == "AUX" and (is_q or first in {"do", "does", "did", "can", "will", "should", "is", "are"}):
        return "yn"
    return "wh" if is_q and any(w.text.lower() in WH_WORDS for w in words) else "declarative"


def convert(sentence: str) -> str:
    """One English sentence -> ASL gloss (space-separated, uppercase)."""
    doc = nlp()(sentence.strip())
    stype = _sentence_type(doc)
    clauses = _items(doc)
    has_time = any(it["k"] in {"time", "time_in_place"} for c in clauses for it in c)
    out: list[str] = []
    for n, c in enumerate(clauses):
        if has_time:
            c = [it for it in c if it["k"] != "future"]
        out.extend(_order_clause(c, stype, first=(n == 0)))
    return " ".join(g for g in out if g)
