"""ASL gloss -> English, rule engine v1: the reverse of
``sb.synthesize.gloss.rules_v2`` (TODO §12.6, stage 3).

``rules_v2`` deletes what ASL gloss does not carry: articles, the copula,
``do``-support, tense endings, most prepositions. It also fronts time words
and moves wh-words to the end. This engine puts those back from the
sentence's glosses and the corpus lexicon's part-of-speech tags
(``sb.recognize.sequences.corpus.load_lexicon``, passed in so this module
needs neither torch nor spaCy):

- **pronouns** by position: ``minemy`` is *my* before a noun, *me* after a
  verb or preposition, *I* otherwise, and *mine* alone. The same goes for
  ``yourself`` (you/your), ``weus`` (we/us/our) and ``hesheit``. GISLR
  has one sign for he/she/it, and the gloss does not say which, so it
  becomes the gender-neutral *they/them/their*;
- **copula** for adjective and locative predicates (*The dog is hungry*,
  *The milk is in the refrigerator*);
- **tense**: past with ``yesterday`` or a completive ``finish``, future
  with ``will``/``tomorrow``/``later``, present otherwise. Agreement and
  ``do``-support under ``not``;
- **articles**: *the* on count nouns without a determiner. None on mass and
  plural nouns, and none on family names used as names (*Mom*, *Grandpa*);
- **word order**: a wh-word goes to the front with ``do``-support or a
  copula (*Where is the gum?*). A clause-initial greeting or response becomes
  an interjection (*Hello, Dad.*). A trailing ``no`` negates the sentence
  (*No chocolate before food.*). Time words go to the front or back.

It is deterministic and runs anywhere, the client included, but it is a
**baseline**. The gloss sentences are telegraphic (``cut napkin scissors``),
and many readings need world knowledge the rules don't have. The LLM arm
(``prompts/v1/gloss2en.txt``) is the realistic comparison. Both are scored on
``evalset/gloss2en.v1.jsonl`` and by round trip through ``rules_v2``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

SURFACE = {
    "thankyou": "thank you", "callonphone": "call", "glasswindow": "window",
    "frenchfries": "French fries", "icecream": "ice cream", "haveto": "have to",
    "TV": "TV", "shhh": "shh", "mom": "Mom", "dad": "Dad", "grandma": "Grandma",
    "grandpa": "Grandpa", "owie": "owie", "potty": "potty",
}
NAMES = {"mom", "dad", "grandma", "grandpa"}  # family words used as names: no article
MASS = {"water", "milk", "food", "cereal", "pizza", "icecream", "gum", "grass", "garbage",
        "chocolate", "rain", "snow", "TV", "home", "fish", "hair", "outside", "time"}
PLURAL = {"jeans", "pajamas", "scissors", "stairs", "nuts", "frenchfries", "feet", "underwear", "lips"}
PLACES_IN = {"bedroom", "room", "closet", "drawer", "refrigerator", "dryer", "car", "boat", "airplane",
             "pool", "store", "farm", "backyard", "tree", "bed", "garbage", "water", "cloud", "hair"}
PLACES_ON = {"table", "chair", "stairs", "flower", "horse"}
NOT_SPLIT = {"hair", "water", "cloud", "horse", "tree"}  # places only after an adjective ("stuck in the hair")
INTERJ = {"hello", "bye", "yes", "no", "please", "thankyou", "shhh"}
TIME_FRONT = {"yesterday", "tomorrow", "now"}
TIME_BACK = {"morning": "in the morning", "night": "at night", "later": "later", "now": "now"}
FUTURE = {"will", "tomorrow", "later"}
PAST = {"yesterday"}
WH = {"who", "where", "why"}
CONJ = {"because", "if"}

# lemma -> (3rd person singular, past); phrasal verbs spelled out
VERBS = {
    "bath": ("takes a bath", "took a bath", "take a bath"),
    "shower": ("takes a shower", "took a shower", "take a shower"),
    "nap": ("takes a nap", "took a nap", "take a nap"),
    "wake": ("wakes up", "woke up", "wake up"),
    "callonphone": ("calls", "called", "call"),
    "blow": ("blows", "blew", "blow"), "clean": ("cleans", "cleaned", "clean"),
    "close": ("closes", "closed", "close"), "cry": ("cries", "cried", "cry"),
    "cut": ("cuts", "cut", "cut"), "dance": ("dances", "danced", "dance"),
    "drink": ("drinks", "drank", "drink"), "drop": ("drops", "dropped", "drop"),
    "dry": ("dries", "dried", "dry"), "fall": ("falls", "fell", "fall"),
    "find": ("finds", "found", "find"), "finish": ("finishes", "finished", "finish"),
    "give": ("gives", "gave", "give"), "go": ("goes to", "went to", "go to"),
    "hate": ("hates", "hated", "hate"), "have": ("has", "had", "have"),
    "hear": ("hears", "heard", "hear"), "hide": ("hides", "hid", "hide"),
    "jump": ("jumps", "jumped", "jump"), "kiss": ("kisses", "kissed", "kiss"),
    "like": ("likes", "liked", "like"), "listen": ("listens to", "listened to", "listen to"),
    "look": ("looks at", "looked at", "look at"), "make": ("makes", "made", "make"),
    "open": ("opens", "opened", "open"), "pretend": ("pretends", "pretended", "pretend"),
    "read": ("reads", "read", "read"), "ride": ("rides", "rode", "ride"),
    "say": ("says", "said", "say"), "see": ("sees", "saw", "see"),
    "sleep": ("sleeps", "slept", "sleep"), "smile": ("smiles at", "smiled at", "smile at"),
    "stay": ("stays", "stayed", "stay"), "talk": ("talks", "talked", "talk"),
    "taste": ("tastes", "tasted", "taste"), "think": ("thinks", "thought", "think"),
    "touch": ("touches", "touched", "touch"), "vacuum": ("vacuums", "vacuumed", "vacuum"),
    "wait": ("waits for", "waited for", "wait for"),
}
OBJ_PREP_VERBS = {"listen", "look", "smile", "wait", "go"}  # the preposition only appears with an object
NO_TO = {"home", "outside"}  # "go home", not "go to home"
TIME_PHRASE = {"morning": "in the morning", "night": "at night"}  # moved to the end of the clause
PRONOUNS = {  # subject, object, possessive, standalone, person-number
    "minemy": ("I", "me", "my", "mine", "1s"),
    "yourself": ("you", "you", "your", "yours", "2"),
    "weus": ("we", "us", "our", "ours", "1p"),
    "hesheit": ("they", "them", "their", "theirs", "3p"),
}


def _tags(g: str, lex: Mapping[str, Sequence[str]]) -> list[str]:
    return list(lex.get(g, ["noun"]))


def _is(g: str, tag: str, lex) -> bool:
    return tag in _tags(g, lex)


def _surface(g: str) -> str:
    return SURFACE.get(g, g)


def _noun_phrase(toks: list[str], lex, role: str) -> tuple[str, str]:
    """Tokens of one NP -> (text, person-number for agreement)."""
    if len(toks) == 1 and toks[0] in PRONOUNS:
        subj, obj, _, _, pn = PRONOUNS[toks[0]]
        return (subj if role == "subj" else obj), pn
    if len(toks) == 1 and toks[0] in {"that", "there"}:
        return toks[0], "3s"
    words, det, head = [], None, toks[-1]
    for t in toks:
        if t in PRONOUNS:
            det = PRONOUNS[t][2]
        elif _is(t, "quant", lex):
            det = {"all": "all the", "another": "another", "any": "any", "every": "every",
                   "many": "many"}.get(t, t)
        elif t in {"that"}:
            det = "that"
        else:
            words.append(_surface(t))
    if not words:  # a lone quantifier / determiner
        return {"all the": "everything", "every": "everyone", "another": "another one"}.get(det or "", det or ""), "3s"
    plural = head in PLURAL or det in {"many", "all the"}
    if det in {"many", "all the"} and head not in PLURAL and head not in MASS and words:
        w = words[-1]
        words[-1] = (w[:-1] + "ies" if w.endswith("y") and w[-2:-1] not in "aeiou"
                     else w + ("es" if w.endswith(("s", "sh", "ch")) else "s"))
    if det is None and head not in NAMES and head not in MASS and head not in PLURAL:
        det = "the"
    if det is None and head in {"food", "hair", "water", "milk", "gum", "TV"} and role == "subj":
        det = "the"
    text = " ".join([det] + words) if det else " ".join(words)
    pn = "3p" if plural else "3s"
    return text, pn


def _chunks(toks: list[str], lex) -> list[tuple[str, list[str]]]:
    """Tokens -> [(kind, tokens)]: np, verb, adj, neg, modal, prep, time, wh, conj."""
    out: list[tuple[str, list[str]]] = []
    np_buf: list[str] = []

    def flush():
        if np_buf:
            out.append(("np", np_buf.copy()))
            np_buf.clear()

    for i, t in enumerate(toks):
        tags = _tags(t, lex)
        prev = out[-1][0] if out else None
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if t in WH:
            flush()
            out.append(("wh", [t]))
        elif t in {"not"} or (t == "no" and "neg" in tags):
            flush()
            out.append(("neg", [t]))
        elif t in {"can", "haveto", "will"}:
            flush()
            out.append(("modal", [t]))
        elif "time" in tags and t not in {"first"} and not (t in {"before", "after"} and nxt):
            flush()
            out.append(("time", [t]))
        elif "prep" in tags and t not in {"outside"}:
            flush()
            out.append(("prep", [t]))
        elif "verb" in tags and ("noun" not in tags or prev in {"np", "neg", "modal", None} and (np_buf or prev)
                                 or (not np_buf and prev is None and nxt is not None)):
            if "adj" in tags and np_buf and nxt is None:
                flush()
                out.append(("adj", [t]))
                continue
            flush()
            out.append(("verb", [t]))
        elif "adj" in tags and len(toks) == 1:
            out.append(("adj", [t]))
        elif "adj" in tags and (np_buf or prev in {"np", "adj", "neg", "verb"}) and not (
                nxt is not None and _is(nxt, "noun", lex) and not _is(nxt, "adj", lex) and not np_buf):
            flush()
            out.append(("adj", [t]))
        else:
            np_buf.append(t)
    flush()
    return _split_nps(out, lex)


def _split_nps(chunks: list[tuple[str, list[str]]], lex) -> list[tuple[str, list[str]]]:
    """Split noun runs ASL leaves unmarked: ``that``/a pronoun followed by
    a second NP (``that minemy toy`` -> that + my toy, ``hesheit minemy
    brother``), and a trailing place noun (``shirt dryer`` -> shirt + in +
    dryer, ``radio car`` -> radio + in + car)."""
    out: list[tuple[str, list[str]]] = []
    for k, t in chunks:
        if k != "np" or len(t) < 2:
            out.append((k, t))
            continue
        if t[0] in {"that", "there"} or (t[0] in PRONOUNS and any(x in PRONOUNS for x in t[1:])):
            out += [("np", [t[0]]), ("np", t[1:])] if t[0] in {"that", "there"} else                 [("np", [t[0]]), ("np", t[1:])]
            continue
        last, prev = t[-1], t[-2]
        if (last in PLACES_IN or last in PLACES_ON) and last not in NOT_SPLIT and _is(prev, "noun", lex) and prev not in PRONOUNS                 and not _is(prev, "quant", lex):
            out += [("np", t[:-1]), ("prep", ["on" if last in PLACES_ON else "in"]), ("np", [last])]
            continue
        out.append((k, t))
    return out


def _conjugate(verb: str, pn: str, tense: str, neg: bool, modal: str | None, has_obj: bool) -> str:
    third, past, base = VERBS.get(verb, (verb + "s", verb + "ed", verb))
    if verb in OBJ_PREP_VERBS and not has_obj:
        third, past, base = third.split()[0], past.split()[0], base.split()[0]
    sg3 = pn == "3s"
    if modal == "can":
        return f"can{'not' if neg else ''} {base}"
    if modal == "haveto":
        if neg:
            return f"{'does' if sg3 else 'do'} not have to {base}"
        return f"{'has' if sg3 else 'have'} to {base}"
    if tense == "future":
        return f"will {'not ' if neg else ''}{base}"
    if tense == "past":
        return f"did not {base}" if neg else past
    if neg:
        return f"{'does' if sg3 else 'do'} not {base}"
    return third if sg3 else base


def _be(pn: str, tense: str) -> str:
    if tense == "past":
        return "was" if pn in {"1s", "3s"} else "were"
    if tense == "future":
        return "will be"
    return {"1s": "am", "3s": "is"}.get(pn, "are")


def _clause(toks: list[str], lex, tense: str, inherit: tuple[str, str] | None = None) -> tuple[str, bool, tuple[str, str] | None]:
    """One clause -> (English without final punctuation, is_question,
    (subject, person-number) for a following subjectless clause)."""
    times = [t for t in toks if t in TIME_PHRASE]
    toks = [t for t in toks if t not in TIME_PHRASE]
    text, q, subj = _clause_core(toks, lex, tense, inherit)
    for t in times:
        text = f"{text} {TIME_PHRASE[t]}" if text else TIME_PHRASE[t]
    return text, q, subj


def _clause_core(toks: list[str], lex, tense: str, inherit) -> tuple[str, bool, tuple[str, str] | None]:
    if [t for t in toks if t not in {"now", "later", "tomorrow", "yesterday"}] in (["rain"], ["snow"]):
        w = "rain" if "rain" in toks else "snow"
        verb = {"past": f"{w}ed", "future": f"will {w}"}.get(tense, f"is {w}ing")
        rest = " ".join(t for t in toks if t not in {"rain", "snow", "yesterday"})
        return f"it {verb}" + (f" {rest}" if rest else ""), False, None
    chunks = _chunks(toks, lex)
    kinds = [k for k, _ in chunks]
    question = "wh" in kinds
    wh = next((c[1][0] for c in chunks if c[0] == "wh"), None)
    chunks = [c for c in chunks if c[0] != "wh"]
    neg = any(k == "neg" for k, _ in chunks)
    modal = next((c[1][0] for c in chunks if c[0] == "modal" and c[1][0] != "will"), None)
    if any(c[0] == "modal" and c[1][0] == "will" for c in chunks):
        tense = "future"
    body = [c for c in chunks if c[0] not in {"neg", "modal"}]
    # completive FINISH after another verb: past tense of that verb
    verbs = [c for c in body if c[0] == "verb"]
    if len(verbs) > 1 and verbs[-1][1][0] == "finish":
        body.remove(verbs[-1])
        tense = "past"
    elif len(verbs) == 1 and verbs[0][1][0] == "finish" and tense == "present":
        tense = "past"

    subj, pn, rest = None, "2", body
    if body and body[0][0] == "np":
        subj_toks = body[0][1]
        rest = body[1:]
        subj, pn = _noun_phrase(subj_toks, lex, "subj")
    elif inherit is not None and body and body[0][0] in {"adj", "verb"}:
        subj, pn = inherit
    elif body and all(k == "adj" for k, _ in body) and not question:
        return "be " + " and ".join(_surface(t[0]) for _, t in body), False, None
    words: list[str] = []
    verb_i = next((i for i, c in enumerate(rest) if c[0] == "verb"), None)
    adj_i = next((i for i, c in enumerate(rest) if c[0] == "adj"), None)

    def tail(items) -> list[str]:
        out: list[str] = []
        for k, t in items:
            if k == "np":
                out.append(_noun_phrase(t, lex, "obj")[0])
            elif k == "prep":
                out.append(t[0])
            elif k == "adj":
                out.append(_surface(t[0]))
            elif k == "verb":
                out.append(VERBS.get(t[0], (None, None, t[0]))[2])
            elif k == "time":
                out.append(TIME_BACK.get(t[0], t[0]))
        return out

    if verb_i is not None and (adj_i is None or verb_i < adj_i or subj is None):
        pre, v, post = rest[:verb_i], rest[verb_i][1][0], rest[verb_i + 1:]
        has_obj = any(k == "np" for k, _ in post) and not (v == "go" and post and post[0][0] == "np"
                                                             and post[0][1][-1] in NO_TO)
        if subj is None and pre and pre[0][0] == "adj" and len(pre) == 1 and post and post[0][0] == "np":
            subj, pn = pre[0][1][0] + " " + _noun_phrase(post[0][1], lex, "subj")[0], "3s"
        if subj is None and not question:  # imperative
            base = VERBS.get(v, (None, None, v))[2]
            if v in OBJ_PREP_VERBS and not has_obj:
                base = base.split()[0]
            words = (["do not"] if neg else []) + [base] + tail(post)
        elif question and wh is not None and wh != "who":
            aux = {"past": "did", "future": "will"}.get(tense, "does" if pn == "3s" else "do")
            if modal == "can":
                aux = "can"
            base = VERBS.get(v, (None, None, v))[2]
            words = [wh, aux] + ([subj] if subj else ["you"]) + (["not"] if neg else []) + [base] + tail(post)
            return " ".join(words), True, None
        else:
            if wh == "who":
                subj, pn = "who", "3s"
            words = [subj or "you", _conjugate(v, pn, tense, neg, modal, has_obj)] + tail(pre) + tail(post)
    elif adj_i is not None:
        pre, post = rest[:adj_i], rest[adj_i:]
        if subj is None:
            subj, pn = ("it", "3s") if not pre else _noun_phrase(pre[0][1], lex, "subj")
            pre = pre[1:] if pre else pre
        adjs = [_surface(t[0]) for k, t in post if k == "adj"]
        others = [(k, t) for k, t in post if k != "adj"]
        adj_text = adjs[0] if len(adjs) == 1 else ", ".join(adjs[:-1]) + " and " + adjs[-1]
        words = [subj, _be(pn, tense)] + (["not"] if neg else []) + [adj_text]
        for k, t in others:
            if k == "np":
                head = t[-1]
                prep = "on" if head in PLACES_ON else "in" if head in PLACES_IN else "with"
                words += [prep, _noun_phrase(t, lex, "obj")[0]]
            else:
                words += tail([(k, t)])
        if question and wh:
            words = [wh, _be(pn, tense), subj] + words[2:]
            return " ".join(words), True, None
    elif subj is not None and rest:
        # verbless: NP + (prep) NP -> locative / identity copula
        k0, t0 = rest[0]
        if question and wh:
            if wh == "who" and subj == "that" or (subj and wh == "who" and toks[0] == "that"):
                return f"whose {_noun_phrase(t0, lex, 'obj')[0].removeprefix('the ')} is that", True, None
            return " ".join([wh, _be(pn, tense), subj] + tail(rest)), True, None
        if k0 == "prep":
            words = [subj, _be(pn, tense)] + (["not"] if neg else []) + tail(rest)
        elif k0 == "np":
            head = t0[-1]
            if subj in {"that", "there", "they", "I", "you", "we"} or toks[0] in PRONOUNS or toks[0] == "that":
                words = [subj, _be(pn, tense)] + (["not"] if neg else []) + tail(rest)
            else:
                prep = "on" if head in PLACES_ON else "in" if head in PLACES_IN else "and"
                if prep == "and":
                    words = [subj, "and"] + tail(rest)
                else:
                    words = [subj, _be(pn, tense), prep] + tail(rest)
        else:
            words = [subj] + tail(rest)
    elif subj is not None:
        if question and wh:
            return f"{wh} {_be(pn, tense)} {subj}", True, None
        words = (["no"] if neg else []) + [subj]
    else:
        words = (["not"] if neg else []) + tail(rest)
        if question and wh:
            return " ".join([wh] + words), True, None
    if question and wh == "who" and words and words[0] != "who":
        words = ["who"] + words[1:]
    carry = None
    if subj and subj not in {"that", "there", "who"}:
        carry = (subj, pn) if subj in {"I", "you", "we", "they"} else ("they", "3p")
    return " ".join(w for w in words if w), question, carry


def convert(glosses: Sequence[str], lexicon: Mapping[str, Sequence[str]]) -> str:
    """One gloss sentence (GISLR labels, ASL order) -> one English sentence."""
    toks = list(glosses)
    if not toks:
        return ""
    lead, trail = [], []
    while toks and toks[0] in INTERJ and (len(toks) == 1 or toks[0] != "no" or toks[1] in INTERJ):
        lead.append(toks.pop(0))
    question = False
    if len(toks) >= 2 and toks[-2:] == ["yes", "no"]:
        toks, question = toks[:-2], True
    if toks and toks[-1] == "please":
        trail.append(toks.pop())
    sentence_no = False
    if len(toks) > 1 and toks[-1] == "no":
        toks, sentence_no = toks[:-1], True
    tense = "past" if PAST & set(toks) else "future" if FUTURE & set(toks) else "present"
    front = [t for t in toks if t in TIME_FRONT and toks.index(t) == 0]
    toks = [t for t in toks if t not in front]
    # clauses split at conjunctions
    parts: list[list[str]] = [[]]
    joins: list[str] = []
    for t in toks:
        if t in CONJ and parts[-1]:
            parts.append([])
            joins.append(t)
        else:
            parts[-1].append(t)
    texts = []
    carry = None
    for p in parts:
        if p:
            txt, q, carry = _clause(p, lexicon, tense, carry)
            texts.append(txt)
            question = question or q
    body = texts[0] if texts else ""
    for j, t in zip(joins, texts[1:]):
        body += f" {j} {t}"
    if sentence_no:
        body = "no " + body
    pieces = [" ".join(_surface(t) for t in lead)] if lead else []
    head = " ".join(_surface(t) for t in front)
    if head:
        body = f"{head} {body}" if body else head
    if body:
        pieces.append(body)
    out = ", ".join(pieces)
    if trail:
        out += ", please"
    out = out.strip()
    if not out:
        return ""
    return out[0].upper() + out[1:] + ("?" if question else ".")


# ============================================================
# Round trip: English -> rules_v2 gloss -> GISLR labels
# ============================================================

#: rules_v2 output tokens (uppercase English gloss) that name a GISLR label differently
TO_GISLR = {
    "ME": "minemy", "MY": "minemy", "MINE": "minemy", "I": "minemy", "MYSELF": "minemy",
    "YOU": "yourself", "YOUR": "yourself", "YOURS": "yourself", "YOURSELF": "yourself",
    "WE": "weus", "US": "weus", "OUR": "weus", "OURS": "weus",
    "HE": "hesheit", "SHE": "hesheit", "IT": "hesheit", "HIM": "hesheit", "HER": "hesheit",
    "THEY": "hesheit", "THEM": "hesheit", "THEIR": "hesheit",
    "MOTHER": "mom", "FATHER": "dad", "CALL": "callonphone", "WINDOW": "glasswindow",
    "SHH": "shhh", "TEETH": "tooth", "FOOT": "feet",
}
MULTI_TO_GISLR = {("THANK", "YOU"): "thankyou", ("ICE", "CREAM"): "icecream",
                  ("FRENCH", "FRIES"): "frenchfries", ("HAVE", "TO"): "haveto"}


def to_gislr(gloss: str, vocab: set[str]) -> list[str]:
    """A ``rules_v2`` gloss string -> GISLR labels, for round-trip scoring.
    Plural ``-s`` is stripped when the singular is a label. Tokens that map
    to no label are kept lowercase, so they count as errors."""
    toks = gloss.upper().split()
    out: list[str] = []
    i = 0
    lower_vocab = {v.lower(): v for v in vocab}
    while i < len(toks):
        if i + 1 < len(toks) and (toks[i], toks[i + 1]) in MULTI_TO_GISLR:
            out.append(MULTI_TO_GISLR[(toks[i], toks[i + 1])])
            i += 2
            continue
        t = toks[i]
        if t in TO_GISLR:
            out.append(TO_GISLR[t])
        elif t.lower() in lower_vocab:
            out.append(lower_vocab[t.lower()])
        elif t.endswith("S") and t[:-1].lower() in lower_vocab:
            out.append(lower_vocab[t[:-1].lower()])
        elif t.endswith("ES") and t[:-2].lower() in lower_vocab:
            out.append(lower_vocab[t[:-2].lower()])
        else:
            out.append(t.lower())
        i += 1
    return out
