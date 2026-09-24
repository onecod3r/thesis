"""English -> ASL gloss, rule engine **v1: frozen** (TODO §13).

Byte-for-byte the logic of the team's ``rule_engine.py`` as copied into cell 3
("STEP 4") of ``chosen_merged_asl_pipeline_hybrid_1.ipynb``. That engine made
the "rule gloss" half of the fine-tuned T5's training input, so it must not
change: a drifted v1 feeds T5 inputs it never saw, and quality drops silently.
Every fix lives in :mod:`sb.synthesize.gloss.rules_v2` instead.

Known defects kept on purpose (audit, ``docs/reports/speech-to-sign-audit.md``
§2): ``I``/``me`` -> ``I``; conjunctions (IF/BUT/BECAUSE/BEFORE) and PLEASE are
dropped; phrasal particles are dropped (TURN OFF -> TURN); no time fronting;
commas stay attached; ``never``/``can't``/``won't`` are not negation; spaCy
runs twice per sentence.

Only changes from the notebook: its print banner, the unused ``import re`` and
the ``rule_engine_gloss`` alias's docstring are left out. ``nlp`` is loaded at
import, exactly as in the original.
"""

import spacy

nlp = spacy.load("en_core_web_sm")

KEEP_POS     = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "PRON"}
WH_WORDS     = {"who", "what", "where", "when", "why", "how", "which"}
COPULA_FORMS = {"is", "am", "are", "was", "were"}
AUX_DO       = {"do", "does", "did"}


def tokenize(sentence: str):
    sentence = sentence.strip().rstrip("?.!")
    return sentence.split()


def has_negation(tokens):
    return any(t.lower() == "not" or t.lower().endswith("n't") for t in tokens)


def detect_sentence_type(tokens):
    if not tokens:
        return "declarative"
    first = tokens[0].lower()
    if first in WH_WORDS:
        return "wh_question"
    if first in AUX_DO or first in COPULA_FORMS or first in {"can", "will", "should"}:
        return "yn_question"
    return "declarative"


def apply_negation_rule(tokens):
    output = []
    for t in tokens:
        low = t.lower()
        if low in AUX_DO:
            continue
        if low in {"not", "don't", "doesn't", "didn't"}:
            output.append("NOT")
        else:
            output.append(t)
    return output


def apply_wh_movement(tokens):
    if tokens and tokens[0].lower() in WH_WORDS:
        wh = tokens[0]
        rest = [t for t in tokens[1:] if t.lower() not in AUX_DO]
        return rest + [wh]
    return tokens


def apply_yn_question_rule(tokens):
    return [t for t in tokens if t.lower() not in AUX_DO]


def apply_future_rule(tokens):
    return ["FUTURE" if t.lower() == "will" else t for t in tokens]


def apply_copula_deletion(tokens):
    return [t for t in tokens if t.lower() not in COPULA_FORMS]


def apply_pos_filter(tokens):
    SPECIAL_TOKENS = {"FUTURE", "NOT"}
    SPECIAL_WH = {"WHO", "WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHICH"}
    doc = nlp(" ".join(tokens))
    filtered = []
    for token in doc:
        word = token.text
        upper_word = word.upper()
        if upper_word in SPECIAL_TOKENS:
            filtered.append(word)
            continue
        if upper_word in SPECIAL_WH:
            filtered.append(word)
            continue
        if token.pos_ == "PRON":
            filtered.append(word)
            continue
        if token.pos_ in KEEP_POS:
            filtered.append(word)
    return filtered


def apply_lemmatization(tokens):
    SPECIAL_TOKENS = {"FUTURE", "NOT", "WHO", "WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHICH"}
    doc = nlp(" ".join(tokens))
    output = []
    for token in doc:
        if token.text.upper() in SPECIAL_TOKENS:
            output.append(token.text.upper())
        else:
            output.append(token.lemma_.upper())
    return output


def convert(sentence: str) -> str:
    tokens = tokenize(sentence)
    if has_negation(tokens):
        tokens = apply_negation_rule(tokens)
    sentence_type = detect_sentence_type(tokens)
    if sentence_type == "wh_question":
        tokens = apply_wh_movement(tokens)
    elif sentence_type == "yn_question":
        tokens = apply_yn_question_rule(tokens)
    tokens = apply_future_rule(tokens)
    tokens = apply_copula_deletion(tokens)
    tokens = apply_pos_filter(tokens)
    tokens = apply_lemmatization(tokens)
    return " ".join(tokens)


def rule_engine_gloss(text: str) -> str:
    return convert(text)
