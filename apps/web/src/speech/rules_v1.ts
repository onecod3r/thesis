/**
 * English -> ASL gloss, rule engine v1 (the team's frozen `rule_engine.py`,
 * `sb.synthesize.gloss.rules_v1`) on wink-nlp instead of spaCy. v1 only reads POS and
 * lemmas, so the port is line for line; differences come from the tagger alone. Its output is
 * half of T5's input, so agreement with Python is tracked in `test/speech.test.ts`.
 */

import { parse } from "./nlp.ts";
import { pySplit } from "./text.ts";

const KEEP_POS = new Set(["NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "PRON"]);
const WH_WORDS = new Set(["who", "what", "where", "when", "why", "how", "which"]);
const COPULA_FORMS = new Set(["is", "am", "are", "was", "were"]);
const AUX_DO = new Set(["do", "does", "did"]);
const SPECIAL_WH = new Set(["WHO", "WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHICH"]);
const SPECIAL = new Set(["FUTURE", "NOT"]);

function tokenize(sentence: string): string[] {
  return pySplit(sentence.trim().replace(/[?.!]+$/, ""));
}

const hasNegation = (ts: string[]) => ts.some((t) => t.toLowerCase() === "not" || t.toLowerCase().endsWith("n't"));

function sentenceType(ts: string[]): "declarative" | "wh_question" | "yn_question" {
  if (!ts.length) return "declarative";
  const first = ts[0].toLowerCase();
  if (WH_WORDS.has(first)) return "wh_question";
  if (AUX_DO.has(first) || COPULA_FORMS.has(first) || ["can", "will", "should"].includes(first)) return "yn_question";
  return "declarative";
}

function negation(ts: string[]): string[] {
  const out: string[] = [];
  for (const t of ts) {
    const low = t.toLowerCase();
    if (AUX_DO.has(low)) continue;
    out.push(["not", "don't", "doesn't", "didn't"].includes(low) ? "NOT" : t);
  }
  return out;
}

function whMovement(ts: string[]): string[] {
  if (ts.length && WH_WORDS.has(ts[0].toLowerCase())) {
    return [...ts.slice(1).filter((t) => !AUX_DO.has(t.toLowerCase())), ts[0]];
  }
  return ts;
}

function posFilter(ts: string[]): string[] {
  return parse(ts.join(" ")).filter((t) => {
    const up = t.text.toUpperCase();
    return SPECIAL.has(up) || SPECIAL_WH.has(up) || t.pos === "PRON" || KEEP_POS.has(t.pos);
  }).map((t) => t.text);
}

function lemmatize(ts: string[]): string[] {
  return parse(ts.join(" ")).map((t) => {
    const up = t.text.toUpperCase();
    return SPECIAL.has(up) || SPECIAL_WH.has(up) ? up : t.lemma.toUpperCase();
  });
}

export function convertV1(sentence: string): string {
  let ts = tokenize(sentence);
  if (hasNegation(ts)) ts = negation(ts);
  const st = sentenceType(ts);
  if (st === "wh_question") ts = whMovement(ts);
  else if (st === "yn_question") ts = ts.filter((t) => !AUX_DO.has(t.toLowerCase()));
  ts = ts.map((t) => (t.toLowerCase() === "will" ? "FUTURE" : t));
  ts = ts.filter((t) => !COPULA_FORMS.has(t.toLowerCase()));
  ts = posFilter(ts);
  ts = lemmatize(ts);
  return ts.join(" ");
}
