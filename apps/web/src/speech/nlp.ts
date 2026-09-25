/**
 * spaCy stand-in for the browser (TODO §13, user decision 2026-09-25: speech -> gloss runs
 * client-side on the Workers Free plan). Pyodide has no spaCy build, so the rule engines run
 * on **wink-nlp** (`wink-eng-lite-web-model`, MIT): UD POS tags and lemmas, contractions split
 * the way spaCy splits them (`ca` + `n't`). It has **no dependency parse and no Penn tags**:
 * `rules_v2.ts` replaces each `dep_`/`tag_` test with a local heuristic, and
 * `test/speech.test.ts` measures how often the ports agree with Python's spaCy engines.
 */

import winkNLP from "wink-nlp";
import type { ItemToken } from "wink-nlp";
import model from "wink-eng-lite-web-model";

export interface Tok {
  i: number;
  text: string;
  lower: string;
  lemma: string;
  pos: string; // UD: NOUN VERB AUX PRON DET ADP PART ...
  isPunct: boolean;
}

let nlp: ReturnType<typeof winkNLP> | null = null;

// spaCy's lemmatizer maps object pronouns to the subject form ("me" -> "I"); wink keeps them.
const PRON_LEMMA: Record<string, string> = { me: "i", him: "he", us: "we", them: "they" };
// Prepositions spaCy tags ADV when nothing follows ("play outside.").
const STRANDED_ADV = new Set(["outside", "inside", "upstairs", "downstairs", "abroad", "ahead", "behind", "nearby",
  "underneath", "home", "overseas", "indoors", "outdoors", "around", "later"]);
const SUBJ = new Set(["i", "you", "he", "she", "we", "they", "it", "there", "this", "that"]);
const OPEN = new Set(["NOUN", "ADJ", "PROPN", "NUM"]);

const clauseEnd = (t: Tok | undefined) => t === undefined || t.isPunct || t.pos === "CCONJ" || t.pos === "SCONJ";

/** Where wink's tagger and spaCy's `en_core_web_sm` systematically disagree, move to spaCy's
 * answer. Each rule was found from `test/speech.test.ts` mismatches (2026-09-25). */
function spacyish(ts: Tok[]): Tok[] {
  for (const t of ts) {
    const prev = ts[t.i - 1];
    const next = ts[t.i + 1];
    if (t.pos === "PRON") {
      if (t.lower in PRON_LEMMA) t.lemma = PRON_LEMMA[t.lower];
      else if (t.lower === "her") t.lemma = next && OPEN.has(next.pos) ? "her" : "she"; // possessive keeps "her"
    }
    // "have"/"do" as the main verb ("we have a meeting", "I do my homework"): AUX only before a verb.
    // AUX stays when a verb (or another auxiliary) follows in the clause, and for do-support + not.
    if (t.pos === "AUX" && ["have", "do"].includes(t.lemma)) {
      const neg = next && (next.lower === "not" || next.lower === "n't");
      let verb = false;
      for (let j = t.i + 1; j < ts.length && !clauseEnd(ts[j]); j++) {
        if (ts[j].pos === "VERB" || ts[j].pos === "AUX") { verb = true; break; }
      }
      if (!verb && !(neg && t.lemma === "do")) t.pos = "VERB";
    }
    if (t.lower === "being" || t.lower === "been") { t.lemma = "be"; t.pos = "AUX"; }
    // "like" after a subject, a negation or an auxiliary is the verb.
    if (t.lower === "like" && t.pos === "ADP" && prev
        && (prev.pos === "PRON" || prev.pos === "AUX" || prev.lower === "not" || prev.lower === "n't" || prev.text === "NOT")) {
      t.pos = "VERB";
      t.lemma = "like";
    }
    if (t.pos === "ADP" && STRANDED_ADV.has(t.lower) && clauseEnd(next)) t.pos = "ADV";
    // complementizer "that" ("told me that I ...", "that the ..."): spaCy SCONJ, wink DET/PRON.
    if (t.lower === "that" && (t.pos === "DET" || t.pos === "PRON") && next
        && ((next.pos === "PRON" && SUBJ.has(next.lower) && next.lower !== "that") || next.pos === "DET")) {
      t.pos = "SCONJ";
    }
  }
  return ts;
}

export function parse(text: string): Tok[] {
  nlp ??= winkNLP(model);
  const its = nlp.its;
  const doc = nlp.readDoc(text);
  const out: Tok[] = [];
  doc.tokens().each((t: ItemToken) => {
    const ttype = t.out(its.type) as string;
    if (ttype === "tabCRLF" || ttype === "spacing") return;
    const tx = t.out() as string;
    const pos = t.out(its.pos) as string;
    // wink-nlp's own .d.ts mistypes `its.lemma` against the SpanItsFunction overload of `out()`.
    const lemma = t.out(its.lemma as Parameters<typeof t.out>[0]) as string;
    out.push({ i: out.length, text: tx, lower: tx.toLowerCase(), lemma: String(lemma ?? tx).toLowerCase(), pos,
      isPunct: pos === "PUNCT" || ttype === "punctuation" });
  });
  return spacyish(out);
}
