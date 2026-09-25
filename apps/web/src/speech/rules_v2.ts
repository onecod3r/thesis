/**
 * English -> ASL gloss, rule engine v2 (`sb.synthesize.gloss.rules_v2`) on wink-nlp.
 *
 * Same item kinds, clause split and ordering as the Python engine. spaCy's dependency labels
 * and Penn tags are not available, so each test that reads them is replaced by a local
 * heuristic, named after the spaCy test it stands in for:
 *
 * | spaCy | here |
 * |---|---|
 * | conj: `CCONJ` or dep `mark`/`cc` | `CCONJ`/`SCONJ`; a clause-introducing `ADP` (before/after/until/since + a verb later in the clause); `when` is never a conj (spaCy parses it as `advmod`, so it is a wh-word there too) |
 * | particle: dep `prt` | a verb + particle pair from `PHRASAL`, or a particle right after its verb (or verb + pronoun) at the end of a clause |
 * | wh `det` head | the noun right after the wh-word |
 * | relative time: a child in this/last/next/every | the previous token |
 * | weather `it` (head or head's head) | `it` followed, across auxiliaries, by a weather verb |
 * | `JJR`/`JJS` | an adjective whose lemma differs from its text |
 * | `NNS` not ending in s | a noun whose lemma differs from its text and that doesn't end in s |
 *
 * Python stays authoritative: `test/speech.test.ts` reports the agreement rate.
 */

import { parse } from "./nlp.ts";
import type { Tok } from "./nlp.ts";

const WH_WORDS = new Set(["who", "what", "where", "when", "why", "how", "which"]);
const TIME_FRONT = new Set(["yesterday", "today", "tomorrow", "tonight", "now", "morning", "afternoon", "evening",
  "night", "weekend", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]);
const TIME_RELATIVE = new Set(["week", "month", "year"]);
const RELATIVE_MARKERS = new Set(["this", "last", "next", "every"]);
const TIME_IN_PLACE = new Set(["later", "soon", "recently", "already"]);
const KEEP_CONJ = new Set(["if", "but", "because", "before", "after", "when", "while", "so", "or", "and", "until",
  "although", "though", "since"]);
const KEEP_INTJ = new Set(["please", "yes", "no", "hello", "hi", "thanks", "sorry", "ok", "okay"]);
const WEATHER_VERBS = ["rain", "snow", "storm", "hail", "drizzle", "pour", "thunder"];
const MODALS: Record<string, string> = { can: "CAN", ca: "CAN", could: "COULD", should: "SHOULD", must: "MUST",
  might: "MIGHT", may: "MAY", would: "WOULD" };
const FUTURE_FORMS = new Set(["will", "'ll", "wo", "shall"]);
const PRONOUN_MAP: Record<string, string> = { i: "ME", me: "ME", myself: "MYSELF" };
const DET_KEEP = new Set(["this", "that", "these", "those", "all", "some", "many", "every", "each", "other"]);
const SUBJECTS = new Set(["i", "you", "he", "she", "we", "they", "it", "there"]);
const CLAUSE_ADP = new Set(["before", "after", "until", "since", "because"]);
const PARTICLES = new Set(["up", "down", "off", "on", "out", "away", "back", "over", "in", "around", "about",
  "through", "along", "by", "together", "apart", "aside", "forward"]);
// verb lemma + particle pairs spaCy parses as `prt` even before an object ("turn off the light").
const PHRASAL = new Set(("turn:off turn:on turn:up turn:down turn:around turn:over wake:up get:up get:on get:off " +
  "get:out get:back get:over sit:down stand:up give:up give:back give:away pick:up look:up look:after put:on " +
  "put:off put:away put:down put:back take:off take:out take:over take:back come:back come:in come:on come:over " +
  "come:out come:up go:out go:back go:away go:on go:over go:down go:up find:out fill:out fill:in clean:up " +
  "shut:up shut:down hang:up hang:out calm:down slow:down grow:up show:up set:up break:down break:up run:out " +
  "run:away throw:away throw:out work:out figure:out check:in check:out log:in log:out sign:up sign:in " +
  "sign:out write:down hurry:up cheer:up dress:up eat:out back:up catch:up make:up mix:up pay:back " +
  "call:back bring:back bring:up carry:on keep:up let:down lie:down move:on move:out move:in pass:away " +
  "point:out switch:on switch:off try:on turn:in warm:up zip:up hold:on"
).split(" "));

type Kind = "word" | "time" | "time_in_place" | "wh" | "modal" | "future" | "conj" | "neg" | "yet";
interface Item { g: string; k: Kind; i?: number; head?: number | null }

const isWeather = (t: Tok) => WEATHER_VERBS.includes(t.lemma) || WEATHER_VERBS.some((w) => t.lower.startsWith(w));

function relativeTime(toks: Tok[], t: Tok): boolean {
  const prev = toks[t.i - 1];
  return TIME_RELATIVE.has(t.lemma) && prev !== undefined && RELATIVE_MARKERS.has(prev.lower);
}

function clauseHasVerb(toks: Tok[], from: number): boolean {
  for (let j = from; j < toks.length; j++) {
    const t = toks[j];
    if (t.isPunct || t.pos === "CCONJ") return false;
    if (t.pos === "VERB" || t.pos === "AUX") return true;
  }
  return false;
}

function isConj(toks: Tok[], t: Tok): boolean {
  if (!KEEP_CONJ.has(t.lower) || t.lower === "when") return false;
  if (t.pos === "CCONJ" || t.pos === "SCONJ") return true;
  if (t.pos === "ADP" && CLAUSE_ADP.has(t.lower)) {
    const next = toks[t.i + 1];
    return next !== undefined && (SUBJECTS.has(next.lower) || clauseHasVerb(toks, t.i + 1));
  }
  return false;
}

function isParticle(toks: Tok[], t: Tok): boolean {
  if (!PARTICLES.has(t.lower)) return false;
  const prev = toks[t.i - 1];
  const prev2 = toks[t.i - 2];
  const verb = prev?.pos === "VERB" ? prev : prev?.pos === "PRON" && prev2?.pos === "VERB" ? prev2 : undefined;
  if (!verb) return false;
  if (PHRASAL.has(`${verb.lemma}:${t.lower}`)) return true;
  const next = toks[t.i + 1];
  return next === undefined || next.isPunct || next.pos === "CCONJ" || next.pos === "SCONJ";
}

function weatherIt(toks: Tok[], t: Tok): boolean {
  for (let j = t.i + 1; j < toks.length; j++) {
    const n = toks[j];
    if (n.pos === "AUX" || n.pos === "PART") continue;
    return (n.pos === "VERB" || n.pos === "NOUN") && isWeather(n);
  }
  return false;
}

function items(toks: Tok[]): Item[][] {
  const clauses: Item[][] = [[]];
  for (const t of toks) {
    const low = t.lower;
    const lemma = t.lemma;
    const cur = clauses[clauses.length - 1];
    if (t.isPunct) continue;
    if (isConj(toks, t)) {
      if (cur.length) clauses.push([]);
      clauses[clauses.length - 1].push({ g: low.toUpperCase(), k: "conj", i: t.i });
      continue;
    }
    if (KEEP_INTJ.has(low) && ["INTJ", "ADV", "VERB", "DET"].includes(t.pos)) {
      cur.push({ g: low.toUpperCase(), k: "word", i: t.i });
      continue;
    }
    if (low === "not" || low === "n't" || lemma === "not") {
      const last = cur[cur.length - 1];
      if (last && last.k === "modal" && last.g === "CAN") cur[cur.length - 1] = { g: "CAN'T", k: "word", i: t.i };
      else cur.push({ g: "NOT", k: "neg", i: t.i });
      continue;
    }
    if (low === "never") { cur.push({ g: "NEVER", k: "neg", i: t.i }); continue; }
    if (low === "yet") { cur.push({ g: "YET", k: "yet", i: t.i }); continue; }
    if (FUTURE_FORMS.has(low) && (t.pos === "AUX" || t.pos === "VERB")) {
      cur.push({ g: "FUTURE", k: "future", i: t.i });
      continue;
    }
    if (low in MODALS && t.pos === "AUX") { cur.push({ g: MODALS[low], k: "modal", i: t.i }); continue; }
    if (lemma === "be" && (t.pos === "AUX" || t.pos === "VERB")) continue;
    if ((lemma === "do" || lemma === "have") && t.pos === "AUX") continue;
    if (low === "to" && t.pos === "PART") continue;
    if (low === "o'clock" || low === "oclock") {
      const last = cur[cur.length - 1];
      if (last && last.k === "word" && toks[t.i - 1]?.pos === "NUM") {
        const num = cur.pop()!.g;
        cur.push({ g: "TIME", k: "time", i: t.i }, { g: num, k: "time", i: t.i });
      }
      continue;
    }
    if (WH_WORDS.has(low) && ["PRON", "ADV", "DET", "SCONJ"].includes(t.pos)) {
      const next = toks[t.i + 1];
      const head = t.pos === "DET" && next && next.pos === "NOUN" ? next.i : null;
      cur.push({ g: low.toUpperCase(), k: "wh", i: t.i, head });
      continue;
    }
    if (TIME_FRONT.has(lemma) || TIME_FRONT.has(low) || relativeTime(toks, t)) {
      cur.push({ g: (TIME_RELATIVE.has(lemma) ? lemma : low).toUpperCase(), k: "time", i: t.i });
      continue;
    }
    if ((RELATIVE_MARKERS.has(low) || low === "that") && ["DET", "ADJ"].includes(t.pos)) {
      const next = toks[t.i + 1];
      if (next && (TIME_FRONT.has(next.lemma) || (TIME_RELATIVE.has(next.lemma) && RELATIVE_MARKERS.has(low)))) {
        cur.push({ g: low.toUpperCase(), k: "time", i: t.i });
        continue;
      }
    }
    if (t.pos === "DET") {
      if (low === "no") cur.push({ g: "NO", k: "neg", i: t.i });
      else if (DET_KEEP.has(low)) cur.push({ g: low.toUpperCase(), k: "word", i: t.i });
      continue;
    }
    if (t.pos === "ADP" || t.pos === "PART") {
      if (isParticle(toks, t)) cur.push({ g: low.toUpperCase(), k: "word", i: t.i });
      continue;
    }
    if (t.pos === "SCONJ" || t.pos === "CCONJ") continue;
    if (t.pos === "PRON") {
      if (low === "it" && weatherIt(toks, t)) continue;
      cur.push({ g: PRONOUN_MAP[low] ?? low.toUpperCase(), k: "word", i: t.i });
      continue;
    }
    if (t.pos === "NUM" || t.pos === "PROPN") { cur.push({ g: t.text.toUpperCase(), k: "word", i: t.i }); continue; }
    if (t.pos === "ADJ" && lemma !== low) { cur.push({ g: t.text.toUpperCase(), k: "word", i: t.i }); continue; }
    if (t.pos === "NOUN" && lemma !== low && !low.endsWith("s")) {
      cur.push({ g: t.text.toUpperCase(), k: "word", i: t.i });
      continue;
    }
    if (["NOUN", "VERB", "ADJ", "ADV", "INTJ", "AUX"].includes(t.pos)) {
      cur.push({ g: lemma.toUpperCase(), k: TIME_IN_PLACE.has(lemma) ? "time_in_place" : "word", i: t.i });
    }
  }
  return clauses.filter((c) => c.length);
}

function orderClause(its: Item[], stype: string, first: boolean): string[] {
  const conj = its.filter((it) => it.k === "conj");
  const time = its.filter((it) => it.k === "time");
  let rest = its.filter((it) => it.k !== "conj" && it.k !== "time");
  if (rest.some((it) => it.k === "yet") && rest.some((it) => it.g === "NOT")) {
    rest = rest.filter((it) => it.k !== "yet").map((it) => (it.g === "NOT" ? { g: "NOT YET", k: "neg" } : it));
  }
  const tail: Item[] = [];
  if (stype === "wh") {
    const wh = rest.find((it) => it.k === "wh");
    if (wh) {
      rest.splice(rest.indexOf(wh), 1);
      const noun = rest.find((it) => it.i !== undefined && it.i === wh.head);
      if (noun) {
        rest.splice(rest.indexOf(noun), 1);
        tail.push(noun);
      }
      tail.push(wh);
    }
  }
  if (stype === "yn" && first && rest.length && rest[0].k === "modal") tail.push(rest.shift()!);
  return [...time, ...conj, ...rest, ...tail].map((it) => it.g);
}

function sentenceType(toks: Tok[], text: string): "wh" | "yn" | "declarative" {
  const words = toks.filter((t) => !t.isPunct);
  if (!words.length) return "declarative";
  const first = words[0].lower;
  const isQ = text.trim().endsWith("?");
  const hasWh = words.some((w) => WH_WORDS.has(w.lower));
  if (WH_WORDS.has(first)) return "wh";
  if (KEEP_INTJ.has(first) || KEEP_CONJ.has(first)) return isQ && hasWh ? "wh" : "declarative";
  if (words[0].pos === "AUX" && (isQ || ["do", "does", "did", "can", "will", "should", "is", "are"].includes(first))) return "yn";
  return isQ && hasWh ? "wh" : "declarative";
}

export function convertV2(sentence: string): string {
  const text = sentence.trim();
  const toks = parse(text);
  const stype = sentenceType(toks, text);
  const clauses = items(toks);
  const hasTime = clauses.some((c) => c.some((it) => it.k === "time" || it.k === "time_in_place"));
  const out: string[] = [];
  clauses.forEach((c, n) => {
    if (hasTime) c = c.filter((it) => it.k !== "future");
    out.push(...orderClause(c, stype, n === 0));
  });
  return out.filter((g) => g).join(" ");
}
