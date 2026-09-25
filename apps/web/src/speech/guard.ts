/**
 * The guarded hybrid, an exact port of `sb.synthesize.gloss.hybrid` (TODO §13): accept T5's
 * gloss only when it keeps the rule gloss's persons, polarity and content words, repeats
 * nothing and invents nothing; otherwise use rules_v2. Parity: `test/speech.test.ts`.
 */

import { pySplit, pyStrip } from "./text.ts";

const PERSON: Record<number, Set<string>> = {
  1: new Set(["I", "ME", "MY", "MINE", "MYSELF", "WE", "US", "OUR", "OURS", "OURSELVES"]),
  2: new Set(["YOU", "YOUR", "YOURS", "YOURSELF", "YOURSELVES"]),
  3: new Set(["HE", "HIM", "HIS", "HIMSELF", "SHE", "HER", "HERS", "HERSELF", "THEY", "THEM", "THEIR", "THEIRS",
    "THEMSELVES"]),
};
const NEGATION = new Set(["NOT", "NEVER", "CAN'T", "NO", "NONE", "NOTHING", "NOBODY"]);
const FUNCTION = new Set([...Object.values(PERSON).flatMap((s) => [...s]), ...NEGATION,
  "FUTURE", "IT", "BE", "DO", "TO", "THE", "A", "AN"]);
const MARKERS = new Set(["ME", "FUTURE", "NOT", "PLEASE", "IF", "BUT", "BEFORE", "AFTER", "BECAUSE", "WHY", "WHAT",
  "WHERE", "WHEN", "WHO", "HOW", "WHICH", "TIME"]);

const toks = (g: string) => pySplit(g.toUpperCase().replaceAll(",", " "));

function persons(ts: string[]): Set<number> {
  const out = new Set<number>();
  for (const t of ts) for (const [p, words] of Object.entries(PERSON)) if (words.has(t)) out.add(Number(p));
  return out;
}

const sameSet = <T>(a: Set<T>, b: Set<T>) => a.size === b.size && [...a].every((x) => b.has(x));

function match(tok: string, pool: Set<string>, n = 4): boolean {
  if (pool.has(tok)) return true;
  if (tok.length < n) return false;
  for (const p of pool) if (p.length >= n && tok.slice(0, n) === p.slice(0, n)) return true;
  return false;
}

function counter(ts: string[]): Map<string, number> {
  const c = new Map<string, number>();
  for (const t of ts) c.set(t, (c.get(t) ?? 0) + 1);
  return c;
}

function sourceWords(english: string): string[] {
  const out: string[] = [];
  for (const raw of pySplit(english)) {
    const w = pyStrip(raw, ".,!?;:\"'").toUpperCase();
    if (w === "CANNOT") out.push("CAN", "NOT");
    else if (w.endsWith("N'T")) {
      const stem = w.slice(0, -3);
      out.push(({ WON: "WILL", CA: "CAN", SHAN: "SHALL" } as Record<string, string>)[stem] ?? stem, "NOT");
    } else if (w) out.push(w);
  }
  return out.map((w) => (w === "I" ? "ME" : w));
}

export interface GuardResult { accepted: boolean; reasons: string[] }

export function guard(english: string, ruleGloss: string, t5Gloss: string, minContentRecall = 0.6): GuardResult {
  const rule = toks(ruleGloss).map((t) => (t === "I" ? "ME" : t));
  const t5 = toks(t5Gloss);
  const words = sourceWords(english);
  const src = new Set([...words, ...words.map((w) => w.replaceAll("'", ""))]);
  const reasons: string[] = [];
  if (!t5.length) return { accepted: false, reasons: ["empty"] };
  const rc = counter(rule), tc = counter(t5), sc = counter(words);
  if ([...tc].some(([t, n]) => n > Math.max(rc.get(t) ?? 0, sc.get(t) ?? 0, 1))) reasons.push("repetition");
  if (!sameSet(persons(rule), persons(t5))) reasons.push("person");
  if (rule.some((t) => NEGATION.has(t)) !== t5.some((t) => NEGATION.has(t))) reasons.push("polarity");
  const content = rule.filter((t) => !FUNCTION.has(t));
  if (content.length) {
    const t5set = new Set(t5);
    const kept = content.filter((t) => match(t, t5set)).length / content.length;
    if (kept < minContentRecall) reasons.push("content");
  }
  const pool = new Set([...src, ...rule]);
  const invented = t5.filter((t) => !MARKERS.has(t) && !match(t, pool));
  if (invented.length) reasons.push("invented:" + invented.join("+"));
  return { accepted: !reasons.length, reasons };
}

export interface HybridOutput {
  gloss: string;
  source: "t5" | "rules_v2";
  ruleV1: string;
  t5: string | null;
  reasons: string[];
}

/** `hybrid.combine`: `t5Gloss === null` means no checkpoint (or not loaded yet). */
export function combine(english: string, ruleV1: string, t5Gloss: string | null, fallback: string,
  minContentRecall = 0.6): HybridOutput {
  if (t5Gloss === null) return { gloss: fallback, source: "rules_v2", ruleV1, t5: null, reasons: ["no_t5"] };
  const g = guard(english, ruleV1, t5Gloss, minContentRecall);
  return { gloss: g.accepted ? t5Gloss : fallback, source: g.accepted ? "t5" : "rules_v2", ruleV1, t5: t5Gloss,
    reasons: g.reasons };
}
