/**
 * Exact ports of the small text helpers of `sb.synthesize` (TODO §13). Parity: `test/speech.test.ts`.
 *
 * - `spellNumbers`: `sb.synthesize.asr.spell_numbers` (Whisper writes digits; the gloss
 *   engines expect words: "3 o'clock" -> "three o'clock").
 * - `splitSentences`: `sb.synthesize.gloss.split_sentences` (the team's `apply_rules_multi` split).
 * - `buildInput`: `sb.synthesize.gloss.t5.build_input`, T5's training-time prompt, word for word.
 */

const ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
  "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

function words(n: number): string {
  if (n < 20) return ONES[n];
  if (n < 100) return TENS[Math.floor(n / 10)] + (n % 10 === 0 ? "" : " " + ONES[n % 10]);
  if (n < 1000) return ONES[Math.floor(n / 100)] + " hundred" + (n % 100 === 0 ? "" : " " + words(n % 100));
  if (n < 1_000_000) return words(Math.floor(n / 1000)) + " thousand" + (n % 1000 === 0 ? "" : " " + words(n % 1000));
  return String(n);
}

export function spellNumbers(text: string): string {
  text = text.replace(/\b(\d{1,2}):(\d{2})\b/g,
    (_, h: string, m: string) => words(Number(h)) + (m === "00" ? "" : " " + words(Number(m))));
  return text.replace(/(?<![\d.])(\d{1,3}(?:,\d{3})+|\d+)(?![\d.]|\.\d)/g,
    (_, d: string) => words(Number(d.replace(/,/g, ""))));
}

export function splitSentences(text: string): string[] {
  return text.split(/[.!?]+/).map((s) => s.trim()).filter((s) => s);
}

export function buildInput(english: string, ruleGloss: string): string {
  return "English: " + english + " Rule gloss: " + ruleGloss + " Produce ASL gloss:";
}

/** Python's `str.split()`: split on runs of whitespace, no empty strings. */
export function pySplit(s: string): string[] {
  return s.split(/\s+/).filter((t) => t);
}

/** Python's `str.strip(chars)`. */
export function pyStrip(s: string, chars: string): string {
  let a = 0;
  let b = s.length;
  while (a < b && chars.includes(s[a])) a++;
  while (b > a && chars.includes(s[b - 1])) b--;
  return s.slice(a, b);
}
