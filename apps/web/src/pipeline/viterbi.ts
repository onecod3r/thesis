/**
 * Whole-sentence decoding over a recognizer's per-segment top-k candidates
 * (TODO §16.2): a port of `sb.rescore.prior.viterbi_rescore`.
 *
 * `fuse.decide`/`decoder.ts`'s `decide` is greedy: it commits each segment before it
 * has seen the next one. This instead searches every combination of each segment's
 * top-`k` candidates for the single sequence of choices that scores best as a *whole
 * sentence* under the n-gram, by dynamic programming -- exact for this `k` and n-gram
 * order (a trigram's branching factor per step is exactly `k`), not a heuristic. Used
 * for the record-then-recognize mode, which has the whole clip (and so every segment)
 * in hand before deciding anything -- no live-decoding latency to trade off, unlike the
 * streaming lattice `LatticeRunner` implements.
 */

import type { Prior } from "./decoder.ts";

/** One segment's classifier vote (probabilities over glosses, sums to 1) -> the chosen
 * class per segment, picked to maximize `sum_i log(q_i[c_i]) + lam * log(prior[c_i]
 * given the choices before it)` over the whole sentence. `beamWidth` bounds the search;
 * with `k` candidates per step an n-gram's reachable state count is exactly `k`, so this
 * is exact in the common case (`beamWidth >= k`), not an approximation. */
export function viterbiRescore(
  candidates: readonly Float32Array[] | readonly Float64Array[],
  prior: Prior,
  lam: number,
  k = 5,
  beamWidth = 64,
): number[] {
  let beams: { hist: number[]; score: number }[] = [{ hist: [], score: 0 }];
  for (const q of candidates) {
    const top = Array.from(q.keys()).sort((a, b) => q[b] - q[a]).slice(0, k);
    const scored = new Map<string, { hist: number[]; score: number }>();
    for (const { hist, score } of beams) {
      const p = prior(hist);
      for (const c of top) {
        const s = score + Math.log(Math.max(q[c], 1e-12)) + lam * Math.log(Math.max(p[c], 1e-12));
        const newHist = [...hist, c];
        const key = newHist.join(",");
        const cur = scored.get(key);
        if (!cur || cur.score < s) scored.set(key, { hist: newHist, score: s });
      }
    }
    beams = Array.from(scored.values()).sort((a, b) => b.score - a.score).slice(0, beamWidth);
  }
  if (!beams.length) return [];
  return beams.reduce((best, b) => (b.score > best.score ? b : best)).hist;
}
