import { test } from "node:test";
import assert from "node:assert/strict";
import { segmentsD3 } from "../src/pipeline/decoder.ts";
import { viterbiRescore } from "../src/pipeline/viterbi.ts";

// segmentsD3: TODO §16.2's batch/offline sibling of OnlineDecoder's D3 branch.
// No Python fixture here (unlike decoder.test.ts) -- these check the function's own
// invariants against a synthetic stream, not parity with a cached Python run.

test("segmentsD3 finds maximal runs of below-nu null probability, at least minLen long", () => {
  const NULL = 2; // 2 glosses + null
  const frame = (a: number, b: number, n: number) => new Float64Array([a, b, n]);
  const gp = [
    frame(0.9, 0.05, 0.05), frame(0.8, 0.1, 0.1), // run 0-2, sign A (null low throughout)
    frame(0.0, 0.0, 1.0), // a single null frame, too short a run on its own (min_len=2) -- irrelevant here, it's null anyway
    frame(0.1, 0.85, 0.05), frame(0.05, 0.9, 0.05), frame(0.1, 0.8, 0.1), // run 3-6, sign B
  ];
  const segs = segmentsD3(gp, 0.5, 2, NULL);
  assert.deepEqual(segs.map((s) => [s.start, s.end]), [[0, 2], [3, 6]]);
  assert.equal(segs[0].q.length, NULL);
  assert.ok(Math.abs(segs[0].q.reduce((a, b) => a + b, 0) - 1) < 1e-9, "q sums to 1");
  assert.ok(segs[0].q[0] > segs[0].q[1], "segment 1 votes gloss 0 (the dominant class in its frames)");
  assert.ok(segs[1].q[1] > segs[1].q[0], "segment 2 votes gloss 1 (the dominant class in its frames)");
});

test("segmentsD3 drops a run shorter than minLen", () => {
  const gp = [
    new Float64Array([0.9, 0.1]), // one non-null frame, alone
    new Float64Array([0.0, 1.0]), // null
    new Float64Array([0.9, 0.1]), new Float64Array([0.8, 0.2]), // a real 2-frame run
  ];
  const segs = segmentsD3(gp, 0.5, 2, 1);
  assert.deepEqual(segs.map((s) => [s.start, s.end]), [[2, 4]]);
});

test("segmentsD3 closes a run still open at the stream's end", () => {
  const gp = [new Float64Array([0.9, 0.1]), new Float64Array([0.8, 0.2])];
  const segs = segmentsD3(gp, 0.5, 1, 1);
  assert.deepEqual(segs.map((s) => [s.start, s.end]), [[0, 2]]);
});

// viterbiRescore: exact for k >= the number of distinct candidates, so it can be
// checked against a brute-force search over every top-k combination directly.
function bruteForce(candidates: Float64Array[], prior: (h: readonly number[]) => Float64Array, lam: number,
                    k: number): number[] {
  const tops = candidates.map((q) => Array.from(q.keys()).sort((a, b) => q[b] - q[a]).slice(0, k));
  let best: number[] = [];
  let bestScore = -Infinity;
  const walk = (i: number, hist: number[], score: number) => {
    if (i === candidates.length) {
      if (score > bestScore) { bestScore = score; best = hist; }
      return;
    }
    const p = prior(hist);
    for (const c of tops[i]) {
      const s = score + Math.log(Math.max(candidates[i][c], 1e-12)) + lam * Math.log(Math.max(p[c], 1e-12));
      walk(i + 1, [...hist, c], s);
    }
  };
  walk(0, [], 0);
  return best;
}

test("viterbiRescore matches a brute-force search over top-k combinations", () => {
  // A tiny 3-gloss "bigram" prior: favors repeating the same class, otherwise uniform.
  const prior = (history: readonly number[]): Float64Array => {
    const p = new Float64Array([0.2, 0.2, 0.2]);
    if (history.length) p[history[history.length - 1]] = 0.6;
    const s = p.reduce((a, b) => a + b, 0);
    for (let i = 0; i < p.length; i++) p[i] /= s;
    return p;
  };
  // Segment 1 clearly favors class 0; segment 2 is ambiguous between 0 and 1 -- the
  // "repeat the same class" prior should be able to pull it toward 0.
  const candidates = [
    new Float64Array([0.7, 0.2, 0.1]),
    new Float64Array([0.34, 0.33, 0.33]),
    new Float64Array([0.1, 0.1, 0.8]),
  ];
  for (const lam of [0, 0.3, 1.0, 2.5]) {
    const got = viterbiRescore(candidates, prior, lam, 3, 64);
    const want = bruteForce(candidates, prior, lam, 3);
    assert.deepEqual(got, want, `lam=${lam}`);
  }
});

test("viterbiRescore's prior can flip an otherwise-argmax-ambiguous segment", () => {
  const prior = (history: readonly number[]): Float64Array => {
    const p = new Float64Array([0.2, 0.2, 0.2]);
    if (history.length) p[history[history.length - 1]] = 0.6;
    const s = p.reduce((a, b) => a + b, 0);
    for (let i = 0; i < p.length; i++) p[i] /= s;
    return p;
  };
  // Segment 1 unambiguously class 1. Segment 2 is a near three-way tie where argmax
  // alone (barely) prefers class 0 -- but the "repeat the previous class" prior favors
  // class 1 strongly enough that a high enough lambda should flip the choice.
  const candidates = [new Float64Array([0.1, 0.8, 0.1]), new Float64Array([0.34, 0.331, 0.329])];
  const noPrior = viterbiRescore(candidates, prior, 0, 3, 64);
  const withPrior = viterbiRescore(candidates, prior, 2.0, 3, 64);
  assert.deepEqual(noPrior, [1, 0], "argmax alone: segment 2 goes to its own top class");
  assert.deepEqual(withPrior, [1, 1], "a strong same-class prior pulls segment 2 to repeat class 1 instead");
});
