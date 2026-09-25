/**
 * Beam search, a port of transformers 5.x `GenerationMixin._beam_search` for one sentence
 * (TODO §13): the vectorized version with `beams_to_keep = 2 * num_beams`, finished beams
 * merged by `score / generated_len ** length_penalty`, and the default `early_stopping=False`
 * heuristic. Logits processors, in HF's order on log-probs: `repetition_penalty`
 * (`RepetitionPenaltyLogitsProcessor`) then `no_repeat_ngram_size`
 * (`NoRepeatNGramLogitsProcessor`). Scores are kept in float32 (`Math.fround`) as in torch.
 *
 * `logits(seqs)` returns the last-position logits of every running sequence (num_beams × V).
 * Parity with `model.generate()`: `test/speech.test.ts` (fp32 ONNX, onnxruntime-node).
 */

export interface GenerateOptions {
  numBeams: number;
  maxLength: number;
  eosTokenId: number;
  decoderStartTokenId: number;
  repetitionPenalty?: number;
  noRepeatNgramSize?: number;
  lengthPenalty?: number;
}

const f = Math.fround;
const NEG = -1.0e9;

function logSoftmax(row: Float32Array): Float32Array {
  let max = -Infinity;
  for (const v of row) if (v > max) max = v;
  let sum = 0;
  for (const v of row) sum += Math.exp(v - max);
  const lse = max + Math.log(sum);
  const out = new Float32Array(row.length);
  for (let j = 0; j < row.length; j++) out[j] = row[j] - lse;
  return out;
}

function process(lp: Float32Array, seq: number[], o: GenerateOptions): void {
  const pen = o.repetitionPenalty ?? 1.0;
  if (pen !== 1.0) {
    for (const t of new Set(seq)) lp[t] = lp[t] < 0 ? f(lp[t] * pen) : f(lp[t] / pen);
  }
  const n = o.noRepeatNgramSize ?? 0;
  if (n > 0 && seq.length >= n) {
    const prefix = seq.slice(seq.length + 1 - n);
    for (let s = 0; s + n <= seq.length; s++) {
      let hit = true;
      for (let k = 0; k < n - 1; k++) if (seq[s + k] !== prefix[k]) { hit = false; break; }
      if (hit) lp[seq[s + n - 1]] = -Infinity;
    }
  }
}

/** Indices of the `k` largest values, largest first (ties: lower index first). */
function topk(values: ArrayLike<number>, k: number): number[] {
  const idx: number[] = [];
  const val: number[] = [];
  for (let j = 0; j < values.length; j++) {
    const v = values[j];
    if (idx.length === k && v <= val[k - 1]) continue;
    let p = idx.length === k ? k - 1 : idx.length;
    while (p > 0 && val[p - 1] < v) p--;
    idx.splice(p, 0, j);
    val.splice(p, 0, v);
    if (idx.length > k) { idx.pop(); val.pop(); }
  }
  return idx;
}

export async function beamSearch(logits: (seqs: number[][]) => Promise<Float32Array[]>,
  o: GenerateOptions): Promise<number[]> {
  const nb = o.numBeams;
  const K = 2 * nb;
  const lpen = o.lengthPenalty ?? 1.0;
  const prompt = 1;
  let running: number[][] = Array.from({ length: nb }, () => [o.decoderStartTokenId]);
  let runScores: number[] = Array.from({ length: nb }, (_, b) => (b === 0 ? 0 : NEG));
  let fin = Array.from({ length: nb }, () => ({ seq: [o.decoderStartTokenId], score: NEG, done: false }));
  let unsatisfied = true;
  let curLen = 1;
  for (;;) {
    const rows = await logits(running);
    const V = rows[0].length;
    const acc = new Float32Array(nb * V);
    for (let b = 0; b < nb; b++) {
      const lp = logSoftmax(rows[b]);
      process(lp, running[b], o);
      for (let j = 0; j < V; j++) acc[b * V + j] = lp[j] + runScores[b];
    }
    const cand = topk(acc, K).map((flat) => {
      const beam = Math.floor(flat / V);
      const tok = flat % V;
      return { seq: [...running[beam], tok], val: acc[flat], hit: tok === o.eosTokenId || curLen + 1 >= o.maxLength };
    });
    // next running beams: the best num_beams candidates that did not stop
    const runVals = cand.map((c) => f(c.val + (c.hit ? NEG : 0)));
    const keep = topk(runVals, nb);
    running = keep.map((k) => cand[k].seq);
    runScores = keep.map((k) => runVals[k]);
    // finished beams
    const denom = (curLen + 1 - prompt) ** lpen;
    const merged = [...fin];
    cand.forEach((c, k) => {
      const done = c.hit && k < nb;
      let s = f(c.val / denom);
      s = f(s + (unsatisfied ? 0 : NEG));
      s = f(s + (done ? 0 : NEG));
      merged.push({ seq: c.seq, score: s, done });
    });
    fin = topk(merged.map((m) => m.score), nb).map((k) => merged[k]);
    curLen += 1;
    const best = f(runScores[0] / (curLen - prompt) ** lpen);
    const worst = Math.min(...fin.map((m) => m.score));
    unsatisfied = unsatisfied && fin.some((m) => best > (m.done ? worst : NEG));
    if (!unsatisfied || cand.every((c) => c.hit)) break;
  }
  return fin[0].seq;
}
