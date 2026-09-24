/**
 * The next-gloss prior: interpolated Kneser-Ney n-gram, a port of
 * `sb.rescore.prior.NgramLM` reading its `to_dict()` tables (`prior.json`).
 * `test/prior.test.ts` checks it against Python on hundreds of histories.
 */

import type { NgramJson } from "../../../shared-ts/src/contracts.ts";

const BOS = "<s>";

export class NgramPrior {
  private readonly vocab: string[];
  private readonly index: Map<string, number>;
  private readonly order: number;
  private readonly discount: number;
  private readonly tables: Map<number, Map<string, [number, number][]>>;
  private readonly levelMemo = new Map<string, Float64Array>();
  private readonly distMemo = new Map<string, Float64Array>();
  /** label index (model order) -> position in the prior's vocabulary */
  private readonly perm: Int32Array;
  private readonly labels: readonly string[];

  constructor(json: NgramJson, labels: readonly string[]) {
    if (json.format !== "kn-ngram/1") throw new Error(`unknown prior format ${json.format}`);
    this.vocab = json.vocab;
    this.index = new Map(json.vocab.map((g, i) => [g, i]));
    this.order = json.order;
    this.discount = json.discount;
    this.tables = new Map();
    for (const [k, t] of Object.entries(json.tables)) {
      const m = new Map<string, [number, number][]>();
      for (const [ctx, counts] of Object.entries(t)) {
        m.set(ctx, Object.entries(counts).map(([w, c]) => [this.index.get(w)!, c]));
      }
      this.tables.set(Number(k), m);
    }
    this.labels = labels;
    this.perm = Int32Array.from(labels, (g) => {
      const i = this.index.get(g);
      if (i === undefined) throw new Error(`gloss ${g} is not in the prior's vocabulary`);
      return i;
    });
  }

  private level(k: number, ctx: readonly string[]): Float64Array {
    const n = this.vocab.length;
    if (k === 0) return new Float64Array(n).fill(1 / n);
    const key = `${k}|${ctx.join(" ")}`;
    const hit = this.levelMemo.get(key);
    if (hit) return hit;
    const lower = this.level(k - 1, ctx.slice(1));
    const cnt = this.tables.get(k)?.get(ctx.join(" "));
    let out: Float64Array;
    if (!cnt || cnt.length === 0) {
      out = lower;
    } else {
      let total = 0;
      for (const [, c] of cnt) total += c;
      out = new Float64Array(n);
      for (const [w, c] of cnt) out[w] = Math.max(c - this.discount, 0) / total;
      const back = (this.discount * cnt.length) / total;
      for (let i = 0; i < n; i++) out[i] += back * lower[i];
    }
    this.levelMemo.set(key, out);
    return out;
  }

  /** P(next token | history) over the prior's vocabulary (glosses + end of sentence). */
  dist(history: readonly string[]): Float64Array {
    const toks = [...new Array(this.order - 1).fill(BOS), ...history];
    const ctx = this.order > 1 ? toks.slice(toks.length - (this.order - 1)) : [];
    return this.level(this.order, ctx);
  }

  /** The decoder's `Prior`: history of label ids -> distribution over labels, end-of-sentence removed. */
  readonly prior = (history: readonly number[]): Float64Array => {
    const key = history.join(",");
    const hit = this.distMemo.get(key);
    if (hit) return hit;
    const d = this.dist(history.map((c) => this.labels[c]));
    let s = 0;
    for (let i = 0; i < d.length - 1; i++) s += d[i];
    const out = new Float64Array(this.perm.length);
    for (let c = 0; c < this.perm.length; c++) out[c] = d[this.perm[c]] / s;
    this.distMemo.set(key, out);
    return out;
  };
}
