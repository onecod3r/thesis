/**
 * The live decoder, one frame at a time: D3 segmentation + fused acceptance.
 *
 * A port of `sb.recognize.continuous.online.OnlineDecoder`, `fuse.decide`
 * (`none` / `rescore` / `agree`) and `select.Lattice` (fixed-lag look-ahead).
 * Python is authoritative. `test/decoder.test.ts` replays C1's cached outputs
 * through both and requires identical emissions.
 *
 * Arithmetic follows numpy where it matters for ties: the per-frame vote term
 * `w * p` is rounded to float32 (numpy's result type for a Python float times a
 * float32 array), and the accumulators are float64.
 */

export type Emission = { cls: number; frame: number; conf: number };
export type Prior = (history: readonly number[]) => Float64Array;

export interface Segment {
  start: number;
  end: number; // exclusive
  q: Float64Array; // recognizer vote over glosses, sums to 1
  mass: number;
  peak: Float64Array;
}

export interface Rule {
  kind: "rule";
  mode: "none" | "rescore" | "agree";
  lam: number;
  theta: number;
  k: number;
  theta_lo: number;
  theta_hi: number;
  max_len: number | null;
}

export interface Lattice {
  kind: "lattice";
  theta: number;
  lam: number;
  k: number;
  lag: number;
  max_len: number | null;
}

export const D3_ONLY: Rule = { kind: "rule", mode: "none", lam: 0, theta: 0, k: 5, theta_lo: 0, theta_hi: 1.01, max_len: null };

function argmax(v: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}

/** `q · p^lam`, normalized, computed in log space. */
export function fused(q: Float64Array, p: Float64Array, lam: number): Float64Array {
  const lf = new Float64Array(q.length);
  let mx = -Infinity;
  for (let i = 0; i < q.length; i++) {
    lf[i] = Math.log(Math.max(q[i], 1e-12)) + lam * Math.log(Math.max(p[i], 1e-12));
    if (lf[i] > mx) mx = lf[i];
  }
  let s = 0;
  for (let i = 0; i < q.length; i++) {
    lf[i] = Math.exp(lf[i] - mx);
    s += lf[i];
  }
  for (let i = 0; i < q.length; i++) lf[i] /= s;
  return lf;
}

/** One segment -> `[class, confidence]` to accept, or null (`fuse.decide`). */
export function decide(seg: Segment, history: readonly number[], prior: Prior | null, rule: Rule): [number, number] | null {
  if (rule.max_len !== null && seg.end - seg.start > rule.max_len) return null;
  const q = seg.q;
  if (rule.mode === "none") {
    const c = argmax(q);
    return q[c] >= rule.theta ? [c, q[c]] : null;
  }
  if (prior === null) throw new Error(`rule ${rule.mode} needs a prior`);
  const p = prior(history);
  const f = fused(q, p, rule.lam);
  if (rule.mode === "rescore") {
    const c = argmax(f);
    return f[c] >= rule.theta ? [c, f[c]] : null;
  }
  const cq = argmax(q);
  if (q[cq] >= rule.theta_hi) return [cq, q[cq]];
  const c = argmax(f);
  let above = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > p[c]) above++;
  return above < rule.k && q[c] >= rule.theta_lo ? [c, f[c]] : null;
}

/** `select.LatticeRunner`: the accepted history and the segments still waiting. */
export class LatticeRunner {
  history: number[] = [];
  private pending: { seg: Segment; opts: [number, number][] }[] = [];
  private readonly lat: Lattice;
  private readonly prior: Prior | null;

  constructor(lat: Lattice, prior: Prior | null) {
    this.lat = lat;
    this.prior = prior;
  }

  /** How many segments are held back, waiting for context. */
  get waiting(): number {
    return this.pending.length;
  }

  private options(seg: Segment): [number, number][] {
    if (this.lat.max_len !== null && seg.end - seg.start > this.lat.max_len) return [];
    const idx = Array.from(seg.q.keys()).sort((a, b) => seg.q[b] - seg.q[a] || a - b).slice(0, this.lat.k);
    const lt = Math.log(this.lat.theta);
    return idx.map((c) => [c, Math.log(Math.max(seg.q[c], 1e-12)) - lt]);
  }

  private priorGain(hist: readonly number[], c: number): number {
    if (this.prior === null || this.lat.lam === 0) return 0;
    const p = this.prior(hist);
    return this.lat.lam * Math.log(Math.max(p[c] * p.length, 1e-12));
  }

  /** Best decision (class or null = skip) for every pending segment; the first maximum wins, in itertools.product order. */
  private best(): (number | null)[] {
    let bestTotal = -Infinity;
    let bestPath: (number | null)[] = [];
    const choices = this.pending.map(({ opts }) => [null, ...opts] as (null | [number, number])[]);
    const path: (null | [number, number])[] = new Array(choices.length);
    const walk = (i: number) => {
      if (i === choices.length) {
        let hist = this.history.slice();
        let total = 0;
        for (const opt of path) {
          if (opt === null) continue;
          total += opt[1] + this.priorGain(hist, opt[0]);
          hist = [...hist, opt[0]];
        }
        if (total > bestTotal) {
          bestTotal = total;
          bestPath = path.map((o) => (o === null ? null : o[0]));
        }
        return;
      }
      for (const opt of choices[i]) {
        path[i] = opt;
        walk(i + 1);
      }
    };
    walk(0);
    return bestPath;
  }

  private commit(choice: number | null): Emission[] {
    const { seg } = this.pending.shift()!;
    if (choice === null) return [];
    this.history = [...this.history, choice];
    return [{ cls: choice, frame: seg.end - 1, conf: seg.q[choice] }];
  }

  push(seg: Segment): Emission[] {
    this.pending.push({ seg, opts: this.options(seg) });
    if (this.pending.length <= this.lat.lag) return [];
    return this.commit(this.best()[0]);
  }

  flush(): Emission[] {
    const out: Emission[] = [];
    while (this.pending.length) out.push(...this.commit(this.best()[0]));
    return out;
  }
}

/** `OnlineDecoder`: frame probabilities in, accepted signs out. */
export class OnlineDecoder {
  history: number[] = [];
  private runner: LatticeRunner | null = null;
  private score!: Float64Array;
  private peak!: Float64Array;
  private mass = 0;
  private start: number | null = null;
  private last: number | null = null;
  readonly nullIndex: number;
  readonly nu: number;
  readonly minLen: number;
  readonly rule: Rule | Lattice;
  readonly prior: Prior | null;
  readonly collapse: boolean;

  constructor(nullIndex: number, nu: number, minLen: number, rule: Rule | Lattice = D3_ONLY,
              prior: Prior | null = null, collapse = true) {
    this.nullIndex = nullIndex;
    this.nu = nu;
    this.minLen = minLen;
    this.rule = rule;
    this.prior = prior;
    this.collapse = collapse;
    this.reset();
  }

  /** A new sentence: empty history, no run in progress. */
  reset(): void {
    this.history = [];
    this.runner = this.rule.kind === "lattice" ? new LatticeRunner(this.rule, this.prior) : null;
    this.score = new Float64Array(this.nullIndex);
    this.peak = new Float64Array(this.nullIndex);
    this.mass = 0;
    this.start = null;
    this.last = null;
  }

  /** True while a sign run is open or the lattice is holding segments. */
  get busy(): boolean {
    return this.start !== null || (this.runner?.waiting ?? 0) > 0;
  }

  /** Segments the lattice is holding back (0 for greedy rules). */
  get waiting(): number {
    return this.runner?.waiting ?? 0;
  }

  private emit(e: Emission): Emission | null {
    if (this.collapse && this.last === e.cls) return null;
    this.last = e.cls;
    return e;
  }

  private close(end: number): Emission | null {
    const start = this.start;
    this.start = null;
    const score = this.score, mass = this.mass, peak = this.peak;
    this.score = new Float64Array(this.nullIndex);
    this.peak = new Float64Array(this.nullIndex);
    this.mass = 0;
    if (start === null || end - start < this.minLen) return null;
    let s = 0;
    for (let i = 0; i < score.length; i++) s += score[i];
    const denom = Math.max(s, 1e-12);
    const q = new Float64Array(score.length);
    for (let i = 0; i < score.length; i++) q[i] = score[i] / denom;
    const seg: Segment = { start, end, q, mass, peak };
    if (this.runner !== null) {
      const got = this.runner.push(seg);
      this.history = this.runner.history;
      return got.length ? this.emit(got[0]) : null;
    }
    const d = decide(seg, this.history, this.prior, this.rule as Rule);
    if (d === null) return null;
    this.history = [...this.history, d[0]];
    return this.emit({ cls: d[0], frame: end - 1, conf: d[1] });
  }

  /** Frame `t`'s probabilities over glosses + null -> the sign accepted at this frame, if any. */
  step(p: ArrayLike<number>, t: number): Emission | null {
    const pn = p[this.nullIndex];
    if (pn < this.nu) {
      const w = 1.0 - pn;
      const w32 = Math.fround(w);
      for (let i = 0; i < this.nullIndex; i++) {
        this.score[i] += Math.fround(w32 * p[i]);
        if (p[i] > this.peak[i]) this.peak[i] = p[i];
      }
      this.mass += w;
      if (this.start === null) this.start = t;
      return null;
    }
    return this.start !== null ? this.close(t) : null;
  }

  /** End of the sentence at frame `t`: close a run in progress and decide every held segment. */
  flushAll(t: number): Emission[] {
    const out: Emission[] = [];
    if (this.start !== null) {
      const e = this.close(t);
      if (e !== null) out.push(e);
    }
    if (this.runner !== null) {
      for (const e of this.runner.flush()) {
        const k = this.emit(e);
        if (k !== null) out.push(k);
      }
      this.history = this.runner.history;
    }
    return out;
  }
}
