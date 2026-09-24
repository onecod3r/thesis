/**
 * One signing session: frames in, accepted signs and finished sentences out.
 *
 * Stages stay separate (user decision, 2026-09-24): recognizer -> decoder with the
 * next-gloss prior -> gloss -> English -> speech. What this file adds to them:
 *
 * - **Sentence end.** `sentence_end_null_frames` consecutive frames that the
 *   recognizer calls null (p_null >= nu) end the sentence: the lattice decides
 *   every held segment, the sentence goes to English, and both the decoder
 *   history and the recurrent state reset (every training stream starts from
 *   zero state; `streaming-confidence.md` found un-reset state to be the
 *   bleed-through problem).
 * - **The model's clock.** C1 learned sign speed at GISLR's rate (about 30 fps).
 *   `Clock` turns a live feed's media times into model ticks at `target_fps`,
 *   repeating a frame when the feed is slower (deployment-research.md §4, risk 1).
 * - **Uncertainty policy** (sign-to-speech-demo-analysis.md §4 A2): a sentence
 *   is spoken automatically only when every gloss is at least
 *   `auto_speak_min_confidence`; otherwise the UI asks first.
 */

import type { Lexicon, PipelineConfig } from "../../../shared-ts/src/contracts.ts";
import { OnlineDecoder } from "./decoder.ts";
import type { Emission, Lattice, Prior, Rule } from "./decoder.ts";
import { glossToEnglish } from "./gloss2en.ts";

export interface Sign {
  gloss: string;
  conf: number;
  frame: number; // the sign's last frame (model ticks)
  decidedAt: number; // the tick the decoder committed it
  uncertain: boolean;
}

export interface Sentence {
  signs: Sign[];
  english: string;
  autoSpeak: boolean;
}

export interface StepResult {
  signs: Sign[];
  sentence: Sentence | null;
  pNull: number;
  waiting: number;
}

export function ruleFromConfig(c: PipelineConfig["rule"]): Rule | Lattice {
  if (c.kind === "lattice") return { kind: "lattice", theta: c.theta, lam: c.lam, k: c.k, lag: c.lag, max_len: c.max_len };
  return { kind: "rule", mode: c.mode, lam: c.lam, theta: c.theta, k: c.k, theta_lo: c.theta_lo, theta_hi: c.theta_hi,
           max_len: c.max_len };
}

export class Session {
  private decoder: OnlineDecoder;
  private tick = 0;
  private nullRun = 0;
  private current: Sign[] = [];
  private readonly cfg: PipelineConfig;
  private readonly glosses: readonly string[];
  private readonly nullIndex: number;
  private readonly lexicon: Lexicon;
  private readonly onReset: () => void;
  /** End sentences on long pauses. Off for replay, which is one sentence per stream, as in Python. */
  autoEnd = true;

  constructor(cfg: PipelineConfig, glosses: readonly string[], nullIndex: number, prior: Prior, lexicon: Lexicon,
              onReset: () => void) {
    this.cfg = cfg;
    this.glosses = glosses;
    this.nullIndex = nullIndex;
    this.lexicon = lexicon;
    this.onReset = onReset;
    this.decoder = new OnlineDecoder(nullIndex, cfg.decoder.nu, cfg.decoder.min_len, ruleFromConfig(cfg.rule), prior,
                                     cfg.collapse);
  }

  get signsSoFar(): readonly Sign[] {
    return this.current;
  }

  private toSign(e: Emission, at: number): Sign {
    return { gloss: this.glosses[e.cls], conf: e.conf, frame: e.frame, decidedAt: at,
             uncertain: e.conf < this.cfg.display.uncertain_below };
  }

  /** Feed one tick's probabilities. */
  push(p: Float32Array): StepResult {
    const t = this.tick++;
    const signs: Sign[] = [];
    const e = this.decoder.step(p, t);
    if (e) signs.push(this.toSign(e, t));
    this.current.push(...signs);
    const pNull = p[this.nullIndex];
    this.nullRun = pNull >= this.cfg.decoder.nu ? this.nullRun + 1 : 0;
    let sentence: Sentence | null = null;
    if (this.autoEnd && this.nullRun >= this.cfg.sentence_end_null_frames && (this.current.length > 0 || this.decoder.busy)) {
      const r = this.endSentence();
      signs.push(...r.tail);
      sentence = r.sentence;
    }
    return { signs, sentence, pNull, waiting: this.decoder.waiting };
  }

  /** End the sentence now (a pause, the end of a file, or the user pressing stop). */
  endSentence(): { tail: Sign[]; sentence: Sentence | null } {
    const tail = this.decoder.flushAll(this.tick).map((e) => this.toSign(e, this.tick));
    this.current.push(...tail);
    const signs = this.current;
    this.current = [];
    this.nullRun = 0;
    this.decoder.reset();
    this.onReset();
    if (!signs.length) return { tail, sentence: null };
    const english = glossToEnglish(signs.map((s) => s.gloss), this.lexicon);
    const autoSpeak = signs.every((s) => s.conf >= this.cfg.speech.auto_speak_min_confidence);
    return { tail, sentence: { signs, english, autoSpeak } };
  }
}

/**
 * Media time -> model ticks at `fps`. A live frame that arrives after k tick
 * boundaries is fed k times (0 when the feed runs faster than the model).
 */
export class Clock {
  private last: number | null = null;
  private readonly fps: number;

  constructor(fps: number) {
    this.fps = fps;
  }

  ticks(mediaTimeS: number): number {
    const n = Math.floor(mediaTimeS * this.fps);
    if (this.last === null) {
      this.last = n;
      return 1;
    }
    const k = Math.max(0, Math.min(n - this.last, this.fps)); // at most 1 s of repeats after a stall
    this.last = Math.max(this.last, n);
    return k;
  }
}

/** Levenshtein errors (substitutions + deletions + insertions) between two gloss lists. */
export function glossErrors(ref: readonly string[], hyp: readonly string[]): number {
  const d = Array.from({ length: ref.length + 1 }, (_, i) => [i, ...new Array(hyp.length).fill(0)]);
  for (let j = 1; j <= hyp.length; j++) d[0][j] = j;
  for (let i = 1; i <= ref.length; i++) {
    for (let j = 1; j <= hyp.length; j++) {
      d[i][j] = Math.min(d[i - 1][j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1), d[i][j - 1] + 1, d[i - 1][j] + 1);
    }
  }
  return d[ref.length][hyp.length];
}
