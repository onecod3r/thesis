/**
 * Individual sign recognition (TODO §12.9): one sign at a time, not a sentence stream.
 *
 * Unlike the continuous C1/C4 family, this model has no null class and no learned boundary
 * head (`sb.recognize.export.step.export_web_isolated`) — the "sign is done" decision is made
 * here in the browser, by two rules, either of which commits the current best guess and resets
 * the recurrent state for the next sign:
 *
 * 1. **Hand leaves the frame** (the user's stop condition): once a hand has been seen, if neither
 *    hand is detected for `HAND_GONE_FRAMES` in a row, the sign just performed is over.
 * 2. **The guess has already stabilized** (the user's early-commit ask: "if a word is candidate
 *    for best next guess and the same sign is repeated, select it right away"): if the top-1
 *    class hasn't changed for `STABLE_FRAMES` in a row and its probability is above
 *    `STABLE_CONF`, commit immediately rather than waiting for the hand to leave — a confidently
 *    held sign doesn't need to wait out the whole gesture.
 *
 * After a commit, a short refractory window (`COOLDOWN_FRAMES`) suppresses a second commit on
 * the same still-held pose, so trigger 2 can't fire twice on one sign.
 */

import { loadAndCompile, loadLiteRt, Tensor } from "@litertjs/core";
import type { IsolatedManifest } from "../../../shared-ts/src/contracts.ts";

const HAND_GONE_FRAMES = 4; // debounce: a blink of missed hand detection isn't "gone"
const STABLE_FRAMES = 8; // consecutive frames the top-1 guess must hold to early-commit
const STABLE_CONF = 0.6; // and its probability must clear this to early-commit
const COOLDOWN_FRAMES = 12; // frames after a commit before another can fire

export interface IsolatedGuess {
  gloss: string;
  conf: number;
  reason: "hand_left" | "stable";
}

let runtime: Promise<void> | null = null;

/** The isolated step model on LiteRT.js: one frame in, class probabilities out (baked-in head,
 * so there is no external class matrix to multiply — simpler than the continuous `Recognizer`). */
export class IsolatedRecognizer {
  private state: Float32Array;
  readonly manifest: IsolatedManifest;
  private readonly model: Awaited<ReturnType<typeof loadAndCompile>>;

  private constructor(model: Awaited<ReturnType<typeof loadAndCompile>>, manifest: IsolatedManifest) {
    this.model = model;
    this.manifest = manifest;
    this.state = new Float32Array(manifest.state_shape[0] * manifest.state_shape[1]);
  }

  static async load(assets: string, wasmDir: string, dir: string): Promise<IsolatedRecognizer> {
    runtime ??= loadLiteRt(wasmDir).then(() => undefined);
    await runtime;
    const manifest = (await (await fetch(`${assets}/${dir}/manifest.json`)).json()) as IsolatedManifest;
    const model = await loadAndCompile(`${assets}/${dir}/model.tflite`, { accelerator: "wasm" });
    return new IsolatedRecognizer(model, manifest);
  }

  reset(): void {
    this.state.fill(0);
  }

  async step(frame: Float32Array): Promise<Float32Array> {
    const [sr, sc] = this.manifest.state_shape;
    const inputs = { frame: new Tensor(frame, [543, 3]), state: new Tensor(this.state, [sr, sc]) };
    const out = (await this.model.run(inputs)) as Record<string, Tensor>;
    const probs = Float32Array.from(out.probs.toTypedArray() as Float32Array);
    this.state = Float32Array.from(out.state_out.toTypedArray() as Float32Array);
    for (const t of [inputs.frame, inputs.state, ...Object.values(out)]) t.delete();
    return probs;
  }

  get glosses(): readonly string[] {
    return this.manifest.glosses;
  }
}

function argmax(v: Float32Array): [number, number] {
  let bi = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[bi]) bi = i;
  return [bi, v[bi]];
}

/** Feeds frames to an `IsolatedRecognizer` and decides, frame by frame, when one sign is over. */
export class IsolatedSession {
  private readonly rec: IsolatedRecognizer;
  private handSeen = false;
  private handGoneStreak = 0;
  private stableClass = -1;
  private stableStreak = 0;
  private cooldown = 0;

  constructor(rec: IsolatedRecognizer) {
    this.rec = rec;
  }

  reset(): void {
    this.rec.reset();
    this.handSeen = false;
    this.handGoneStreak = 0;
    this.stableClass = -1;
    this.stableStreak = 0;
    this.cooldown = 0;
  }

  /** One frame -> a committed guess if this frame ends a sign, else `null`. `handPresent` is
   * whether either hand was detected this frame (from the raw Holistic result, before the frame
   * is even assembled — cheaper and more direct than re-deriving it from landmark values). */
  async feed(frame: Float32Array, handPresent: boolean): Promise<IsolatedGuess | null> {
    const probs = await this.rec.step(frame);
    if (this.cooldown > 0) this.cooldown--;

    if (handPresent) {
      this.handSeen = true;
      this.handGoneStreak = 0;
    } else {
      this.handGoneStreak++;
    }

    const [cls, conf] = argmax(probs);
    if (cls === this.stableClass) this.stableStreak++;
    else {
      this.stableClass = cls;
      this.stableStreak = 1;
    }

    if (this.cooldown > 0) return null;

    // Trigger 1: the hand that was signing has left the frame.
    if (this.handSeen && this.handGoneStreak >= HAND_GONE_FRAMES) {
      const guess: IsolatedGuess = { gloss: this.rec.glosses[cls], conf, reason: "hand_left" };
      this.commit();
      return guess;
    }
    // Trigger 2: the guess has already stabilized -- no need to wait for the hand to leave.
    if (this.handSeen && this.stableStreak >= STABLE_FRAMES && conf >= STABLE_CONF) {
      const guess: IsolatedGuess = { gloss: this.rec.glosses[cls], conf, reason: "stable" };
      this.commit();
      return guess;
    }
    return null;
  }

  private commit(): void {
    this.rec.reset();
    this.handSeen = false;
    this.handGoneStreak = 0;
    this.stableClass = -1;
    this.stableStreak = 0;
    this.cooldown = COOLDOWN_FRAMES;
  }
}
