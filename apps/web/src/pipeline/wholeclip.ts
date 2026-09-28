/**
 * Whole-clip classifiers for record-then-recognize (TODO §16.2): the opposite input
 * contract from every step model in this app (`recognizer.ts`, `isolated.ts`) -- a
 * clip's raw `(543,3)` frames go in all at once, one softmax distribution over glosses
 * comes out, because a bidirectional model (`bilstm`) needs the whole sequence before
 * it can say anything.
 */

import { loadAndCompile, loadLiteRt, Tensor } from "@litertjs/core";
import type { IsolatedRecognizer } from "./isolated.ts";

export interface WholeClipManifest {
  glosses: string[];
  landmarks: number[];
  coords: string;
}

let runtime: Promise<void> | null = null; // LiteRT's WASM is loaded once per page

/** `bilstm`'s whole-clip export (`sb.recognize.export.step.export_web_wholeclip`):
 * dynamic-length input, no external class matrix (the softmax head is baked in, same
 * as `isolated.ts`'s classifiers). */
export class WholeClipRecognizer {
  readonly manifest: WholeClipManifest;
  private readonly model: Awaited<ReturnType<typeof loadAndCompile>>;

  private constructor(model: Awaited<ReturnType<typeof loadAndCompile>>, manifest: WholeClipManifest) {
    this.model = model;
    this.manifest = manifest;
  }

  static async load(assets: string, wasmDir: string, dir: string): Promise<WholeClipRecognizer> {
    runtime ??= loadLiteRt(wasmDir).then(() => undefined);
    await runtime;
    const manifest = (await (await fetch(`${assets}/${dir}/manifest.json`)).json()) as WholeClipManifest;
    const model = await loadAndCompile(`${assets}/${dir}/model.tflite`, { accelerator: "wasm" });
    return new WholeClipRecognizer(model, manifest);
  }

  /** `frames`: `t` concatenated `(543,3)` frames, flat and in order -> probabilities
   * over `glosses`. */
  async classify(frames: Float32Array, t: number): Promise<Float32Array> {
    const input = new Tensor(frames, [t, 543, 3]);
    const out = (await this.model.run({ frames: input })) as Record<string, Tensor>;
    const probs = Float32Array.from(out.probs.toTypedArray() as Float32Array);
    for (const tt of [input, ...Object.values(out)]) tt.delete();
    return probs;
  }

  get glosses(): readonly string[] {
    return this.manifest.glosses;
  }
}

/** A causal step model's whole-clip readout: reset, step every frame in order, keep
 * the last frame's output -- the same thing PyTorch's `forward_full` computes for a
 * causal architecture, since it never looks ahead. Lets `gru_phono_raw`'s *existing*
 * step export (§15, no new export needed) join the record-then-recognize ensemble. */
export async function classifyWithStepModel(rec: IsolatedRecognizer, frames: readonly Float32Array[]): Promise<Float32Array> {
  rec.reset();
  let probs: Float32Array<ArrayBufferLike> = new Float32Array(rec.glosses.length);
  for (const frame of frames) probs = await rec.step(frame);
  return probs;
}
