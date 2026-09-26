/**
 * The C1 step model on LiteRT.js (WASM): one (543, 3) frame + recurrent state in,
 * a probability over the 250 glosses + null out.
 *
 * The same computation as the demo notebook's `frame_probs`:
 * `logits = cos_scale · W·embedding` (both unit length), masked, softmax. The class
 * matrix `W` stays outside the graph (`classes.f32`), so a custom sign is one
 * appended row (TODO §12.7), not a re-export.
 */

import { loadAndCompile, loadLiteRt, Tensor } from "@litertjs/core";
import type { StepManifest } from "../../../shared-ts/src/contracts.ts";

type Compiled = Awaited<ReturnType<typeof loadAndCompile>>;

/** What a session steps: one frame in, probabilities over glosses + null out. */
export interface StepModel {
  readonly manifest: StepManifest;
  readonly glosses: readonly string[];
  step(frame: Float32Array): Promise<Float32Array>;
  reset(): void;
}

let runtime: Promise<void> | null = null; // LiteRT's WASM is loaded once per page

export class Recognizer implements StepModel {
  private state: Float32Array;
  private readonly logits: Float64Array;
  readonly manifest: StepManifest;
  private readonly model: Compiled;
  private readonly W: Float32Array;
  private readonly mask: Uint8Array;

  private constructor(model: Compiled, W: Float32Array, manifest: StepManifest) {
    this.model = model;
    this.W = W;
    this.manifest = manifest;
    const n = W.length / manifest.embed_dim;
    this.mask = Uint8Array.from({ length: n }, (_, i) => (i < manifest.class_mask.length ? Number(manifest.class_mask[i]) : 1));
    this.logits = new Float64Array(n);
    this.state = new Float32Array(manifest.state_shape[0] * manifest.state_shape[1]);
  }

  /** `dir` is the bundle folder under `assets`: `model` (the deployed C1) or `models/<run>`. */
  static async load(assets: string, wasmDir: string, dir = "model"): Promise<Recognizer> {
    runtime ??= loadLiteRt(wasmDir).then(() => undefined);
    await runtime;
    const manifest = (await (await fetch(`${assets}/${dir}/manifest.json`)).json()) as StepManifest;
    const model = await loadAndCompile(`${assets}/${dir}/model.tflite`, { accelerator: "wasm" });
    const W = new Float32Array(await (await fetch(`${assets}/${dir}/classes.f32`)).arrayBuffer());
    if (W.length % manifest.embed_dim !== 0) throw new Error("classes.f32 does not match the manifest's embed_dim");
    return new Recognizer(model, W, manifest);
  }

  /** Start a new sentence: zero recurrent state, as every training stream starts. */
  reset(): void {
    this.state.fill(0);
  }

  /** One frame -> probabilities over glosses + null (length = rows of W). */
  async step(frame: Float32Array): Promise<Float32Array> {
    const [sr, sc] = this.manifest.state_shape;
    const inputs = { frame: new Tensor(frame, [543, 3]), state: new Tensor(this.state, [sr, sc]) };
    const out = (await this.model.run(inputs)) as Record<string, Tensor>;
    const emb = out.embedding.toTypedArray() as Float32Array;
    this.state = Float32Array.from(out.state_out.toTypedArray() as Float32Array);
    for (const t of [inputs.frame, inputs.state, ...Object.values(out)]) t.delete();

    const d = this.manifest.embed_dim, scale = this.manifest.cos_scale;
    let mx = -Infinity;
    for (let c = 0; c < this.logits.length; c++) {
      if (!this.mask[c]) {
        this.logits[c] = -Infinity;
        continue;
      }
      let s = 0;
      for (let j = 0; j < d; j++) s += this.W[c * d + j] * emb[j];
      this.logits[c] = scale * s;
      if (this.logits[c] > mx) mx = this.logits[c];
    }
    const p = new Float32Array(this.logits.length);
    let z = 0;
    for (let c = 0; c < p.length; c++) z += (this.logits[c] = Math.exp(this.logits[c] - mx));
    for (let c = 0; c < p.length; c++) p[c] = this.logits[c] / z;
    return p;
  }

  get glosses(): readonly string[] {
    return this.manifest.glosses;
  }
}

/**
 * Several step models averaged per frame (TODO §12.8, `docs/reports/window-ensembles.md`): each member
 * keeps its own recurrent state and steps the same frame; their gloss + null probabilities are
 * averaged. C1 + C2 this way scored GER 0.244 offline vs C1's 0.276. Members must share the label
 * space (checked here).
 */
export class Ensemble implements StepModel {
  private readonly members: StepModel[];

  constructor(members: StepModel[]) {
    if (!members.length) throw new Error("an ensemble needs at least one model");
    const g = members[0].glosses.join("|");
    for (const m of members) {
      if (m.glosses.join("|") !== g || m.manifest.null_index !== members[0].manifest.null_index) {
        throw new Error(`run ${m.manifest.run_id} has a different label space`);
      }
    }
    this.members = members;
  }

  get manifest(): StepManifest {
    return this.members[0].manifest;
  }

  get glosses(): readonly string[] {
    return this.members[0].glosses;
  }

  reset(): void {
    for (const m of this.members) m.reset();
  }

  async step(frame: Float32Array): Promise<Float32Array> {
    if (this.members.length === 1) return this.members[0].step(frame);
    const outs = [];
    for (const m of this.members) outs.push(await m.step(frame));
    const p = new Float32Array(outs[0].length);
    for (const o of outs) for (let c = 0; c < p.length; c++) p[c] += o[c];
    for (let c = 0; c < p.length; c++) p[c] /= outs.length;
    return p;
  }
}
