/**
 * Record-then-recognize (TODO §16.2): record a clip, then process it once, offline --
 * not a live streaming decode. Design measured in
 * `docs/reports/bilstm-wholeclip-eval.md`: segment the recorded clip with C4's own D3
 * decoding (it already works, GER 0.278 deployed -- movement/stillness segmentation
 * alone was tried and abandoned, Part 1 of that report), classify each segment with an
 * ensemble of `bilstm` (whole-clip) and `gru_phono_raw` (its existing step model,
 * driven to the segment's last frame), and decode the whole sentence's top-5
 * candidates per segment with `viterbiRescore` over the deployed trigram. Full corpus:
 * GER 0.177, sentence accuracy 0.619 (vs. deployed C4 alone, 0.278 / ~0.37).
 */

import type { IsolatedRecognizer } from "./isolated.ts";
import { segmentsD3 } from "./decoder.ts";
import type { Prior } from "./decoder.ts";
import type { StepModel } from "./recognizer.ts";
import { viterbiRescore } from "./viterbi.ts";
import { classifyWithStepModel, WholeClipRecognizer } from "./wholeclip.ts";

// D3 nu/min_len and the Viterbi lambda/k are the values measured on GISLR-Sentences
// (docs/reports/bilstm-wholeclip-eval.md): nu/min_len match C4's own deployed D3
// setting (pipeline.config.json), lambda=0.7 is the retuned optimum (0.5 was an
// unconfirmed grid edge), k=5 is the user's requested top-5.
const NU = 0.5;
const MIN_LEN = 4;
const LAM = 0.7;
const K = 5;

export interface Candidate {
  gloss: string;
  prob: number;
}

export interface RecordedSign {
  gloss: string;
  frame: number; // the segment's last frame (index into the recorded clip)
  candidates: Candidate[]; // top-K before sentence-level rescoring, most likely first
}

function topK(q: Float64Array, glosses: readonly string[], k: number): Candidate[] {
  return Array.from(q.keys())
    .sort((a, b) => q[b] - q[a])
    .slice(0, k)
    .map((c) => ({ gloss: glosses[c], prob: q[c] }));
}

/** `frames`: one already-assembled `(543,3)` frame per recorded tick (real detections
 * only -- no "nobody in view" gaps, the caller filters those out same as the live
 * pipeline's own reset-on-empty-frame rule). `c4` segments; `bilstm` + `phono`
 * classify; `prior` (the deployed trigram) picks the whole sentence. All four models'
 * glosses must be the same 250-class GISLR label space, in the same order -- the
 * caller checks this once at load time, not per call. */
export async function recognizeRecording(
  frames: readonly Float32Array[],
  c4: StepModel,
  bilstm: WholeClipRecognizer,
  phono: IsolatedRecognizer,
  prior: Prior,
): Promise<RecordedSign[]> {
  if (!frames.length) return [];
  c4.reset();
  const gp: Float64Array[] = [];
  for (const frame of frames) gp.push(Float64Array.from(await c4.step(frame)));
  const segs = segmentsD3(gp, NU, MIN_LEN, c4.manifest.null_index);
  if (!segs.length) return [];

  const glosses = c4.glosses;
  const candidateQs: Float64Array[] = [];
  for (const seg of segs) {
    const segFrames = frames.slice(seg.start, seg.end);
    const flat = new Float32Array(segFrames.length * 543 * 3);
    let off = 0;
    for (const f of segFrames) {
      flat.set(f, off);
      off += f.length;
    }
    const qb = await bilstm.classify(flat, segFrames.length);
    const qp = await classifyWithStepModel(phono, segFrames);
    const q = new Float64Array(qb.length);
    for (let c = 0; c < q.length; c++) q[c] = (qb[c] + qp[c]) / 2;
    candidateQs.push(q);
  }

  const choice = viterbiRescore(candidateQs, prior, LAM, K);
  return segs.map((seg, i) => ({
    gloss: glosses[choice[i]],
    frame: seg.end - 1,
    candidates: topK(candidateQs[i], glosses, K),
  }));
}
