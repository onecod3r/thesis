/**
 * LANDMARK_TENSOR v1 — the stage-1 → stage-2 contract, in TypeScript.
 *
 * This is a **port, not a new definition**. `packages/sb-core/src/sb/core/schema.py`
 * is authoritative; this file exists so the Deno extractor can produce artifacts
 * the Python stack already knows how to read, and `parity.py` checks the two
 * agree. If they ever diverge, the Python one is right.
 *
 * Row layout (543 rows/frame, xyz each):
 *
 *     rows   0-467  face      (468 face-mesh landmarks; row === face-mesh index)
 *     rows 468-488  left_hand (21)
 *     rows 489-521  pose      (33; holistic row = 489 + pose index)
 *     rows 522-542  right_hand(21)
 *
 * That ordering is load-bearing: it is what makes the `subsets.py` index lists
 * (ME_126, FP_118, …) valid for POPSIGN as well as GISLR. Get it wrong and every
 * downstream subset silently selects the wrong landmarks.
 *
 * **NaN is data, not corruption.** An undetected group is NaN for that frame, and
 * that is what the quality proxies measure and what the 1st-place feature pipeline
 * normalises over. Never substitute zeros.
 */

export const SPEC_VERSION = 1;
export const N_LANDMARKS = 543;
export const N_COORDS = 3; // x, y, z — NOT visibility; the spec is 3 channels
export const STORAGE_DTYPE = "float16";

export interface LandmarkGroup {
  readonly name: string;
  readonly offset: number;
  readonly size: number;
}

/** The one row layout. Order and offsets must match sb.core.schema.GROUPS. */
export const GROUPS: readonly LandmarkGroup[] = [
  { name: "face", offset: 0, size: 468 },
  { name: "left_hand", offset: 468, size: 21 },
  { name: "pose", offset: 489, size: 33 },
  { name: "right_hand", offset: 522, size: 21 },
] as const;

// fails at import time rather than after a 30-hour extraction run
const tiled = GROUPS.reduce((n, g) => n + g.size, 0);
if (tiled !== N_LANDMARKS) {
  throw new Error(`group sizes tile to ${tiled}, expected ${N_LANDMARKS}`);
}
let cursor = 0;
for (const g of GROUPS) {
  if (g.offset !== cursor) {
    throw new Error(`group ${g.name} starts at ${g.offset}, expected ${cursor}`);
  }
  cursor += g.size;
}

export class SpecError extends Error {}

/** One landmark as MediaPipe Tasks reports it. */
export interface Landmark {
  x: number;
  y: number;
  z: number;
}

/**
 * Assemble one frame into the 543-row layout.
 *
 * Every row starts as NaN and is overwritten only where a group was detected, so
 * "not detected" and "detected at the origin" stay distinguishable — the whole
 * reason the storage policy is NaN rather than 0.
 */
export function frameToRows(
  groups: Partial<Record<string, readonly Landmark[] | undefined>>,
): Float32Array {
  const frame = new Float32Array(N_LANDMARKS * N_COORDS).fill(NaN);
  for (const g of GROUPS) {
    const lms = groups[g.name];
    if (!lms || lms.length === 0) continue;
    if (lms.length > g.size) {
      throw new SpecError(
        `${g.name}: got ${lms.length} landmarks, layout allows ${g.size}`,
      );
    }
    for (let i = 0; i < lms.length; i++) {
      const base = (g.offset + i) * N_COORDS;
      frame[base] = lms[i].x;
      frame[base + 1] = lms[i].y;
      frame[base + 2] = lms[i].z;
    }
  }
  return frame;
}

/**
 * Structural check of a `(T, 543, 3)` tensor held flat.
 *
 * Mirrors `sb.core.schema.validate_tensor`, including its two judgements:
 * `T === 0` is valid (a clip whose frames all failed to decode is a recorded
 * outcome), and an all-NaN tensor is valid too (nothing detected is a *quality*
 * signal, not a malformed file). Infinities are rejected — they cannot come out
 * of the pipeline and they destroy any downstream normalisation.
 */
export function validateTensor(flat: Float32Array, nFrames: number): void {
  const expected = nFrames * N_LANDMARKS * N_COORDS;
  if (flat.length !== expected) {
    throw new SpecError(
      `expected ${expected} values for ${nFrames} frames ` +
        `(T x ${N_LANDMARKS} x ${N_COORDS}), got ${flat.length}`,
    );
  }
  for (let i = 0; i < flat.length; i++) {
    if (flat[i] === Infinity || flat[i] === -Infinity) {
      throw new SpecError(
        `infinity at flat index ${i} — undetected landmarks must be NaN`,
      );
    }
  }
}

/** The contract as data, for embedding in a manifest. */
export function spec(): Record<string, unknown> {
  return {
    spec: "LANDMARK_TENSOR",
    version: SPEC_VERSION,
    n_landmarks: N_LANDMARKS,
    n_coords: N_COORDS,
    storage_dtype: STORAGE_DTYPE,
    missing_value: "nan",
    row_order: "gislr-holistic",
    groups: GROUPS.map((g) => ({ name: g.name, offset: g.offset, size: g.size })),
    npz_keys: { landmarks: "float16", fps: "float32", num_frames: "int32" },
  };
}
