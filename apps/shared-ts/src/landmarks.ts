/**
 * MediaPipe Holistic result -> one (543, 3) GISLR-layout frame.
 *
 * The layout itself is not restated here: it is `packages/sb-extract-ts/src/schema.ts`,
 * the existing port of `sb.core.schema` (apps/README.md: "reuse it; don't write a
 * third copy"). This file only adapts a Tasks `HolisticLandmarkerResult` to it:
 *
 * - The Tasks face mesh has 478 points (468 + 10 iris). GISLR uses the first 468,
 *   and `frameToRows` rejects anything longer, so the face is cut to 468.
 * - `mirror`: flip x and swap the hands, for a feed whose handedness is the
 *   reverse of GISLR's front-camera recordings (deployment-research.md §4, risk 2).
 */

import { frameToRows, N_COORDS, N_LANDMARKS } from "../../../packages/sb-extract-ts/src/schema.ts";
import type { Landmark } from "../../../packages/sb-extract-ts/src/schema.ts";

export { frameToRows, N_COORDS, N_LANDMARKS };
export type { Landmark };

export interface HolisticLike {
  faceLandmarks?: readonly (readonly Landmark[])[];
  leftHandLandmarks?: readonly (readonly Landmark[])[];
  poseLandmarks?: readonly (readonly Landmark[])[];
  rightHandLandmarks?: readonly (readonly Landmark[])[];
}

const FACE_ROWS = 468;

function flipped(lms: readonly Landmark[] | undefined): Landmark[] | undefined {
  return lms?.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z }));
}

/** One Holistic result -> a flat Float32Array of 543*3, NaN where undetected. */
export function holisticToFrame(result: HolisticLike, mirror = false): Float32Array {
  let face: readonly Landmark[] | undefined = result.faceLandmarks?.[0]?.slice(0, FACE_ROWS);
  let left: readonly Landmark[] | undefined = result.leftHandLandmarks?.[0];
  let pose: readonly Landmark[] | undefined = result.poseLandmarks?.[0];
  let right: readonly Landmark[] | undefined = result.rightHandLandmarks?.[0];
  if (mirror) {
    [face, pose] = [flipped(face), flipped(pose)];
    [left, right] = [flipped(right), flipped(left)];
  }
  return frameToRows({ face, left_hand: left, pose, right_hand: right });
}

/** Rebuild a full frame from a replay stream's stored landmarks (xy of `rows`, NaN elsewhere). */
export function rowsToFrame(xy: Float32Array, rows: readonly number[], t: number): Float32Array {
  const frame = new Float32Array(N_LANDMARKS * N_COORDS).fill(NaN);
  const base = t * rows.length * 2;
  for (let i = 0; i < rows.length; i++) {
    frame[rows[i] * N_COORDS] = xy[base + 2 * i];
    frame[rows[i] * N_COORDS + 1] = xy[base + 2 * i + 1];
  }
  return frame;
}
