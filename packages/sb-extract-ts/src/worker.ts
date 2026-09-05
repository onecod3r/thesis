/**
 * Extraction worker: **one landmarker, many videos**.
 *
 * The pool sends `{type: "job"}` messages and this replies with `{type: "done"}`
 * or `{type: "error"}`. It keeps the `HolisticLandmarker` alive across jobs —
 * spawning a worker per video would re-download and re-compile the WASM graph
 * every time, which dominates the cost for short clips. The Python extractor
 * makes the same trade with `MAXTASKSPERCHILD=64`; a worker here is recycled by
 * the pool on the same principle.
 *
 * Timestamps must increase monotonically *within* a landmarker instance, so the
 * clock carries an offset across jobs rather than restarting at zero.
 *
 * **The landmarker is rebuilt on a resolution change**, which is not an
 * optimisation but a correctness requirement the Python extractor already paid
 * for: POPSIGN mixes 1944x2592 and 1080x1920, and the graph's
 * segmentation-smoothing calculator compares each frame against the previous
 * one, so feeding a differently-sized video into a reused landmarker fails with
 * INTERNAL "RET_CHECK ... current_mat->rows == previous_mat->rows".
 */

import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision";

import { dirname } from "@std/path";

import { explainIfWebglMissing, installDomShim, withoutProcess } from "./dom_shim.ts";

import { probe, readFrames } from "./frames.ts";
import { frameToRows, N_COORDS, N_LANDMARKS, validateTensor } from "./schema.ts";
import { writeLandmarkNpz } from "./npz.ts";

export interface Job {
  type: "job";
  id: string;
  videoPath: string;
  outPath: string;
  /** `null` = the video's own; a number forces a rescale. */
  width: number | null;
  height: number | null;
  /** `null` = the video's own frame sequence; a number resamples. */
  fps: number | null;
  wasmBase: string;
  /** Remote `.task` URL. Ignored when `modelAssetPath` is set. */
  modelAsset: string;
  /** Local `.task` file — read into a buffer, so no network and no drift. */
  modelAssetPath?: string;
}

export interface DoneMessage {
  type: "done";
  id: string;
  frames: number;
  seconds: number;
  width: number;
  height: number;
  fps: number;
  detection: Record<string, number>;
}

export interface ErrorMessage {
  type: "error";
  id: string;
  error: string;
}

let landmarker: HolisticLandmarker | null = null;
let timestampOffset = 0;
let lastShape: string | null = null;

async function getLandmarker(
  job: Job,
  shape: string,
): Promise<HolisticLandmarker> {
  if (landmarker && lastShape === shape) return landmarker;
  if (landmarker) {
    // resolution change: a reused graph would fail the RET_CHECK above
    landmarker.close();
    landmarker = null;
    timestampOffset = 0;
  }
  const baseOptions = job.modelAssetPath
    ? { modelAssetBuffer: await Deno.readFile(job.modelAssetPath), delegate: "CPU" as const }
    : { modelAssetPath: job.modelAsset, delegate: "CPU" as const };

  // MediaPipe here is the *web* build: it wants browser globals, and its graph
  // creates a WebGL context on construction. The shim supplies the globals so
  // the failure, when it comes, names the real requirement (dom_shim.ts).
  installDomShim(job.wasmBase);
  try {
    landmarker = await withoutProcess(async () => {
      const vision = await FilesetResolver.forVisionTasks(job.wasmBase);
      return await HolisticLandmarker.createFromOptions(vision, {
        baseOptions,
        runningMode: "VIDEO",
      });
    });
  } catch (err) {
    const diagnosis = explainIfWebglMissing(err);
    throw diagnosis ? new Error(diagnosis, { cause: err }) : err;
  }
  lastShape = shape;
  return landmarker;
}

async function run(job: Job): Promise<DoneMessage> {
  const started = performance.now();

  // native geometry unless the caller overrode it; parity with the Python
  // extractor depends on this being the video's own size and rate
  const native = await probe(job.videoPath);
  const width = job.width ?? native.width;
  const height = job.height ?? native.height;
  const rescale = width !== native.width || height !== native.height;
  const fps = job.fps ?? native.fps;
  const resample = job.fps !== null && job.fps !== native.fps;

  const detector = await getLandmarker(job, `${width}x${height}`);

  const frames: Float32Array[] = [];
  let index = 0;
  for await (const rgba of readFrames(job.videoPath, {
    width,
    height,
    fps: resample ? fps : null,
    rescale,
  })) {
    const image = new ImageData(rgba, width, height);
    const ts = timestampOffset + Math.round((index / fps) * 1000);
    // deno-lint-ignore no-explicit-any -- ImageData is a valid ImageSource at
    // runtime; the published types only enumerate the browser-canvas ones
    const result = detector.detectForVideo(image as any, ts);
    frames.push(
      frameToRows({
        face: result.faceLandmarks?.[0],
        left_hand: result.leftHandLandmarks?.[0],
        pose: result.poseLandmarks?.[0],
        right_hand: result.rightHandLandmarks?.[0],
      }),
    );
    index++;
  }
  timestampOffset += Math.round((index / fps) * 1000) + 1000;

  const flat = new Float32Array(frames.length * N_LANDMARKS * N_COORDS);
  frames.forEach((frame, i) => flat.set(frame, i * N_LANDMARKS * N_COORDS));
  validateTensor(flat, frames.length);

  await Deno.mkdir(dirname(job.outPath), { recursive: true });
  await writeLandmarkNpz(job.outPath, flat, frames.length, fps);

  // per-group detection rate, so the manifest records quality alongside progress
  const detection: Record<string, number> = {};
  for (const g of [
    { name: "face", offset: 0 },
    { name: "left_hand", offset: 468 },
    { name: "pose", offset: 489 },
    { name: "right_hand", offset: 522 },
  ]) {
    let seen = 0;
    for (const frame of frames) {
      if (!Number.isNaN(frame[g.offset * N_COORDS])) seen++;
    }
    detection[g.name] = frames.length ? seen / frames.length : 0;
  }

  return {
    type: "done",
    id: job.id,
    frames: frames.length,
    seconds: (performance.now() - started) / 1000,
    width,
    height,
    fps,
    detection,
  };
}

self.onmessage = async (event: MessageEvent<Job | { type: "shutdown" }>) => {
  const msg = event.data;
  if (msg.type === "shutdown") {
    landmarker?.close();
    landmarker = null;
    self.close();
    return;
  }
  try {
    self.postMessage(await run(msg));
  } catch (err) {
    // one bad video must not take the pool down; the driver records it as
    // failed in the manifest and retries it on a later pass
    self.postMessage({
      type: "error",
      id: msg.id,
      error: err instanceof Error ? err.message : String(err),
    } satisfies ErrorMessage);
  }
};
