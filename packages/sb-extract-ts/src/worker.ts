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
 */

import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision";

import { readFrames } from "./frames.ts";
import { frameToRows, N_COORDS, N_LANDMARKS, validateTensor } from "./schema.ts";
import { writeLandmarkNpz } from "./npz.ts";

export interface Job {
  type: "job";
  id: string;
  videoPath: string;
  outPath: string;
  width: number;
  height: number;
  fps: number;
  wasmBase: string;
  modelAsset: string;
}

export interface DoneMessage {
  type: "done";
  id: string;
  frames: number;
  seconds: number;
  detection: Record<string, number>;
}

export interface ErrorMessage {
  type: "error";
  id: string;
  error: string;
}

let landmarker: HolisticLandmarker | null = null;
let timestampOffset = 0;

async function getLandmarker(job: Job): Promise<HolisticLandmarker> {
  if (landmarker) return landmarker;
  const vision = await FilesetResolver.forVisionTasks(job.wasmBase);
  landmarker = await HolisticLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: job.modelAsset, delegate: "CPU" },
    runningMode: "VIDEO",
  });
  return landmarker;
}

async function run(job: Job): Promise<DoneMessage> {
  const started = performance.now();
  const detector = await getLandmarker(job);

  const frames: Float32Array[] = [];
  let index = 0;
  for await (const rgba of readFrames(job.videoPath, {
    width: job.width,
    height: job.height,
    fps: job.fps,
  })) {
    const image = new ImageData(rgba, job.width, job.height);
    const ts = timestampOffset + Math.round((index / job.fps) * 1000);
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
  timestampOffset += Math.round((index / job.fps) * 1000) + 1000;

  const flat = new Float32Array(frames.length * N_LANDMARKS * N_COORDS);
  frames.forEach((frame, i) => flat.set(frame, i * N_LANDMARKS * N_COORDS));
  validateTensor(flat, frames.length);

  await Deno.mkdir(job.outPath.replace(/[/\\][^/\\]+$/, ""), { recursive: true });
  await writeLandmarkNpz(job.outPath, flat, frames.length, job.fps);

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
