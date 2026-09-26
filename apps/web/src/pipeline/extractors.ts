/**
 * Three ways to get sign-relevant landmarks from a live camera, for the landmark test page
 * (`landmarks.html`, TODO §12.8): the app uses Holistic today, and the live-camera diagnosis suspects
 * its hand labels, so this compares it with MediaPipe's separate task models.
 *
 * - `holistic`: one HolisticLandmarker (face 478 + pose 33 + left/right hand 21 each). What the app runs.
 * - `separate`: PoseLandmarker + FaceLandmarker + HandLandmarker (2 hands), run one after the other on
 *   the same frame. Hands come with a Left/Right *handedness* label and score instead of fixed slots.
 * - `hands_face`: FaceLandmarker + HandLandmarker only (no body).
 *
 * All run in VIDEO mode on the frame the page passes, with strictly increasing timestamps per model.
 */

import { FaceLandmarker, FilesetResolver, HandLandmarker, HolisticLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

export type Mode = "holistic" | "separate" | "hands_face";
export type Delegate = "CPU" | "GPU";
export type PoseSize = "lite" | "full" | "heavy";

export const MODE_LABEL: Record<Mode, string> = {
  holistic: "Holistic (one model)",
  separate: "Hands + Face + Pose (three models)",
  hands_face: "Hands + Face (two models)",
};

const CDN = "https://storage.googleapis.com/mediapipe-models";
export const MODEL_URLS = {
  holistic: `${CDN}/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task`,
  face: `${CDN}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`,
  hand: `${CDN}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`,
  pose: (size: PoseSize) => `${CDN}/pose_landmarker/pose_landmarker_${size}/float16/latest/pose_landmarker_${size}.task`,
};

export interface Hand {
  landmarks: NormalizedLandmark[];
  /** The model's own label: Holistic's slot ("left"/"right" hand output), or the HandLandmarker's handedness. */
  label: "Left" | "Right";
  /** Handedness score (HandLandmarker only; Holistic has fixed slots and no score). */
  score: number | null;
}

export interface Detection {
  face: NormalizedLandmark[] | null;
  pose: NormalizedLandmark[] | null;
  hands: Hand[];
  /** Milliseconds per model for this frame. */
  timings: Record<string, number>;
}

type Runner = { detectForVideo(frame: HTMLVideoElement, ts: number): unknown; close(): void };

async function create<T>(make: (delegate: Delegate) => Promise<T>, prefer: Delegate): Promise<{ task: T; delegate: Delegate }> {
  const order: Delegate[] = prefer === "GPU" ? ["GPU", "CPU"] : ["CPU"];
  let last: unknown;
  for (const d of order) {
    try {
      return { task: await make(d), delegate: d };
    } catch (e) {
      last = e;
    }
  }
  throw new Error(`MediaPipe model failed to load: ${String(last)}`);
}

export class Extractor {
  readonly mode: Mode;
  readonly delegate: Delegate;
  private readonly runners: Record<string, Runner>;
  private lastTs = -1;

  private constructor(mode: Mode, runners: Record<string, Runner>, delegate: Delegate) {
    this.mode = mode;
    this.runners = runners;
    this.delegate = delegate;
  }

  /** `holisticUrls` are tried in order (a local copy first, then the CDN). */
  static async load(mode: Mode, wasmDir: string, prefer: Delegate, poseSize: PoseSize, holisticUrls: string[]): Promise<Extractor> {
    const fileset = await FilesetResolver.forVisionTasks(wasmDir);
    const base = (url: string, delegate: Delegate) => ({ baseOptions: { modelAssetPath: url, delegate }, runningMode: "VIDEO" as const });
    const runners: Record<string, Runner> = {};
    let delegate: Delegate = prefer;
    if (mode === "holistic") {
      let err: unknown;
      for (const url of holisticUrls) {
        try {
          const r = await create((d) => HolisticLandmarker.createFromOptions(fileset, base(url, d)), prefer);
          runners.holistic = r.task;
          delegate = r.delegate;
          break;
        } catch (e) {
          err = e;
        }
      }
      if (!runners.holistic) throw new Error(String(err));
    } else {
      if (mode === "separate") {
        const p = await create((d) => PoseLandmarker.createFromOptions(fileset, { ...base(MODEL_URLS.pose(poseSize), d), numPoses: 1 }), prefer);
        runners.pose = p.task;
        delegate = p.delegate;
      }
      const f = await create((d) => FaceLandmarker.createFromOptions(fileset, { ...base(MODEL_URLS.face, d), numFaces: 1 }), prefer);
      const h = await create((d) => HandLandmarker.createFromOptions(fileset, { ...base(MODEL_URLS.hand, d), numHands: 2 }), prefer);
      runners.face = f.task;
      runners.hand = h.task;
      if (f.delegate === "CPU" || h.delegate === "CPU") delegate = "CPU";
    }
    return new Extractor(mode, runners, delegate);
  }

  /** A new source restarts every model's timestamp clock. */
  restart(): void {
    this.lastTs = -1;
  }

  detect(video: HTMLVideoElement, mediaTimeMs: number): Detection {
    const ts = Math.max(Math.round(mediaTimeMs), this.lastTs + 1);
    this.lastTs = ts;
    const timings: Record<string, number> = {};
    const time = <T>(name: string, fn: () => T): T => {
      const t0 = performance.now();
      const out = fn();
      timings[name] = performance.now() - t0;
      return out;
    };
    if (this.mode === "holistic") {
      const r = time("holistic", () => (this.runners.holistic as HolisticLandmarker).detectForVideo(video, ts));
      const hands: Hand[] = [];
      if (r.leftHandLandmarks[0]?.length) hands.push({ landmarks: r.leftHandLandmarks[0], label: "Left", score: null });
      if (r.rightHandLandmarks[0]?.length) hands.push({ landmarks: r.rightHandLandmarks[0], label: "Right", score: null });
      return { face: r.faceLandmarks[0] ?? null, pose: r.poseLandmarks[0] ?? null, hands, timings };
    }
    const pose = this.runners.pose
      ? time("pose", () => (this.runners.pose as PoseLandmarker).detectForVideo(video, ts)).landmarks[0] ?? null
      : null;
    const face = time("face", () => (this.runners.face as FaceLandmarker).detectForVideo(video, ts)).faceLandmarks[0] ?? null;
    const hr = time("hands", () => (this.runners.hand as HandLandmarker).detectForVideo(video, ts));
    const hands: Hand[] = hr.landmarks.map((lm, i) => ({
      landmarks: lm,
      label: (hr.handedness[i]?.[0]?.categoryName === "Left" ? "Left" : "Right") as "Left" | "Right",
      score: hr.handedness[i]?.[0]?.score ?? null,
    }));
    return { face, pose, hands, timings };
  }

  close(): void {
    for (const r of Object.values(this.runners)) r.close();
  }
}

/**
 * Which body side each hand is on, judged by geometry: the pose wrist (15 = body left, 16 = body right)
 * its own wrist is nearest to. The same rule as the C4 stream normalization (`StreamNormFrontend`), so
 * the page shows whether a model's hand labels agree with the body. `null` without a pose.
 */
export function bodySide(hand: Hand, pose: NormalizedLandmark[] | null): "Left" | "Right" | null {
  if (!pose || pose.length < 17) return null;
  const w = hand.landmarks[0];
  const d = (p: NormalizedLandmark) => Math.hypot(p.x - w.x, p.y - w.y);
  return d(pose[15]) <= d(pose[16]) ? "Left" : "Right";
}
