/**
 * MediaPipe Tasks HolisticLandmarker on a <video> element, in VIDEO mode: the app
 * calls `detect` on each frame the video delivers, with that frame's media time.
 * VIDEO mode (not LIVE_STREAM's callback) keeps one result per processed frame,
 * which is what the recognizer's clock needs (deployment-research.md §4).
 */

import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision";
import type { HolisticLandmarkerResult } from "@mediapipe/tasks-vision";

export type HolisticResult = HolisticLandmarkerResult;

export class Holistic {
  private lastTs = -1;
  readonly delegate: "GPU" | "CPU";
  private readonly lm: HolisticLandmarker;

  private constructor(lm: HolisticLandmarker, delegate: "GPU" | "CPU") {
    this.lm = lm;
    this.delegate = delegate;
  }

  /**
   * `modelUrls` are tried in order (a local copy first, then Google's CDN). `prefer` "CPU" uses
   * only the CPU; "GPU" tries the GPU first and falls back to the CPU if it fails.
   */
  static async load(wasmDir: string, modelUrls: string[], prefer: "CPU" | "GPU" = "CPU"): Promise<Holistic> {
    const fileset = await FilesetResolver.forVisionTasks(wasmDir);
    const delegates = prefer === "GPU" ? (["GPU", "CPU"] as const) : (["CPU"] as const);
    let lastError: unknown;
    for (const url of modelUrls) {
      for (const delegate of delegates) {
        try {
          const lm = await HolisticLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: url, delegate },
            runningMode: "VIDEO",
          });
          return new Holistic(lm, delegate);
        } catch (e) {
          lastError = e;
        }
      }
    }
    throw new Error(`HolisticLandmarker failed to load: ${String(lastError)}`);
  }

  /** Landmarks for the video's current frame. Timestamps must increase strictly. */
  detect(video: HTMLVideoElement, mediaTimeMs: number): HolisticResult {
    const ts = Math.max(Math.round(mediaTimeMs), this.lastTs + 1);
    this.lastTs = ts;
    return this.lm.detectForVideo(video, ts);
  }

  /** A new source (camera <-> file) restarts the timestamp clock. */
  restart(): void {
    this.lastTs = -1;
  }
}
