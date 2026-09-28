/**
 * "No overall movement" detection for continuous streaming (TODO §16.1): closes a sign
 * run whose boundary the model's own null head missed. Runs beside the model's
 * null-gated close (`decoder.ts`'s p_null >= nu), not instead of it -- either can end a
 * run, the same way `isolated.ts` layers a hand-left-frame trigger on top of a model
 * that has no boundary signal of its own.
 *
 * Purely a browser-side heuristic with no Python reference: `runReplay` in `main.ts`
 * never calls it, so it does not touch the Python-parity tests (`npm test`,
 * `scripts/browser-check.ts`).
 */

const POINT_NOISE_EPS = 0.0015; // per-point xy displacement below this is jitter, not motion
const SMOOTHING_ALPHA = 0.3; // EMA weight on each new frame's movement reading
const STILL_THRESHOLD = 0.003; // smoothed movement below this counts as "still"
const STILL_FRAMES = 10; // consecutive still real (non-interpolated) frames before a run ends

/** Mean per-point xy displacement between two (543,3) frames, over landmarks present in
 * both. z is dropped (CLAUDE.md: mostly noise for pose/hand landmarks). Sub-jitter
 * per-point displacements are clamped to 0 before averaging, so a hand trembling in
 * place doesn't read as movement. */
export function frameMovement(prev: Float32Array, curr: Float32Array): number {
  const n = prev.length / 3;
  let sum = 0, count = 0;
  for (let i = 0; i < n; i++) {
    const px = prev[i * 3], py = prev[i * 3 + 1];
    const cx = curr[i * 3], cy = curr[i * 3 + 1];
    if (Number.isNaN(px) || Number.isNaN(py) || Number.isNaN(cx) || Number.isNaN(cy)) continue;
    const d = Math.hypot(cx - px, cy - py);
    sum += d < POINT_NOISE_EPS ? 0 : d;
    count++;
  }
  return count ? sum / count : 0;
}

/** Smooths `frameMovement` over time (EMA, noise filtering) and reports whether the
 * signer has been still for `STILL_FRAMES` real frames in a row. */
export class MovementGate {
  private ema = 0;
  private stillStreak = 0;
  private prev: Float32Array | null = null;

  reset(): void {
    this.ema = 0;
    this.stillStreak = 0;
    this.prev = null;
  }

  /** Feed one real (non-interpolated) frame; true once stillness has held long enough. */
  push(frame: Float32Array): boolean {
    if (this.prev === null) {
      this.prev = frame;
      return false;
    }
    const m = frameMovement(this.prev, frame);
    this.prev = frame;
    this.ema = SMOOTHING_ALPHA * m + (1 - SMOOTHING_ALPHA) * this.ema;
    this.stillStreak = this.ema < STILL_THRESHOLD ? this.stillStreak + 1 : 0;
    return this.stillStreak >= STILL_FRAMES;
  }
}
