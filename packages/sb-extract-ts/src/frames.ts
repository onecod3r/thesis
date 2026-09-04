/**
 * Video → RGBA frames, via a raw-video ffmpeg pipe.
 *
 * **The buffering here is the whole point of this module.** A pipe delivers chunks
 * on socket boundaries, which have nothing to do with frame boundaries: a chunk
 * routinely ends mid-frame. Code that does
 *
 *     while (offset + frameSize <= chunk.length) { ...take a frame... }
 *
 * and moves on to the next chunk silently discards the trailing partial frame and
 * then reads the *next* frame starting from the wrong byte. Every frame after the
 * first short chunk is a sheared mix of two frames, MediaPipe still returns
 * plausible-looking landmarks for them, and nothing downstream can tell. This
 * accumulates a carry buffer instead and only ever emits whole frames.
 *
 * ffmpeg is spawned directly with `Deno.Command` rather than through
 * fluent-ffmpeg: one fewer npm dependency, and a real `ReadableStream` instead of
 * Node stream-event semantics.
 */

export interface FrameSourceOptions {
  width: number;
  height: number;
  /** Frames are resampled to this rate, so timestamps are exact. */
  fps: number;
  /** ffmpeg binary; override if it is not on PATH. */
  ffmpeg?: string;
}

export class FfmpegError extends Error {}

/**
 * Yield `Uint8ClampedArray` RGBA frames, one per decoded frame, in order.
 *
 * RGBA rather than RGB because that is what `ImageData` takes, and Deno provides
 * `ImageData` natively — the reason this pipeline does not need a `canvas` build.
 */
export async function* readFrames(
  videoPath: string,
  opts: FrameSourceOptions,
): AsyncGenerator<Uint8ClampedArray, void, unknown> {
  const { width, height, fps, ffmpeg = "ffmpeg" } = opts;
  const rgbFrameSize = width * height * 3;

  const command = new Deno.Command(ffmpeg, {
    args: [
      "-hide_banner",
      "-loglevel", "error",
      "-i", videoPath,
      "-vf", `scale=${width}:${height}`,
      "-r", String(fps),
      "-f", "rawvideo",
      "-pix_fmt", "rgb24",
      "-", // stdout
    ],
    stdout: "piped",
    stderr: "piped",
  });

  const child = command.spawn();
  const stderrChunks: Uint8Array[] = [];
  const stderrDone = (async () => {
    for await (const chunk of child.stderr) stderrChunks.push(chunk);
  })();

  // carry holds the bytes of a frame that spanned a chunk boundary
  let carry = new Uint8Array(0);
  const rgba = new Uint8ClampedArray(width * height * 4);

  try {
    for await (const chunk of child.stdout) {
      let buf: Uint8Array;
      if (carry.length === 0) {
        buf = chunk;
      } else {
        buf = new Uint8Array(carry.length + chunk.length);
        buf.set(carry, 0);
        buf.set(chunk, carry.length);
      }

      let offset = 0;
      while (offset + rgbFrameSize <= buf.length) {
        const rgb = buf.subarray(offset, offset + rgbFrameSize);
        for (let i = 0, j = 0; i < rgbFrameSize; i += 3, j += 4) {
          rgba[j] = rgb[i];
          rgba[j + 1] = rgb[i + 1];
          rgba[j + 2] = rgb[i + 2];
          rgba[j + 3] = 255;
        }
        // a copy per frame: the consumer holds it across an await, and `rgba`
        // is reused on the next iteration
        yield rgba.slice();
        offset += rgbFrameSize;
      }
      // whatever is left is the head of the next frame — keep it
      carry = offset < buf.length ? buf.slice(offset) : new Uint8Array(0);
    }
  } finally {
    await stderrDone;
    const status = await child.status;
    if (!status.success) {
      const msg = new TextDecoder().decode(
        stderrChunks.length === 1
          ? stderrChunks[0]
          : stderrChunks.reduce((acc, c) => {
            const merged = new Uint8Array(acc.length + c.length);
            merged.set(acc);
            merged.set(c, acc.length);
            return merged;
          }, new Uint8Array(0)),
      );
      throw new FfmpegError(
        `ffmpeg exited ${status.code} for ${videoPath}: ${msg.trim() || "(no stderr)"}`,
      );
    }
    if (carry.length !== 0) {
      // ffmpeg succeeded but left a partial frame: the stream is not a whole
      // number of frames, which means the geometry assumption is wrong
      throw new FfmpegError(
        `${videoPath}: ${carry.length} trailing bytes, not a multiple of the ` +
          `${rgbFrameSize}-byte frame size — check width/height`,
      );
    }
  }
}

/** Probe a video's native frame rate and duration, for the manifest. */
export async function probe(
  videoPath: string,
  ffprobe = "ffprobe",
): Promise<{ fps: number; duration: number }> {
  const out = await new Deno.Command(ffprobe, {
    args: [
      "-v", "error",
      "-select_streams", "v:0",
      "-show_entries", "stream=r_frame_rate:format=duration",
      "-of", "json",
      videoPath,
    ],
    stdout: "piped",
    stderr: "piped",
  }).output();
  if (!out.success) {
    throw new FfmpegError(
      `ffprobe failed for ${videoPath}: ${new TextDecoder().decode(out.stderr)}`,
    );
  }
  const parsed = JSON.parse(new TextDecoder().decode(out.stdout));
  const rate = parsed.streams?.[0]?.r_frame_rate ?? "0/1";
  const [num, den] = String(rate).split("/").map(Number);
  return {
    fps: den ? num / den : 0,
    duration: Number(parsed.format?.duration ?? 0),
  };
}
