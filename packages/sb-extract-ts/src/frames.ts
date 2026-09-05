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
 * **Geometry defaults to the video's own.** The Python extractor feeds cv2's
 * native frames at the video's own frame rate; anything else here would make the
 * two extractors incomparable *by construction* — POPSIGN is 1944x2592 portrait
 * at 30/120 fps, so a fixed `scale=640:480 -r 30` both inverts the aspect ratio
 * and resamples the time axis. `rescale: false` omits the scale filter and
 * `fps: null` omits `-r`, which is what parity requires; both are still
 * overridable for a deliberately cheaper run.
 *
 * ffmpeg is spawned directly with `Deno.Command` rather than through
 * fluent-ffmpeg: one fewer npm dependency, and a real `ReadableStream` instead of
 * Node stream-event semantics.
 */

export interface FrameSourceOptions {
  /** Decoded frame width — must be what ffmpeg actually emits. */
  width: number;
  /** Decoded frame height — must be what ffmpeg actually emits. */
  height: number;
  /** Resample to this rate; `null` keeps the video's own frame sequence. */
  fps: number | null;
  /** Rescale to width x height; `false` means the video is already that size. */
  rescale: boolean;
  /** ffmpeg binary; override if it is not on PATH. */
  ffmpeg?: string;
}

export interface VideoGeometry {
  width: number;
  height: number;
  fps: number;
  duration: number;
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
): AsyncGenerator<Uint8ClampedArray<ArrayBuffer>, void, unknown> {
  const { width, height, fps, rescale, ffmpeg = "ffmpeg" } = opts;
  const rgbFrameSize = width * height * 3;

  const args = ["-hide_banner", "-loglevel", "error", "-i", videoPath];
  if (rescale) args.push("-vf", `scale=${width}:${height}`);
  if (fps !== null) args.push("-r", String(fps));
  args.push("-f", "rawvideo", "-pix_fmt", "rgb24", "-");

  const command = new Deno.Command(ffmpeg, {
    args,
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

/**
 * Probe a video's native geometry — the default the extractor runs at.
 *
 * `r_frame_rate` rather than `avg_frame_rate` because it is the rate of the
 * frame sequence ffmpeg will emit, which is what has to line up with cv2's
 * `CAP_PROP_FPS` on the Python side.
 */
export async function probe(
  videoPath: string,
  ffprobe = "ffprobe",
): Promise<VideoGeometry> {
  const out = await new Deno.Command(ffprobe, {
    args: [
      "-v", "error",
      "-select_streams", "v:0",
      "-show_entries", "stream=width,height,r_frame_rate:format=duration",
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
  const stream = parsed.streams?.[0];
  if (!stream?.width || !stream?.height) {
    throw new FfmpegError(`${videoPath}: ffprobe reported no video stream`);
  }
  const [num, den] = String(stream.r_frame_rate ?? "0/1").split("/").map(Number);
  return {
    width: Number(stream.width),
    height: Number(stream.height),
    fps: den ? num / den : 0,
    duration: Number(parsed.format?.duration ?? 0),
  };
}
