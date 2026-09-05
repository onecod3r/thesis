# sb-extract-ts — landmark extraction in Deno

Stage 1 (video → landmarks) in TypeScript, as an alternative to
`packages/sb-extract` (Python + MediaPipe Tasks).

> **This does not run. Do not reach for it.**
> `@mediapipe/tasks-vision` creates a **WebGL** context during graph
> construction — before any frame is submitted, and regardless of
> `delegate: "CPU"` — and Deno has no WebGL implementation. Measured
> 2026-09-05 on 12 clips: 0/12 extracted, every one with
> `ReferenceError: WebGLRenderingContext is not defined`. Use
> `packages/sb-extract` (Python MediaPipe, native pipeline, no GL) for
> extraction. Full measurement and the routes that remain open:
> [`docs/reports/extractor-parity.md`](../../docs/reports/extractor-parity.md).

Deno was chosen here because it implements `ImageData` and Web Workers natively,
so MediaPipe's WASM build could be fed frames without a `canvas` native module —
Node needs `canvas` compiled from C++ just to construct an `ImageData`, and that
is the kind of dependency that leaks memory in a long-running frame loop. That
reasoning holds; it simply was not the binding constraint. WebGL was, and Node
is no better placed, because the requirement belongs to MediaPipe's *web*
distribution rather than to the runtime.

```bash
deno task check                       # typecheck, no network needed
deno task extract --input ./videos --out ./landmarks
deno task extract --input ./videos --out ./landmarks --workers 4 --retry-failed
deno task extract --input ./videos --out ./landmarks --model ./holistic.task
```

Needs `deno`, `ffmpeg` and `ffprobe` on PATH — all three are installed on the
development machine as of 2026-09-05. Geometry defaults to **each video's own**
size and frame rate; `--width/--height/--fps` are overrides, and using them makes
the output incomparable with the Python extractor's.

## Status: not a replacement yet

Beyond the runtime blocker above, this produces the same artifact as the Python
extractor but is **not** known to produce the same *numbers*. Before it extracts
anything that will be trained on:

```bash
# both extractors over the same clips, then the comparison
python -m sb.extract.parity_run --limit 12
```

**Why this gate exists.** 33,599 POPSIGN test clips are already extracted with the
Python path. If the train split is extracted with this one and the two disagree
systematically, that difference sits *between the splits*, a model learns it, and
no accuracy metric will ever reveal it. The two are not the same code — different
MediaPipe distribution, different delegate, possibly a different model revision.
`parity.py` checks structure (frame counts, detected/undetected masks) before
geometry, because a landmark present in one and NaN in the other is a categorical
difference that changes the quality proxies and the NaN-aware feature pipeline.

## What this does differently from the obvious sketch

Three things, each of which would otherwise be a silent data bug rather than a
crash.

**1. Frames are reassembled across pipe chunks.** A pipe delivers chunks on socket
boundaries, not frame boundaries, so a chunk routinely ends mid-frame. The natural
loop —

```ts
while (offset + frameSize <= chunk.length) { /* take a frame */ }
```

— discards the trailing partial frame and then starts the next chunk mid-frame.
Every frame after the first short chunk is a shear of two frames, MediaPipe
returns perfectly plausible landmarks for them, and nothing downstream can tell.
`frames.ts` keeps a carry buffer and emits only whole frames.

**2. The output is the `LANDMARK_TENSOR` v1 contract**, not JSON of four arrays:
`(T, 543, 3)` float16 npz with `fps` and `num_frames`, in GISLR holistic row order
(face 0–467, left hand 468–488, pose 489–521, right hand 522–542). That row order
is what makes the `subsets.py` index lists valid for POPSIGN, and it is why the
four MediaPipe groups are interleaved into one array rather than kept separate.
`npz.ts` is a small NPY/ZIP writer so no conversion step can drift.

**3. NaN, and three channels.** An undetected landmark is NaN, never 0 — the
quality proxies and the 1st-place feature pipeline both depend on being able to
tell "absent" from "at the origin". `visibility` is dropped: the spec is xyz.

**4. Geometry is each video's own.** An earlier default of `scale=640:480 -r 30`
would have made parity impossible by construction: POPSIGN is 1944×2592 *portrait*
at 30, ~29.92 or 120 fps depending on the clip, so that both inverted the aspect
ratio and resampled the time axis, while the Python extractor feeds cv2's native
frames at the video's own rate. `worker.ts` probes each clip; the flags are
overrides.

Two further departures from the sketch, for cost rather than correctness: workers
are **reused across videos** (a worker per video re-downloads and re-compiles the
WASM graph each time), and the run is **manifest-driven and resumable** — the npz
is written before the unit is marked `done`, `done` is skipped, `failed` is
retried with `--retry-failed`.

The landmarker is also **rebuilt on a resolution change**, which is correctness
rather than hygiene: POPSIGN mixes 1944×2592 and 1080×1920, and the graph's
segmentation-smoothing calculator compares each frame against the previous one,
so a reused landmarker fails with INTERNAL `RET_CHECK ... current_mat->rows ==
previous_mat->rows`. The Python extractor already paid for this lesson.

## Layout

| file | role |
|---|---|
| `src/schema.ts` | LANDMARK_TENSOR v1, ported from `sb.core.schema` — **that file is authoritative** |
| `src/npz.ts` | float32→float16, NPY 1.0, stored-ZIP `.npz`, atomic write |
| `src/frames.ts` | ffmpeg rawvideo → RGBA frames, with the carry buffer |
| `src/worker.ts` | one `HolisticLandmarker`, many videos; probes native geometry |
| `src/dom_shim.ts` | the browser globals MediaPipe wants — **not a fix**, see below |
| `src/cli.ts` | pool + resumable manifest |

## Known gaps

- **It does not run.** See the banner at the top. `src/dom_shim.ts` supplies
  `document`, a browser-looking `window`, and a `<script>` loader that fetches
  and evaluates MediaPipe's WASM loader; it exists **so the failure names the
  real requirement**. Without it the first error is `document is not defined`,
  which reads like a missing polyfill; with it, the runtime gets far enough to
  say `WebGLRenderingContext is not defined` from inside
  `_emscripten_webgl_do_create_context`. It is documentation that executes.
- **Livestream mode is not here.** `runningMode: "VIDEO"` with monotonically
  increasing timestamps is a batch contract; `LIVE_STREAM` uses a result callback
  and drops frames under load, which is right for a camera and wrong for a corpus.
  It belongs with the app surface (`apps/`), against the same `schema.ts` — and
  it is also the browser-hosted route that would make this package run at all.
- **The default model asset is the bucket's `latest`**, the only published path,
  so it is unpinned and can change under you. `--model <file.task>` takes a local
  copy instead, and is what parity requires: the Python extractor loads
  `data/external/mediapipe/tasks/holistic_landmarker.task`, and a comparison
  against different weights would measure the model rather than the code.
- **`npz.ts` is validated at the algorithm level, not the execution level.**
  `tools/verify_npz_format.py` transcribes `encodeNpy`/`encodeNpz`/`toFloat16`
  into Python and confirms numpy reads the bytes — NPY header padding, ZIP
  central-directory offsets, float16 with NaN preserved and exact binary
  fractions intact. That catches format-logic bugs without Deno, and it passes.
  It cannot prove the TypeScript *runs* correctly; for that, extract one clip and
  run `python tools/verify_npz_format.py --file <clip>.npz`.
