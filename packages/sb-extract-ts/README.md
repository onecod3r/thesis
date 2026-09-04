# sb-extract-ts — landmark extraction in Deno

Stage 1 (video → landmarks) in TypeScript, as an alternative to
`packages/sb-extract` (Python + MediaPipe Tasks).

Deno earns its place here for one specific reason: it implements `ImageData` and
Web Workers natively, so MediaPipe's WASM build can be fed frames without a
`canvas` native module. Node needs `canvas` compiled from C++ just to construct
an `ImageData`, and that is exactly the kind of dependency that leaks memory in a
long-running frame loop.

```bash
deno task check                       # typecheck, no network needed
deno task extract --input ./videos --out ./landmarks
deno task extract --input ./videos --out ./landmarks --workers 4 --retry-failed
```

Needs `ffmpeg` and `ffprobe` on PATH, and `deno`. **Neither was installed on the
development machine**, so everything below is written against the specs and
typechecked, not executed. Treat the first real run as the test.

## Status: not a replacement yet

This produces the same artifact as the Python extractor but is **not** known to
produce the same *numbers*. Before it extracts anything that will be trained on:

```bash
# extract the same clips both ways, then
python -m sb.extract.parity --python-dir data/raw/popsign/test \
                            --ts-dir data/temp/parity_ts --limit 50
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

Two further departures from the sketch, for cost rather than correctness: workers
are **reused across videos** (a worker per video re-downloads and re-compiles the
WASM graph each time), and the run is **manifest-driven and resumable** — the npz
is written before the unit is marked `done`, `done` is skipped, `failed` is
retried with `--retry-failed`.

## Layout

| file | role |
|---|---|
| `src/schema.ts` | LANDMARK_TENSOR v1, ported from `sb.core.schema` — **that file is authoritative** |
| `src/npz.ts` | float32→float16, NPY 1.0, stored-ZIP `.npz`, atomic write |
| `src/frames.ts` | ffmpeg rawvideo → RGBA frames, with the carry buffer |
| `src/worker.ts` | one `HolisticLandmarker`, many videos |
| `src/cli.ts` | pool + resumable manifest |

## Known gaps

- **Livestream mode is not here.** `runningMode: "VIDEO"` with monotonically
  increasing timestamps is a batch contract; `LIVE_STREAM` uses a result callback
  and drops frames under load, which is right for a camera and wrong for a corpus.
  It belongs with the app surface (`apps/`), against the same `schema.ts`.
- **Model asset is pinned to `latest`** in the Google bucket URL, which is the
  only published path. If reproducibility of the extraction itself matters, mirror
  the `.task` file and pin a copy.
- **`npz.ts` is validated at the algorithm level, not the execution level.**
  `tools/verify_npz_format.py` transcribes `encodeNpy`/`encodeNpz`/`toFloat16`
  into Python and confirms numpy reads the bytes — NPY header padding, ZIP
  central-directory offsets, float16 with NaN preserved and exact binary
  fractions intact. That catches format-logic bugs without Deno, and it passes.
  It cannot prove the TypeScript *runs* correctly; for that, extract one clip and
  run `python tools/verify_npz_format.py --file <clip>.npz`.
