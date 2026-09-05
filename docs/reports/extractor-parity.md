# Extractor parity: the TypeScript path cannot run on Deno

**Status: the comparison did not happen, and cannot yet.** The parity harness is
built and runs end to end, the Python reference tree exists, and the TypeScript
extractor was executed for the first time. It produced no landmarks — not
because of a bug in it, but because `@mediapipe/tasks-vision` requires a WebGL
context that Deno does not have. That is a property of the MediaPipe *web*
distribution, not of this repo's code, and no amount of fixing `sb-extract-ts`
changes it.

| | |
|---|---|
| **Question** | Do `packages/sb-extract` (Python) and `packages/sb-extract-ts` (Deno) produce the same landmarks? — the gate in TODO §10.1 before the TS path may touch anything trainable |
| **Instrument** | `python -m sb.extract.parity_run --limit 12` (selection → staging → both extractors → npz format check → `sb.extract.parity`) |
| **Sample** | 12 clips, one per label, seeded 42, recorded to `data/cache/popsign/parity/selection.json` |
| **Clips** | `after, animal, awake, bad, bee, boy, bug, can, cloud, cowboy, dad, drink` — POPSIGN **train a–e** |
| **Result** | Python 12/12 extracted in 47.7 s · TypeScript **0/12**, all with the same error |
| **Data** | `data/temp/parity/` (throwaway); selection and report under `data/cache/popsign/parity/` |

---

## 1. The blocker

```
ReferenceError: WebGLRenderingContext is not defined
    at OffscreenCanvas.fixedGetContext [as getContext]
    at Object.createContext
    at _emscripten_webgl_do_create_context
    at Module._changeBinaryGraph
```

`_changeBinaryGraph` is graph **construction** — this happens before a single
frame is submitted. Two things follow, and both were assumptions worth losing:

**`delegate: "CPU"` does not mean "no GPU work".** It selects the CPU inference
backend. The graph's image pipeline in the web build is GL-backed regardless, so
the WASM module creates a WebGL context on startup whichever delegate is asked
for. There is no option that turns this off.

**Deno's `ImageData` was never the hard part.** The package README argued Deno
over Node because Deno provides `ImageData` and Web Workers natively, so
MediaPipe's WASM build would need no `canvas` native module. That reasoning is
correct as far as it goes, and it is not the constraint that binds:

| global | Deno 2.9.6 |
|---|---|
| `ImageData` | yes |
| `OffscreenCanvas` | yes |
| `createImageBitmap` | yes |
| `GPU` (WebGPU) | yes |
| **`WebGLRenderingContext`** | **no** |
| **`WebGL2RenderingContext`** | **no** |
| `OffscreenCanvas.getContext("webgl2")` | returns `null` |

Deno has the canvas objects and no WebGL behind them. Node is no better placed:
the requirement belongs to the web WASM build, so it would need `headless-gl` —
a native module, which is the exact dependency the Deno choice existed to avoid.

## 2. Getting to the real error took three layers

The first failure was `document is not defined`, which reads like a missing
polyfill and invites an afternoon of shimming. Three distinct layers sat between
that and the actual requirement, and each one looked like the last problem:

1. **`document`** — the loader appends a `<script>` tag to fetch its WASM
   loader. A shim whose `body.appendChild` fetches and `eval`s the source, then
   fires `load`, satisfies it.
2. **`require is not defined`** — Emscripten sniffs globals to pick an I/O
   layer, and Deno satisfies its *node* test (`process.versions.node`), so it
   calls `require("fs")`. Presenting a `window` and hiding `process` across
   graph construction puts it on the web path.
3. **`both async and sync fetching of the wasm failed`** — Emscripten
   deliberately skips `fetch` for `file://` URIs and falls back to a
   `readBinary` that exists only in its node path. A local mirror of the `.wasm`
   cannot be used; the base must be `http(s)`.

Only past all three does the WebGL requirement surface. `src/dom_shim.ts` keeps
those three layers **precisely so the failure names the real constraint** — it
is documentation that executes, not an attempted fix, and the file says so.

## 3. What did work

Everything up to the MediaPipe call is now exercised rather than merely
typechecked:

- `deno check` passes (it did **not** before: `Uint8ClampedArray<ArrayBufferLike>`
  is not assignable to `ImageData`'s `ArrayBuffer` under Deno 2.9.6 / TS 6)
- ffmpeg spawns, decodes, and the pool dispatches all 12 clips across 4 workers
- the resumable manifest records all 12 units with the failure reason
- `verify_npz_format.py --algorithm` still passes

The Python half of the harness produced a clean reference tree:

| | |
|---|---|
| clips | 12/12, 47.7 s, 4 workers |
| frame counts | 50–345, tracking each clip's own rate (29.83 / 29.92 / 30 / 31 / 120 fps) |
| face detection | 1.00 on every clip |
| pose detection | 1.00 on every clip |
| hand detection | 0.00–0.73, varying by sign — the expected spread |

Per the `data/temp/` policy that tree was deleted after the run, but it is
**reproducible rather than lost**: the selection is seeded and cached to
`data/cache/popsign/parity/selection.json`, so re-running the harness rebuilds
exactly those 12 clips in ~48 s. `parity.py` needs only for the TS side to
appear.

## 4. Two findings the run produced independently of the blocker

**The TS extractor's default geometry could never have matched.** It hardcoded
`scale=640:480 -r 30`. POPSIGN is **1944×2592 portrait** at 30, ~29.92 or 120 fps
depending on the clip, while the Python extractor feeds cv2's native frames at
the video's own rate. So the default both inverted the aspect ratio and
resampled the time axis — the frame counts alone would have differed by up to
4×, and every landmark would have moved. Fixed: geometry now defaults to each
video's own, probed per clip, with `--width/--height/--fps` demoted to explicit
overrides. `probe()` existed for this and had never been called.

**The two sides were going to load different models.** The Python extractor uses
`data/external/mediapipe/tasks/holistic_landmarker.task`; the TS one defaulted to
the bucket's `latest`, a different and unpinned artifact. A comparison between
them would have measured the model, not the code. `--model <file.task>` now
points the TS side at the same local weights, and the harness always passes it.

Neither would have been visible without trying to run the thing.

## 5. Where this leaves the TypeScript extractor

The npz writer, the frame-carry buffer, the row-order contract, the resumable
manifest and the worker pool are all sound and now partly exercised. What is
missing is a MediaPipe runtime, and there are three honest options:

1. **Drive the web build from a real browser** (headless Chrome via Playwright /
   Puppeteer). This is where the WebGL context actually exists, and it is also
   the runtime the eventual `LIVE_STREAM` app surface (TODO §10.2) will target,
   so the work is not thrown away. It is the only route that keeps the
   TypeScript path.
2. **Keep extraction on `packages/sb-extract`.** The Python `mediapipe` package
   uses the native C++ pipeline with no GL requirement, which is why it has
   already extracted 33,599 clips. Nothing about POPSIGN extraction is blocked.
3. **Native GL in Deno** (`headless-gl` equivalent). This reintroduces the
   native module the Deno choice was made to avoid, and is not recommended.

Until one of those lands, **the TS extractor must not extract anything
trainable** — the TODO §10.1 gate stands, unchanged and now for a second reason:
the parity measurement it demands has still never been made.

## 6. Reproducing

```bash
.venv/Scripts/python.exe -m sb.extract.parity_run --limit 12
.venv/Scripts/python.exe -m sb.extract.parity_run --clean    # drop data/temp/parity
```

Resumable: the selection is cached and both extractors skip units their own
manifest marks `done`, so an interrupt costs the clips in flight. The run needs
`deno` and `ffmpeg` on PATH — both are installed on this machine as of
2026-09-05, which is itself a change from what TODO §10.1 recorded.
