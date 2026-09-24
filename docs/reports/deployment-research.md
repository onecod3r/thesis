# Sign → speech deployment research: client-side recognition, Cloudflare edge

**TODO §12.5** · 2026-09-24 · status: research done, **plan awaiting review** (§9)

**Question.** How do we deploy the full sign → speech pipeline so that **landmark
extraction and recognition run on the client** (the user's constraint, 2026-09-24),
and everything else runs on Cloudflare? This covers the TFLite export, the browser
runtime, MediaPipe on the web, and Cloudflare Workers. It also covers the options
behind "remote functions".

## 0. Summary

- **The recognizer runs in the browser, and this is proven.** The existing TFLite
  export cannot be used for this. So a new **single-step export**
  (`sb.recognize.export.step`) turns C1 (`1790143122`) into a 3.46 MB `.tflite` that
  uses builtin ops only. It takes one frame plus the recurrent state and returns an
  embedding, a boundary logit and the new state. It runs in **headless Chrome on
  LiteRT.js (WASM) at 0.20 ms/frame median (p95 0.30 ms)** and matches PyTorch to
  within 6e-7 (§3).
- **MediaPipe Holistic runs in the browser too** (`@mediapipe/tasks-vision`
  `HolisticLandmarker`, 13.7 MB model). The code that maps its output to the 543-row
  layout already exists in the paused TS extractor (`frameToRows`). MediaPipe, not
  the recognizer, sets the frame budget (§4).
- **Decoding also runs on the client.** D3 is about 20 lines of TypeScript. The client
  sends the edge **one message per accepted sign**, not per frame, so no network
  round trip ever sits inside the per-frame loop.
- **This changes §12.6's plan: the next-gloss prior should also run on the client.**
  Workers AI documents no `logprobs`, so a hosted LLM cannot cheaply return a
  distribution over 250 glosses. A small n-gram LM (or a tiny in-browser LM) over the
  gloss vocabulary gives the fused-acceptance prior at no latency cost. The LLM is then
  needed for exactly one job: **gloss sequence → fluent English** (§5).
- **The edge does two things: gloss → English (Workers AI LLM) and TTS (Workers AI
  Aura / MeloTTS).** Session state lives in a Durable Object. On price, a spoken
  sentence costs about $0.002 with `aura-2-en`. With `melotts` it is effectively free
  (§6).
- **Workers RPC is not needed (user, 2026-09-24).** The browser can't call it, and
  with recognition on the client the only browser ↔ edge traffic is one message per
  accepted sign plus the audio back. That is a plain WebSocket to the Worker/session DO
  (§7).

## 1. The pipeline as it would ship

```
┌──────────────────────────── browser (apps/web) ────────────────────────────┐
│ camera ─► MediaPipe HolisticLandmarker (LIVE_STREAM, WebGL)                 │
│            553 landmarks ─► frameToRows ─► (543,3) frame, NaN = undetected   │
│                                   │                                          │
│            LiteRT.js step model   ▼  (0.2 ms)                                │
│            frame + state ─► embedding, boundary, state'                      │
│            logits = 16 · W·emb   (W = 250 glosses + null + custom signs)     │
│                                   │                                          │
│            D3 decoder (+ D1 fallback) ─► candidate sign (top-k, confidence)  │
│            n-gram next-gloss prior  ─► fused accept   ─► accepted gloss      │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ WebSocket, 1 message per accepted sign
┌──────────────────────────────────────▼──── Cloudflare (apps/edge) ──────────┐
│ Worker (static assets: app, .tflite, .task, wasm)                            │
│ Agent / Durable Object per session: gloss history, custom-sign meanings      │
│   sentence end (pause) ─► Workers AI LLM: glosses → English                  │
│                        ─► Workers AI TTS ─► audio stream ─► browser plays    │
└──────────────────────────────────────────────────────────────────────────────┘
```

| stage | where | why |
|---|---|---|
| camera, MediaPipe | client | user constraint. Runs every frame |
| recognizer (step model) | client | user constraint. 0.2 ms/frame measured |
| class matrix, custom-sign rows | client (persisted per user, §8) | enrolling a sign is appending a row, no re-export |
| decoder (D3/D1) | client | pure arithmetic over per-frame outputs |
| next-gloss prior + fused acceptance | **client** (changed from §12.6) | runs per candidate. No logprobs on Workers AI (§5) |
| gloss → English | edge (Workers AI) | needs an LLM. Runs once per sentence |
| TTS | edge (Workers AI) | runs once per sentence |
| session state | edge (Durable Object) | gloss history, the LLM's context |

## 2. Why the existing TFLite export does not work here

`sb.recognize.export.tflite` is the Kaggle-submission path. It serves the wrong contract
for live use, in three ways:

1. **It takes the whole clip and returns one prediction** (`(T,543,3) → (250,)`). Live
   recognition sees one frame at a time and needs a per-frame output.
2. **Its recurrence is a TFLite `WHILE` loop that needs `SELECT_TF_OPS` (Flex)**
   (`export_tflite`: "recurrent dynamic-loop ops need the fallback"). LiteRT.js ships
   **no Flex kernels**, only builtins.
3. **It has no continuous-model support.** It lacks the null class, the boundary head
   and the cosine head with its `class_mask`. `keras.BUILDERS` has no
   `gru_continuous` entry, as the §12 gap check found.

The fix is a different graph, not a patch to that one.

## 3. The step export (built and measured)

`packages/sb-recognize/src/sb/recognize/export/step.py`, with entry point `export_web(run_dir)`.

| | |
|---|---|
| inputs | `frame (543,3)` raw holistic frame with NaNs · `state (2,256)` |
| outputs | `embedding (256,)` unit-norm · `boundary (1,)` logit · `state_out (2,256)` |
| inside the graph | NaN→0, the ME_132 row gather, the xy column gather, input LayerNorm, 2 GRU layers, embed LayerNorm+Linear+L2 normalization, boundary head |
| outside the graph | class matrix `W (251,256)` → `classes.f32` (257 KB). The client computes `logits = 16·W·emb` and masks rows |
| ops | 17 builtins: `ADD CONCATENATION FULLY_CONNECTED GATHER L2_NORMALIZATION LOGISTIC MEAN MUL NOT_EQUAL RESHAPE RSQRT SELECT_V2 SPLIT SQUARED_DIFFERENCE STRIDED_SLICE SUB TANH`. **No Flex, no WHILE** |
| size | **3.46 MB** fp32. fp16 weights would roughly halve it (not tried) |
| output | `registry/runs/<id>/export/web/{model.tflite, classes.f32, manifest.json}` (gitignored). The manifest carries glosses, `null_index`, `cos_scale`, `class_mask`, the decoder settings, the op list, parity and sha256 |

**Design choices:**
- **The GRU is written as matmuls from PyTorch's weights, in PyTorch's gate order.**
  No Keras layer is involved, so the gate-reordering risk that `keras.py` has to guard
  against does not exist here.
- **The class matrix is kept out of the graph.** Custom signs (§12.4/§12.7) are then a
  row the client appends to `W`, the JS equivalent of `CosineGlossHead.enroll()`. No
  re-export and no server are involved.
- **The client owns the state.** A reset is `state = zeros`, which matches
  `RecurrentSession.reset()`.

**Measurements (2026-09-24, C1 `1790143122`):**

| check | result |
|---|---|
| TFLite (Python) stepped 200 frames vs PyTorch batch `forward_frames`, 6% NaN input | max prob diff **2.4e-6**, boundary logit diff 8e-6, argmax agreement 100% |
| **LiteRT.js 2.5.3, WASM, headless Chrome**, same 200 frames vs the TFLite reference | max embedding diff **6.1e-7**, boundary 5.8e-6 |
| LiteRT.js step latency (WASM, includes tensor create/copy-out) | **0.20 ms median, 0.30 ms p95** |
| LiteRT.js WebGPU | failed to compile in headless Chrome, probably because there was no GPU adapter. **Not needed**: the model is tiny and sequential, so WASM is the right backend |
| LiteRT.js in Node | does not work. The loader injects a `<script>` tag (`document` required), so it is browser-only |

**Cost on the client:** the LiteRT.js WASM runtime is about **9 MB**
(`litert_wasm_internal.wasm`; the threaded and JSPI variants are similar). Add 3.5 MB of
model and 13.7 MB of Holistic `.task`, so the first load is about 26 MB. After that it
is served from cache. Every file is under the 25 MiB per-asset cap for Workers static
assets.

**Runtime alternatives considered and not needed:** TF.js and ONNX Runtime Web. The
TFLite route already works and keeps one export format for web and mobile. It also
avoids ONNX, which this repo abandoned in §6.2.

## 4. MediaPipe on the client

- **`@mediapipe/tasks-vision` `HolisticLandmarker` supports Web**, in IMAGE, VIDEO and
  **LIVE_STREAM** modes. It is float16, a single 13.7 MB `holistic_landmarker.task`.
  The repo's paused TS extractor already drives it (`packages/sb-extract-ts/src/worker.ts`,
  pinned `tasks-vision@0.10.18`).
- **It returns 553 landmarks, not 543.** Its face mesh has 478 points: 468 plus 10
  iris points. GISLR's layout uses the first 468. `schema.ts`'s `frameToRows` already
  truncates the face to 468 and places the groups in GISLR row order (face 0–467,
  left hand 468–488, pose 489–521, right hand 522–542). **Reuse it through
  `apps/shared-ts`; don't port it a third time.**
- **The frame budget is MediaPipe's, not the recognizer's.** Reported figures are about
  15–20 fps on CPU and much higher with the GPU delegate. These are secondary sources,
  **not measured here** (§9, step 1).

**Risks that need a live measurement, not reasoning:**
1. **Frame rate mismatch.** The model learned sign dynamics at GISLR's capture rate
   (phone recordings; the rate isn't stored per clip). A browser that runs Holistic
   at 12 fps shows the model signs that look twice as fast. The client should
   **timestamp frames and resample to the training rate**, or at least measure the
   achieved fps and report it with every result.
2. **Mirroring / handedness.** GISLR was recorded with front cameras. Whether a
   mirrored `<video>` feed swaps `leftHand`/`rightHand` relative to training decides
   whether the hand rows are right. Check this on one known sign before anything else.
3. **Extractor shift.** GISLR's landmarks came from Google's own pipeline. Live
   landmarks come from Tasks HolisticLandmarker. Any systematic difference (smoothing,
   the z scale; z is unused here since C1 is xy) is a domain shift. Worth one small
   A/B: record a few signs, compare live accuracy with the eval numbers.
4. **The rest pose.** C1 was trained with synthetic rest and transition frames (§12.1:
   `p_lowered_rest` 0.7). A real user's hands-down rest may not look like them. D3
   depends on those null frames. GISLR-Sentences v2's "realistic lowered rest" follow-up
   (§12.3) targets this.

## 5. Fused acceptance: why the prior moves to the client

§12.6 planned "LLM next-gloss distribution constrained to the 250-gloss vocabulary ×
recognizer confidence". That needs a probability for each candidate gloss.

- **Workers AI documents no `logprobs`/`top_logprobs`.** Neither the OpenAI-compatible
  endpoint page nor the JSON-mode page mentions them. JSON mode is limited to a handful
  of models and "doesn't support streaming".
- Without logprobs, the options are one generation per candidate (k round trips per
  sign) or asking the LLM to "rank these k glosses" (slow, and not a calibrated
  probability). Both put a network round trip of hundreds of ms on the accept path of
  every sign.
- **An n-gram (or tiny neural) LM over gloss IDs** trained on 12.1's corpus is a few
  KB. It gives a real distribution and runs in microseconds on the client. §12.6
  already planned it as the baseline ("if it gets most of the gain, the LLM earns its
  place only for step (4)"). **Recommendation: make it the design, and measure the
  hosted-LLM prior only if the n-gram prior leaves a large gap.** The circularity
  caveat (12.1's sentences are Claude-written) applies to both.

## 6. Cloudflare: what runs there and within which limits

**Workers platform limits** (developers.cloudflare.com/workers/platform/limits, fetched 2026-09-24):

| limit | Free | Paid |
|---|---|---|
| CPU time / request | 10 ms | 30 s default, 5 min max |
| memory / isolate | 128 MB | 128 MB |
| script size (uncompressed) | 64 MiB | 64 MiB |
| startup (global scope) | 1 s | 1 s |
| subrequests | 50 | 10,000 |
| static assets | 20,000 files, 25 MiB each | 100,000 files, 25 MiB each |

Static asset requests are **free and unlimited**. The ~26 MB client payload costs
nothing to serve. The Worker's own CPU is tiny: awaiting Workers AI is I/O, not CPU.
So even the free plan's 10 ms is plausible for the Worker itself. Workers AI usage is
billed separately.

**Workers AI models for this pipeline** (catalogue and pricing pages, 2026-09-24):

| job | candidates | price |
|---|---|---|
| gloss → English (LLM) | `llama-3.2-3b-instruct`, `llama-3.1-8b-instruct-fp8`, `qwen3-30b-a3b-fp8`, `glm-4.7-flash`, `gpt-oss-20b` | $0.05–0.20/M in, $0.30–0.40/M out. A sentence is about 200 tokens, which is negligible |
| TTS | `@cf/deepgram/aura-2-en` (39 voices, streams audio) · `aura-1` · `@cf/myshell-ai/melotts` | aura-2: $0.030/1k chars · aura-1: $0.015/1k chars · melotts: $0.0002/audio-minute |
| (free tier) | 10,000 neurons/day, then $0.011/1k neurons | |

A 60-character spoken sentence costs about **$0.0018 with `aura-2-en`** and about
$0.0001 with MeloTTS. The LLM share is under $0.0001. **Which LLM:** pick it on
`sb-rescore`'s eval set (§12.6). Don't choose it here. Gloss → English is a small
rewriting task, so start with the 3B/8B models and only move up if they fail the eval.

**Session state:** a Durable Object per session holds the accepted-gloss history (the
LLM's context) and the user's custom-sign meanings. With the WebSocket Hibernation API,
an idle session is **not billed for duration**, while the client stays connected.
Per-connection attachments are capped at 16 KB; larger state goes to DO storage.

**Custom-sign prototypes:** a 256-float row is 1 KB, so a user's whole custom
vocabulary fits in DO storage or D1. R2 is not needed (and was not activated on this
account as of 2026-09-04, §9.3). A client-side IndexedDB copy is the offline cache.

## 7. "Remote functions": the four meanings, and which to use

| meaning | what it is | browser can call it? | fit |
|---|---|---|---|
| **Workers RPC** (service bindings, DO methods) | JS-native RPC between Workers / DOs on one account. Structured-clone payloads up to 32 MiB | **no**. Worker ↔ Worker only | internal only (e.g. a Worker calling the session DO) |
| **Remote bindings** (`remote: true` in wrangler) | local dev (`wrangler dev`) talks to real production resources (e.g. Workers AI) | n/a (dev tooling) | yes, for development: Workers AI has no local simulator |
| **Cap'n Web** (`capnweb`, Cloudflare) | object-capability RPC for **browser ↔ Worker**, over WebSocket/HTTP, promise pipelining, <16 kB, no schemas | **yes** | a good light transport |
| **Agents SDK** `Agent` + `@callable()` | a Durable Object whose `@callable` methods the browser calls through `AgentClient`/`useAgent`, with WebSocket state sync (`setState`) and built-in SQLite | **yes** | **recommended**: session DO, RPC, state sync and storage in one |
| **SvelteKit remote functions** (`query`/`command`/`form`) | framework-level typed server functions. **Experimental** (opt-in flags). Fetch-based; `query.live` streams, no WebSocket | yes (via the SvelteKit app) | only if the web client is SvelteKit; not needed for a single WebSocket |

**Recommendation:** an Agents SDK `Agent` per session. The client calls
`acceptSign(gloss, confidence, t)` and `endSentence()`. The Agent keeps the history,
calls Workers AI and streams TTS audio back. If the Agents SDK turns out to be too
heavy, Cap'n Web over a WebSocket to a plain DO does the same job. **Resolved 2026-09-24:** the user meant Workers RPC. It is not needed as a design element. The browser can't call it, and with the recognizer on the client the only browser↔edge traffic is one message per accepted sign plus the audio back. That is a plain WebSocket (or HTTP) to the Worker. Workers RPC survives only as the ordinary way the Worker calls its session Durable Object's methods.

## 8. Latency budget (sign accepted → audio)

| step | budget | basis |
|---|---|---|
| MediaPipe per frame | ~15–50 ms | **unmeasured**, secondary sources. Set by device and GPU |
| recognizer step | 0.2 ms | **measured** (LiteRT.js WASM) |
| D3 commit after sign end | median +1 frame (~33 ms at 30 fps) | **measured** offline, `continuous-models.md` |
| client → edge message | 1 RTT, ~20–80 ms | typical. Once per sign |
| sentence end detection | pause length (e.g. 0.5–1 s of null) | design choice (§12.6 "when to emit") |
| LLM gloss → English | ~0.3–1 s | **unmeasured**. Depends on the model |
| TTS first audio | ~0.2–0.5 s | **unmeasured**. aura-2 streams |

The dominant terms are the **pause used to detect a sentence end** and the LLM. Both
are edge/UX choices. The client-side recognizer is not a bottleneck.

## 9. Proposed build plan (for review; nothing past step 0 is built)

| step | what | output | TODO |
|---|---|---|---|
| **0 (done)** | step export + LiteRT.js browser parity | `sb.recognize.export.step`, this report | §12.5 |
| 1 | **`apps/web` live prototype**: camera → HolisticLandmarker → `frameToRows` → LiteRT.js step model → D3 → on-screen glosses. Measure achieved fps and check mirroring on known signs | a page served by `wrangler dev` | §12.5, §10.2 |
| 2 | `apps/shared-ts`: generated ports of the gloss list, ME_132 rows and the manifest schema, plus a parity script against Python (`sb.core.*`) | TS module + check | §12.5 |
| 3 | n-gram gloss prior + fused acceptance, **offline first** in a notebook on 12.1's streams, compared against C1 D3's GER | notebook + GER table | §12.6 |
| 4 | `apps/edge`: Agent (DO) session, gloss → English on Workers AI with a versioned prompt, TTS stream back | Worker + `wrangler.jsonc` | §12.6 |
| 5 | custom signs: capture → embed with the step model → append a row to `W`, persisted per user | web + edge | §12.4 → §12.7 |

Step 1 is the next action. It is the first time real webcam landmarks reach the
model, and risks 1–4 of §4 can only be answered there.

**Open questions for you:**
1. ~~"Cloudflare remote functions"~~: **answered**. The user meant Workers RPC. Not needed (§7).
2. **Plan**: Free or Paid Workers plan? The Worker fits either. The Paid plan matters
   for Workers AI volume beyond 10k neurons/day and for longer CPU if the edge grows.
3. **Web framework** for `apps/web`: plain TS + Vite (my default, smallest), or
   SvelteKit/React?
4. **Moving the next-gloss prior to the client (§5)**: accept this change to §12.6?

## Sources

- Cloudflare Workers limits: https://developers.cloudflare.com/workers/platform/limits/
- Static assets billing: https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/
- Workers AI models: https://developers.cloudflare.com/workers-ai/models/ · pricing: https://developers.cloudflare.com/workers-ai/platform/pricing/
- Workers AI OpenAI compatibility: https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/ · JSON mode: https://developers.cloudflare.com/workers-ai/features/json-mode/
- Aura-2-en: https://developers.cloudflare.com/workers-ai/models/aura-2-en/
- Durable Objects WebSockets / hibernation: https://developers.cloudflare.com/durable-objects/best-practices/websockets/
- Workers RPC: https://developers.cloudflare.com/workers/runtime-apis/rpc/
- Agents SDK: https://developers.cloudflare.com/agents/api-reference/agents-api/
- Cap'n Web: https://github.com/cloudflare/capnweb · https://blog.cloudflare.com/capnweb-javascript-rpc-library/
- SvelteKit remote functions: https://svelte.dev/docs/kit/remote-functions
- LiteRT.js: https://developers.googleblog.com/litertjs-googles-high-performance-web-ai-inference/ · https://developers.google.com/edge/litert/web/get_started · https://www.npmjs.com/package/@litertjs/core
- MediaPipe Holistic Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker
