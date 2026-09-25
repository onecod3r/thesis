# apps/ — deployment surfaces

The live pipeline: camera → landmarks → recognizer → LLM → English → speech.
Target platform: **Cloudflare Workers** (TODO §12.5). **Deployed 2026-09-25:**
https://signbridge.onecoder1.workers.dev (Workers Free plan, `onecoder1` account).

**Sign → speech runs in the browser as of 2026-09-24** (`web/`, see its README):
- camera or video → Holistic → C1 step model (LiteRT.js) → lattice decoder + trigram
  prior → rule-based English → `speechSynthesis`;
- every Python-derived piece has a parity check, and a headless Chrome replay of 24
  held-out streams matches Python exactly;
- `edge/` serves it as static assets, with an optional LLM English route.
- **Live overlay improved 2026-09-25**: camera mode shows the raw video with a bright
  pose skeleton (the model's own ME-126 upper-body landmarks) and clearer hand markers
  drawn live on top, not just faint dots.

**Speech → gloss runs in the browser as of 2026-09-25** (`web/speech.html`, user decision:
Free plan, no server-side Containers): mic or an audio file → `apps/edge` `/api/asr`
(Workers AI `whisper-large-v3-turbo`) → English → `rules_v1`/`rules_v2` (ported from
`sb.synthesize.gloss` to TypeScript on **wink-nlp**, since Pyodide has no spaCy) → the
team's T5 refiner (**mixed-precision** ONNX in a Worker — int8 encoder + fp32 decoder,
`num_beams=2` — see TODO §13 for the "reduce inference time" tuning) → the guarded hybrid.
Gloss → *signs* (rendering) is still deferred: the gloss engine now runs, but there are no
sign videos to play.

```
apps/
├── web/         # browser client: camera, MediaPipe landmarks, streaming recognizer, custom-sign capture
├── edge/        # Cloudflare Worker(s): session state, downstream LLM, TTS
└── shared-ts/   # TypeScript contracts both sides import: landmark layout, gloss vocab, wire messages
```

## Data flow (working assumption — §12.5 confirms or changes it)

```
camera ──► [web] MediaPipe Holistic (per frame, 543×3)
             │  subset + xy (ME-126, same as training)
             ▼
           [web] streaming recognizer (ContinuousGRU, one state per session)
             │  per-frame gloss distribution + boundary score
             ▼  candidate events (top-k glosses + confidence), not raw frames
           [edge] session (Durable Object): accepted-gloss context
             │  LLM next-gloss prior × recognizer confidence ─► accept (§12.6)
             │  accepted glosses ─► LLM ─► fluent English
             ▼
           [edge] TTS ─► audio ─► [web] playback
```

Why the recognizer is assumed to run in the browser: it runs every frame, and a
network round trip per frame would dominate latency. Only accepted-sign
candidates cross the network. §12.5 quantifies this.

## Invariants (fixed regardless of §12.5's answers)

- **The Python side stays authoritative.** Landmark layout comes from
  `sb.core.schema` (`LANDMARK_TENSOR` v1), the vocabulary from `sb.core.vocab`,
  and the subsets from `sb.core.subsets`. `shared-ts/` holds *ports* of them and
  needs a parity check (compare `packages/sb-extract-ts/src/schema.ts` and `sb.extract.parity`).
- **Causal only.** Anything deployed here is a streaming model (`streaming: true`
  in `meta.json`). BiLSTM and the other offline models never ship.
- **The model comes from the registry.** Export from a promoted run
  (`sb-promote`), never from an ad-hoc checkpoint.
- **Custom signs are per-user data, not model weights.** They are prototypes
  enrolled into the cosine head (§12.4, §12.7).

## Status

| part | TODO | state |
|---|---|---|
| deployment research | §12.5 | **done 2026-09-24**: `docs/reports/deployment-research.md`. Recognizer confirmed client-side (LiteRT.js, step export) |
| **web app, sign → speech** | §12.5 step 1, §12.6 | **built 2026-09-24, camera + overlay verified live 2026-09-25** (`web/`): parity tests pass, browser replay 24/24 identical to Python, headless-Chrome-verified camera stream + Holistic + a clearly visible pose/hand overlay |
| **web app, speech → gloss** | §13 Phase 4 | **built + deployed 2026-09-25, with the real T5 checkpoint, tuned for latency** (`web/speech.html`, live at `/speech`): ASR + gloss run client-side, `npm test` passes. T5 ships **mixed precision** (int8 encoder + fp32 decoder, `num_beams=2`) — same guard-acceptance as full fp32 at ~30% less download and fewer decode steps |
| edge Worker | §12.5 step 4, §13 | **built + deployed 2026-09-24/25** (`edge/`): static assets + `/api/english` + `/api/asr` (Workers AI), both live. Found + fixed a missing `assets.binding` that 500'd every real 404 |
| LLM + TTS | §12.6 | rules English + browser voices in the app; LLM route in `edge/`; Workers AI TTS not wired |
| custom signs | §12.4 / §12.7 | plan pending |
| live MediaPipe mode | §10.2 | folded into §12.5 |
