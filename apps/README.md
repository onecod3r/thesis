# apps/ — deployment surfaces

The live pipeline: camera → landmarks → recognizer → LLM → English → speech.
Target platform: **Cloudflare Workers** (TODO §12.5).

**Sign → speech runs in the browser as of 2026-09-24** (`web/`, see its README):
- camera or video → Holistic → C1 step model (LiteRT.js) → lattice decoder + trigram
  prior → rule-based English → `speechSynthesis`;
- every Python-derived piece has a parity check, and a headless Chrome replay of 24
  held-out streams matches Python exactly;
- `edge/` serves it as static assets, with an optional LLM English route.

Speech → sign is not in the app yet (user decision, 2026-09-24). Its gloss engine needs spaCy,
and there are no sign videos to play.

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
| **web app, sign → speech** | §12.5 step 1, §12.6 | **built 2026-09-24** (`web/`): parity tests pass, browser replay 24/24 identical to Python. **Not yet tried on a camera** |
| edge Worker | §12.5 step 4 | **built 2026-09-24** (`edge/`): static assets + `/api/english` (Workers AI, prompt v1). Verified offline; LLM route and deploy need Cloudflare credentials |
| LLM + TTS | §12.6 | rules English + browser voices in the app; LLM route in `edge/`; Workers AI TTS not wired |
| custom signs | §12.4 / §12.7 | plan pending |
| live MediaPipe mode | §10.2 | folded into §12.5 |
