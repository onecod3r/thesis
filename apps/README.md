# apps/ — deployment surfaces

The live pipeline: camera → landmarks → recognizer → LLM → English → speech.
Target platform: **Cloudflare Workers** (TODO §12.5). Nothing here runs yet.
This tree fixes the *split* and the *contracts*. The tooling (bundler,
`wrangler` config, which inference runtime) is chosen by the §12.5 research, not
before it.

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
| deployment research | §12.5 | **done 2026-09-24**: `docs/reports/deployment-research.md`. Recognizer confirmed client-side (LiteRT.js, step export). Build plan awaiting review |
| LLM + TTS | §12.6 | plan pending |
| custom signs | §12.4 / §12.7 | plan pending |
| live MediaPipe mode | §10.2 | folded into §12.5 |
