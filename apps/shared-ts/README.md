# apps/shared-ts — contracts shared by web/ and edge/

TypeScript **ports** of the Python contracts, plus the wire messages between
the client and the Worker. The Python definitions are authoritative. If a port
disagrees, the port is wrong.

| contract | authoritative source | notes |
|---|---|---|
| landmark layout (543 rows × xyz) | `sb.core.schema` | a port already exists in `packages/sb-extract-ts/src/schema.ts` (paused package). Reuse it; don't write a third copy |
| subsets (ME-126 indices) | `sb.core.subsets` | should be *generated* from Python, not hand-copied |
| gloss vocabulary (250 + null) | `sb.core.vocab` + GISLR's `sign_to_prediction_index_map.json` | custom signs extend it per user at runtime |
| wire messages web ↔ edge | defined here | candidate event (top-k glosses, confidence, boundary, frame time) → accept/English/audio back |

Each port needs a parity check, like `sb.extract.parity` does for the Deno extractor.

**In use (2026-09-24):**
- `src/landmarks.ts`: Holistic result → (543, 3) frame. It re-exports `frameToRows` from
  `packages/sb-extract-ts/src/schema.ts` (no third copy), cuts the 478-point Tasks face to 468,
  and optionally mirrors. It also rebuilds replay frames from stored rows.
- `src/contracts.ts`: types for the files `apps/web/tools/export.py` writes (step manifest,
  pipeline config, n-gram JSON, lexicon, replay index).

The vocabulary and the landmark subset are not restated here. They travel in the step export's
`manifest.json`, which Python writes.
