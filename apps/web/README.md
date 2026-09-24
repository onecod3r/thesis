# apps/web — browser client

Camera capture, MediaPipe landmarks and the streaming recognizer, all running
per frame on the device. Also the capture flow for custom signs (TODO §12.7).

**Decided 2026-09-24 (§12.5):** vanilla **TypeScript + Vite** (SolidJS or QwikCity
later), recognizer on **LiteRT.js** (WASM) from `sb.recognize.export.step`, served as
**Workers static assets**. The next-gloss prior (n-gram tables) and fusion rule also
run here (§12.6).

Responsibilities:
- MediaPipe Holistic in `LIVE_STREAM` mode. This is a callback contract that
  drops frames under load, which is not the batch `VIDEO` mode the extractors use
  (§10.2).
- Build the model input exactly as training does: ME-126 subset, xy only,
  NaN→0 at the model boundary. Import the layout from `shared-ts/`, never
  restate it.
- Hold one recurrent state per session, and reset it on the decoder's
  accept/boundary signal (§11, §12.3).
- Send candidate events, not frames, to `edge/`.
- Custom signs: record N examples, embed them with the recognizer, and enroll
  the prototype locally (§12.4 decides the method).
