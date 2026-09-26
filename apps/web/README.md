# apps/web — sign → speech in the browser

Camera or video file → MediaPipe Holistic → the C1 step model (LiteRT.js, WASM) →
D3 + look-ahead lattice with the trigram prior → glosses → English (rules) → the
browser's own voice. **Everything runs on the device** (TODO §12.5, §12.6). Speech → sign is
not in the app yet (user decision, 2026-09-24).

Decided 2026-09-24: vanilla **TypeScript + Vite** (SolidJS or QwikCity later), served as
**Workers static assets** by `apps/edge` on the Free plan.

## Run it

```bash
# 1. Python writes the model, prior, lexicon, replay streams and test fixtures (gitignored)
.venv/Scripts/python.exe apps/web/tools/export.py            # assets + fixtures, about 1 min

# 2. the app
cd apps/web
npm install
npm test          # parity with Python: decoder, n-gram prior, gloss -> English (no browser)
npm run dev       # http://localhost:5173 (copies WASM, downloads the Holistic model once)
npm run build     # dist/, what apps/edge serves
node scripts/browser-check.ts   # headless Chrome: replays every held-out stream, compares with Python
```

**Models from Kaggle instead of Python** (public, MIT; versions pinned in `kaggle.models.json`):

```bash
npm run models            # sign -> speech bundle + T5 web bundle -> public/assets/, sha256-checked
```

Or leave the models off the server entirely: open either page with **`?models=kaggle`** and it
fetches its model bundle from Kaggle at runtime (Kaggle serves both download hops with CORS).
The replay streams and MediaPipe's `.task` still come from this site. Publishing:
`.venv/Scripts/python.exe apps/web/tools/publish_kaggle.py --apply`, then bump the version in
`kaggle.models.json`.

**Pages** (one header bar, 2026-09-26): **Sign → Speech** (`/`), **Speech → Sign** (`/speech`) and
**Landmark test** (`/landmarks`, TODO §12.8). The landmark test runs the camera (or a video file)
through three MediaPipe setups side by side: Holistic (what the app uses), Hands + Face + Pose as
three separate task models, and Hands + Face. Each panel draws the exact frame its models read
plus the overlay in one canvas, so the overlay cannot drift off the body; mirroring flips both.
Per panel: frame rate, detection time per model, render time, camera-to-overlay latency, pose
visibility, hand labels and handedness scores, how often each part was found, and whether each
hand's label agrees with the pose wrist it is nearest (the C4 re-slot rule).

**Recognizer variants** (Sign → Speech, "Recognizer"; `variants` in `pipeline.config.json`): C1,
C2, and **C1 + C2 averaged per frame** (offline GER 0.244 vs C1's 0.276,
`docs/reports/window-ensembles.md`); C4, C5, C4 + C5 and C1 + C4 appear once those runs are
trained and `tools/export.py assets` is re-run. Each variant uses the D3 setting chosen for it on
the selection signers; the lattice and prior are shared. `?variant=C1%2BC2` selects one from the URL,
and `?check&variant=…` replays the held-out streams against Python for that variant.

With no camera (a remote machine, for example), choose **Held-out replay**. It feeds GISLR
landmarks of evaluation-signer streams straight to the recognizer, and **Check all**
compares every stream with Python.

## What is where

| file | what | Python it ports | parity |
|---|---|---|---|
| `src/pipeline/recognizer.ts` | step model on LiteRT.js; `logits = 16·W·emb`, softmax; `Ensemble` averages several per frame | the demo notebook's `frame_probs`; `sb.recognize.sequences.windows.average` | browser replay: **24/24 streams identical for C1, C2 and C1+C2** (2026-09-26) |
| `src/pipeline/extractors.ts`, `src/landmarks-main.ts` | the landmark test page: Holistic / pose+face+hands / face+hands task models, overlay, stats | – | headless Chrome on a signing video: all three run, overlay on the body, hand labels agree with the pose 100% |
| `src/pipeline/decoder.ts` | `OnlineDecoder`, `decide` (none/rescore/agree), `Lattice` | `sb.recognize.continuous.{online,fuse,select}` | `npm test`: 60 streams × 6 rules identical |
| `src/pipeline/prior.ts` | Kneser-Ney n-gram from `prior.json` | `sb.rescore.prior.NgramLM` | `npm test`: 592 histories, max diff 3e-15 |
| `src/pipeline/gloss2en.ts` | gloss → English rules (offline) | `sb.rescore.gloss2en.convert` | `npm test`: 5,724 sequences identical |
| `src/pipeline/holistic.ts` | MediaPipe HolisticLandmarker, VIDEO mode, **CPU delegate** (`holistic_delegate`; `?delegate=GPU` to compare) | `packages/sb-extract-ts` worker | headless smoke test: loads, runs (GPU delegate) |
| `src/pipeline/session.ts` | sentence end on a pause, 30 fps clock, speak policy | new (app behaviour) | – |
| `src/pipeline/speech.ts` | `speechSynthesis` | the SAPI arm of the TTS notebook | – |
| `../shared-ts/src/landmarks.ts` | Holistic result → (543, 3) frame | reuses `packages/sb-extract-ts/src/schema.ts` | – |
| `pipeline.config.json` | the deployed settings (rule, thresholds, fps) | – | – |
| `tools/export.py` | writes `public/assets/` and `test/fixtures/` | – | – |

## Deployed settings (`pipeline.config.json`)

- **Rule: the lag-2 lattice** (k=5, λ=0.2, θ=0.269). It beat the plain floor on missed,
  wrong and extra signs and on noise, on the evaluation signers
  (`docs/reports/sign-to-speech-downstream.md` §2.2). An uncertain sign is decided once two more
  signs arrive or the sentence ends. The page says when it is waiting.
- **Uncertainty is shown, not hidden** (`docs/reports/sign-to-speech-demo-analysis.md` §4):
  - glosses under 0.5 confidence are greyed out with a "?";
  - a sentence is spoken automatically only if every gloss is ≥ 0.5; otherwise "Speak anyway";
  - the glosses always stay next to the English.
- **Sentence end:** 45 consecutive null frames (about 1.5 s). The lattice then decides what it
  holds, English is produced, and the decoder and the recurrent state reset.
- **30 fps clock:** a slower live feed repeats frames, so signs keep the speed the model learned.
  Toggle on the page.
- **Nobody in view → pause.** On an all-NaN frame (no body detected) the recognizer says
  p(null) ≈ 0.01, i.e. "a sign is happening". GISLR frames always contain a body. So the app
  treats such frames as null and resets the state, instead of letting them produce signs.

## Not verified yet (needs a camera and a person)

These are the §4 risks of `docs/reports/deployment-research.md`, which only a live test answers:
1. **Accuracy on real signing.** Every number so far is on composed GISLR clips. Expect worse.
2. **Mirroring.** Whether a front camera's hands match GISLR's `left_hand`/`right_hand` rows.
   Try a one-handed sign with and without **Mirror input**.
3. **Achieved frame rate.** It is shown live. Below about 15 fps the 30 fps clock repeats a lot of frames.
4. **Rest pose.** Whether the model calls a real "hands down" rest null (D3 depends on it).
