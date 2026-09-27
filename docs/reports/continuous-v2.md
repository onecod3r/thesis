# C1 v2: making the continuous model usable for real sentences — research + build

**Status: built 2026-09-26, trained 2026-09-27, not yet scored on the criteria that matter (§6).** TODO
§12.8 Fix 3. Two runs, **C4** (`gru_continuous_norm`) and **C5** (its ablation), trained by
`experiments/recognition/gislr.1.models.continuous.ipynb` §6b (user ran it 2026-09-27) — still need
`gislr.3.streaming.continuous-eval.ipynb` and `gislr.3.streaming.live-robustness.ipynb` before §4's
judgment can be made.

| | |
|---|---|
| **Question** | C1 scores GER 0.293 on synthetic GISLR-Sentences but is "not working at all" on sentences from a live camera. What would make it usable, and what can be fixed by training? |
| **Inputs** | our measurements (`continuous-models.md`, `live-streaming-gap.md`, TODO §12.8 diagnosis), plus the literature below |
| **Output** | a v2 training recipe, one change per measured failure, built and checked, handed over for training |

---

## 1. What is wrong with C1, measured

| failure | evidence (C1 + D3, evaluation signers) | cause |
|---|---|---|
| **signs run together** | hard-cut streams (no pause between signs): GER **0.546** vs 0.293 with pauses | D3 only commits after null frames; C1 never saw signs back to back or shortened, as they are in fluent signing |
| **low fps** | 15 fps with repeated frames: **0.504**, missed signs 0.353 | repeated frames look like pauses, which cut signs into fragments |
| **jitter** | 1% landmark noise: **0.516**, extra signs 0.165 | noise on still hands reads as motion |
| **framing** | landscape aspect 0.56: **0.651**; farther away (scale 0.7): 0.375 | C1 reads raw image coordinates; GISLR is portrait and close |
| **mirror / hand labels** | mirrored or swapped hands: **0.990** (isolated 3%) | hand slots and x direction are part of the input |
| **long sessions** | 6 chained sentences: 0.415; recall 0.73 → 0.67 along the chain | training streams stopped at 600 frames |
| **non-sign movement** | fidget/hold/reverse noise: GER 0.982 unfiltered (`sign-to-speech-downstream.md`) | nothing in training was ever movement without a gloss |
| **idle never ends a sentence** | live phone screenshot: p(null) = 0.001 with hands out of view | not reproducible offline; the leading suspect is hand labels (Tasks vs legacy Holistic), i.e. the mirror row |

On clean synthetic streams the classifier is not the problem: substitutions are already at the oracle's
level (0.213 vs 0.221). Almost everything above is **distribution shift between composed training
streams and a live camera**.

## 2. What the literature says

- **Stitched isolated signs are too long and too clean.** Composing continuous signing from isolated
  clips by linear interpolation makes sentences **1.63×** the real length. A learned co-articulation model
  (BRAID) brings this to 1.02× and cuts pose error 18%
  ([Towards Continuous Sign Language Conversation from Isolated Signs, 2026](https://arxiv.org/html/2605.14705)).
  Real signs inside sentences are shortened, and their edges merge into the next sign.
- **Trimming the frames around an isolated sign** is a cheap substitute for co-articulation used in
  co-articulated recognition work (see e.g. [BSL-1K, Albanie et al., ECCV 2020](https://arxiv.org/abs/2007.12131),
  which studies recognition of co-articulated signs cut from continuous video).
- **A background class for transitions and non-signs**, trained on real continuous data, took online WER
  from 38.4% to 22.1% ([Zuo et al., EMNLP 2024](https://arxiv.org/abs/2401.05336)). The same design
  appears in pose-based sign spotting, where co-articulation frames are a discarded background category
  ([Pose-Based Sign Language Spotting, 2025](https://arxiv.org/abs/2512.08738)). C1 already has the null
  class; it has only been shown synthetic transitions and rest, never movement that is not a sign.
- **Models trained only on isolated data can segment continuous signing** if trained to mark
  boundaries explicitly: boundary-prediction and latent-curvature losses on synthetic concatenations
  ([Recognizing words in continuous sign language with a model trained only on isolated words, 2026](https://www.sciencedirect.com/science/article/pii/S2590123026024692)).
  C1's boundary head is this idea's boundary part.
- **Normalizing to the body** (shoulder-centred, shoulder-width scaled) is what the 1st-place GISLR
  solution does ([repo](https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution)),
  and what our own phonology models do. It removes framing instead of calibrating it away.

## 3. The v2 recipe (built)

Each change answers one row of §1. All of them are **training-time**, and C4's input is **the same raw
ME_132 xy as C1**, so the web app feeds it without any change of its own. The normalization is a fixed
layer inside the model.

| change | answers | where |
|---|---|---|
| **Stream normalization** (C4 only): per frame, each hand re-slotted to the pose wrist it is nearest; shoulder-centred; divided by shoulder width; hand presence flags | framing, scale, **hand-label convention** | `architectures.StreamNormFrontend`, arch `gru_continuous_norm` |
| **Clip trim** (p 0.5, up to 20% off each end) and **speed-up** (p 0.5, ×0.8–1.5) | signs run together; fluent signing is shorter | `continuous.augment.Augmenter.clip` |
| **Hard-cut streams** (25%: no gap frames at all) | signs run together; teaches the boundary head to split without a pause | `Augmenter.stream_cfg` |
| **Noise as null** (30% of streams: one fidget / hold / reverse block from other clips) | non-sign movement; false signs in rest | `Augmenter._noise` (reuses `sequences.noise`) |
| **Long streams**: up to 16 signs / 1,500 frames (was 6 / 600) | long-session drift | `composer_overrides` |
| **Camera augmentation**, one draw per stream: scale 0.7–1.35, x aspect 0.55–1.25, rotation ±8°, shift ±0.12, mirror (p 0.5, with hands, arms and face points swapped) | framing, mirror | `Augmenter._camera` |
| ... and per frame: jitter up to 0.8% (p 0.5), hand dropout up to 20% (p 0.4), **15/10 fps** with repeat or interpolation (p 0.3) | jitter, low fps, flickering hands | same |

**C5** takes the same v2 streams with C1's raw input (`gru_continuous`), so the results say how much comes
from the normalization (C4 vs C5) and how much from the training streams (C5 vs C1). Everything else
(GRU 2×256, cosine head, boundary head, loss, optimizer, early stopping) is C1's, so comparisons stay
all-else-equal.

### Checks passed before handing over (2026-09-26)

| check | result |
|---|---|
| re-slot rule on GISLR's own labels (2.86M frames) | a hand sits a median 0.09 shoulder widths from its own pose wrist, 1.6 from the other; the rule relabels 0.20% / 0.08% of one-hand frames (crossings), 3.2% of the 0.3% two-hand frames |
| stream norm under zoom 0.7 + shift | output unchanged (max \|diff\| 6e-7) |
| stream norm with every hand label swapped | output identical on 100% of frames |
| `RecurrentSession` (frame by frame) vs batch forward, C4 | max \|diff\| 1e-7 |
| mirror permutation | exact involution; 7.9% of face points self-paired (the midline) |
| 300 augmented streams | every sign label still on its own frames; null count exact; 24% hard-cut, 30% with noise; length p50 375, p95 788 frames |
| driver smoke run, C4 and C5 (3 batches each) | trains; 8,484 streams / epoch; C4 929k params (C1 928k) |
| v1 runs (C1–C3, Copen, P1) | augmentation off (defaults), so their recipe is unchanged |

## 4. How v2 will be judged

On the same protocol as C1 (16 evaluation signers, decoder chosen on the 5 selection signers):

1. **Clean GISLR-Sentences GER** should stay near C1's 0.293. v2 trains on harder streams, so a small
   cost here is acceptable if items 2–3 improve a lot.
2. **Hard-cut GER** should beat C1's 0.546: that is the fluent-signing case.
3. **Live-robustness probes** (C1's decoder settings for all runs): `live_like` should fall well below C1's
   0.592 (0.481 with every app-side fix); `mirror`, `aspect_0.56` and `scale_0.7` should land near clean;
   `fps_15` and `jitter_0.01` should lose far less than C1's +0.21 / +0.22.
4. **Canonical isolated accuracy** should stay near C1's 0.7188.

If C4 wins, the web app needs one change: the TFLite export must carry the normalization layer
(`apps/web/tools/export.py` rebuilds the model in Keras, so `StreamNormFrontend` needs a Keras port with a
parity fixture). The app's D3 decoder, prior and lattice stay as they are.

## 5. What training cannot fix

- **Real transitions.** Every training and test stream is still composed from isolated clips. The
  strongest published lever, clips cut from **real** continuous signing (Zuo et al.), needs real data: the
  landmark recorder in the web app (TODO §12.8 Fix 2), then forced alignment of known sentences with C4's
  own per-frame probabilities, or ASLLRP DAI (continuous ASL with time-aligned glosses).
- **The live idle bug** (p(null) 0.001) has not been reproduced. C4's re-slotting and mirror training cover
  the leading suspect (hand labels); the recorder is still needed to confirm it on a real session.
- **Decoding without pauses.** D3 cannot split two signs with no null frame between them; D1 (boundary
  head) can. A combined decoder D5 = D3 ∪ D1 is still open (TODO §12.3) and matters more once hard-cut
  training has sharpened the boundary head.
- **A learned co-articulation model** (BRAID-style inpainting of the joins) would replace linear gaps with
  realistic ones. It is a research project of its own, not a first step.

## 6. Training results (2026-09-27)

Both runs auto-resumed to completion via early stop (plateau 12/12). Each process then exited with Windows
code `3221226505` (0xC0000409) *after* printing its own `DONE` line — almost certainly a benign CUDA/
DataLoader teardown crash on Windows, not a training failure: checkpoints, `assets/history.json` and the
canonical `sb-evaluate` pass that ran immediately afterward are all intact and internally consistent, so the
numbers below are trusted as-is.

| run | epochs | best seg acc | best epoch | frame acc@best | boundary F1@best | canonical accuracy | canonical macro |
|---|---|---|---|---|---|---|---|
| C1 (v1, reference) | 95 | 0.7233 | 83 | 0.6820 | 0.4271 | 0.7188 | 0.7163 |
| **C4** (norm + v2 streams) | 126 | 0.6929 | 114 | 0.6246 | 0.3568 | **0.7339** | **0.7315** |
| **C5** (v2 streams, raw input) | 105 | 0.6489 | 93 | 0.5829 | 0.3550 | 0.6795 | 0.6770 |

- **C4 beats C5 on every metric** — the stream normalization itself is buying something beyond just the
  harder v2 training streams.
- **Both v2 runs score below C1 on segment/frame/boundary accuracy, measured on their own (harder)
  validation streams** — expected and, per §4, not disqualifying on its own: v2 trains on deliberately
  harder streams (hard-cut, noise-as-null, camera augmentation), so a cost here is the price of the
  robustness §4's real criteria are meant to check for.
- **C4's canonical isolated accuracy (0.7339) is already above C1's (0.7188)** — the one number scored
  identically to every other run on the leaderboard. That's a good sign, but it is not one of §4's actual
  judging criteria (clean/hard-cut GER, live-robustness probes) — those still require
  `gislr.3.streaming.continuous-eval.ipynb` and `gislr.3.streaming.live-robustness.ipynb` on C4/C5, not yet
  run. **Training succeeding is necessary, not sufficient, to know whether v2 fixed the live-camera
  problem.**
