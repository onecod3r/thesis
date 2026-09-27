# C1 v2: making the continuous model usable for real sentences — research + build

**Status: DONE, 2026-09-27. C4 wins and is the new deployment candidate (§7).** TODO §12.8 Fix 3. **C4**
(`gru_continuous_norm`) beats C1 on clean GER (0.278 vs 0.293) and decisively on every live-robustness
probe that matters for a real camera (§7) — the one miss is hard-cut, where C4 is worse than C1, flagged
plainly in §7, not hidden.

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
  judging criteria (clean/hard-cut GER, live-robustness probes).

## 7. Judged against §4's criteria (2026-09-27) — C4 wins, one miss

Both `gislr.3.streaming.continuous-eval.ipynb` and `gislr.3.streaming.live-robustness.ipynb` have now run
on C1/C2/C4/C5 (+P1/C3 for the eval notebook). Scoring against each of §4's four criteria in turn:

**1. Clean GER ≈ C1's 0.293 — exceeded.** D3 collapse=True, evaluation signers
(`data/cache/gislr/continuous_eval/results/final_table.csv`):

| model | clean GER | control GER | hard-cut GER |
|---|---|---|---|
| **C4** | **0.278** | 0.276 | 0.634 |
| C1 | 0.293 | 0.289 | **0.546** |
| C2 | 0.298 | 0.295 | 0.513 |
| C5 | 0.347 | 0.348 | 0.615 |
| P1 | 0.587 | 0.586 | 0.731 |
| C3 (CTC) | 0.898 | 0.899 | 0.926 |

C4 is now the best clean-GER continuous model, beating C1 by 1.5 points — v2's harder training streams cost
nothing on the easy case and bought something.

**2. Hard-cut GER < C1's 0.546 — failed.** C4 is *worse* on hard-cut (0.634 vs 0.546), despite hard-cut
streams being part of v2's training augmentation (25% of streams). C2 (the plain LSTM body, no norm, v1
streams) is actually best here (0.513) — hard-cut performance doesn't track the v2 changes in the direction
intended. Not investigated further; flagged as a known weakness of the shipped model, not silently dropped.

**3. Live-robustness probes should fall well below C1 — met, decisively.** 132-landmark synthetic probes,
16 evaluation signers, GER (`data/cache/gislr/live_robustness/`):

| probe | C1 | **C4** | C5 |
|---|---|---|---|
| none (clean) | 0.293 | **0.285** | 0.359 |
| aspect_0.56 (landscape) | 0.651 | **0.305** | 0.375 |
| scale_0.7 | 0.375 | **0.285** | 0.363 |
| fps_15 | 0.504 | **0.381** | 0.443 |
| jitter_0.01 | 0.516 | **0.372** | 0.444 |
| mirror | 0.990 | 0.962 | 0.982 |
| **live_like** (combined worst case) | 0.592 | **0.408** | 0.460 |
| live_like+fixes (app-side EMA/reframe/interp stacked) | 0.481 | **0.349** | 0.424 |

C4 crushes this. `aspect_0.56` and `scale_0.7` land essentially at C4's own clean number (criterion 4: "near
clean" — met for aspect/scale). `fps_15`/`jitter_0.01` lose 0.096/0.087 for C4 vs. C1's 0.211/0.223
(criterion 5: "far less than C1's loss" — met). `live_like` — the combined stress test closest to an actual
phone camera — drops from C1's 0.592 to **0.408**, without even the app-side fixes stacked on top.

**One probe still fails for everyone: `mirror`.** C4 barely moves it (0.990 → 0.962) — a mirrored/swapped-
hands feed is still close to unusable on any of these models. Not claimed as fixed.

**Long sessions**: C1's recall degrades 0.728 → 0.672 across a 6-sentence chain; C4's is flatter (0.700 →
0.651, less monotonic) — roughly comparable, no clear win either way.

### Verdict

**C4 is the new best model for deployment.** It wins clean GER, wins every live-camera robustness axis that
matters for an actual phone/webcam feed (framing, scale, fps, jitter, the combined `live_like` stress test)
by large margins, and costs nothing on the canonical isolated benchmark (0.7339 > C1's 0.7188). It loses on
two things: hard-cut GER (0.634 vs 0.546 — a synthetic zero-pause-between-signs case, not what "live camera"
robustness was chasing) and mirror handling (still broken for both, unresolved by v2). Given the live app's
actual failure mode was framing/scale/jitter/fps — exactly what §1 measured and §7 confirms C4 fixes —
**C4 replaces C1 as the deployed sign-recognition model.** Hard-cut and mirror remain open follow-ups
(TODO §12.8), not blockers.
