# Improvements: live continuous recognition and sign patterns — research + recommendations

**Status: research note, 2026-09-25.** Follows [live-streaming-gap.md](live-streaming-gap.md) (TODO §12.8)
and [sign-patterns.md](sign-patterns.md) (TODO §3.8). Nothing here has been built or run yet; each item says
what it would be tested against.

---

## Part A — continuous recognition on a live camera

What we measured: the model knows the signs but loses them in the stream. Low fps makes it miss signs,
jitter and long sessions make it add signs, and landscape framing breaks everything. App-side fixes
recover part of it (a combined live-like case goes 0.592 → 0.481 GER). Real transitions were never
tested.

### A1. Input conditioning in the app (days, no retraining)

| # | change | evidence | test |
|---|---|---|---|
| 1 | **One Euro filter** instead of a fixed EMA: it smooths heavily when a landmark is still and lightly when it moves | Casiez et al. 2012. A reported MediaPipe comparison: still-hand shimmer 1.29 → 1.00 px RMS, lag 62 ms → 4–17 ms vs a fixed moving average ([OneEuro](https://mohamedalirashad.github.io/FreeFaceMoCap/2021-12-25-filters-for-stability/), [MVP Factory](https://mvpfactory.io/blog/wiring-android-s-camerax-to-a-quantized-hand-gesture-model-for-real-time-sign)) | EMA α 0.5 already took jitter 0.516 → 0.337 but cost +0.016 on clean. One Euro should keep the gain without that cost. Add it as a probe. |
| 2 | **Reframe** calibration (per-session shoulder/nose affine), plus an on-screen framing guide | our probe: aspect 0.651 → 0.334, scale 0.375 → 0.334 | done offline; port to TS with a parity fixture |
| 3 | **Interpolate, don't repeat** frames in `Clock`; also raise the real fps (smaller Holistic input, fewer UI redraws) | our probe: 15 fps 0.504 → 0.365 | measure the live fps first (user) |
| 4 | **Reset the state** at the next null frame after ~600 frames | our probe: recall 0.73 → 0.67 over 6 chained sentences | re-run `long_session` with the reset |

### A2. Real continuous data: the single biggest lever

**Why.** The closest published setting is online CSLR built on an isolated-sign classifier
([Zuo et al., EMNLP 2024](https://arxiv.org/abs/2401.05336)). They segment real continuous video into
per-sign clips with a CTC model's forced alignment. They train the isolated classifier on those clips plus
a **background class for co-articulation**, then decode online with a 16-frame sliding window and 7-window
majority voting. Training on clips cut from real continuous signing, instead of isolated recordings, took
online WER from **38.4% to 22.1%** on Phoenix-2014T. Our C1 only ever saw synthetic transitions (linear
interpolation), the one thing our probes could not test.

| # | source | what it gives | cost |
|---|---|---|---|
| 1 | **Our own recordings with known sentences.** Add a landmark recorder to `apps/web`; the user signs 30–50 corpus sentences, each with a known gloss order | a real continuous **test set** (the first); then training data via forced alignment: Viterbi over C1's per-frame probabilities, constrained to the known gloss order, yields sign spans and real transition frames labeled null | recorder ≈ a day; recording ≈ an hour |
| 2 | **ASLLRP DAI** (Rutgers): continuous ASL sentences with **time-aligned sign glosses**, downloadable as XML with the video ([DAI](https://dai.cs.rutgers.edu/dai/s/aboutwlasl)) | real native continuous signing. Extract landmarks with `sb-extract` (paused, still works) and keep sentences whose glosses fall in GISLR's 250 | licence check + extraction |
| 3 | **ASL Citizen**: 83,399 isolated videos, 2,731 signs, 52 Deaf/HoH signers, **recorded on webcams at home** ([paper](https://arxiv.org/abs/2304.05934), Microsoft Research licence) | isolated training data from the *same camera domain* as the web app; covers most of GISLR's vocabulary plus more for §12.4 | extraction ≈ GISLR-scale |
| 4 | How2Sign: 80 h continuous ASL, sentence-level English alignment only ([paper](https://arxiv.org/abs/2008.08143)) | unlabeled real transitions (for null/background pre-training) | low priority, no sign timings |

### A3. Retrain C1 v2 (needs the user's go)

1. **Normalized input**: shoulder-centred, shoulder-width scaled, per-frame, as in the 1st-place GISLR
   solution ([repo](https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution)).
   It makes framing irrelevant instead of calibrating it away.
2. **Augmentations mirroring the probes**:
   - fps drop + repeat, speed 0.7–1.5×, landmark jitter, hand dropout;
   - scale 0.7–1.3 (the 1st place also uses 0.7–1.3);
   - x/y shift and aspect;
   - **handedness mirroring** (42% of GISLR clips have the "left" hand dominant).
3. **Longer streams** (up to ~2,000 frames, 10–20 signs) so a session never outlasts training.
4. **Background/null for real transitions** from A2.1–2 once they exist, and noise as null (§12.6 Phase 3).
5. **Consider a fixed-context model**: a causal conv/transformer over the last ~2 s, or C1 with a periodic
   reset. Zuo et al.'s window plus voting has no state to drift. Our own B4 sliding window (0.507)
   lost to C1 (0.293) on synthetic streams, but under drift and jitter the ranking may flip. Score both on
   A2.1's recordings.

Gate every change on the live-robustness probes **and** on A2.1's real recordings. The synthetic
GISLR-Sentences number alone has already proved optimistic.

---

## Part B — sign patterns

Our result: orderless per-clip statistics of arm angles, distances, derivative variances and touches do
not make a pattern per sign (negative silhouette everywhere, template top-1 ≤ 4.9%). The strongest cue was
where the hand is relative to the face.

### B1. Look for patterns at the level of phonological parameters, not whole signs

Signs are built from a small inventory of reused parameters: **handshape, location, movement, palm
orientation** (plus non-manual markers) ([ASL phonology](https://en.wikipedia.org/wiki/American_Sign_Language_phonology)).
**ASL-LEX 2.0** codes these for 2,723 signs: sign type, selected fingers, flexion, major/minor location,
movement ([ASL-LEX](https://asl-lex.org/), [2.0 paper](https://academic.oup.com/jdsde/article/26/2/263/6142509)).
**PhonSSM** ([arXiv 2604.08761](https://arxiv.org/abs/2604.08761)) factorizes skeleton features into
per-parameter subspaces with prototype classification. It reports 72.1% on WLASL2000 (+18.4 pp over prior
skeleton SOTA), +225% relative in few-shot, and zero-shot transfer to ASL Citizen. That is the learned
version of the user's idea.

**Proposed test (no training, extends §3.8):**
- map GISLR's 250 glosses to ASL-LEX codes;
- rerun the same intra/inter test with *parameter values* as the groups (e.g. major location = head / body /
  arm / hand / neutral; one- vs two-handed; movement type).

If the variables cluster by parameter, a sign is a combination of parameter patterns. §3.8's top
variable (hand-to-face contact) already points at location. This also informs §12.4/§12.7: a new custom
sign could be enrolled as a parameter combination from one or two examples.

### B2. Add the parameters the current variables miss

| parameter | variables to add | source |
|---|---|---|
| handshape | 20 finger flexion angles + fingertip-to-palm distances of the dominant hand (already in `sb.recognize.interp.kinematics`) | ASL-LEX selected fingers / flexion |
| palm orientation | palm-normal direction from wrist, index MCP, pinky MCP (`_palm_facing_deg` exists) | ASL-LEX orientation |
| location | the dominant hand's position binned into body regions (head, chin, chest, neutral) over time | ASL-LEX major location |
| movement | direction and path shape of the dominant hand, not only its variance | ASL-LEX movement |

### B3. Keep time

Resample each clip to a fixed length and compare trajectories with **DTW** against per-gloss templates.
This is the classic one-shot sign-matching approach ([DTW + handshape features](https://dl.acm.org/doi/10.1145/2674396.2674421);
a MediaPipe example that learns a new sign from two recordings with learned embeddings + partial DTW:
[handy-slr](https://github.com/yoyo222/handy-slr)). Canonicalize handedness (done in §3.8) and signing
speed first. Scored the same way as §3.8: nearest template, train → test, top-1/top-5.

---

## Recommended order

1. **User, 5 min:** report the live fps, check mirroring, sit to the GISLR framing, and retry. This tells
   us which of A1's causes dominate.
2. **A1** (One Euro, reframe, interpolation, periodic reset) + **the landmark recorder**. Then the user
   records 30–50 known sentences, which become the real test set.
3. **A3 retrain** with normalized input + augmentations, gated on the probes and the real test set; then
   forced-alignment fine-tuning on the recordings (A2.1).
4. In parallel, no training: **B1** (ASL-LEX parameter-level patterns) → B2 → B3.
5. Later: ASL Citizen (webcam domain, larger vocabulary) and ASLLRP (native continuous signing).
