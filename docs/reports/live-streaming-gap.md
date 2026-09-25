# Why continuous recognition breaks on a live camera: input-robustness probes for C1

**Status: probes run 2026-09-25 (inference only, no training).** TODO §12.8. Narrative:
[docs/logs/daily/2026-09-25.md](../logs/daily/2026-09-25.md).

| | |
|---|---|
| **Question** | The first camera test of the web app (user, 2026-09-25): isolated signs are recognized well, but continuous sentences are "garbage", with missing, extra and wrong glosses. Offline, C1 + D3 scores GER 0.293. What does the live input do differently, and what fixes it? |
| **Instrument** | `experiments/recognition/gislr.3.streaming.live-robustness.ipynb` |
| **Data** | 400 seeded `sentence` streams from the 16 evaluation signers of GISLR-Sentences v1 (1,251 signs) |
| **Model / decoder** | C1 `1790143122` (`gru_continuous`, ME_132, raw `xy`) + D3 (ν 0.7, min_len 4, collapsed), the offline best |
| **Metric** | stream GER (sub/del/ins), plus **isolated accuracy**: each true sign segment fed alone from zero state |
| **Artifacts** | `data/cache/gislr/live_robustness/{summary.csv, framing.json, parts/, assets/probes.png}` |

---

## 1. Headline

**Four live-camera differences break the stream while isolated recognition stays intact.** That is
the symptom the user reported. In these four cases the model still knows the signs; what fails is
finding where they start and end:

| probe | stream GER | missed | extra | isolated acc |
|---|---|---|---|---|
| clean | **0.293** | 0.068 | 0.017 | 0.737 |
| camera at 15 fps (app repeats frames to 30) | 0.504 | **0.353** | 0.010 | 0.719 |
| camera at 10 fps | 0.515 | **0.336** | 0.014 | 0.703 |
| landmark jitter, 1% of the image | 0.516 | 0.090 | **0.165** | 0.730 |
| jitter 0.5% | 0.372 | 0.069 | 0.080 | 0.730 |
| 6 sentences, no state reset | 0.415 | 0.045 | **0.107** | – |

- **Low fps → missed signs.** Repeated frames make the motion stall every other tick. The null
  detector reads each stall as a pause, which cuts signs into pieces shorter than `min_len`.
- **Jitter → extra signs.** Noise on still hands looks like motion, so rest and transitions are
  read as signs.
- **Long sessions → extra signs and slowly falling recall.** Recall by position in a 6-sentence chain
  goes 0.73, 0.72, 0.70, 0.66, 0.66, 0.67. Training streams never exceeded 600 frames. The app resets
  the state only after 45 null frames, and with jitter those rarely come.

**Two more differences break both the stream and isolated recognition:**

| probe | stream GER | isolated acc |
|---|---|---|
| x squashed to 0.56 (landscape webcam vs GISLR's portrait framing) | **0.651** | 0.416 |
| mirrored / swapped hands | **0.990** | 0.032 |
| farther from the camera (scale 0.7) | 0.375 | 0.671 |
| closer (scale 1.3) | 0.336 | 0.711 |
| off-centre by 0.1 (x or y) | 0.313–0.325 | 0.700–0.728 |
| a hand drops out of 20% of frames | 0.380 | 0.717 |
| signing 1.5× faster | 0.380 | 0.686 |

The model reads **raw image coordinates** (ME_132 `xy`, no normalization), so framing is part of the
input. GISLR's framing (§3) is portrait and close: shoulders span 0.57 of the image width. The app
requests a **640×480 landscape** camera. At the same vertical framing that squashes x to about
0.4–0.55 of what the model learned, which is exactly the range of the 0.56 probe.

## 2. App-side fixes, measured (no retraining)

| condition | without the fix | with the fix | fix |
|---|---|---|---|
| 15 fps | 0.504 | **0.365** | interpolate between real frames instead of repeating (one frame of latency) |
| 10 fps | 0.515 | 0.434 | same |
| jitter 1% | 0.516 | **0.337** | causal EMA per landmark, α = 0.5 (α = 0.3 is worse, 0.373) |
| x squashed 0.56 | 0.651 | **0.334** | **reframe**: per-session affine that maps the signer's shoulder midpoint, shoulder width (x) and shoulder-to-nose height (y) to GISLR's medians |
| scale 0.7 | 0.375 | **0.334** | reframe |
| clean | 0.293 | 0.309 / 0.334 | EMA / reframe cost on clean data |
| **live-like** (scale 0.8, lower in frame, 15 fps, 0.5% jitter, 10% hand dropout) | 0.592 | **0.481** | interp + reframe + EMA |

Reframing makes the model **invariant to framing**: aspect, scale and clean all land on the same 0.334.
The cost is +0.04 on clean GISLR, where each signer's own framing already matched training. Even with every
fix, the combined live-like case (0.481) stays far from 0.293. The fixes are worth shipping, but they do
not replace a retrain.

## 3. What the model expects: GISLR framing

Measured on the sign frames of the sampled streams (`framing.json`, image-normalized coordinates):

| | p5 | median | p95 |
|---|---|---|---|
| shoulder midpoint x | 0.35 | 0.49 | 0.62 |
| shoulder midpoint y | 0.53 | 0.61 | 0.72 |
| shoulder width | 0.47 | 0.57 | 0.67 |
| nose y | 0.30 | 0.38 | 0.51 |

Each hand is detected in only about 30% of sign frames. Until reframing ships, a live test should match
this framing: close to the camera, shoulders spanning about half the image width, nose about 40% from the top.

## 4. What this does not cover

These probes put live-like distortions on synthetic GISLR-Sentences streams. What they cannot measure:

- **Real transitions and rest.** C1 learned transitions from linear interpolation and rest from lowered or
  missing hands. Real co-articulation (a hand shape changing mid-flight, reduced signs) is untested. It is
  the most likely remaining cause after the fixes, and the only way to measure it is real recorded
  sentences.
- **The Tasks HolisticLandmarker vs the legacy Holistic that produced GISLR.** Hand-detection rates and
  jitter differ between them. The live fps and jitter are unknown until the user reports them.

## 5. Recommendations, in order

0. **Check the live test (5 min, user).** Note the fps the UI shows. Sign one one-handed sign with
   mirroring on and then off (a wrong setting drops even isolated accuracy to 3%). Sit so the shoulders
   span about half the frame width.
1. **App-side, no retraining:**
   - reframe calibration over the first ~2 s of a session (a TS port of the probe's `reframe`);
   - interpolate instead of repeating frames in `Clock`;
   - an EMA with α = 0.5 on landmarks;
   - a forced state reset at the next null frame after ~600 frames.
   Each needs a parity fixture like the existing ones.
2. **Record real sentences.** Add a landmark recorder to the web app (download the `(T, 543, 3)` frames +
   timestamps) and record 30–50 known sentences. That is the first real continuous test set. It measures
   §4's unknowns, and every later fix is scored on it.
3. **Retrain C1 v2 (needs the user's go):**
   - **normalized input** (shoulder-centred, shoulder-width scaled, as the 1st-place port does), which
     removes framing from the problem instead of calibrating it away;
   - composer augmentations matching §1's probes: fps drop + repeat, speed 0.7–1.5×, jitter, hand dropout,
     scale/shift/aspect;
   - streams up to ~2,000 frames;
   - noise as null (§12.6 Phase 3, already pending).
   Score it on this notebook's probes and on the recordings from step 2.
