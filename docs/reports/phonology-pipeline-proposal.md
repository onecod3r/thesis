# A phonology-first recognition pipeline: proposal, critique, and open questions

**Status: design discussion, 2026-09-27. Nothing here has been built.** TODO §12.9. Follows
[sign-patterns.md](sign-patterns.md) (TODO §3.8), [phonology-models.md](phonology-models.md) and
[asl-phonology-features.md](asl-phonology-features.md) (TODO §3.9–3.10), [window-ensembles.md](window-ensembles.md)
and [live-streaming-gap.md](live-streaming-gap.md) (TODO §12.8). This is a conversation record, not a build
plan — it exists so the next session can pick up exactly where this one stopped.

## 1. The proposal (as given by the user, 2026-09-27)

A camera pipeline: MediaPipe extracts landmarks; coordinates are centered so head/camera movement is
tolerated; a first model reads the landmark stream frame by frame and emits phonological features into an
ordered array; every 1-2 seconds that array is fed to a second model, which predicts a gloss sequence, using
context from what was recognized before; an LLM then turns the gloss sequence into a full sentence (verb,
tense, negation, question), also holding memory of prior context; how much memory/context is explicitly
left open ("something to be experimented"); training data is a sentence dataset built from GISLR, restricted
to glosses that also exist in ASL-LEX.

Revised in the same session: stage 1 (the phonological-feature detector) should be **rule/definition-based
rather than learned** — e.g. "two hands are close together" fires in one frame; "dominant hand touches chin,
then moves away" fires over several frames and, combined with "eyebrow raised," should surface as two
separate signals (a THANK-YOU-shaped manual event, and a question-marking non-manual event) for stage 2 to
read.

## 2. Critique, part 1: centering / jitter tolerance

**Original ask:** center coordinates on the nose in the first frame, tolerant to jitter.

**Problem:** this solves neither goal.

- A first-frame-only anchor does not track motion after frame 1 — any sustained shift (posture, lean,
  camera movement, all routine in live use) drifts every later frame's coordinates against a stale
  reference. `sb.recognize.phonology._extract` already anchors every frame independently, to the mid-shoulder
  midpoint (`phonology.py:242-243`), falling back to the clip's median anchor only when shoulders are
  undetected — that already tracks sustained motion, which a fixed first-frame anchor cannot.
- The nose is a bad anchor point precisely because of head movement: a nod or turn moves the nose in the
  image without the hands or torso translating, so anchoring to it turns an innocent head movement into a
  spurious shift of the whole coordinate frame — the opposite of the stated goal. It is also frequently
  occluded, since several `MajorLocation` values (forehead, chin, mouth) put the signing hand directly over
  it. Shoulders are essentially never occluded by the signer's own hand, which is presumably why the
  existing extractor anchors there instead.

**What actually gives jitter tolerance** is temporal smoothing, already recommended in
`improvements-research.md` A1: a fixed EMA (α=0.5) took synthetic-jitter GER 0.516 → 0.337; a **One Euro
filter** (Casiez et al. 2012) is preferred over EMA because it adapts smoothing to signal speed — heavy when
still, light when moving — removing shake without adding lag to real motion. This layers on top of the
existing per-frame mid-shoulder anchor; it does not replace the anchor's reference point.

**Resolution (this session): the user accepted this — no first-frame/nose anchor.** Scale normalization
(shoulder-width division) was also flagged as needed for a live camera at variable distance, since the
repo's "scale doesn't matter" finding is specific to GISLR's uniform framing.

## 3. Critique, part 2: stage 1 as a learned per-frame feature detector

**Original ask:** train a GRU to identify phonological features frame by frame.

Two problems raised:

1. **Not every ASL-LEX parameter is a single-frame quantity.** Handshape, selected fingers, flexion, spread,
   thumb position/contact and location are instantaneous. Movement, repeated movement, flexion change,
   spread change and wrist twist are defined over the *whole sign* — "repeated movement" has no value at one
   frame alone. `sb.recognize.phonology.codes()` computes exactly these by comparing onset-third vs.
   final-third and accumulating path turning/reversals across the whole detected span
   (`phonology.py:545-568`); there is no way to make that a per-frame instantaneous output without either
   (a) emitting continuous derivatives per frame (velocity, one-frame delta-flexion/orientation — genuinely
   causal, and already built in `model_features()`, `phonology.py:622-635`) and letting stage 2's recurrence
   integrate them, or (b) giving stage 1 its own internal window before it commits to a discrete code, which
   quietly re-introduces the two-stage split the proposal is trying to keep clean.
2. **No training signal exists for a learned "identify phonological features" model.** Per-frame
   phonological-feature labels don't exist. The only candidate ground truth is ASL-LEX's gloss-level codes,
   propagated down to every frame of a gloss as a weak label — which only covers 233/250 GISLR glosses (see
   the gloss-mapping comparison, this conversation, 2026-09-26) and is exactly what the deterministic
   extractor already computes for free. Training such a layer with only the gloss label as signal (no
   phonology supervision) doesn't produce an interpretable feature detector — it produces a hidden layer
   shaped like one, functionally similar to a deeper GRU classifier. The current leaderboard leader,
   `gru_phono_raw` (0.7632), already uses the deterministic, non-learned extractor concatenated with raw
   landmarks — if the underlying geometry is the bottleneck, recalibrating its thresholds (`EXT_T`,
   `SPREAD_T`, etc. — tuned by eye on GISLR framing) is cheaper than training a whole model with no ground
   truth.

## 4. Refinement: stage 1 as a rule/definition-based hybrid (this session, 2026-09-27)

The user's revision: keep stage 1 rule-based, but let each rule fire over whatever window it needs — one
frame for an instantaneous fact ("two hands close together"), a few frames for a short event ("dominant
hand touches chin, then moves away"), plus non-manual channels (eyebrow raise) as their own signal, feeding
stage 2 a sequence of discrete phonological *events* rather than a dense per-frame vector.

**This resolves the training-signal problem in §3 outright** — a rule engine needs no labels, so it inherits
none of the "what supervises this" issue. It is also a natural fit for how sign phonology is actually
modeled: `phonology.py`'s own `SEGMENTS = ("onset", "medial", "final")` docstring cites Liddell & Johnson's
**hold-movement-hold** structure — a rule-based event stream is a fairly direct online implementation of
that idea (HOLD events where a parameter is stable for ≥k frames, MOVEMENT events where it changes), just
computed causally with a trailing buffer instead of after the whole clip is in hand.

**A concrete addition this surfaces**: ASL-LEX's 18 parameters are lexical and manual-only — it does not
code grammatical non-manual markers at all (brow raise for yes/no questions, headshake for negation, etc.,
per Neidle-style ASL syntax literature). `sb.recognize.phonology.CATALOG` already measures the raw
non-manual channels the user's eyebrow example needs (`brow_raise_r/l`, `eye_open_r/l`, `mouth_open`,
`mouth_width`, `head_yaw/pitch/roll`, `shoulder_tilt` — `phonology.py:116-125`), but `codes()` currently
turns none of them into discrete events; only the manual parameters get a discrete code. Building the
grammatical-marker rules is new work, and — unlike the manual parameters — there is **no ASL-LEX ground
truth to validate against at all**, and GISLR's isolated single-sign clips are unlikely to contain balanced
examples of sentence-level grammar markers in the first place, since those arise from sentential context.
Validating this half of stage 1 will need either a different annotated source or accepting no accuracy
number, only spot-checking.

**What this design still needs to resolve, not yet decided:**

- **Debounce/hysteresis per event type.** "Two hands close" can fire in 1 frame; "moves away" needs enough
  frames to distinguish real recession from jitter; "repeated" needs to see multiple cycles. Each rule's
  window length is a separate tuning problem, and — same as the deterministic thresholds already in
  `codes()` — calibrated on GISLR framing/fps, so it inherits the live-camera calibration risk documented in
  `live-streaming-gap.md` (now with an added timing dimension: a live camera's variable fps changes how many
  frames a k-frame debounce actually spans in wall-clock time).
- **Stage 2's input contract.** A sparse, irregularly-timed event stream doesn't match a GRU's one-vector-
  per-timestep expectation. The simpler option is to keep stepping stage 2 every frame with a mostly-zero
  event slot that goes hot when a rule fires (compatible with the existing streaming stack — D3, the
  continuous GRU — with no redesign of the sequence model's input); the alternative, an event-driven RNN that
  only steps on an event, is more novel and loses implicit timing information unless a delta-time feature is
  added back in. Recommend the former unless there's a specific reason to prefer the latter.
- **Discretization still discards information a continuous representation would keep** (e.g. an exact
  hand-to-chin distance collapses to a binary CONTACT flag). The Fisher-ratio results in
  `asl-phonology-features.md` suggest this is fine for identity-defining parameters like handshape, but the
  P1 failure (§5 below) shows how sensitive this pipeline already is to feature-space vs. landmark-space
  mismatches — each new discretization step is one more place train-time rule calibration can diverge from
  live-camera behavior.

**This does not change the critique in §5-6 below** — whether stage 1 is rule-based or learned is orthogonal
to how stage 2 is decoded, and the batching cadence ("every 1-2 seconds") is the part with a clear negative
result already on record.

## 5. Critique, part 3: "feed the array to a second model every 1-2 seconds"

**This exact scheme was already tested and lost decisively.** `window-ensembles.md` (TODO §12.8, run
2026-09-26) evaluated 1-3 second windows fed to two models offset by ~500ms, combined, fused with a
next-gloss prior — the user's scheme, run before this conversation restated it — against the deployed
per-frame continuous decoder (C1):

| approach | GER (+ prior) |
|---|---|
| per-frame continuous decoding (deployed, C1) | 0.276 |
| the user's scheme, 1 s windows | 0.523 |
| the user's scheme, 2 s | 0.687 |
| the user's scheme, 3 s | 0.777 |

Windows lose, and lose worse the longer they are: a median GISLR sign is ~20 frames (0.7s), so even a 1s
window sometimes splits a sign or catches two, and a 3s window holding 3-4 signs still returns one label per
window from a classifier trained on one sign per clip — about 70% of signs are never emitted at 3s. The
report's own conclusion: *"Windows would need windows under a sign's length and a start every few frames,
which is the per-frame model again at many times the cost."*

**The instinct to combine models is validated, at a different granularity.** Combining models helped inside
the windowed approach too (one model < two offset copies < two different models — the same ranking the user
proposed), but never closed the gap to per-frame decoding. A **per-frame ensemble** (two continuous models
stepping on the same frame, probabilities averaged) is the new best result in the same report, GER 0.244,
beating the single deployed model by 12%. **Recommendation: keep the "combine multiple models' opinions"
idea, drop the "wait 1-2 seconds and batch" part.**

## 6. Critique, part 4: a continuous phonological model was already tried, and failed — but recoverably

`P1` (`gru_continuous_phono130` — a continuous model reading the 130 phonological features, close to what
the proposal's stage 2 would be) was built and evaluated on real sentence streams
(`phonology-models.md` §9): **GER 0.587, roughly twice C1's 0.293**, worse than a naive reset-on-accept
baseline. This was not a failure of phonological features as signal — P1's isolated per-sign accuracy
(0.726) is *higher* than C1's raw-landmark equivalent (0.719). The diagnosis: P1's training streams were
synthesized by interpolating between feature vectors and a synthetic rest pose, while evaluation ran the
real extractor on landmark-composed sentences, where real transitions produce contacts/velocities/locations
the synthetic composer never showed the model — a train/test mismatch in how streams were built, not a flaw
in the phonology-first idea. The recorded (not yet built) fix is to compose training streams in **landmark
space** and run the real extractor on them, the same way C1's training data is built. **Any new phonological
continuous model should fix this first**, or it will likely reproduce P1's failure regardless of whether
stage 1 is rule-based or learned.

## 7. Critique, part 5: memory/context, and the sentence dataset

- **Gloss-sequence context** (using recent history to resolve an ambiguous sign) is already implemented:
  `sb.recognize.continuous.fuse` carries a `history: tuple[int, ...]` into the acceptance rule and prior
  (`fuse.py:116-172`), and the deployed prior is a held-out trigram over the sentence corpus. The "how much
  memory" question the user left open is therefore already bounded by an existing, cheaply-sweepable
  parameter (bigram vs. trigram vs. higher), not a fresh experiment design.
- **LLM sentence-construction memory** (coreference, tense continuity across sentences) is a separate,
  mostly-unbuilt idea, since the existing gloss→text pipeline (T5 + rules/guard) works per sentence. Whether
  the recognizer's gloss-history tuple could be passed straight into the LLM's prompt, instead of building
  two independent context mechanisms, is an open question worth resolving before building anything.
- **The sentence dataset**: `GISLR-Sentences` v1 already exists (12,727 sequences, built 2026-09-23 from
  `test.csv`), unfiltered by ASL-LEX. Restricting a new dataset to the 233/250 ASL-LEX-mapped glosses cuts
  17 signs (`say`, `look`, `every`, `snack`, `puzzle`, ... — see the gloss-mapping comparison, this
  conversation, 2026-09-26) from anything trained on it. Whether that restriction should gate the sentence
  *vocabulary* (a real capability cut) or only an *auxiliary phonology-supervision loss* (harmless) is not
  yet decided. Separately, GISLR-Sentences is synthesized — real isolated clips concatenated with synthetic
  transitions and rest, not naturally continuous signing — the same caveat attached to every §12 result
  including P1's and the window-ensembles' numbers above: rankings between approaches are trustworthy,
  absolute numbers are optimistic versus live signing.

## 8. Open questions for the user

1. When stage 1 fires a rule that needs several frames (e.g. "moves away"), what should the emitted event's
   timestamp be — frame the rule started evaluating, or frame it fired? This affects stage 2's ability to
   line events up with the LLM's/prior's timing expectations.
2. Should the ASL-LEX-gloss restriction gate the sentence dataset's vocabulary, or only which clips get an
   auxiliary phonology-supervision label?
3. Should stage 2 read a dense per-frame vector with occasional hot event slots (compatible with the
   existing D3/continuous-GRU stack), or an event-driven representation (novel, needs its own timing
   feature)?
4. For LLM sentence memory: is passing the recognizer's existing gloss-history tuple into the prompt
   sufficient, or is a second, independently-tuned context window actually needed?
5. Is a new phonological continuous model worth building before P1's landmark-space composition fix lands,
   given P1 already failed for a reason unrelated to whether stage 1 is learned or rule-based?

## References

- Casiez, G., Godbout, B., Roussel, N. (2012). 1€ Filter: A Simple Speed-based Low-pass Filter for Noisy
  Input in Interactive Systems.
- Liddell, S. K. & Johnson, R. E. (1989). American Sign Language: The Phonological Base. *Sign Language
  Studies* 64.
- Zuo et al. (2024), EMNLP — online CSLR from an isolated-sign classifier with forced alignment + background
  class, cited via `improvements-research.md`.
- Sehyr, Z., Caselli, N., Cohen-Goldberg, A. & Emmorey, K. (2021). The ASL-LEX 2.0 Project.
