# recognition_2: the rule-based ASL-LEX engine, retuned, and a pose ablation

**Status: run 2026-09-27, extended 2026-09-28 (§5).** TODO §14. Follows the first run (`README.md`/TODO
§14, same date) and the error analysis that found the issues fixed here. Module:
`sb.recognize.recognition2.rules`/`validate`. Notebook:
`experiments/recognition2/gislr.1.pipeline.stage1-rules.ipynb`. Isolated as before — nothing in
`sb.recognize.phonology`/`aslex` is touched.

## 1. Two bugs found and fixed

**Bug 1 — `major_location`'s "Hand" and "Head" checks used a median distance, not a minimum.**
`repeated_movement`/`contact` already used `np.nanmin` over the nucleus; `major_location` used `_med`
(median) for the same kind of "is this hand close to X" check. A location touch is often brief within a
sign — a median over the whole nucleus washes it out almost completely. Effect before the fix: `Hand`
predicted 5 times out of 1,308 clips against 108 true cases. Fixed to `np.nanmin`, matching `contact`.

**Bug 2 — the pose-ablation's `Arm` check compared a hand to its *own* elbow/wrist.** When `use_pose=True`
was first added (for §2 below), `MajorLocation=Arm` was checked against the *ipsilateral* (same-side)
elbow and wrist. A hand's own wrist landmark nearly coincides with its own pose wrist by anatomy — it's the
same joint — so this fired on almost every clip: `Arm` predicted 729/1,414 times against ~24 true cases.
ASL-LEX's `Arm` means the hand touching the *other* (passive) arm, so this needed the *contralateral* elbow,
and the wrist was dropped from the check entirely (same coincidence one joint up would recur less severely).
Fixed: `Arm` predicted 4/60 times on a 30-gloss spot check, a sane rate.

A related, non-bug finding while fixing these: the two variants' distance thresholds were not actually
comparable, because `use_pose=True` switched the scale unit from inter-eye distance to shoulder width
(several times larger) for *every* threshold, not just the pose-specific ones. Fixed by always computing
true inter-eye distance for threshold scaling (it's translation- and mirror-invariant, so this is safe in
either anchor mode) — `scale` (shoulder width under `use_pose`) is now anchor bookkeeping only, never a
threshold unit.

## 2. Retuning `stage1-rules.json` (no-pose)

From the first run's confusion analysis: `spread_t` 12°→20° (was over-triggering `spread=1`, 478 predicted
vs. 266 true), `amp_t` 0.08→0.04 and a new `apt_amp_t` 0.15 (repeated-movement recall on true repeats was
13%), `flat_base_t` 40°→25° (two known Flat-truth glosses measured base flexion 28–42°, both missed at 40°).

**Net effect on the no-pose table, full 233-gloss run:**

| parameter | before | after | Δ |
|---|---|---|---|
| `spread` | 0.487 | **0.554** | +0.067 |
| `major_location` | 0.456 | **0.475** | +0.019 (the min-not-median fix) |
| `marked_handshape` | 0.590 | 0.564 | −0.026 |
| `minor_location` | 0.266 | 0.249 | −0.017 |
| `flexion` | 0.302 | 0.274 | −0.028 |
| `repeated_movement` | 0.531 | 0.532 | +0.001 (retuning barely moved it) |
| everything else | unchanged | unchanged | 0 |

**Honest reading**: the `spread` and `major_location` fixes clearly helped. `flat_base_t`'s change net-hurt
`flexion` across the full sample despite fixing the two anecdotal cases it was based on — a real
methodological caveat: retuning thresholds against the same 233 glosses being evaluated is not a held-out
test, so a fix motivated by 1–2 examples can regress on the rest. `repeated_movement`'s recall problem is
evidently not primarily about `amp_t`/`apt_amp_t`'s amplitude — more likely GISLR's short clips (~20 frames
median) genuinely don't contain enough distinguishable cycles for reversal-counting to work well, which
threshold tuning alone won't fix.

## 3. The pose ablation: does adding pose help? (user ask, 2026-09-27)

Both variants use identical clips, identical thresholds, and hands/face as the primary signal for every
hand-internal parameter — pose only changes the anchor/scale and adds `Body`/`Arm` to `major_location` plus
a pose-wrist fallback for `movement`/`repeated_movement` when the hand drops out of the nucleus for more
than half its span (`hand_absent_frac`).

| parameter | no_pose | with_pose | Δ |
|---|---|---|---|
| every hand-internal parameter (`handshape`, `selected_fingers`, `flexion`, `flexion_change`, `spread`, `spread_change`, `thumb_position`, `thumb_contact`, `ulnar_rotation`, `non_dominant_handshape`, `marked_handshape`, `sign_type`, `contact`, `minor_location`) | — | — | **exactly 0** |
| `major_location` | 0.475 | 0.474 | **−0.001** (`Body`/`Arm` fire correctly but rarely — true `Body`/`Arm` are themselves rare, 126/24 of 1,308 — not enough volume to move the number either way) |
| `repeated_movement` | 0.532 | 0.532 | +0.001 |
| **`movement`** | **0.492** | **0.377** | **−0.116** |
| `second_minor_location` | 0% coverage | 0% coverage | 0 (torso/arm sub-sites not attempted either way — out of scope this pass) |

**Headline: pose does not help, and for `movement` it measurably hurts.** The zero-delta on every
hand-internal row confirms the implementation genuinely keeps hands/face primary, as asked — pose isn't
quietly doing more than intended anywhere except where it's supposed to. `major_location`'s new categories
are correct but too rare in this dataset to matter. `movement`'s drop is the one real, actionable finding:
the pose-wrist fallback triggers far more often than expected (**37% of a 30-gloss spot check** — GISLR's
hand tracking has real gaps even within a clip's own detected span, not just at the edges) and it makes
`movement` classification measurably worse when it fires. The likely reason: ASL-LEX's `Movement` is about
the *hand's* spatial path, and the wrist is a proxy for it, not identical to it — a sign with a fixed wrist
but a rotating/curving hand would read as `Straight` from the wrist and something else from the hand
centroid. A gappy hand-centroid signal is evidently still a better source for this parameter than a complete
but different-signal wrist trajectory.

**Conclusion for the "is discarding pose feasible" question**: yes, for every parameter this pass covers.
The only place pose helps in principle (`Body`/`Arm`) is real but too rare to register in this sample, and
the one place it was tried more substantively (movement's wrist fallback) made things worse. Not tried:
using pose for `SecondMinorLocation`'s torso/arm sub-sites, or a smarter fallback than "swap to wrist
entirely" (e.g. blending, or requiring near-total absence rather than >50%) — both are possible follow-ups,
not done here given the fallback already looks like the wrong lever for `movement` specifically.

## 4. Artifacts

`data/cache/gislr_aslex_rules/`: `agreement_no_pose.csv`, `agreement_with_pose.csv`,
`agreement_comparison.csv`, `records_no_pose.json`, `records_with_pose.json` (every scored clip's truth and
prediction, both variants, for further error analysis).

## 5. Extended 2026-09-28: full canonical test split + the two remaining gaps

User ask: build/extract every ASL-LEX feature and check it against ASL-LEX, "one by one." Sections 1–4
above already cover 16 of 18 parameters with real numbers; this section closes the last two
(`second_minor_location`, `non_dominant_handshape`) and reruns everything on far more data.

**Sample size.** The runs above scored a 6-clips/gloss sample of `train.csv` (1,398 clips). GISLR actually
has 239–332 train clips and 60–83 test clips per one of the 233 ASL-LEX-mapped glosses. This run scores
**every clip in `test.csv`** (the same fixed 80/20 canonical split used for GISLR model evaluation
elsewhere in this repo) — 16,741–16,772 clips per parameter (`n_with_ground_truth` varies per parameter
since an ambiguous/unmapped gloss is excluded per-parameter, same policy as `aslex.gloss_codes`), for both
variants. Rule evaluation is pure numpy geometry (~16ms/clip measured), no training, so both variants
together ran in a few minutes.

**Result: the numbers from the 6-clip sample hold up.** Every parameter's agreement is within ±0.02 of the
first run's, on ~12x the data — the small sample was already a fair estimate, not noise. `movement`'s
`with_pose` penalty (−0.107 here vs. −0.116 before) and `major_location`'s flat pose delta both replicate.

**`second_minor_location` (new — was 0% coverage, no rule existed).** ASL-LEX's own value set here, checked
against `signdata.csv`, is dominated by `Neutral`/`HeadAway`/`HandAway`/`BodyAway` — not a second face
site, but whether the hand moves *away* from wherever `MajorLocation` found contact by the sign's end.
Implemented as a path-departure check (`rules.py::_away`): find the hand's closest approach to that site
across its whole detected span, then check whether it ends up more than `away_t` (0.5x inter-eye distance)
further away by the last tracked frame. Result: **coverage ~100%, agreement 0.390 vs. chance 0.048** (8.2x
chance) — a genuinely strong new signal, not just "attempted." Finer sub-site categories (`TorsoMid`,
`ElbowBack`, `Other`, ...) are still `NOT_COMPUTABLE`, not guessed; `Body`→`BodyAway` needs pose, matching
`major_location`'s own gating.

**`non_dominant_handshape` (fix — mechanism works, ceiling is the data, not the bug).** The bug: its
features were sliced using the *dominant* hand's active window (`nucleus`), which on a two-handed sign is
frequently a window where hand 2 itself isn't tracked, even though hand 2 IS tracked elsewhere in the clip.
Fixed by giving hand 2 its own independently-computed nucleus (`rules.py::own_nucleus`). Effect: attempted
predictions went from 2/1,308 (0.4 %, old 6-clip run) to 36/16,741 (0.6%, full run) — mechanically working,
7x more clips now get an attempt, but still near-chance agreement (0.028 vs. chance 0.032, 1/36 correct).
**The real bottleneck isn't the bug — it's that GISLR's vocabulary is overwhelmingly one-handed or has a
passive/non-varying second hand**: only ~2.5% of a spot-checked sample even has the non-dominant hand
tracked for ≥5% of frames, and ASL-LEX ground truth itself only exists for 108/290 mapped entries (the rest
are coded one-handed, no second handshape to have). This parameter's ceiling on GISLR is structural, not a
threshold-tuning problem.

**Updated artifacts**: same file names as §4, now holding the full-test-split run (previous 6-clip-sample
files were overwritten — the sample-size finding above is the record of what changed).

## 6. Can a trained classifier beat the hand-picked thresholds? (2026-09-28)

User ask: train one small classifier per ASL-LEX parameter on the same *continuous* pre-threshold
measurements the rule engine computes internally (not its binarized code), to see whether the
near-chance parameters are a measurement problem or a threshold problem.

**Setup**: `rules.continuous_features()` (new, 23 scalars/clip — finger extension/flexion/spread
angles, thumb distances, rotation, movement path stats, location distances, two-handedness
symmetry) extracted for 46,600 clips across all 233 mapped glosses (up to 200/gloss, train+test
combined). One `HistGradientBoostingClassifier` per parameter (`class_weight="balanced"`, native
NaN handling), evaluated on a **held-out 20% of glosses** — signs never seen in training at all, not
just held-out clips of a seen sign, so the test is whether the continuous features generalize to
new vocabulary rather than memorizing which sign is which.
Notebook: `experiments/recognition2/gislr.2.pipeline.stage2-classifiers.ipynb` — Claude built and
ran the deterministic setup/feature-load/gloss-split cells (§1–2, no training), the user ran the
training cell (§3) themselves per this repo's convention.

**Result: mixed, and informative.** Comparing classifier accuracy against both the rule engine's
agreement and a trivial majority-class baseline on the *same* held-out glosses:

| parameter | classifier | rule engine | majority baseline | verdict |
|---|---|---|---|---|
| `spread` | 0.599 | 0.523 | 0.429 | **real win** — was near-chance for the rule engine |
| `handshape` | 0.198 | 0.110 | 0.040 | real win |
| `selected_fingers` | 0.585 | 0.476 | 0.462 | real win |
| `thumb_position` | 0.709 | 0.609 | 0.654 | real win |
| `marked_handshape` | 0.608 | 0.551 | 0.560 | real win |
| `non_dominant_handshape` | 0.149 | 0.028 | 0.125 | real win (still low absolute, 16 classes) |
| `flexion` | 0.445 | 0.252 | 0.565 | beats rules, loses to majority baseline |
| `repeated_movement`, `movement`, `contact`, `flexion_change` | — | beats rules | loses to majority | same pattern — real signal, small held-out set |
| `ulnar_rotation` | 0.678 | **0.847** | 0.846 | loses — the rule engine's strongest parameter, little room to gain |
| `sign_type` | 0.503 | 0.667 | 0.708 | loses badly |
| `minor_location`, `second_minor_location`, `spread_change`, `thumb_contact`, `major_location` | — | loses to rules | mostly loses to majority too | |

**6 parameters show a genuine, generalizing win** — beating both the rule engine *and* the trivial
baseline on glosses the classifier never trained on. `spread` is the headline: it was one of the
weakest rule-engine parameters (0.523, barely above chance) and the classifier fixed it
(0.599, balanced accuracy 0.617 — not just riding the majority class).

**`second_minor_location`'s loss (−0.143) is a feature-engineering gap, not a ceiling**:
`continuous_features()` only exposes the medial/nucleus-window scalars the rule engine's other
parameters use — it never computed the one signal `_away()` actually uses (displacement between
the hand's closest approach and the clip's *last* tracked frame, over the whole span). The
classifier structurally cannot see what the rule sees for this parameter; adding that as an
explicit feature is the obvious next step, not evidence the parameter is unlearnable.

**`sign_type`'s loss (−0.164) likely has the same root cause**: the rule engine decides symmetry
partly from whether the two hands' *looked-up handshapes* match exactly — the classifier's
`sign_type_sym` feature is only a velocity-correlation proxy, with no direct handshape-identity
signal and heavy NaN for one-handed signs (which are most of the dataset).

**Honest caveat repeated from §5**: held-out-gloss evaluation here uses only ~47 test glosses out
of 233 — small enough that a handful of hard/easy glosses can swing a parameter's number
noticeably; read a close margin as "roughly tied," not literally decided.

**Artifacts**: `data/cache/gislr_aslex_rules/continuous_features.jsonl` (input),
`classifier_results.csv` (this table), `classifier_gloss_split.json` (which glosses were held out).
