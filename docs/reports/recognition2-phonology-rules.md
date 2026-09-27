# recognition_2: the rule-based ASL-LEX engine, retuned, and a pose ablation

**Status: run 2026-09-27.** TODO §14. Follows the first run (`README.md`/TODO §14, same date) and the error
analysis that found the issues fixed here. Module: `sb.recognize.recognition2.rules`/`validate`. Notebook:
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
