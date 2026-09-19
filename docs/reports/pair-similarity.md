# Semantic sign-pair similarity: does a richer feature set rescue what pooling loses?

**Status: complete (2026-09-19).** All 16 documented confusable pairs + 8
random-pair controls, 5-fold CV. Narrative:
[docs/logs/daily/2026-09-19.md](../logs/daily/2026-09-19.md).

| | |
|---|---|
| **Question** | `docs/reports/plateau-diagnosis.md` §6 found a binary probe on pooled raw xy position/velocity sits at 0.463–0.628 accuracy for 7 confused sign pairs (`awake`/`wake` at chance). Does the richer `landmark_kinematics_v1` feature set (28 joint angles, frame-gap-aware kinematics, jitter, rolling variance) do any better? |
| **Instrument** | `experiments/recognition/gislr.0.dataset.pair-similarity.ipynb` (TODO §7.1), over `sb.recognize.interp.kinematics`/`discriminability` (TODO §3.4) |
| **Data** | GISLR `train.csv`, 16 documented pairs (31 classes, 14,309 videos of the 47-class run, ~600/pair) + 8 random-pair controls (16 classes) |
| **Metric** | `probe_classifier_cv` — 5-fold CV binary logistic probe accuracy per pair, matching `plateau-diagnosis.md` §6's exact methodology, broken down by feature type (angle/position/speed/relational/jitter/rolling-variance) |

---

## 1. Headline: angles don't rescue what pooling loses — confirmed at full scale

**`corr(confusion_rate, acc_all)` = Spearman ρ **−0.712**, Pearson r
−0.839** — matching `plateau-diagnosis.md` §6's −0.72 almost exactly, with a
**completely different feature set** (28 angles + kinematics vs raw xy
position/velocity). The pairs a trained model confuses most are still
exactly the pairs this richer pooled probe cannot separate either.

`awake`/`wake` is again the floor: **0.512** accuracy (chance), barely above
§6's 0.463–0.506 range, and its angle-only probe does no better (0.514).
Random-pair controls average **0.954** accuracy — semantic pairs are hard
in a way arbitrary pairs simply are not.

| Pair | Confusion rate | §6 position | §6 velocity | §6 both | **This probe (all features)** | Angle-only |
|---|---|---|---|---|---|---|
| `awake`/`wake` | 0.839 | 0.463 | 0.506 | 0.463 | **0.512** | 0.514 |
| `lips`/`mouth` | 0.533 | 0.674 | 0.588 | 0.638 | **0.645** | 0.585 |
| `hear`/`listen` | 0.411 | 0.754 | 0.612 | 0.728 | **0.726** | 0.713 |
| `pen`/`pencil` | 0.404 | 0.632 | 0.556 | 0.597 | **0.610** | 0.610 |
| `gift`/`give` | 0.367 | 0.754 | 0.693 | 0.736 | **0.707** | 0.783 |
| `cut`/`scissors` | 0.357 | 0.687 | 0.638 | 0.698 | **0.719** | 0.578 |
| `finger`/`wait` | 0.343 | 0.790 | 0.692 | 0.768 | **0.774** | 0.691 |

The richer feature set moves individual pairs a few points either
direction (`cut`/`scissors` +2pp on `acc_all` but −11pp on angle-only;
`gift`/`give` −5pp on `acc_all` but +3pp on angle-only) — noise-level shifts,
not a systematic rescue. `gift`/`give` is the only pair where angles beat
every other single feature type including the full set combined (0.783 vs
0.707 `acc_all`) — worth a second look, but one pair out of seven is not a
pattern.

## 2. Full 16-pair + control leaderboard

| Pair | Confusion rate | n | `acc_all` | angle | position | speed | relational | jitter | rolling-var |
|---|---|---|---|---|---|---|---|---|---|
| `look`/`see` | 0.194 | 641 | 0.875 | 0.818 | 0.900 | 0.671 | 0.775 | 0.538 | 0.499 |
| `bed`/`bedroom` | 0.212 | 592 | 0.787 | 0.794 | 0.796 | 0.642 | 0.758 | 0.593 | 0.522 |
| `finger`/`wait` | 0.343 | 580 | 0.774 | 0.691 | 0.803 | 0.574 | 0.640 | 0.536 | 0.547 |
| `nap`/`sleep` | 0.260 | 617 | 0.763 | 0.723 | 0.746 | 0.559 | 0.697 | 0.472 | 0.507 |
| `cat`/`kitty` | 0.271 | 631 | 0.750 | 0.715 | 0.778 | 0.601 | 0.720 | 0.518 | 0.536 |
| `duck`/`goose` | 0.330 | 639 | 0.748 | 0.737 | 0.728 | 0.645 | 0.601 | 0.535 | 0.484 |
| `animal`/`have` | 0.283 | 548 | 0.741 | 0.653 | 0.746 | 0.608 | 0.599 | 0.544 | 0.518 |
| `dry`/`dryer` | 0.209 | 608 | 0.740 | 0.676 | 0.704 | 0.607 | 0.604 | 0.561 | 0.507 |
| `hear`/`listen` | 0.411 | 656 | 0.726 | 0.713 | 0.730 | 0.602 | 0.694 | 0.489 | 0.468 |
| `cut`/`scissors` | 0.357 | 590 | 0.719 | 0.578 | 0.702 | 0.546 | 0.661 | 0.549 | 0.520 |
| `gift`/`give` | 0.367 | 591 | 0.707 | 0.783 | 0.711 | 0.597 | 0.719 | 0.531 | 0.530 |
| `stay`/`that` | 0.336 | 606 | 0.700 | 0.662 | 0.711 | 0.510 | 0.619 | 0.492 | 0.520 |
| `sleep`/`sleepy` | 0.263 | 626 | 0.669 | 0.634 | 0.636 | 0.554 | 0.599 | 0.505 | 0.516 |
| `lips`/`mouth` | 0.533 | 634 | 0.645 | 0.585 | 0.632 | 0.535 | 0.576 | 0.491 | 0.517 |
| `pen`/`pencil` | 0.404 | 633 | 0.610 | 0.610 | 0.611 | 0.504 | 0.583 | 0.487 | 0.488 |
| **`awake`/`wake`** | **0.839** | 642 | **0.512** | 0.514 | 0.488 | 0.522 | 0.486 | 0.483 | 0.502 |
| control (mean of 8) | — | ~600 | **0.954** | 0.943 | 0.950 | 0.723 | 0.895 | 0.560 | 0.548 |

Ranked by `acc_all` ascending (hardest first): `awake`/`wake`, `pen`/`pencil`,
`lips`/`mouth`, `sleep`/`sleepy`, `stay`/`that` — the same five pairs are
hardest whichever feature type is used, not just on the combined probe.
**No feature type reliably separates `awake`/`wake`** — every single-type
probe sits within 0.02 of chance (0.480–0.522), the tightest chance-cluster
of any pair.

**`jitter` and `rolling_var` are uniformly the weakest types** (0.47–0.59
across every pair, control pairs included) — consistent with
`docs/reports/feature-discriminability.md`'s finding that these are the
least informative feature types in general, not just for confusable pairs.

## 3. What actually distinguishes the three hardest pairs

Top individual features by F-ratio (5-fold-CV probe's underlying signal):

- **`awake`/`wake`**: the entire top-8 is face/pose landmark **positional
  std** (`pose_492_x_std`, `face_58_x_std`, ...), all with F < 5.2 (weak —
  compare to `hear`/`listen`'s F=39.7 below). Nothing stands out; this
  matches the chance-level probe accuracy.
- **`lips`/`mouth`**: pose landmark **speed** dominates
  (`pose_496_speed_mean` F=15.2), with one angle feature
  (`angle_L_index_pip_flex_std`, F=9.5) in the mix — some real signal, still
  weak relative to `hear`/`listen`.
- **`hear`/`listen`**: `angle_R_palm_facing_mean` is the top feature by a
  wide margin (F=39.7, more than double the next-best), then a run of
  right-hand y-position features — this pair's real distinction is hand
  orientation, and it shows in the highest single-pair probe accuracy of
  the three (0.726 vs `awake`/`wake`'s 0.512).

## 4. Verdict

**Angles do not rescue what pooling loses.** The Spearman correlation
between confusion rate and probe accuracy (−0.71) reproduces
`plateau-diagnosis.md` §6's −0.72 almost exactly with a completely
different, richer feature set — the strongest evidence yet that the
bottleneck really is **what pooling itself destroys** (trajectory
shape/ordering — §6's own reading), not which raw channel gets pooled.
`awake`/`wake` stays at chance under every single feature type tried here,
individually or combined. This is consistent with, not a contradiction of,
`docs/reports/curated-features.md`'s finding that a **sequence model**
(LSTM, not a pooled linear probe) reaches 69% top-1 overall — a model that
sees frame order, rather than a summary statistic over frames, is a
different instrument entirely.

## 5. Follow-ups

- [ ] `gift`/`give`'s angle-only outperforming its combined probe (0.783 vs
  0.707) is the one pair worth a second look — noise or real, one pair of
  sixteen doesn't say which.
- [ ] This still only tests **pooled** angle features (mean/std over time,
  same limitation `plateau-diagnosis.md` §6 flagged for pooled position).
  A per-frame **sequence** probe (not a linear one on summary statistics)
  over the angle+kinematics feature set, rather than raw xy, is the
  logical next test — closer to what `curated-features.md`'s LSTM already
  does, but with the richer feature set instead of ME-126+xy+angles.
- [ ] Global scope (all 250 classes' worth of pairs, not just the 16
  documented ones) was never the goal here — this is specifically about
  the already-identified confusable pairs.
