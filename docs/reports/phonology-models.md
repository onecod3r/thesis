# Phonology front-end models + the current-split architecture grid (2026-09-26)

**Question (user, 2026-09-25, TODO §3.9):** does putting the sign-pattern findings (handshape,
palm orientation, location, movement; `sign-patterns.md` §7–§8) in front of a model beat the
raw-landmark models, and which features carry it? Same notebook run also covered TODO §4.3's
architecture grid on the current split.

**Answer: yes. `gru_phono_raw` on ME_134 is the new best model on the current split, 0.7632
canonical, and it streams.** That is +1.2 points over the best raw-landmark `gru` (0.7517) at
about the same size (929k vs 861k parameters). The phonology features alone (`gru_phono`,
0.7010) do not beat raw landmarks; the gain comes from giving the model both.

Source: `experiments/recognition/gislr.1.models.training.ipynb` (run by the user, 2026-09-25
15:00 → 23:25), every run scored with `sb-evaluate` on the canonical split (GISLR_Stratified's
fixed 18,896-video val set) on 2026-09-26.

## 1. What ran, and what didn't

The notebook ran §4 (caches), §5b `gru_deep`, §6 `lstm`, §7 `bilstm`, §8b–§8e (phonology),
and a re-run of §5 `gru`. It did **not** finish:

| cell | state | consequence |
|---|---|---|
| §5 `gru` | interrupted (KeyboardInterrupt) | 4 duplicate `gru` runs, plus `1790352803` stopped at epoch 9 (0.6098); ignore that one |
| §7 `bilstm` FP_118 | stopped at epoch 22, still at the starting learning rate | `1790351435` 0.7049 is not a final number |
| §8 `cnn1d` × 3 | never run | no current-split `cnn1d` exists |
| §8e `bilstm_phono` | stopped at epoch 14, still at the starting learning rate | `1790356714` 0.7054 is not a final number |
| §9 comparison, §10 eval handoff | never run | done here instead (`sb-evaluate` per run) |

The three stopped runs were scored once for this report, then set back to `eval_status:
pending`. A resumed run continues in the same folder, and `write_meta` keeps canonical metrics
across training rewrites, so leaving them `canonical` would have kept the mid-training score
after the run finished.

## 2. Phonology arms (TODO §3.9)

| run | arch | input | features/frame | params | canonical acc | macro | classes < 50% |
|---|---|---|---|---|---|---|---|
| `1790355555` | `gru_phono_raw` | 84 phonology + ME_134 raw xy | 352 | 928,698 | **0.7632** | 0.7614 | 4 |
| `1790354810` | `gru_phono_raw` | 84 phonology + PH_55 raw xy | 194 | 807,038 | 0.7429 | 0.7410 | 10 |
| `1790352947` | `gru_phono` | 84 phonology only (PH_55 landmarks) | 84 | 722,338 | 0.7010 | 0.6988 | 19 |
| `1790356714` | `bilstm_phono` | as `1790355555`, bidirectional | 352 | 2,956,218 | 0.7054 *(epoch 14 of ?)* | – | – |
| *reference* `1789559734` | `gru` | ME_132 raw xy | 264 | 860,938 | 0.7517 | 0.7499 | 5 |

- **Phonology + raw beats raw alone.** ME_134 is ME_132 plus chin and forehead, which the
  front-end needs as location anchors, so the raw part is nearly the reference's input. The
  +1.2 points comes from adding the 84 phonology features, not from a bigger landmark set.
- **Phonology alone loses 5 points** (0.7010). 84 hand-built numbers recover most, not all, of
  what the model reads from raw coordinates.
- **It also has fewer weak classes**: 4 classes under 50% accuracy vs 5 for the best `gru`, 10
  for PH_55.
- **`bilstm_phono` is unfinished.** At epoch 14 it was at 0.7054, ahead of every plain
  `bilstm` at the same epoch (0.667–0.677) but behind `gru_phono_raw` (0.7303). Its final
  number, and so the offline ceiling of this input, is still unknown.

## 3. Which features carry `gru_phono_raw` (§8d, permutation importance)

Each group's features shuffled across clips on the val split; `drop` = accuracy lost.

| group | features | accuracy with it shuffled | drop |
|---|---|---|---|
| raw hands (xy) | 84 | 0.0824 | **0.681** |
| right handshape | 25 | 0.3205 | **0.443** |
| left handshape | 25 | 0.4229 | 0.340 |
| right palm orientation | 6 | 0.5494 | 0.214 |
| raw face (xy) | 156 | 0.5805 | 0.183 |
| left palm orientation | 6 | 0.6180 | 0.145 |
| raw pose (xy) | 28 | 0.7006 | 0.063 |
| left location | 9 | 0.7111 | 0.052 |
| right location | 9 | 0.7227 | 0.041 |
| elbow angles | 2 | 0.7429 | 0.020 |
| right / left hand present | 1 / 1 | 0.755 / 0.757 | 0.008 / 0.006 |

- **The model leans on the engineered handshape features even though raw hand coordinates are
  there too**: shuffling right handshape alone costs 44 points. So they are not a redundant
  re-encoding the model ignores. This is the evidence for why the combination wins.
- Palm orientation is next (21 / 15 points). Location is small (4–5 points).
- The dominant hand (right, after mirroring the left) matters more than the other hand for
  every parameter, as expected for ASL.
- Drops are not additive: correlated groups share information, so each drop understates that
  group's total contribution.

Artifacts: `registry/runs/1790355555/assets/feature_group_importance.{csv,png}`.

## 4. The architecture grid on the current split (TODO §4.3)

Canonical accuracy, xy coordinates, `v2-plateau-300` regime, seed 42:

| architecture | streaming | ME_126 | ME_132 | FP_118 | params |
|---|---|---|---|---|---|
| `gru` | yes | 0.7450 | **0.7517** | 0.7425 | 0.85M |
| `bilstm` | no | 0.7502 | 0.7417 | *0.7049 (unfinished)* | 2.75M |
| `lstm` | yes | 0.7366 | 0.7261 | 0.7261 | 1.10M |
| `gru_deep` (384×4) | yes | 0.7343 | 0.7332 | 0.7304 | 3.49M |
| `cnn1d` | yes | – | – | – | not run |

- **`gru` is still the best raw-landmark model, streaming included.** Bidirectionality buys
  nothing on this split, which agrees with `bilstm-curated.md` and `five-arch-benchmark.md`.
- **Capacity is not the bottleneck**: `gru_deep` has 4× the parameters and loses 1–2 points on
  every subset. This answers TODO §4.1 ("does depth close the gap to BiLSTM"): no.
- **Subset differences are within about 1 point** for every architecture.
- **Reproducibility for free**: the duplicate `gru` runs reproduce the 2026-09-16 numbers
  exactly on ME_126 and FP_118 (0.745025, 0.742485, same seed); ME_132 differs by 0.0007.

## 5. A registry defect found while analysing this run (fixed)

Ten runs trained on 2026-09-13 on the **retired** 9,448-val split (`gru`, `gru_deep`, `lstm` ×
3 subsets, `bilstm` ME_126) had been scored by the 2026-09-18 eval backfill (`300e7a7`) on the
**current** 18,896-val split. The two splits are different random partitions, so part of the
current val set was in those runs' training data. They read 0.847–0.872 and were marked
`canonical`. Any leaderboard view that included legacy runs, which is the default, ranked them
first, and `1789258464` (0.8719) had already been exported to TFLite as the "best" model.

Fixed:
- `sb.recognize.evaluate.evaluate_run` now refuses a run whose recorded `split.n_val` differs
  from the canonical split's.
- `gislr.2.models.evaluation.ipynb` §3's backfill skips legacy-split runs (`AND NOT legacy_split`).
- The ten runs are back to `eval_status: pending` with their metrics cleared, a note in
  `meta.json`, and the four contaminated eval files removed (still in git history).

No report quoted the inflated numbers. The TFLite file in `1789258464/export/` is a valid
model; only the claim that it was the best one was wrong.

## 6. Similar-word scoring

Scored with "a similar word counts as correct" in `label-merging.md` (2026-09-26): the best
model goes **0.7632 strict → 0.7774 counting synonyms** (the 9 `MERGE_GROUPS`) **→ 0.7822
lenient** (also any WordNet-related pair, an upper bound). The +1.4-point synonym lift is the
same for every architecture, so it does not change the ranking.

## 7. Next

- Finish `bilstm_phono` (the offline ceiling of this input) and `bilstm` FP_118: re-run §8e
  and §7 (auto-resume picks up the stopped runs).
- `cnn1d` × 3 (§8), if the grid should be complete.
- Port the winner to continuous signing (TODO §3.9): a `gru_continuous_phono` run on ME_134,
  then the step-model export, which needs a Keras port of the front-end before the web app can
  use it.
