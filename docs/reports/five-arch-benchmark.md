# Five architectures, one feature pipeline: GRU vs LSTM vs BiLSTM vs CNN vs DNN

**Status: complete.** Five arms trained and evaluated on the current
GISLR_Stratified canonical split (18,896-video held-out `test.csv`).

| | |
|---|---|
| **Question** | With feature pipeline and hyperparameter regime held fixed, how do the five architectures actually rank — and where does `dnn` (never measured on this pipeline before) land? Does the new mean-true-class-confidence metric agree with ranked accuracy or diverge? |
| **Instrument** | `experiments/recognition/gislr.1.models.five-arch-benchmark.ipynb` (TODO §3.7) |
| **Data** | GISLR_Stratified, ME-126 subset, xy coordinates, `base_v1` (252-dim raw), one hyperparameter regime copied from `gislr.training.json`'s `shared` block |
| **Arms** | `gru`, `lstm`, `bilstm`, `cnn`, `dnn` — see §1 |

---

## 1. Method

Single final fit per arm (small internal-val carve-out for early stopping,
no k-fold — same convention as `bilstm-curated.ipynb`/`curated-features.ipynb`),
held-out `test.csv` eval. `gru`/`lstm`/`bilstm`/`cnn` are the unmodified
production classes from `sb.recognize.architectures`; `cnn`'s `num_layers`
overrides to 5 (TODO §5.5.1's receptive-field fix — at the shared
`num_layers=2` a dilated-conv block collapses to a 13-frame receptive
field). `dnn` is `sb.recognize.interp.models.LandmarkDNN`, a per-frame
memory-free classifier, fed the same flat 252-dim vector with no
angle/relational blocks (`base_v1` doesn't compute them); its
video-level prediction is the softmax averaged over valid frames.

**Mean true-class confidence** (new metric): `probs[i, true_label[i]]`,
averaged overall and per gloss — the probability mass the model puts on
the *correct* class specifically, independent of its rank. Computed
alongside top-1/3/5 from the same prediction matrix.

Note: `topn_summary.csv`/`per_gloss_confidence.csv` are the only artifacts
the notebook itself wrote to disk; the confusable-pair and registry-comparison
sections (§4-§5 below) print in-kernel and were not saved anywhere by the
notebook, so this write-up recomputes both from each arm's cached
`test_predictions.npz` (labels/probs/ranked-top5), which is why every number
here matches what a re-run of those notebook cells would show.

## 2. Results

### 2.1 Headline numbers (held-out `test.csv`, 18,896 videos)

| arm | top-1 | top-3 | top-5 | mean true-class conf. | n_params | streaming? |
|---|---|---|---|---|---|---|
| `bilstm` | **0.7392** | 0.8653 | 0.8924 | 0.5898 | 2,751,218 | no (offline reference) |
| `gru` | 0.7380 | **0.8669** | **0.8973** | 0.5983 | 851,698 | **yes** |
| `lstm` | 0.7286 | 0.8546 | 0.8861 | **0.6059** | 1,113,842 | **yes** |
| `cnn` | 0.6696 | 0.8333 | 0.8740 | 0.5919 | 1,702,386 | **yes** |
| `dnn` | 0.6485 | 0.8355 | 0.8816 | 0.2387 | 328,432 | **yes** (per-frame) |

Among the three streaming-viable recurrent/conv architectures, **`gru` leads**
(0.7380) — `bilstm` edges it out by 0.12pp but is offline-only, so it doesn't
change the deployment ranking. `cnn` trails the recurrent arms by 6-7pp
despite the `num_layers=5` receptive-field fix. `dnn`, with no memory at all,
gives up a further 2pp of top-1 to `cnn` but is competitive on top-3/top-5
(0.8355/0.8816 — ahead of `cnn`'s 0.8333/0.8740) — its ranked accuracy holds
up reasonably once the correct class only has to be in the top few, not
necessarily first.

**The standout finding is `dnn`'s mean true-class confidence: 0.2387, a
quarter of every other arm's (0.59-0.61)**, despite top-3/5 numbers in the
same range as `cnn`. Ranked accuracy and confidence decouple here: `dnn` is
about as likely as the recurrent models to have the right answer in its
top few guesses, but the probability mass it assigns to that answer is far
lower — consistent with a per-frame classifier whose video-level softmax is
an *average* over per-frame softmaxes, which flattens the distribution
relative to a single sequence-level decision the recurrent/conv arms make
once.

### 2.2 Training dynamics — who overfits

| arm | epochs run | best epoch | train acc @ last | internal val acc @ best | train/val gap |
|---|---|---|---|---|---|
| `gru` | 67 | 66 | 0.8994 | 0.7291 | 0.170 |
| `lstm` | 93 | 92 | 0.9356 | 0.7341 | 0.201 |
| `bilstm` | 80 | 79 | 0.9908 | 0.7407 | 0.250 |
| `cnn` | 138 | 137 | 0.7047 | 0.6688 | 0.036 |
| `dnn` | 136 | 135 | 0.6525 | 0.6476 | 0.005 |

The recurrent arms reproduce the familiar plateau-diagnosis pattern
(`docs/reports/plateau-diagnosis.md`) — large train/val gaps that grow with
capacity/bidirectionality (`gru` 0.170 → `lstm` 0.201 → `bilstm` 0.250).
`cnn` and `dnn` barely overfit at all (gaps of 0.036 and 0.005) — they ran
far more epochs before early stopping (137-138 vs 66-79) without their
train accuracy ever pulling much ahead of validation, which reads as
**underfitting** relative to the recurrent arms rather than a generalization
win: both cap out well below `gru`/`lstm`/`bilstm` on the metric that
matters (top-1).

### 2.3 Confusable pairs

Same 16 documented pairs as `curated-features.ipynb`/`pair-similarity.md`
(mean top-1 accuracy over both directions of each pair):

| arm | mean confusable-pair accuracy | overall top-1 |
|---|---|---|
| `gru` | **0.6256** | 0.7380 |
| `bilstm` | 0.6169 | 0.7392 |
| `lstm` | 0.6066 | 0.7286 |
| `cnn` | 0.5503 | 0.6696 |
| `dnn` | 0.5135 | 0.6485 |

Ranking tracks overall top-1 closely, with one flip: `gru` edges out
`bilstm` on confusable pairs specifically (0.6256 vs 0.6169) despite
`bilstm` having the higher overall top-1 — a small effect, but bidirectional
context doesn't buy extra help on the semantic near-synonyms that dominate
this repo's error analysis (`docs/reports/plateau-diagnosis.md`).

**`give`/`gift` is the hardest pair for every architecture** (mean 0.272 —
`give` scores 0.116-0.391 depending on arm, the single worst per-class
number of any pair in either direction) and, independently, `give` is also
the **lowest mean-true-class-confidence gloss for all five arms** (§2.4) —
consistent cross-validation from two different metrics that this is
genuinely the hardest sign in the benchmark, not an artifact of either
measurement. `see`/`look` is the easiest pair for every arm (mean 0.859).

### 2.4 Lowest-confidence glosses

Averaged across all five arms, the five lowest mean-true-class-confidence
signs are **`give` (0.192), `there` (0.213), `beside` (0.232), `go`
(0.239), `nap` (0.248)** — the same handful shows up near the bottom for
every individual arm (`give`, `there`, `beside`, `go` all place in every
arm's own worst-5), so this isn't one architecture's idiosyncrasy. `dnn`'s
worst glosses are far more extreme than the other arms' (`give` 0.028,
`beside` 0.033, `into` 0.039) — the same confidence-compression from §2.1
showing up at the per-gloss level, not just in the overall average.

Full table: `data/cache/gislr/five_arch_benchmark_runs/per_gloss_confidence.csv`
(not yet promoted into `docs/reports/assets/` — cache-only pending a decision
on whether this benchmark gets its own asset folder).

## 3. Comparison against the registry leaderboard

Every canonical registry run on the current split (`registry/index.csv`,
queried live via `sb.mlops.query.query_runs`):

| architecture | subset | coords | accuracy | source |
|---|---|---|---|---|
| `gru` | ME_132 | xy | 0.7517 | registry (canonical) |
| `gru` | ME_126 | xy | 0.7450 | registry (canonical) |
| `gru` | FP_118 | xy | 0.7425 | registry (canonical) |
| `bilstm` | ME_126 | xy | 0.7392 | **this notebook** |
| `gru` | ME_126 | xy | 0.7380 | **this notebook** |
| `bilstm_base` | ME_126 | xy | 0.7371 | `bilstm-curated.ipynb` (2026-09-21) |
| `lstm` | ME_126 | xy | 0.7286 | **this notebook** |
| `cnn` | ME_126 | xy | 0.6696 | **this notebook** |
| `dnn` | ME_126 | xy | 0.6485 | **this notebook** |

This notebook's `gru`/ME_126/xy (0.7380, single final fit) lands **0.70pp
below the registry's canonical `gru`/ME_126/xy run (0.7450)** — the two
protocols are not identical (this notebook does one final fit with an
internal-val carve-out for early stopping; the registry-producing driver's
protocol may differ in exact split/seed), so the two are in the same
ballpark but not a strict apples-to-apples check. `bilstm` here (0.7392)
and `bilstm_base` from the separate `bilstm-curated.ipynb` run the same
day (0.7371) agree to within 0.2pp on the identical `base_v1`/ME-126/xy
pipeline — a useful sanity check that the single-final-fit protocol is
reproducible run-to-run.

**Every arm here — including the accuracy leader, `bilstm`, which is
offline-only — sits below or at the low end of the three canonical `gru`
runs.** There is still no architecture in this benchmark, or anywhere in
the current-split registry, that beats `gru`/ME_132/xy's 0.7517 while
remaining streaming-viable.

## 4. `dnn` across three feature pipelines

`dnn` (`LandmarkDNN`) has now been measured on three different feature
pipelines across three separate notebooks. Not a controlled ablation (the
pipelines differ in width, engineered content, and training protocol), but
the pattern is worth recording:

| feature pipeline | dims | top-1 | source |
|---|---|---|---|
| `landmark_interp_v1` (full-543, engineered) | 5,442 | **0.7102** | `landmark-importance.md` |
| `base_v1` (ME-126, raw xy) | 252 | 0.6485 | **this notebook** |
| `landmark_curated_v1` (ME-126, curated) | 922 | 0.5776 | `curated-features.md` |

Non-monotonic in dimensionality: the richest, widest engineered pipeline
(full-543) scores highest, the narrowest raw pipeline (this notebook) is
middling, and the mid-sized curated pipeline scores lowest of the three —
consistent with `bilstm-curated.md`'s finding that the curated feature set
is a poor fit for architectures without an attention/projection gate ahead
of it (`dnn` here has none). Not investigated further; filed as a
follow-up (§5).

## 5. Follow-ups

- [ ] Cross-check this notebook's `gru`/`lstm`/`bilstm`/`cnn` against their
  canonical registry counterparts once TODO §4.3's 12-run gap
  (`gru_deep`/`lstm`/`bilstm`/`cnn1d` × 3 subsets) is filled — right now only
  `gru` has a canonical registry comparison point, and it's a single-final-fit
  vs registry-driver protocol difference, not a clean same-protocol check.
- [ ] `dnn`'s mean-true-class-confidence collapse (0.24 vs 0.59-0.61
  elsewhere) despite competitive top-3/5 — is this specific to averaging
  per-frame softmaxes, or would a per-frame-max / last-frame readout (closer
  to what a streaming deployment would actually see) close the gap? Relevant
  to whether mean-true-class-confidence is comparable at all between
  per-frame and sequence-level architectures, or needs a different
  aggregation for `dnn`.
- [ ] Why does `dnn` do *worse* on the 922-dim curated pipeline than on the
  252-dim raw pipeline here (§4) — the curated pipeline is supposed to be a
  distillation of what matters, not noise. Cross-reference
  `bilstm-curated.md`'s decomposition (curated features cost architectures
  without a projection/attention gate the most) — `dnn` has no such gate
  either.
- [ ] Promote `per_gloss_confidence.csv` and a confusable-pair table into
  `docs/reports/assets/five-arch-benchmark/` if this comparison gets
  revisited, rather than leaving them cache-only.
- [ ] Fill in the notebook's own §4b/§5/§6/§7 cells by actually running them
  (they currently only print in-kernel) so re-running reproduces this
  write-up's tables directly, instead of requiring the `test_predictions.npz`
  reconstruction this report used.
