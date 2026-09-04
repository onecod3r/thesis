# Why the models stop at ~75%

**Question:** are semantically similar signs (`awake`/`wake`, `lips`/`mouth`, `pen`/`pencil`)
the reason GISLR accuracy plateaus at ~75%?

**Answer: no.** Semantic confusion is real, specific and reproducible, but it accounts for
about **a tenth** of the error — solving it perfectly would move the plateaued family from
0.7433 to 0.7704. The plateau is a **generalization failure**, not a label ceiling: the models
separate these pairs almost perfectly on data they were trained on (mean symmetric confusion
**0.012 on train vs 0.273 on val**) and fail to carry it over.

Produced by `experiments/recognition/gislr.2.models.plateau-diagnosis.ipynb` over **31 canonical
runs** at ≥0.70 accuracy (mean 0.7433). `cnn1d` (~0.54) is excluded: it has a different problem
and would drag every aggregate.

---

## 1. The pairs are real and they replicate

Aggregate confusion, row-normalised per run before averaging so no class or run dominates.
Pairs scored symmetrically (`C[i,j] + C[j,i]`) — a symmetric confusion is the signature of two
classes that genuinely overlap, as opposed to a weak class collapsing into a strong neighbour.

| pair | symmetric rate | | pair | symmetric rate |
|---|---|---|---|---|
| `awake` ↔ `wake` | 0.839 | | `duck` ↔ `goose` | 0.330 |
| `lips` ↔ `mouth` | 0.533 | | `animal` ↔ `have` | 0.283 |
| `hear` ↔ `listen` | 0.411 | | `cat` ↔ `kitty` | 0.271 |
| `pen` ↔ `pencil` | 0.404 | | `sleep` ↔ `sleepy` | 0.263 |
| `gift` ↔ `give` | 0.367 | | `nap` ↔ `sleep` | 0.260 |
| `cut` ↔ `scissors` | 0.357 | | `bed` ↔ `bedroom` | 0.212 |
| `finger` ↔ `wait` | 0.343 | | `dry` ↔ `dryer` | 0.209 |
| `stay` ↔ `that` | 0.336 | | `look` ↔ `see` | 0.194 |

Almost all are near-synonyms or morphological relatives. Two — `finger`/`wait` and
`animal`/`have` — are not semantically related at all, so whatever drives those is not meaning.

## 2. But they are only a tenth of the error

Merge the top-N confusable pairs into single classes and re-score: that is the accuracy the
models would reach **if semantic confusion were solved perfectly**. It is an upper bound.

The **random-pair control** is what makes it interpretable — merging any N pairs can only raise
accuracy, so the semantic figure alone says nothing.

| pairs solved | semantic | random control | gain over control |
|---|---|---|---|
| 5 | 0.7540 | 0.7434 | **+0.011** |
| 10 | 0.7605 | 0.7435 | **+0.017** |
| 20 | 0.7704 | 0.7435 | **+0.027** |
| 30 | 0.7768 | 0.7436 | **+0.033** |
| 50 | 0.7873 | 0.7438 | **+0.043** |
| 75 | 0.7966 | 0.7445 | **+0.052** |

Baseline 0.7433. **The top 20 pairs absorb 10.5% of all errors.** The effect is unmistakably
real — the control gains essentially nothing, so these are not arbitrary merges — but it is
small. Even declaring 50 pairs (100 of the 250 classes) solved leaves accuracy under 0.79.

A ceiling caused by semantic ambiguity would look nothing like this. It would show the
confusable pairs absorbing most of the error mass, not a tenth of it.

## 3. Semantic errors are near-misses; the rest are not

Top-k recovery, split by error type (run `1784453891`, gru/ME_126/xy, 0.7565):

| error type | n | recovered at rank 2 | recovered at rank 5 |
|---|---|---|---|
| semantic pair | 223 | **77%** | **95%** |
| everything else | 2078 | 31% | 56% |

Where the model confuses a near-synonym it very nearly has the answer. That is the profile of a
*decision* problem, and it is what a rescoring layer (TODO §8) could plausibly capture — bounded
by §2 at roughly 2.7 points.

For the other 90% of errors the true label is outside the top 5 more often than not. No
re-ranking recovers those.

## 4. The boring explanations are ruled out

- **Class imbalance is not it.** `corr(support, accuracy) = +0.34`, but the canonical val set
  spans only 30–42 videos per class — there is barely any imbalance to correlate with.
- **The error is diffuse.** The 10 worst classes carry 8.6% of all error (uniform would be
  4.0%); the worst 50 carry 33.8% against a uniform 20%.

## 5. The decisive result: it is generalization, not a ceiling

Run `1784447187` (gru/ME_126/xy, 0.7565), inference over a seeded stratified sample of the
**train** split (8 clips/class, 2,000 total), scoring the same 20 pairs:

| | mean symmetric confusion |
|---|---|
| validation | **0.273** |
| train | **0.012** |

**19 of the 20 pairs sit at exactly zero confusion on training data.** The single exception is
`awake`→`wake` at 0.25, and even that is one-directional (`wake`→`awake` is 0.0).

The models can separate these pairs. They do it on examples they have seen and fail to do it on
examples they have not — which is the definition of a generalization gap, and consistent with
the train 90–99% vs val ~75% figure recorded on 2026-07-19. This is **not** a label ceiling and
**not** an irreducible ambiguity in the sign vocabulary.

The caveat worth stating: a model that memorises will show ~0 train confusion on anything it
memorised, so this demonstrates capacity to fit, not that the features generalisably separate
the pairs. §6 is the check on that.

## 6. What the landmarks actually carry

A binary logistic regression that only ever sees one pair, on time-pooled features
(mean/std/min/max over frames, ME-126 xy, 5-fold CV):

| pair | position | velocity | both | confusion rate |
|---|---|---|---|---|
| `awake`/`wake` | **0.463** | 0.506 | 0.463 | 0.839 |
| `lips`/`mouth` | 0.674 | 0.588 | 0.638 | 0.533 |
| `pen`/`pencil` | 0.632 | 0.556 | 0.597 | 0.404 |
| `cut`/`scissors` | 0.687 | 0.638 | 0.698 | 0.357 |
| `hear`/`listen` | 0.754 | 0.612 | 0.728 | 0.411 |
| `gift`/`give` | 0.754 | 0.693 | 0.736 | 0.367 |
| `finger`/`wait` | 0.790 | 0.692 | 0.768 | 0.343 |

Median position probe 0.628 on a binary task. **`awake`/`wake` sits at chance** — and stays
there under feature standardisation and stronger regularisation (0.463 → 0.464 at `C=0.01`), so
it is not a scaling artifact. `corr(confusion_rate, probe_accuracy) = −0.72`: the pairs the
models confuse most are exactly the pairs a pooled probe cannot separate.

**Pooled velocity does not rescue it.** Adding motion summaries changes nothing (`velocity_gain`
is ≈0 or slightly negative for 6 of 7 pairs). So the distinction is not in pooled position *or*
pooled motion.

Read this precisely. It bounds **time-pooled summaries**, not the landmarks: pooling destroys
ordering and trajectory shape, which is exactly where `awake` (repeated) and `wake` (single
motion) differ. The sequence models do better than these probes — they reach ~57% recall on
`awake`/`wake` against the probe's 46% — so they are extracting temporal structure that pooling
throws away. It just does not survive to unseen signers.

**This does not refute TODO §7.3.** §7.3 proposes per-frame velocity *channels* into a sequence
model; §6 here tests pooled velocity summaries in a linear model. Different claim.

## 7. Conclusions

1. **Semantic similarity is not why the models stop at 75%.** It is worth ~2.7 points of the
   ~25 that are missing. Quote it that way.
2. **The plateau is a generalization gap.** Train confusion 0.012 vs val 0.273 on the very pairs
   that were the ceiling candidate, alongside train 90–99% vs val ~75% overall.
3. **The levers this points at are §7.2 and §7.4** — removing nuisance variance from the input
   (normalization: nothing in the stack currently removes signer appearance or position) and
   expanding the training distribution (augmentation/regularization). Both attack a
   generalization gap directly.
4. **A rescoring layer (§8) is bounded at ~2.7 points** and only helps the near-miss tenth.
   Worth doing on its own terms, not as a plateau fix.
5. **`finger`/`wait` and `animal`/`have` are the interesting anomalies** — confused as heavily
   as the near-synonyms without being semantically related, and `finger`/`wait` has the *highest*
   probe score (0.790) of the set. Worth an eyeball on raw sequences.

## Method notes

- Row-normalising each run's confusion matrix before averaging is deliberate: without it, runs
  and classes with more samples would dominate.
- The oracle merge uses **one** union-find map applied to both labels and predictions. Building
  separate maps produces accuracies *below* baseline, which is impossible for a merge — an early
  version had that bug and it inflated the apparent effect roughly fivefold.
- The train-split sample is seeded and stratified (`GroupBy.sample`, 8/class), so §5 is a rate
  estimate on 2,000 clips rather than the full 85,029.
- Only 1 of 31 runs has top-k stored (`sb-evaluate` began saving it 2026-09-04), so §3 rests on
  a single run. Re-run `sb-evaluate` on more leaders to widen it.
