# Why the models stop at ~75%

**Question:** are semantically similar signs (`awake`/`wake`, `lips`/`mouth`, `pen`/`pencil`)
the reason GISLR accuracy plateaus at ~75%?

**Answer so far: no — not mainly.** Semantic confusion is real, specific and reproducible, but
it accounts for about **a tenth** of the error. Solving it perfectly would move the plateaued
family from 0.7433 to 0.7704. The other ~22 points of error are diffuse, and most of them are
not near-misses at all.

Produced by `experiments/recognition/gislr.2.models.plateau-diagnosis.ipynb` over **31
canonical runs** at ≥0.70 accuracy (mean 0.7433). `cnn1d` (~0.54) is excluded: it has a
different problem and would drag every aggregate.

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

Almost all are near-synonyms or morphological relatives. Two (`finger`/`wait`, `animal`/`have`)
are not semantically related at all, which is worth noting: whatever drives those is *not*
meaning.

## 2. But they are only a tenth of the error

Merge the top-N confusable pairs into single classes and re-score: that is the accuracy the
models would reach **if semantic confusion were solved perfectly**. It is an upper bound.

The **random-pair control** is what makes it interpretable — merging any N pairs can only
raise accuracy, so the semantic figure alone says nothing.

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

A ceiling at 75% caused by semantic ambiguity would look nothing like this. It would show the
confusable pairs absorbing most of the error mass, not a tenth of it.

## 3. Semantic errors are near-misses; the rest are not

Top-k recovery, split by error type (run `1784453891`, gru/ME_126/xy, 0.7565):

| error type | n | recovered at rank 2 | recovered at rank 5 |
|---|---|---|---|
| semantic pair | 223 | **77%** | **95%** |
| everything else | 2078 | 31% | 56% |

Where the model confuses a near-synonym it very nearly has the answer — 95% of those errors
have the true label within the top 5. That is the profile of a *decision* problem, and it is
what a rescoring layer (TODO §8) could plausibly capture.

For the other 90% of errors, the true label is outside the top 5 more often than not. No
re-ranking recovers those; the representation does not carry the answer.

## 4. The boring explanations are already ruled out

- **Class imbalance is not it.** `corr(support, accuracy) = +0.34`, but the canonical val set
  spans only 30–42 videos per class — there is barely any imbalance to correlate with.
- **The error is diffuse, not concentrated.** The 10 worst classes carry 8.6% of all error
  (a uniform spread would be 4.0%); the worst 50 carry 33.8% against a uniform 20%. Mildly
  concentrated, nowhere near "a handful of broken classes".

## 5. What this leaves

The plateau is ~22 points of error spread across the label space, in which the model usually
does not have the right answer anywhere near the top. Two readings remain, and they are
distinguishable — §7 and §8 of the notebook are exactly that test, and both need a GPU:

- **§7 — train-split confusion.** Does the model confuse these pairs on data it was trained
  on? Train accuracy is 90–99% against val ~75% (`docs/logs/daily/2026-07-19.md`), so it
  separates *something*. If train confusion on these pairs is ~zero, the pairs are separable
  and this is a generalization failure (→ §7.4 augmentation). If train confusion is also high,
  it is a representation ceiling (→ §7.2/§7.3 features).
- **§8 — per-pair separability probe.** A binary classifier that only ever sees `awake` and
  `wake`. Near 0.95 means the signal is in the landmarks and the 250-way head is not using it;
  near 0.55 means those two signs really are near-identical in this representation.

**The unexplored lever this points at** is §7.2 — normalization. Nothing in the current stack
removes signer appearance or position; every run trains on raw or reference-point-shifted
coordinates. A representation carrying signer identity as strongly as it carries sign identity
would produce exactly this: high train accuracy, diffuse val error, and the right answer often
nowhere near the top.

## Method notes

- Row-normalising each run's confusion matrix before averaging is deliberate: without it, runs
  and classes with more samples would dominate the aggregate.
- The oracle merge uses **one** union-find map applied to both labels and predictions. Building
  separate maps produces accuracies *below* baseline, which is impossible for a merge — an
  early version of this analysis had that bug and it inflated the apparent effect roughly
  fivefold.
- Only 1 of 31 runs currently has top-k stored (added to `sb-evaluate` on 2026-09-04). §3's
  numbers are therefore from a single run; re-run `sb-evaluate` on more to widen it.
