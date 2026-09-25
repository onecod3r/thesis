# Semantically duplicate GISLR labels: which to merge, and what it's worth

**Question (user, 2026-09-25):** do any of GISLR's 250 gloss labels mean the same (or a
near-synonymous) thing, such that the model's confusion between them is a labeling
question rather than a recognition failure — and if so, what does merging them do to
accuracy for the current top models?

**Answer: nine groups (19 words folded down to 9 canonical labels) both mean nearly the
same thing and are genuinely confused by the model** — not assumed from either signal
alone. Produced by `experiments/recognition/gislr.2.models.label-merging.ipynb`. This
extends `plateau-diagnosis.md` §1's hand-picked 16-pair list (pre-registry-reset, the old
90/10 split) with a systematic pass over all 250×249/2 pairs on the *current* canonical
split, and packages the result as reusable code
(`sb.recognize.label_merge.MERGE_GROUPS`) rather than a one-off notebook table.

## Method: two independent checks, both required

1. **Semantic**: WordNet similarity (Wu-Palmer, dominant sense per part of speech) plus a
   lexicographic read. WordNet alone is a poor filter for this specific question — it
   rates sibling categories (`cat`/`dog`, any two colors, `horse`/`zebra`) just as
   similar as true near-synonyms, because it measures taxonomic tree distance, not
   synonymy. It's supporting evidence.
2. **Empirical**: the aggregate confusion matrix, rebuilt from every current-split
   canonical run (`sb.mlops.query.leaderboard(include_legacy=False)`), not the possibly
   pre-reset cached `confusion_all_normalized.npy`.

A pair is merged only when **both** signals agree. This ruled out plausible-looking
guesses on the semantic side alone (`not`/`no`, `cute`/`pretty`, `shower`/`bath`,
`talk`/`say`, `fall`/`drop`, `bad`/`yucky`, `loud`/`noisy`: all near-zero in the matrix —
the model already tells them apart, so merging would remove a real distinction for no
accuracy reason) and on the empirical side alone (`cut`/`scissors`, `goose`/`duck`,
`bedroom`/`bed`, `wait`/`finger`, `touch`/`find`, `please`/`minemy`, `bad`/`thankyou`,
`stay`/`that`, `animal`/`have`: real confusion, but a tool vs. its action, a compound
sign sharing a component, or an unrelated meaning that likely shares a handshape — not a
labeling question, since the two concepts differ regardless of how the signs look).
**A direct landmark-similarity check (`sign-patterns.md` §9) found most of these
"rejected" pairs are in fact just as kinematically near-identical as the merge groups**
(mean confusability 1.060 vs. the merge groups' 1.043, both far below random pairs'
1.271) — true near-homophones for unrelated concepts. The decision to keep them as
separate labels is still right; the earlier framing here (that merging them "would hide
a model weakness instead of fixing a semantic one") assumed they were kinematically
separable, which turned out not to be the case for most of them.

## The nine merge groups

| group | canonical | why (semantic + empirical) |
|---|---|---|
| `awake` ↔ `wake` | `wake` | same state/action, tense variants; the single most confused pair in the matrix |
| `give` ↔ `gift` | `gift` | verb/noun of one concept — the sign for the object and the act of handing it over |
| `listen` ↔ `hear` | `listen` | near-synonymous perception verbs |
| `nap` ↔ `sleep` ↔ `sleepy` | `sleep` | a nap is a sleep, sleepy is the adjective for the same state |
| `kitty` ↔ `cat` | `cat` | casual synonym, not a life-stage distinction in this vocabulary |
| `pencil` ↔ `pen` | `pencil` | near-interchangeable writing tools in casual/children's usage |
| `look` ↔ `see` | `see` | near-synonymous visual-perception verbs |
| `mouth` ↔ `lips` | `lips` | lips are part of / near-synonymous with mouth in casual usage |
| `puppy` ↔ `dog` | `dog` | near-synonym in a children's vocabulary, not a meaningful life-stage split here |

Canonical name per group: whichever member has the higher true-class recall in the
aggregate matrix (the model's own more reliably-recognized form) — objective, not a
linguistic preference. Exception: `{nap, sleep, sleepy}` is `sleep` regardless of recall,
since `sleepy` (an adjective) reads oddly as the umbrella term for `nap` clips.

Full rationale, including every rejected candidate and why, lives in
`sb.recognize.label_merge`'s module docstring (the code and this report should never
disagree — if they do, the notebook is the source of truth and the report is stale).

## Relationship to `plateau-diagnosis.md`

That report (pre-registry-reset, old 90/10 split, 31 runs) found 14 of the same pairs from
a hand-picked list of 16, and separately concluded semantic confusion is real but small —
solving the top 20 pairs perfectly moved accuracy by +2.7 points (0.7433 → 0.7704), about a
tenth of the error, against a random-pair control that gained almost nothing. This report
doesn't re-run that random-control experiment (see the plateau report for it); it exists
to (a) refresh the pair list on the current split, (b) turn the hand list into reusable
code, and (c) report the merge's effect on the *current* top models, once they exist.

## Current numbers (regenerated by this notebook)

7 current-split canonical runs in the aggregate confusion matrix: [1789559734, 1789558839, 1789560829, 1790143122, 1790144582, 1790146838, 1790142624].

| run | architecture | subset | strict | merged | delta |
|---|---|---|---|---|---|
| `1789559734` | gru | None | 0.7517 | 0.7657 | +0.0139 |
| `1789558839` | gru | None | 0.7450 | 0.7591 | +0.0141 |
| `1789560829` | gru | None | 0.7425 | 0.7573 | +0.0148 |
| `1790143122` | gru_continuous | None | 0.7188 | 0.7324 | +0.0137 |
| `1790144582` | lstm_continuous | None | 0.7144 | 0.7297 | +0.0152 |
| `1790146838` | gru_continuous | None | 0.6696 | 0.6854 | +0.0158 |
| `1790142624` | gru_continuous | None | 0.3354 | 0.3420 | +0.0066 |

**Provisional in two ways, not just pending TODO §4.3's runs.** Only 7 current-split
canonical runs exist right now (`1790142624` is a broken/outlier run at 0.335 accuracy,
included for transparency, not excluded); the aggregate confusion matrix built from them
is noisier than `plateau-diagnosis.md`'s 31-run one. It surfaced a few pairs worth
watching once more runs land, currently below the bar for a merge (`mouth`/`tooth` 0.063,
`lips`/`tooth` 0.057 — both plausibly "tooth is part of the mouth," similar in kind to the
rejected `bedroom`/`bed`; `ear`/`hear` 0.092, already considered and rejected as
verb/noun). Re-run §2–§4 after `gislr.2.models.evaluation.ipynb` backfills TODO §4.3's
grid; the merge lift so far (+1.4 to +1.6 points, consistent across every architecture
here) is in the same range plateau-diagnosis.md found for a comparable pair count, which
is reassuring rather than a new finding — the delta is not expected to change much,
though which run is "top" will.
