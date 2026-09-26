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

25 current-split canonical runs in the aggregate confusion matrix: [1790355555, 1789559734, 1790327942, 1790347646, 1790327033, 1789558839, 1790330305, 1790354810, 1789560829, 1790332037, 1790349689, 1790338279, 1790333188, 1790334584, 1790336517, 1790339908, 1790341214, 1790143122, 1790144582, 1790356714, 1790351435, 1790352947, 1790146838, 1790352803, 1790142624].

| run | architecture | subset | streaming | strict | merged (synonyms) | lenient (WordNet) |
|---|---|---|---|---|---|---|
| `1790355555` | gru_phono_raw | ME_134 | yes | 0.7632 | 0.7774 (+0.0141) | 0.7822 |
| `1789559734` | gru | ME_132 | yes | 0.7517 | 0.7657 (+0.0139) | 0.7707 |
| `1790327942` | gru | ME_132 | yes | 0.7511 | 0.7651 (+0.0140) | 0.7703 |
| `1790347646` | bilstm | ME_126 | no | 0.7502 | 0.7648 (+0.0146) | 0.7698 |
| `1790327033` | gru | ME_126 | yes | 0.7450 | 0.7591 (+0.0141) | 0.7643 |
| `1789558839` | gru | ME_126 | yes | 0.7450 | 0.7591 (+0.0141) | 0.7643 |
| `1790330305` | gru | ME_126 | yes | 0.7450 | 0.7591 (+0.0141) | 0.7643 |
| `1790354810` | gru_phono_raw | PH_55 | yes | 0.7429 | 0.7573 (+0.0143) | 0.7628 |
| `1789560829` | gru | FP_118 | yes | 0.7425 | 0.7573 (+0.0148) | 0.7621 |
| `1790332037` | gru | FP_118 | yes | 0.7425 | 0.7573 (+0.0148) | 0.7621 |

Lenient = merged + 60 WordNet pairs at Wu-Palmer >= 0.85 (related concepts such as horse/zebra, not synonyms): an upper bound, not a claim that those predictions are right.
