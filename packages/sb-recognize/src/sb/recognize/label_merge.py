"""Semantically-duplicate GISLR labels, merged for evaluation (TODO §7.1).

**Not a change to the training labels or the dataset.** GISLR's 250 gloss strings are
Kaggle's own ground truth; nothing here edits `sign_to_prediction_index_map.json` or any
clip. This module says: for a handful of label *pairs whose meaning is the same or a
near-synonym*, and which the model also spends real accuracy confusing with each other,
score them as one class at evaluation time — a merged accuracy that answers "does the
model get the *concept* right," alongside the strict 250-way number.

**Selection was evidence-gated, not vibes.** `experiments/recognition/gislr.2.models.
label-merging.ipynb` ran two independent checks and only kept a pair when both agreed:

1. **Semantic**: WordNet similarity (dominant sense per POS) plus a lexicographic read.
   WordNet alone is a poor filter here — Wu-Palmer similarity rates sibling categories
   (`cat`/`dog`, any two colors, `horse`/`zebra`) just as high as true near-synonyms,
   because it measures taxonomic distance, not synonymy. It's supporting evidence, not
   the decision.
2. **Empirical**: the aggregate canonical confusion matrix
   (`data/cache/gislr/evaluation/confusion_all_normalized.npy`, averaged over every
   evaluated run) — is the model actually spending accuracy on this pair, not just some
   other pair that happens to share a word stem?

Both mattered. Semantically-plausible pairs the model doesn't actually confuse (`not`/
`no`, `cute`/`pretty`, `shower`/`bath`, `talk`/`say`, `fall`/`drop`, `bad`/`yucky`,
`loud`/`noisy` — all near-zero in the matrix) are excluded: the signs are apparently
distinct enough that merging would just be throwing away a real distinction. Empirically
confused pairs with no semantic relation (`cut`/`scissors`, `goose`/`duck`, `bedroom`/
`bed`, `wait`/`finger`, `touch`/`find`, `please`/`minemy`, `bad`/`thankyou`, `stay`/
`that`, `animal`/`have` — a tool vs. its action, a compound sign sharing a component, or
just similar handshapes) are also excluded: real confusion, but not a labeling question —
`cut` and `scissors` mean different things regardless of how alike the signs look, so
merging them would hide the distinction a downstream consumer needs, not fix anything.
**A direct landmark check found most of these are in fact just as kinematically
near-identical as the merge groups** (`docs/reports/sign-patterns.md` §9,
`gislr.0.dataset.sign-patterns.ipynb`) -- true near-homophones for unrelated concepts,
not a separate, fixable model weakness the way an earlier draft of this module guessed.

**Canonical name per group**: whichever member has the higher true-class recall in the
aggregate matrix (the model's own more reliably-recognized form of the shared concept) —
objective and reproducible, not a linguistic preference. The one exception is `{nap,
sleep, sleepy}`, kept as `sleep` regardless of recall: `sleepy` (an adjective) reads
oddly as the umbrella term for `nap` clips (a noun/verb).
"""

from __future__ import annotations

from dataclasses import dataclass

# Each group: the words that share a concept, tightest-first isn't meaningful here (order
# doesn't matter) -- see the notebook for the WordNet + confusion-matrix evidence behind
# each one. `canonical` must be a member of `words`.
@dataclass(frozen=True)
class MergeGroup:
    words: frozenset[str]
    canonical: str
    note: str


MERGE_GROUPS: tuple[MergeGroup, ...] = (
    MergeGroup(frozenset({"awake", "wake"}), "wake",
               "same state/action, tense variants; mean confusion 0.356 (the single most "
               "confused pair in the matrix)"),
    MergeGroup(frozenset({"give", "gift"}), "gift",
               "verb/noun of one concept (the sign for the object and the act of handing "
               "it over); give->gift 0.267"),
    MergeGroup(frozenset({"listen", "hear"}), "listen",
               "near-synonymous perception verbs; symmetric ~0.18 both directions"),
    MergeGroup(frozenset({"nap", "sleep", "sleepy"}), "sleep",
               "a nap is a sleep, sleepy is the adjective for the same state; "
               "nap<->sleep 0.12-0.18, sleep<->sleepy 0.09-0.14, nap<->sleepy 0.04-0.06"),
    MergeGroup(frozenset({"kitty", "cat"}), "cat",
               "kitty is a casual synonym for cat, not a life-stage distinction here; "
               "kitty->cat 0.160"),
    MergeGroup(frozenset({"pencil", "pen"}), "pencil",
               "near-interchangeable writing tools in casual/children's usage; "
               "symmetric ~0.18-0.21"),
    MergeGroup(frozenset({"look", "see"}), "see",
               "near-synonymous visual-perception verbs; look->see 0.133"),
    MergeGroup(frozenset({"mouth", "lips"}), "lips",
               "lips are part of / near-synonymous with mouth in casual usage; "
               "symmetric ~0.18-0.26 (third-highest pair overall)"),
    MergeGroup(frozenset({"puppy", "dog"}), "dog",
               "puppy is a near-synonym for dog in a children's vocabulary (not a "
               "meaningful life-stage distinction here); puppy->dog 0.114"),
)

# Considered and rejected -- semantically plausible but the model doesn't actually confuse
# them (near-zero in the aggregate matrix), so merging would remove a real distinction for
# no accuracy reason: not/no, cute/pretty, shower/bath, talk/say, fall/drop, bad/yucky,
# loud/noisy, later/after, that/there, quiet/shhh (weak: 0.05/0.00).
#
# Considered and rejected -- genuinely confused but not a semantic duplicate (a tool vs.
# its action, a compound sign sharing one part, or same-category siblings): cut/scissors,
# goose/duck, bedroom/bed, bedroom/room, dryer/dry, hear/ear, wait/finger, touch/find,
# please/minemy, bad/thankyou, stay/that, animal/have, animal/bath, bug/bee, lion/tiger,
# wolf/dog, horse/donkey.


def word_to_canonical() -> dict[str, str]:
    """Every word that belongs to a merge group -> that group's canonical word. Words not
    in any group are simply absent (the caller's default is "unchanged")."""
    return {w: g.canonical for g in MERGE_GROUPS for w in g.words}


def index_remap(label_map: dict[str, int]) -> dict[int, int]:
    """``{original_index: merged_index}`` for all 250 classes. A class outside any merge
    group maps to itself; a merged class maps to its group's canonical index. Apply with
    ``np.vectorize`` or a lookup array -- see :func:`remap_array`."""
    canon = word_to_canonical()
    return {i: label_map[canon.get(w, w)] for w, i in label_map.items()}


def remap_array(label_map: dict[str, int], n_classes: int = 250):
    """A ``(n_classes,)`` int array ``r`` with ``r[i]`` = the merged index for original
    index ``i`` -- the fast form of :func:`index_remap` for indexing into label/pred arrays."""
    import numpy as np

    remap = index_remap(label_map)
    arr = np.arange(n_classes)
    for i, j in remap.items():
        arr[i] = j
    return arr


def merged_accuracy(labels, preds, label_map: dict[str, int]) -> float:
    """Top-1 accuracy after folding merged groups together. ``labels``/``preds`` are the
    original (unmerged) class indices from ``assets/val_predictions.npz``."""
    import numpy as np

    r = remap_array(label_map)
    return float((r[np.asarray(labels)] == r[np.asarray(preds)]).mean())


def merge_report(labels, preds, label_map: dict[str, int]) -> dict:
    """Strict vs. merged accuracy, plus how many validation examples belong to a merged
    group (context for how much the gap *can* move)."""
    import numpy as np

    labels, preds = np.asarray(labels), np.asarray(preds)
    strict = float((labels == preds).mean())
    merged = merged_accuracy(labels, preds, label_map)
    grouped_idx = {label_map[w] for w in word_to_canonical()}
    in_group = float(np.isin(labels, list(grouped_idx)).mean())
    return {"strict_accuracy": strict, "merged_accuracy": merged, "delta": merged - strict,
            "fraction_of_val_in_a_merge_group": in_group, "n_groups": len(MERGE_GROUPS)}
