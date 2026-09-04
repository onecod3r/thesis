"""Model client for the rescoring layer — NOT IMPLEMENTED.

Deliberately empty until TODO §8's scope question is answered: this repo does
isolated single-sign classification (one clip -> one of 250 labels), so there is
no stage that assembles predicted signs into a sentence for a model to have
context over. Wiring a client before that decision would build the wrong thing.

The two readings §8 is choosing between:

1. **Continuous / sentence-level** — needs segmentation and sequence assembly
   first, and a dataset with sentence-level labels to evaluate against. Neither
   GISLR nor POPSIGN has one as extracted here.
2. **N-best re-ranking within one prediction** — no real "sentence": re-rank the
   top-k using label co-occurrence and the confusability prior already measured
   in the aggregate confusion matrix. Testable today, offline, in a notebook.

Whichever it becomes, the surface here stays: prompts come from
``prompts/<version>/`` and are hashed into the run record, and the delta is
measured against the frozen set in ``evalset/``.
"""


def rescore(*args, **kwargs):
    raise NotImplementedError(
        "sb-rescore is a scoped skeleton — TODO §8 has not decided between "
        "sentence-level rescoring and n-best re-ranking. See this module's "
        "docstring before implementing either."
    )
