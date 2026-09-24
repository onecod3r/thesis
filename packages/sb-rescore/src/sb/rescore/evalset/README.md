# Frozen evaluation sets

Cases every prompt/engine version is scored against. Frozen means
append-only: removing or editing a case silently changes what "better" means
and makes two versions incomparable. That is the same reason the recognizer's
canonical split is seeded and its size asserted.

| file | what | status |
|---|---|---|
| `gloss2en.v1.jsonl` | 132 GISLR gloss sentences (12 per theme, seeded `numpy` rng 42 from `sb.recognize.sequences.corpus` v1) with 1–2 English references each | **draft 2026-09-24**. References written by Claude, **awaiting user review** (TODO §12.6). A reviewed set becomes v2; v1 stays as written |

Few-shot examples in `prompts/v1/gloss2en.txt` are drawn from sentences
**not** in this set.
