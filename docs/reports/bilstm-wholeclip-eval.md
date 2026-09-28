# BiLSTM whole-clip evaluation: record-then-recognize (TODO §16.2)

2026-09-28. `experiments/recognition/gislr.3.streaming.bilstm-wholeclip-eval.ipynb`,
full run (21 signers, 5 selection / 16 evaluation, `sequences.csv`'s 6,616 `sentence` +
6,111 `control` sequences). Model: `bilstm`, run `1784447175` (ME_126, xy), 0.7569
canonical — the accuracy leader (§4.1/§4.3), exported Flex-free via
`sb.recognize.export.step.export_web_wholeclip` (see TODO §16.2 for the export itself).

## Headline (evaluation signers, `sentence` split)

| arm | segmentation | prior | GER | sentence acc |
|---|---|---|---|---|
| B0 | oracle (true sign boundaries) | none | **0.106** | 0.718 |
| B0 + prior | oracle | trigram rescore | **0.089** | 0.763 |
| B-mv | `movement_segments` | none | **0.953** | 0.000 |
| B-mv + prior | `movement_segments` | trigram rescore | **0.827** | 0.015 |

References, different modes, context not a leaderboard (`sentence-baselines.md`,
`continuous-models.md`): oracle isolated-model upper bound (`gru_reg` B1) 0.221; best
streaming baseline (`gru` sliding window) 0.507; best continuous model (C1 D3
collapsed) 0.293; deployed (C4 D3 collapsed) 0.278.

## Two findings, pulling in opposite directions

**1. `bilstm`'s whole-clip classification is excellent — better than every existing
number in this repo — when segmentation is accurate.** B0's 0.089–0.106 beats the
isolated-model oracle (0.221) by more than half. This is not surprising in hindsight:
`bilstm` is already the accuracy leader canonically (0.7569 vs `gru`'s 0.7517), and B0
removes segmentation error entirely, so it's measuring classification quality alone.
The 91.8% top-5 hit rate (oracle spans, 4,895 of them) says there is real headroom for
a downstream re-ranker too, and the trigram prior recovers about 16% of B0's remaining
error (0.106 → 0.089) even on this small a vocabulary/corpus.

**2. Movement-only segmentation is not adequate for this classifier, and three
attempted fixes did not close the gap.** B-mv's GER (0.95, sentence accuracy ≈0) is
close to useless despite a *reasonable-looking* segmentation diagnostic (segment count
3.21 vs true 3.13 signs/sequence, 95.7% mean 50%-overlap coverage) — the spans are
roughly the right count and roughly in the right place, but not tight enough for a
bidirectional classifier trained on cleanly-trimmed isolated clips (GISLR clips have 0
lead-in/out hand-absent frames, §12.1). Diagnosed with three follow-up probes (300-clip
subsample, selection signers, not written back to the notebook):

- **Trimming the trailing stillness off each segment** (the `still_frames`-long settled
  tail right before a cut fires) made no difference (GER 0.966–0.968 across trim 0–15
  frames). The tail isn't the problem.
- **Restricting each movement span to its own `frame_kind == SIGN` sub-range** (an
  oracle-assisted variant — real deployment has no `frame_kind` to consult) recovered
  some accuracy (GER 0.97 → 0.80) but is still nowhere near B0 (0.17 on the same
  300-clip subset), and its error shifted from substitution-dominated to
  **deletion-dominated** (0.458): many true signs never got their own segment at all,
  because two signs close together with a quick transition don't produce a stillness
  gap the gate can catch.
- This is the **same failure mode §12.2 already diagnosed and fixed for the continuous
  GRU family**: "with the state reset at each TRUE sign start, the fixed-hold trigger
  commits the right gloss on only 60% of signs vs 76% isolated... No single hold serves
  both [short and long signs]... this is the design brief for 12.3: per-frame
  supervision + a model-signalled boundary, not a fixed hold"
  (`sentence-baselines.md` §5). Movement/stillness gating is structurally a fixed-hold
  heuristic in different clothing, and it hits the same wall independently here.

## Part 2: the hybrid (2026-09-28, same day — the user asked "can we not use this for
the model?")

Segmentation is the actual blocker (Part 1), not classification, so §12.8's already-
**deployed** C4 (a learned boundary/null decoder, GER 0.278 with its own lattice+prior)
is a segmentation source that already works — no new training, no new export, it's the
live model. This section reuses C4's D3 segments (`nu=0.5`, `min_len=4`, the deployed
values, `pipeline.config.json`) and classifies each one with `bilstm`'s whole-clip
readout instead of C4's own per-frame vote. §7 goes one step further: `bilstm`'s top-5
per segment, picked by **exact whole-sentence decoding** (`sb.rescore.prior
.viterbi_rescore`, new — dynamic programming over every combination of each segment's
top-5 candidates against the same deployed trigram, not greedy left-to-right rescoring)
instead of one gloss at a time.

### Headline (evaluation signers, `sentence` split, full corpus — 5,054 sequences)

| arm | segmentation | classifier | decision | GER | sentence acc |
|---|---|---|---|---|---|
| C4 alone | C4 D3 | C4's own vote | argmax, no prior | 0.310 | 0.374 |
| hybrid | C4 D3 | `bilstm` | argmax, no prior | 0.251 | 0.479 |
| hybrid + prior | C4 D3 | `bilstm` | greedy rescore (`fuse.decide`) | 0.220 | 0.525 |
| **hybrid + top-5 + Viterbi** | C4 D3 | `bilstm` top-5 | **exact sentence decode** | **0.195** | **0.586** |
| *(reference)* deployed C4 | C4 D3 + lattice | C4's own vote | tuned lag-2 lattice | 0.278 | — |
| *(reference)* B0 oracle + prior | true boundaries | `bilstm` | greedy rescore | 0.089 | 0.763 |

**Swapping the classifier alone (hybrid vs. C4 alone) already beats deployed C4**
(0.251 vs 0.278) with no prior at all — segmentation was never `bilstm`'s problem, and
C4's segments, while not as tight as oracle boundaries, are tight enough for `bilstm`
to do much better on than its own movement-cut spans (Part 1) ever were. **Global
sentence decoding over the top-5 (§7) is the single biggest additional lever** — 0.220
→ 0.195, a bigger jump than moving from no-prior to greedy-prior gave (0.251 → 0.220).
Selection-signer λ sweep for the Viterbi arm picked λ=0.5 (the top of the swept grid —
an interior optimum was not reached; a wider sweep is a cheap follow-up, not done here).

**Why an n-gram and not an LLM for the "best sentence" search:** this repo already
researched a hosted-LLM prior for this exact role (§12.6, `deployment-research.md` §5)
and found Cloudflare Workers AI exposes no per-token logprobs — without those, using an
LLM here means either one generation per candidate combination or asking it to "rank
these," neither a calibrated probability a real search can use. The n-gram +
`viterbi_rescore` is what actually implements "pick the sign that makes the best
sentence" cheaply and exactly today; the documented trigger for trying an LLM instead
is if this still leaves a real gap, which — at 0.195 vs. deployed's 0.278 — it does not
look like it does.

## Recommendation

**Worth building.** The hybrid + top-5 + Viterbi arm (GER 0.195) beats the currently
deployed continuous pipeline (0.278) by about 30% relative, using only models and code
that already exist (C4, `bilstm`'s parity-verified export, `fuse.segments_d3`, one new
~50-line search function) plus one architectural change: **record-then-recognize now
means "run C4 to get segments, then reclassify each with `bilstm` and decode the whole
sentence," not "segment by stillness alone"** (Part 1's original plan). TODO §16.2's UI
checklist is unblocked, scoped to this design.

## Artifacts

`data/cache/gislr/bilstm_wholeclip_eval/results/` — `select_*_noprior.json`,
`sweep_*_prior.json`, `sweep_viterbi_lam.json`, `b0_top5.json`, `final.json`,
`final_hybrid.json`, `final_viterbi.json`, `final_table.csv`.
