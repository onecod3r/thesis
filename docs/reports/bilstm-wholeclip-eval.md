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

## Recommendation

Don't build the record-then-recognize UI (TODO §16.2's remaining checklist) on pure
movement segmentation — the measured accuracy doesn't clear a usable bar. Two ways
forward, neither built:

1. **Hybrid**: reuse C4's own learned boundary/null decoding (already deployed, already
   segments continuous streams at 0.278 GER) to produce segment proposals, then
   classify each accepted segment with `bilstm`'s whole-clip readout instead of (or
   averaged with) C4's own per-frame vote — costs an extra forward pass per accepted
   segment, on a model already exported and parity-verified. Untested; the natural next
   experiment if this mode stays wanted.
2. **A dedicated boundary signal for whole-clip mode**, mirroring §12.3's per-frame
   supervision — meaningfully more work (a new training run), not justified without
   first trying (1).

## Artifacts

`data/cache/gislr/bilstm_wholeclip_eval/results/` — `select_*_noprior.json`,
`sweep_*_prior.json`, `b0_top5.json`, `final.json`, `final_table.csv`.
