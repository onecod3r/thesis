# Model registry — best runs per dataset × architecture

One row per (dataset, architecture): the current best-scoring run and where it
came from. Every run lives at `models/<dataset>/<architecture>/<timestamp>/`
with its own `README.md` (training conditions + metrics), `data.md` (exact
data/subset/split), `assets/` (plots) and `cache/` (eval artifacts). Weights
are gitignored; the docs are the record.

## Leaderboard

| dataset | architecture | best run | input | params | val acc (overall) | macro | notes |
|---|---|---|---|---|---|---|---|
| gislr | gru | [20260715-190729](gislr/gru/20260715-190729/README.md) | **ME-126** subset × xyz (378) | 0.95M | **73.73%** | 73.49% | landmark-subset ablation winner; +3.14 over full-543 |
| gislr | gru *(prev best)* | [20260713-213000](gislr/gru/20260713-213000/README.md) | all 543 × xyz (1,629) | 1.91M | 70.59% | 70.36% | full-input baseline |

All runs above share the canonical evaluation: stratified 90/10 split
(`random_state=42`), 9,448-video val set, per-class accuracy from raw parquet.
A new run displaces the leader only on the same split and metric.

## Context

- **ME-126** = Kaggle-1st-place 118 landmarks (lips, hands, nose, eyes) ∪
  upper-body pose {11–16, 23, 24}. Derivation, motion-energy evidence and the
  1st-place cross-check: `docs/2026-07-15.md`.
- Planned entries: exact 1st-place-118 GRU, ME-126 xy-only, lag-feature GRU
  (TODO §3.1); 1D-CNN + Transformer port and other architectures (TODO §4).
