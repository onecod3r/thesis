# docs/

Two kinds of document live here: **logs** (time-ordered — what happened when)
and **reports** (standalone — what a test or analysis found).

```
docs/
├── logs/
│   ├── daily/<YYYY-MM-DD>.md     # figures in logs/daily/assets/<YYYY-MM-DD>/
│   └── weekly/<YYYY>-<WW>.md     # week starts SUNDAY; figures in logs/weekly/assets/<YYYY>-<WW>/
└── reports/<topic-slug>.md       # figures in reports/assets/<topic-slug>/
```

| folder | content | naming |
|---|---|---|
| `logs/daily/` | day-by-day work logs and dated narrative | `<YYYY-MM-DD>.md`, figures in `logs/daily/assets/<YYYY-MM-DD>/` |
| `logs/weekly/` | weekly summaries | `<YYYY>-<WW>.md`, e.g. `2026-30.md`; figures in `logs/weekly/assets/<YYYY>-<WW>/` |

**Week numbering.** Weeks run **Sunday → Saturday** (not ISO, which starts Monday), and week 1
is the week containing January 1. So week 30 of 2026 is **2026-07-19 → 2026-07-25**; ISO would
number that same Sunday as the last day of its week 29. Always state the date range in the
weekly file's title so the numbering never has to be re-derived. The running week's file is
written as work happens and marked **in progress** until its Saturday.
| `reports/` | standalone topic reports from a test or analysis — motion-energy, subset-comparison, confidence-tuning, plateau-breakout, … | `<topic-slug>.md`, figures in `reports/assets/<topic-slug>/` |

**Which one to write.** If the document is "here is what I did on this date",
it's a log. If it is "here is what this experiment measured and concluded",
it's a report — and the daily log for that date should link to it rather than
restate it. Long-lived findings belong in `reports/` so they stay findable
without knowing the date they were produced.

Every substantial analysis or experiment gets a write-up here; figures always
live under the matching `assets/` subfolder, never inline-only in notebooks.

## Daily logs

Chronological. `logs/daily/<YYYY-MM-DD>.md`.

| date | log | contents |
|---|---|---|
| 2026-07-15 | [logs/daily/2026-07-15.md](logs/daily/2026-07-15.md) | GISLR landmark motion-energy analysis (z-noise finding, keep/discard recommendation) · 1st-place solution landmark cross-check · GRU training in detail: full-543 baseline vs ME-126 subset (+3.1 pts at half the parameters) |
| 2026-07-16 | [logs/daily/2026-07-16.md](logs/daily/2026-07-16.md) | Landmark-subset discriminability comparison (F-ratio / MI / probe classifier, 3 scopes): **ME-126 wins**; discriminability ≠ motion energy (rho −0.12) · POPSIGN extraction module + driver notebook (resource-capped, resumable) built & validated |
| 2026-07-17 | [logs/daily/2026-07-17.md](logs/daily/2026-07-17.md) | Registry tooling v1: per-run metadata + queryable index · fresh-run-folder policy · training regime **v2-plateau-300** · LSTM / BiLSTM / CausalConv1D benchmark notebooks built · eval script generalized (arch dispatch + xy mode) |
| 2026-07-18 | [logs/daily/2026-07-18.md](logs/daily/2026-07-18.md) | **Repo restructure**: unified `modules/model` training stack · flat epoch-seconds model registry + meta.json schema v2 (registry reset) · `data/{raw,cache,temp,models}` tree + temp-cleanup policy · docs daily/weekly/reports split · single-progress-bar training |
| 2026-07-19 | [logs/daily/2026-07-19.md](logs/daily/2026-07-19.md) | **Plateau diagnosed as overfitting** (train 90–99% vs val ~75%, gap 0.16–0.24) · most-confused pairs are semantic near-synonyms · GISLR **stage-2 evaluation/submission notebook** + arch-generic TFLite export + meta.json **schema v3** (`submission.tested`, submission queue as a query) · `docs/` split into logs/ vs reports/ · POPSIGN **confidence-tuning** first results ([report](reports/confidence-tuning.md)) — thresholds barely matter, the quality proxies are measuring clip padding · POPSIGN extraction driver given a **CLI handoff** (`extract_popsign.py`) · **bulk test-split extraction running** (15,549/33,600, 1 failed, 1.46 videos/s) + the on-disk npz format recorded |

## Weekly logs

Chronological. `logs/weekly/<YYYY>-<WW>.md`.

| week | dates | log | contents |
|---|---|---|---|
| 2026-29 | Jul 12 – Jul 18 | [logs/weekly/2026-29.md](logs/weekly/2026-29.md) | motion energy → discriminability (ME-126 wins on both, rho −0.12 between them) · xy beats xyz · registry v1 → the restructure (registry reset, pre-reset weights gone) · 4 architecture notebooks |
| 2026-30 | Jul 19 – Jul 25 | [logs/weekly/2026-30.md](logs/weekly/2026-30.md) | *in progress*: plateau diagnosed as **overfitting** · TFLite export working for all 4 archs (Keras rebuild) · training consolidated to one notebook + config · POPSIGN **test-split extraction finished** (33,599/33,600) and all 4 train dataset parts now downloaded (train manifest regeneration + bulk run still pending) |

## Reports

Standalone; not date-ordered — findable by topic.

| report | contents |
|---|---|
| [reports/motion-energy.md](reports/motion-energy.md) | GISLR per-landmark motion analysis (three scopes, 94,477 videos): **~92% of pose "motion" is z-axis noise** · seeded 50-video samples reproduce the global ranking (rho 0.95+) · ME-126 keep/discard recommendation, cross-checked against the Kaggle 1st-place subset |
| [reports/subset-comparison.md](reports/subset-comparison.md) | Landmark-subset discriminability (F-ratio / MI / probe classifier, 3 scopes): **ME-126 wins** the 6-subset leaderboard (49.9% global probe) · discriminability ≈ uncorrelated with motion energy (rho −0.12) · probe difficulty profile tracks the trained GRU's (rho 0.640) |
| [reports/confidence-tuning.md](reports/confidence-tuning.md) | POPSIGN extraction-quality threshold sweep (**partial — 2 of 7 arms**): `min_hand_landmarks_confidence` is inert and the *pose* thresholds gate the hands · thresholds move hand detection by only ~0.02 · **the quality proxies are dominated by clip padding** — ~half of every clip is non-signing lead-in/lead-out, and detection is 0.85–0.94 within the signing span |
