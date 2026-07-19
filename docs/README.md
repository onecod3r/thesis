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
