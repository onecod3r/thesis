# docs/

| folder | content | naming |
|---|---|---|
| `daily/` | day-by-day work logs and dated analysis write-ups | `<YYYY-MM-DD>.md`, figures in `daily/assets/<YYYY-MM-DD>/` |
| `weekly/` | weekly summaries | `<YYYY>-<WW>.md` — ISO week number, e.g. `2026-29.md` for the 29th week of 2026; figures in `weekly/assets/<YYYY>-<WW>/` |
| `reports/` | standalone topic reports (e.g. a motion-energy analysis, a subset comparison) | `<topic-slug>.md`, figures in `reports/assets/<topic-slug>/` |

Every substantial analysis or experiment gets a write-up here; figures always
live under the matching `assets/` subfolder, never inline-only in notebooks.
