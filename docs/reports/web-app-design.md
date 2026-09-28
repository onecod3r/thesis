# Web app UI/UX design

**Status:** design-only deliverable (no `apps/web` code touched). Built 2026-09-27; extended with
mobile coverage the same day, then with the remaining desktop pages 2026-09-28 — **all four pages
now have both desktop and mobile artboards**. Canonical reference: this file, plus a live, editable
canvas artifact (mockups are pixel-real HTML/CSS, not sketches) — **private**, not yet shared:
<https://claude.ai/artifact/DuuUMVMPf7MUCHyxMiy6Zt>. The artifact is the source of truth for exact
markup/spacing; this file is the durable, repo-tracked record of what it contains and why, since
Artifacts live outside git and outside this session. Note: the user has since rearranged some
mobile artboards' positions on the canvas itself (cosmetic, in-artifact only — not reflected below).

## Why this exists

The user asked to "design a good web app for the project," then to add mobile coverage and save
it durably, then to "generate the remaining pages" (the two desktop screens that were still
missing). `apps/web` already has real, working pages (`apps/web/README.md`) — Sign → Speech,
Speech → Sign, Landmark test — so the brief was a visual-design pass over real, documented
functionality, not invention from scratch (per the design tool's "root hi-fi mockups in context"
rule).

## Page inventory (8 artboards — complete)

| Artboard | Viewport | Covers |
|---|---|---|
| Home | 1440×960 (desktop) | Landing/marketing page — not a page in the current app, proposed as a front door |
| Home (mobile) | 390×844 | same |
| Sign → Speech | 1440×900 (desktop) | `apps/web`'s `/` — the flagship live demo |
| Sign → Speech (mobile) | 390×844 | same |
| Speech → Sign | 1440×900 (desktop) | `apps/web`'s `/speech` — text/speech → gloss via T5, session history, guarded-preset settings |
| Speech → Sign (mobile) | 390×844 | same |
| Landmark test | 1440×900 (desktop) | `apps/web`'s `/landmarks` — 3-mode MediaPipe diagnostic, per-mode stats + session table |
| Landmark test (mobile) | 390×844 | same |

Every page in `apps/web` now has a desktop and a mobile mockup.

## Design system (original, not Anthropic's)

No design system was attached to the account and no brand references were given, so one was
committed to per the design tool's "no brand governs" guidance — deliberately **not** Anthropic's
Clay/Ivory palette, since this is signbridge's own product identity, not a Claude surface.

**Palette** (warm, not Anthropic's exact hexes):
| Token | Value | Use |
|---|---|---|
| `--paper` | `#FBF8F2` | page background |
| `--ink` | `#201E1A` | primary text |
| `--ink-dim` / `--ink-faint` | `#6B675E` / `#9A968B` | secondary/tertiary text |
| `--card` | `#FFFFFF` | card surfaces |
| `--line` | `rgba(32,30,26,0.12)` | hairline borders |
| `--accent` (teal) | `#2F6F62` | confident/active state, primary actions |
| `--accent-soft` | `#E4EEEC` | accent tint fills |
| `--warn` (amber) | `#C97A2B` | uncertain/attention state |
| `--warn-soft` | `#F7E8D6` | warn tint fills |

Teal/amber (blue-leaning vs. orange-leaning) was chosen over a green/red confidence pairing
deliberately — the design tool's accessibility rule prefers hues that differ in lightness as well
as hue, which red/green often fails for colorblind users.

**Type**: Source Serif 4 (display headings, editorial voice) over IBM Plex Sans (UI text) over IBM
Plex Mono (stats, gloss chips, fps/latency readouts) — all Google Fonts, loaded per-artboard.

## What's mocked, grounded in real numbers/features

Every stat and control on the mockups is either a real number from `README.md`/`apps/web/README.md`
or a real documented feature, not invented content:

- **Home**: 0.7632 canonical accuracy (`gru_phono_raw`/ME_134), 0.293 sentence GER, 24/24 browser↔Python
  parity streams; a 4-step pipeline explainer (camera → landmarks/phonology → streaming GRU → lattice → English).
- **Sign → Speech**: the lag-2 lattice uncertainty UI (glosses under 0.5 confidence shown greyed with
  "?", "Speak anyway" fallback), mirror/landmark-overlay toggles, 30 fps clock indicator, recognizer
  variant picker (C1 / C2 / C1+C2, with the real "C1+C2 GER 0.244 vs C1's 0.276" note), individual
  sign mode toggle, input source (camera / video file / held-out replay), live session stats (fps,
  camera→overlay latency, hand-label agreement). The variant picker, individual-sign toggle and input-source
  picker are **actually interactive** in the artifact (real click state), not just static color swatches.
- **Speech → Sign**: mic + text input → "Refine to gloss" → a gloss-chip output, T5 int8-encoder /
  beam=2 / max_length=56 "guarded preset" badge (from TODO §13's shipped tuning) — with an honest
  amber notice that sign-video playback isn't built yet, since `apps/web/README.md` is explicit that
  speech → *sign* (the animation) isn't in the app; only speech → gloss is.
- **Landmark test**: the real 3-way mode tab bar (Holistic / Hands+Pose+Face / Hands+Face), and the
  real per-mode stat set from `apps/web/README.md` (fps, detection ms, render ms, camera→overlay
  latency, pose visibility, hand-label agreement) — plus the real caveat that only one mode runs at
  a time because running all three biased the comparison.

## Open items for the user

- Review the artifact and decide what (if anything) gets implemented in `apps/web` — this pass
  never touched app code or ran the dev server.
- The Home page is a new landing/marketing surface with no current route in `apps/web` — decide
  whether it should become a real page or stay reference-only.
- The palette/type direction was picked unilaterally (auto-mode, no references given) — flag if you
  want a different aesthetic; it's a cheap change now, before any implementation.
- Speech → Sign's session history and Landmark test's session table are illustrative sample data
  (a mockup, not a live app) — real content once wired to `apps/web`.
