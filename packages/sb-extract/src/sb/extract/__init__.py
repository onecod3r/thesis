"""Stage 1: video -> landmark tensors.

**PAUSED (2026-09-22): POPSIGN, this package's only active consumer, is
deprecated** (see ``TODO.md`` §2) — GISLR ships pre-extracted npz and needs
no extraction stage. Not deleted: the extraction engine itself is
dataset-agnostic and could serve a future raw-video dataset or a live-camera
deployment path (``TODO.md`` §10.2), but nothing currently exercises it and
it is not being maintained until a new consumer exists.

MediaPipe Holistic over a worker pool, resumable through a manifest, capped so a
bulk run leaves the machine usable. Every artifact it writes conforms to
``sb.core.schema`` — the contract stage 2 reads.

- ``holistic``  — the extractor: worker pool, manifest, resource caps
- ``sources``   — per-dataset adapters (popsign, gislr), not branches in the core
- ``quality``   — extraction-quality proxies + composite score
- ``overlay``   — landmark-on-video rendering, the visual quality test
- ``cli``       — `sb-extract`: pilot benchmark + resumable bulk run
- ``tune``      — detector-threshold sweep
"""
