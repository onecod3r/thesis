"""Stage 1: video -> landmark tensors.

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
