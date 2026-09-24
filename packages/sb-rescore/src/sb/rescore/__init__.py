"""Downstream of the recognizer: next-gloss prediction and gloss -> English
(TODO §12.6). Stage 2 and stage 3 of the sign -> speech pipeline.

TODO §8's scope question was answered on 2026-09-24: continuous,
sentence-level signing, with an LLM and TTS downstream. The stages are kept
separate (user, 2026-09-24): MediaPipe -> recognizer + next-gloss prediction
-> gloss -> English -> TTS. A single model from video to text is future work.

- :mod:`sb.rescore.prior`: next-gloss predictors (Kneser-Ney n-grams, no
  torch) plus held-out protocols and intrinsic metrics. The recognizer side
  of the fusion lives in ``sb.recognize.continuous.fuse``.
- :mod:`sb.rescore.neural`: a small GRU language model, the neural arm
  (``neural`` extra: torch).
- :mod:`sb.rescore.gloss2en`: rule-based gloss -> English, the reverse of
  ``sb.synthesize.gloss.rules_v2``, plus the round-trip mapping.
- :mod:`sb.rescore.client`: Workers AI (REST) with an on-disk cache, for the
  LLM gloss -> English arm.

Prompts live as versioned files under ``prompts/<version>/`` and are hashed
into every result. ``evalset/`` holds the frozen evaluation set
(append-only).
"""
