"""Continuous multi-sign sequences built from isolated GISLR clips (TODO §12).

GISLR is isolated-sign only: one tightly-trimmed clip per gloss, no lead-in,
no lead-out, no transitions (measured 2026-09-23: 0 hand-absent lead-in/out
frames over a 1,500-clip sample). This package supplies what the dataset
lacks, in two separable halves:

- :mod:`.corpus` -- the committed, versioned sentence corpus (ASL gloss order,
  250-gloss vocabulary only) and its part-of-speech lexicon. Pure text; no
  landmarks.
- :mod:`.compose` -- turns sentences into landmark sequences: assigns real
  clips (one signer per sequence, every source clip used), synthesizes the
  non-sign frames GISLR never recorded (interpolated transitions between
  signs, hands-absent rest at either end), and records per-frame labels plus
  every sign's start/end frame.

The same composer serves the uploaded test-derived dataset
(``gislr.0.dataset.sentences.ipynb``) and, later, on-the-fly training
sequences from ``train.csv`` -- one definition of "a synthetic sentence".
"""
