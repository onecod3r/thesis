"""Continuous-signing models (TODO §12.3): trained on multi-sign streams
composed on the fly from ``train.csv``, with a per-frame gloss + null output
and a sign-boundary output (``sb.recognize.architectures.ContinuousRNN``).

- :mod:`.data` -- NaN-preserving clip bank, held-out splits, and the
  feature-space stream composer (transitions, lowered-hands / hands-absent
  rest), with per-frame targets.
- :mod:`.train` -- the training driver (auto-resume, one progress bar,
  registry run per training), one call per run of
  ``configs/gislr.continuous.json``.
- :mod:`.decode` -- stream decoders for these models (boundary commit, the
  user's threshold loop, null-gated, CTC greedy).

Kept apart from ``sb.recognize.train`` because the unit of training is a
composed stream with per-frame targets, not an isolated clip with one label;
the run record, checkpoint keys and canonical isolated evaluation are the
same.
"""
