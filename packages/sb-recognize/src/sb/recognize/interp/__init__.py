"""Landmark-importance interpretability track (TODO §3).

A parallel, self-contained stage-1 track — same relationship to the shared
training stack as ``sb.recognize.features.firstplace_v1``: its own feature
pipeline (:mod:`sb.recognize.interp.features`) and its own model classes
(:mod:`sb.recognize.interp.models`), kept out of ``architectures.py``/``ARCHS``
and the config-driven ``train_from_config`` because neither the feature
pipeline (per-landmark engineered channels, full FULL_543) nor the training
scheme (rotating k-fold, not the fixed canonical split alone) fits that
contract. It still reads the same canonical GISLR split/label map so its final
test-set number means the same thing as everything else in the registry, even
though no run from here is written to ``registry/``.
"""
