"""Builds gislr.1.models.gapdetect.ipynb (TODO §17): trains GapGRU, the
sign-vs-gap detector, on GISLR-GapCorpus.

Run: .venv/Scripts/python.exe experiments/recognition/build_gapdetect_training_nb.py

Regenerate after touching sb.recognize.architectures.{KinematicFrontend,GapGRU},
sb.recognize.gapdetect.{data,train}, or configs/gislr.gapdetect.json.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "gislr.1.models.gapdetect.ipynb"


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# GapGRU: a sign-vs-gap detector (TODO §17)

Trains `GapGRU` (`sb.recognize.architectures`) on GISLR-GapCorpus
(`sb.recognize.sequences.gapcorpus_kaggle_notebook`): a causal GRU behind a
non-learned kinematic front-end (velocity/acceleration/jerk/jitter/distance-travelled
per hand+arm landmark), predicting a per-frame sign-vs-gap probability.

**Purpose**: a segmentation source decoupled from any classifier, for both the
deployed continuous GRU/LSTM (§12.3) and the record-then-recognize BiLSTM hybrid
(§16.2) to use in place of their current segmenters.

**Not a registry run**: see the note above `GapGRU`'s definition and
`sb.recognize.gapdetect.train`'s module docstring for why. Runs land in
`experiments/recognition/gapdetect_runs/<run_id>/`, not `registry/runs/`.

**Design decisions vs. the TODO spec**:
- Corpus built on Kaggle (`gislr.0.dataset.gapcorpus-kaggle.ipynb`), downloaded here
  via `sb.core.paths.gislr_gapcorpus_dir()` (falls back to a local build if the
  Kaggle dataset isn't published yet).
- Hyperparameters live only in `configs/gislr.gapdetect.json` (never in a cell),
  following the repo-wide convention.
- **I (Claude) ran §§1-2 and the smoke check below myself** -- deterministic setup
  and a wiring check (a few batches, no real optimization), never the training cell
  itself, per this repo's standing rule (`CLAUDE.md`: never run model training).
"""),
    md("## 1. Setup"),
    code("""
# ============================================================
# Setup: config, corpus resolution
# ============================================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sb.core.paths import gislr_gapcorpus_dir
from sb.recognize.gapdetect.train import CONFIG_PATH, load_config, run_hyp, train_gap

RUN_NAME = "A"           # which configs/gislr.gapdetect.json["runs"] entry to train
CORPUS_DIR = None        # None = gislr_gapcorpus_dir(); set to a local path before the
                         # Kaggle dataset is published (e.g. a copy of a smoke-scale build)

cfg = load_config(CONFIG_PATH)
hyp = run_hyp(cfg, RUN_NAME)
root = Path(CORPUS_DIR) if CORPUS_DIR else gislr_gapcorpus_dir(cfg["corpus_version"])
build_info = json.loads((root / "build_info.json").read_text(encoding="utf-8"))
print(f"corpus: {root}")
print(f"run {RUN_NAME}: {hyp}")
print(f"corpus build: {build_info['dataset']} {build_info['dataset_version']}, "
      f"{build_info['size_bytes'] / 1e9:.2f} GB, mean gap fraction "
      f"{build_info.get('mean_gap_fraction_sampled', float('nan')):.3f}")
"""),
    md("""
## 2. Data sanity

Sequence-length and gap-fraction distribution for both splits, and one exemplar
sequence's gap curve against the true segments -- a visual check that `train.csv`
and the npz files agree with each other before spending a training run on them.
"""),
    code("""
# ============================================================
# Data sanity: length/gap-fraction distributions, one exemplar
# ============================================================
from sb.recognize.gapdetect.data import GapCorpus
from sb.recognize.sequences.compose import read_sequence

train_df = pd.read_csv(root / "train.csv", keep_default_na=False)
test_df = pd.read_csv(root / "test.csv", keep_default_na=False)
print(f"train: {len(train_df)} sequences, {train_df.n_frames.sum():,} frames")
print(f"test:  {len(test_df)} sequences, {test_df.n_frames.sum():,} frames")

sample = train_df.sample(min(300, len(train_df)), random_state=0)
gap_fracs = [read_sequence(root / p)["gap"].mean() for p in sample["npz_relpath"]]
print(f"gap fraction (sample of {len(sample)} train sequences): "
      f"mean {np.mean(gap_fracs):.3f}, std {np.std(gap_fracs):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].hist([train_df.n_frames, test_df.n_frames], bins=40, label=["train", "test"])
axes[0].set_title("sequence length (frames)")
axes[0].legend()
axes[1].hist(gap_fracs, bins=30, color="#dd8452")
axes[1].set_title("gap fraction per sequence (train sample)")
fig.tight_layout()
plt.show()

ex = train_df.iloc[0]
arr = read_sequence(root / ex["npz_relpath"])
fig, ax = plt.subplots(figsize=(13, 3))
ax.plot(arr["gap"], drawstyle="steps-post", label="gap label")
for s, e in arr["segments"]:
    ax.axvspan(s, e, color="#4c72b0", alpha=0.15)
ax.set_title(f"{ex['seq_id']}: gap label vs. true sign segments (shaded)")
ax.legend()
fig.tight_layout()
plt.show()

train_ds, test_ds = GapCorpus(root, "train"), GapCorpus(root, "test")
assert len(train_ds) == len(train_df) and len(test_ds) == len(test_df)
print("GapCorpus dataset lengths match train.csv/test.csv")
"""),
    md("""
## 3. Wiring check (smoke) -- not training

A handful of real batches through a freshly-initialized model: forward, loss,
backward, optimizer step, then a small validation pass. Confirms the whole path
(data -> collate -> `KinematicFrontend` -> GRU -> loss -> metrics) runs without
error and produces finite, sane numbers. This is what I (Claude) ran -- the loss
after 5 batches is expected to barely have moved; it is not a training result.
"""),
    code("""
# ============================================================
# Smoke check: a few batches, no real optimization
# ============================================================
smoke = train_gap(RUN_NAME, CONFIG_PATH, corpus_dir=CORPUS_DIR, smoke=5)
print(json.dumps({k: v for k, v in smoke.items() if k != "train_losses"}, indent=2, default=str))
print("train_losses (5 batches):", [round(x, 4) for x in smoke["train_losses"]])
assert all(np.isfinite(x) for x in smoke["train_losses"]), "non-finite loss -- do not train on this"
"""),
    md("""
## 4. Train (user runs this cell)

Full training run: `configs/gislr.gapdetect.json["runs"]["A"]`'s hyperparameters,
early-stopped on validation gap-F1. Auto-resumes from `last.pt` if interrupted.
**This is the cell CLAUDE.md says I never execute.**
"""),
    code("""
# ============================================================
# Train (run this yourself -- see CLAUDE.md: Claude never trains models)
# ============================================================
run_dir = train_gap(RUN_NAME, CONFIG_PATH, corpus_dir=CORPUS_DIR)
print(f"run_dir: {run_dir}")
"""),
    md("""
## 5. Results

Learning curves from the finished run's `history.json`. Reads back whatever
`run_dir` §4 produced (or set `RUN_DIR` explicitly to inspect an older run).
"""),
    code("""
# ============================================================
# Learning curves
# ============================================================
RUN_DIR = None  # None = the run_dir §4 just produced
rd = RUN_DIR or run_dir
history = json.loads((rd / "history.json").read_text(encoding="utf-8"))
meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
print(f"run {rd.name}: {meta['n_params']:,} params, {len(history['val_f1'])} epochs")
print(f"best val gap-F1: {max(history['val_f1']):.4f} at epoch {int(np.argmax(history['val_f1'])) + 1}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("loss")
axes[0].legend()
for k in ("val_acc", "val_precision", "val_recall", "val_f1"):
    axes[1].plot(history[k], label=k.replace("val_", ""))
axes[1].set_title("validation metrics (gap class)")
axes[1].legend()
fig.tight_layout()
plt.show()
"""),
]

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 4,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")
