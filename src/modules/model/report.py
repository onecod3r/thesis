"""Run reporting: learning-curve figures into a run's assets/ + comparison rows."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from modules.model import registry as R


def load_history(run_dir: Path) -> dict:
    """Per-epoch history for a run: {train_loss, train_acc, val_loss, val_acc, lr}.

    Prefers assets/history.json (written every epoch since schema v3) and falls
    back to last.pt for runs trained before that asset existed — the evaluation
    notebook plots curves for every run either way, and only needs the (large,
    gitignored) checkpoint for the older ones.
    """
    hist_path = run_dir / "assets" / "history.json"
    if hist_path.is_file():
        return json.loads(hist_path.read_text())
    ck_path = run_dir / R.CKPT_LAST
    if not ck_path.is_file():
        raise FileNotFoundError(
            f"{run_dir.name}: no assets/history.json and no {R.CKPT_LAST} — "
            "learning curves unavailable for this run"
        )
    history = torch.load(ck_path, map_location="cpu", weights_only=False)["history"]
    hist_path.parent.mkdir(exist_ok=True)
    hist_path.write_text(json.dumps(history))  # backfill so the next read is cheap
    return history


def save_learning_curves(run_dir: Path, title: str | None = None):
    """Plot loss/accuracy/LR curves from the run's last checkpoint, save to
    assets/learning_curves.png, register the asset in meta.json, and return the
    figure (for inline display in a notebook). Reads only from disk."""
    ck = torch.load(run_dir / R.CKPT_LAST, map_location="cpu", weights_only=False)
    h = ck["history"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(h["train_loss"], label="train")
    axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title(f"{title or run_dir.name} — loss")
    axes[0].legend()
    axes[1].plot(h["train_acc"], label="train")
    axes[1].plot(h["val_acc"], label="val")
    axes[1].set_title(f"{title or run_dir.name} — accuracy")
    axes[1].legend()
    axes[2].plot(h["lr"])
    axes[2].set_title("LR schedule")
    fig.tight_layout()
    fig.savefig(run_dir / "assets" / "learning_curves.png", dpi=110)
    R.register_assets(run_dir, learning_curves="assets/learning_curves.png")
    return fig


def confusion_matrix(run_dir: Path, n_classes: int = 250, normalize: bool = True):
    """Confusion matrix from a run's cached canonical-eval predictions.

    Built from ``assets/val_predictions.npz`` (written by eval_gru.py), never by
    re-running inference — one inference pass per run, ever.

    Row *i* = true class *i*: normalized rows sum to 1, so the diagonal is the
    class's recall and off-diagonal mass is where it goes wrong. Classes with no
    val samples stay all-zero rather than NaN, so matrices from different runs
    can be averaged without special-casing.
    """
    import numpy as np

    npz_path = run_dir / "assets" / "val_predictions.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"{run_dir.name}: no val_predictions.npz — run the canonical eval "
            f"(modules/scripts/eval_gru.py) for this run first"
        )
    d = np.load(npz_path)
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(cm, (d["labels"].astype(int), d["preds"].astype(int)), 1)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    return cm


def plot_confusion(cm, title: str, ax):
    """Draw a (250x250) confusion matrix on ``ax``; returns the image handle.

    Log color scale on purpose: off-diagonal cells are 1-2 orders of magnitude
    smaller than the diagonal and are invisible on a linear scale — and the
    off-diagonal is the entire point of looking at one of these.
    """
    from matplotlib.colors import LogNorm

    im = ax.imshow(cm, cmap="magma", norm=LogNorm(vmin=1e-3, vmax=1.0), aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    return im


def comparison_row(run_dir: Path) -> dict:
    """One leaderboard-style row per run for the notebooks' comparison table."""
    meta = R.load_meta(run_dir)
    m, t = meta["metrics"], meta["training"]
    return dict(
        run_id=meta["run_id"],
        subset=meta["subset"],
        coords=meta["coords"],
        n_landmarks=meta["n_landmarks"],
        n_params=meta["n_params"],
        epochs=t["epochs_trained"],
        # an interrupted run reads as a bad result unless its state is visible:
        # finished=False means "stopped mid-training", not "converged here"
        finished=t["finished"],
        early_stopped=t["early_stopped"],
        train_val_acc=m["train_val_acc"],
        eval_status=m["eval_status"],
        overall_accuracy=m["overall_accuracy"],
    )
