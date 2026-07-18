"""Run reporting: learning-curve figures into a run's assets/ + comparison rows."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from modules.model import registry as R


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
        early_stopped=t["early_stopped"],
        train_val_acc=m["train_val_acc"],
        eval_status=m["eval_status"],
        overall_accuracy=m["overall_accuracy"],
    )
