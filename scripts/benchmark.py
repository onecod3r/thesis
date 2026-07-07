"""
Benchmarking suite run after training each model:
  - Confusion matrix (full, saved as an image)
  - Precision: overall (macro/micro/weighted) and per-class
  - Learning curve: loss/accuracy over epochs with LR overlaid
  - Input contribution: gradient-saliency, aggregated by landmark group
    (face / left_hand / pose / right_hand) and by top individual landmarks,
    answering "which part of the input drives the output most."
"""
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, classification_report

from gislr_dataset import NUM_LANDMARKS


LANDMARK_GROUPS = {
    "face": (0, 468),
    "left_hand": (468, 489),
    "pose": (489, 522),
    "right_hand": (522, 543),
}


# ---------------------------------------------------------------------------
# Evaluation / classification metrics
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_classification_metrics(preds, labels, label_map, results_dir, model_name):
    idx_to_sign = {v: k for k, v in label_map.items()}
    present_classes = sorted(set(labels.tolist()) | set(preds.tolist()))
    class_names = [idx_to_sign[c] for c in present_classes]

    overall_acc = float((preds == labels).mean())
    macro_precision = float(precision_score(labels, preds, average="macro", zero_division=0))
    micro_precision = float(precision_score(labels, preds, average="micro", zero_division=0))
    weighted_precision = float(precision_score(labels, preds, average="weighted", zero_division=0))
    per_class_precision = precision_score(labels, preds, average=None, zero_division=0,
                                           labels=present_classes)
    per_class = {class_names[i]: float(p) for i, p in enumerate(per_class_precision)}

    report = classification_report(labels, preds, labels=present_classes,
                                    target_names=class_names, zero_division=0)

    metrics = {
        "overall_accuracy": overall_acc,
        "macro_precision": macro_precision,
        "micro_precision": micro_precision,
        "weighted_precision": weighted_precision,
        "per_class_precision": per_class,
    }

    with open(os.path.join(results_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(results_dir, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report)

    print(f"[{model_name}] overall_acc={overall_acc:.4f} "
          f"macro_precision={macro_precision:.4f} micro_precision={micro_precision:.4f}")

    return metrics, present_classes, class_names


def plot_confusion_matrix(preds, labels, present_classes, class_names, results_dir, model_name):
    cm = confusion_matrix(labels, preds, labels=present_classes)
    n = len(present_classes)

    # ~250 classes makes per-cell annotation unreadable; render as a plain
    # heatmap without per-cell text, sized to stay legible.
    fig_size = max(10, n * 0.12)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_title(f"{model_name} — confusion matrix ({n} classes)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if n <= 40:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    save_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    np.save(os.path.join(results_dir, f"{model_name}_confusion_matrix.npy"), cm)
    return save_path


# ---------------------------------------------------------------------------
# Learning curve (loss / accuracy / LR)
# ---------------------------------------------------------------------------
def plot_learning_curve(history, results_dir, model_name):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].plot(epochs, history["train_acc"], label="train acc", linestyle="--")
    axes[0].plot(epochs, history["val_acc"], label="val acc", linestyle="--")
    axes[0].set_xlabel("epoch")
    axes[0].set_title(f"{model_name} — loss / accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history["lr"], color="tab:red")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title(f"{model_name} — LR schedule")
    axes[1].set_yscale("log")

    save_path = os.path.join(results_dir, f"{model_name}_learning_curve.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Input contribution via gradient saliency
# ---------------------------------------------------------------------------
def landmark_saliency(model, loader, device, use_z, results_dir, model_name, max_batches=10):
    """Gradient-based saliency: |d(logit_true_class)/d(input)|, averaged
    over a sample of the validation set, then summed per landmark group
    (face / left_hand / pose / right_hand) and per individual landmark.
    This is the same family of technique (gradient saliency) you'd already
    scoped out alongside SHAP for the thesis XAI work — simpler and much
    cheaper to run across every model here than SHAP would be."""
    model.eval()
    coords = 3 if use_z else 2

    group_totals = {g: 0.0 for g in LANDMARK_GROUPS}
    per_landmark_totals = np.zeros(NUM_LANDMARKS, dtype=np.float64)
    n_batches = 0

    for x, y in loader:
        if n_batches >= max_batches:
            break
        x = x.to(device).requires_grad_(True)
        y = y.to(device)

        logits = model(x)
        target_logits = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        model.zero_grad(set_to_none=True)
        target_logits.sum().backward()

        grad = x.grad.detach().abs()  # (batch, max_len, feature_dim)
        batch, T, F = grad.shape
        grad = grad.view(batch, T, NUM_LANDMARKS, coords)
        # Sum over batch, time, coords -> (num_landmarks,) importance
        landmark_importance = grad.sum(dim=(0, 1, 3)).cpu().numpy()
        per_landmark_totals += landmark_importance
        n_batches += 1

    total = per_landmark_totals.sum()
    if total > 0:
        per_landmark_pct = per_landmark_totals / total * 100.0
    else:
        per_landmark_pct = per_landmark_totals

    for group, (start, end) in LANDMARK_GROUPS.items():
        group_totals[group] = float(per_landmark_pct[start:end].sum())

    with open(os.path.join(results_dir, f"{model_name}_saliency_groups.json"), "w") as f:
        json.dump(group_totals, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(group_totals.keys()), list(group_totals.values()),
           color=["#e07b39", "#3f8efc", "#2ca02c", "#9467bd"])
    ax.set_ylabel("% of total gradient magnitude")
    ax.set_title(f"{model_name} — input contribution by landmark group")
    save_path_group = os.path.join(results_dir, f"{model_name}_saliency_by_group.png")
    fig.tight_layout()
    fig.savefig(save_path_group, dpi=150)
    plt.close(fig)

    top_idx = np.argsort(per_landmark_pct)[::-1][:20]
    top_labels = []
    for i in top_idx:
        for group, (start, end) in LANDMARK_GROUPS.items():
            if start <= i < end:
                top_labels.append(f"{group}[{i - start}]")
                break

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(top_idx)), per_landmark_pct[top_idx])
    ax.set_xticks(range(len(top_idx)))
    ax.set_xticklabels(top_labels, rotation=90, fontsize=8)
    ax.set_ylabel("% of total gradient magnitude")
    ax.set_title(f"{model_name} — top 20 individual landmarks by contribution")
    save_path_top = os.path.join(results_dir, f"{model_name}_saliency_top20.png")
    fig.tight_layout()
    fig.savefig(save_path_top, dpi=150)
    plt.close(fig)

    print(f"[{model_name}] landmark group contribution (%): "
          + ", ".join(f"{g}={v:.1f}" for g, v in group_totals.items()))

    return group_totals, save_path_group, save_path_top


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def run_full_benchmark(model_name, train_result, args):
    os.makedirs(args.results_dir, exist_ok=True)

    model = train_result["model"]
    dataset = train_result["dataset"]
    val_loader = train_result["val_loader"]
    history = train_result["history"]
    device = train_result["device"]

    preds, labels = evaluate(model, val_loader, device)
    metrics, present_classes, class_names = compute_classification_metrics(
        preds, labels, dataset.label_map, args.results_dir, model_name
    )
    plot_confusion_matrix(preds, labels, present_classes, class_names, args.results_dir, model_name)
    plot_learning_curve(history, args.results_dir, model_name)
    landmark_saliency(model, val_loader, device, args.use_z, args.results_dir, model_name)

    metrics["best_val_acc"] = train_result["best_val_acc"]
    return metrics
