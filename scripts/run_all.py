"""
Usage:
    python run_all.py --epochs 30
    python run_all.py --epochs 30 --models lstm bilstm gru   # subset

Trains every model in models.MODEL_REGISTRY (or a chosen subset) one after
another, runs the full benchmark suite on each, and writes a final
cross-model comparison CSV/printout at the end.
"""
import argparse
import csv
import os

from models import MODEL_REGISTRY
from train import build_arg_parser, resolve_paths, train_model
from benchmark import run_full_benchmark


def main():
    parser = argparse.ArgumentParser(parents=[build_arg_parser()], add_help=False, conflict_handler="resolve")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                         choices=list(MODEL_REGISTRY.keys()),
                         help="Subset of models to run; defaults to all six.")
    args = parser.parse_args()
    args = resolve_paths(args)

    os.makedirs(args.results_dir, exist_ok=True)

    summary = []
    for model_name in args.models:
        print(f"\n{'=' * 60}\nRunning: {model_name}\n{'=' * 60}")
        args.model = model_name

        result = train_model(args)
        metrics = run_full_benchmark(model_name, result, args)

        summary.append({
            "model": model_name,
            "best_val_acc": metrics["best_val_acc"],
            "overall_accuracy": metrics["overall_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "micro_precision": metrics["micro_precision"],
            "weighted_precision": metrics["weighted_precision"],
        })

    summary_path = os.path.join(args.results_dir, "model_comparison.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n{'=' * 60}\nFinal comparison ({summary_path}):\n{'=' * 60}")
    header = f"{'model':<14}{'val_acc':>10}{'macro_prec':>12}{'micro_prec':>12}{'weighted_prec':>15}"
    print(header)
    for row in summary:
        print(f"{row['model']:<14}{row['best_val_acc']:>10.4f}"
              f"{row['macro_precision']:>12.4f}{row['micro_precision']:>12.4f}"
              f"{row['weighted_precision']:>15.4f}")


if __name__ == "__main__":
    main()
