"""Rebuild src/data/models/index.csv from the per-run meta.json files, and query it.

Every registry run folder (src/data/models/<run_id>/, run_id = epoch seconds)
carries a meta.json — written every epoch by the training driver
(modules/model/train.py) and promoted to eval_status="canonical" by
modules/scripts/eval_gru.py. This script flattens them into one table so
"best 3 gru runs on gislr" or "all ME_126 runs" is a one-liner instead of a
folder crawl.

Runs with any CWD (bootstraps its own imports):

    .venv/Scripts/python.exe src/modules/scripts/build_model_index.py                 # rebuild + full leaderboard
    .venv/Scripts/python.exe src/modules/scripts/build_model_index.py --dataset gislr --architecture gru --top 3
    .venv/Scripts/python.exe src/modules/scripts/build_model_index.py --subset ME_126

Sorting uses val_acc = the canonical-eval overall accuracy when available
(eval_status "canonical"), falling back to the training-loop best val accuracy
(eval_status "pending"). Run folders missing meta.json are warned about but
never block the rebuild; an empty registry writes a header-only index.csv.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> src/

import pandas as pd

from modules.model import registry as R
from modules.paths import MODEL_INDEX, MODELS_DIR

# column order of index.csv: identity, then results, then config; the flattened
# hyp_* columns follow in whatever order the meta files introduce them
LEAD_COLUMNS = [
    "dataset", "architecture", "run_id", "created", "subset", "coords",
    "val_acc", "eval_status", "overall_accuracy", "macro_accuracy",
    "median_class_accuracy", "n_classes_below_50pct", "train_val_acc",
    "n_landmarks", "feature_dim", "n_params", "n_classes",
    "model_name", "streaming",
    "split_strategy", "split_random_state", "split_n_val",
    "training_regime", "training_source", "training_epoch_cap",
    "training_epochs_trained", "training_best_epoch", "training_early_stopped",
    "training_finished", "training_wall_time_min",
    "submission_tested", "submission_platform", "submission_public_score",
    "submission_private_score", "submission_submitted_at", "submission_reference",
]


def _flatten(meta: dict) -> dict:
    row = {k: v for k, v in meta.items()
           if k not in ("schema_version", "split", "training", "hyperparameters",
                        "metrics", "checkpoints", "assets", "submission")}
    row.update({f"split_{k}": v for k, v in meta.get("split", {}).items()})
    row.update({f"training_{k}": v for k, v in meta.get("training", {}).items()})
    row.update({f"hyp_{k}": v for k, v in meta.get("hyperparameters", {}).items()})
    row.update(meta.get("metrics", {}))
    # pre-v3 runs have no submission block — an absent block means "never
    # submitted", so default rather than leaving the column ragged
    sub = {**{"tested": False, "platform": None, "public_score": None,
              "private_score": None, "submitted_at": None, "reference": None},
           **meta.get("submission", {})}
    row.update({f"submission_{k}": v for k, v in sub.items() if k != "notes"})
    return row


def load_runs() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(p for p in MODELS_DIR.iterdir()
                          if p.is_dir() and p.name.isdigit()):
        meta_path = run_dir / "meta.json"
        if not meta_path.is_file():
            print(f"WARNING: no meta.json in {run_dir} — run not indexed")
            continue
        rows.append(_flatten(json.loads(meta_path.read_text())))

    if not rows:
        return pd.DataFrame(columns=[*LEAD_COLUMNS, "notes"])
    df = pd.DataFrame(rows)
    df["val_acc"] = df["overall_accuracy"].fillna(df["train_val_acc"])
    ordered = [c for c in LEAD_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in ordered and c != "notes"]
    df = df[ordered + rest + ["notes"]]
    return df.sort_values(["dataset", "architecture", "val_acc"],
                          ascending=[True, True, False]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", help="filter printed view by dataset (e.g. gislr)")
    ap.add_argument("--architecture", "--arch", dest="architecture",
                    help="filter printed view by architecture (e.g. gru)")
    ap.add_argument("--subset", help="filter printed view by landmark subset (e.g. ME_126)")
    ap.add_argument("--top", type=int, help="print only the top N rows of the filtered view")
    ap.add_argument("--untested", action="store_true",
                    help="show only runs not yet scored on the official test set "
                         "(submission.tested = false) — the submission queue")
    ap.add_argument("--no-migrate", action="store_true",
                    help="skip the pre-v3 meta.json schema backfill")
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_migrate:
        migrated = R.migrate_all(MODELS_DIR)
        if migrated:
            print(f"migrated {len(migrated)} meta.json file(s) to schema "
                  f"v{R.SCHEMA_VERSION}\n")
    df = load_runs()
    df.to_csv(MODEL_INDEX, index=False)
    print(f"wrote {MODEL_INDEX} ({len(df)} runs)\n")
    if df.empty:
        print("registry is empty — no runs recorded yet")
        return

    view = df
    for col in ("dataset", "architecture", "subset"):
        if getattr(args, col):
            view = view[view[col].str.lower() == getattr(args, col).lower()]
    if args.untested:
        view = view[~view["submission_tested"].fillna(False).astype(bool)]
    if args.top:
        view = view.head(args.top)

    show = ["dataset", "architecture", "run_id", "subset", "coords",
            "val_acc", "eval_status", "submission_tested", "n_params", "created"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(view[show].to_string(index=False))


if __name__ == "__main__":
    main()
