"""Canonical per-class evaluation of a registry run on the val split.

Handles every architecture in modules.model.architectures.ARCHS (gru, lstm,
bilstm, cnn1d) by dispatching on the checkpoint's "arch" key; coordinate
channels follow its "coords" key ("xyz" or "xy"), landmark selection its
"landmarks" key. The model classes are imported from modules/model — the same
definitions the notebooks train — so state_dicts can never drift.

Reproduces the canonical split (stratified 10%, seed 42, 9,448 videos) and the
dataset preprocessing (NaN->0, uniform subsample to MAX_SEQ_LEN frames),
straight from the raw parquet files — no feature cache needed.

Usage (any CWD — the script bootstraps its own imports):

    .venv/Scripts/python.exe src/modules/scripts/eval_gru.py <run_dir> [--checkpoint best.pt]

<run_dir> is a registry folder (src/data/models/<run_id>/). Writes
assets/per_class_accuracy.{csv,png} + assets/eval_summary.json +
assets/val_predictions.npz, promotes meta.json metrics to
eval_status="canonical", and registers the new assets — rebuild the index
afterwards with modules/scripts/build_model_index.py.

val_predictions.npz (labels + preds over the canonical val split, in split
order) is what makes confusion matrices cheap: gislr.2.models.evaluation.ipynb
builds every matrix from these files instead of re-running inference.

Importable as well as runnable — the evaluation notebook calls

    from modules.scripts.eval_gru import evaluate_run
    summary = evaluate_run(run_dir)
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> src/ (imports work from any CWD)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from modules.model import registry as R
from modules.model.architectures import build_model
from modules.model.data import MAX_SEQ_LEN, ROWS_PER_FRAME, get_canonical_split, load_label_map
from modules.paths import gislr_dir

BATCH = 256


def load_video(path, landmarks, coords):
    cols = list(coords)
    table = pq.read_table(path, columns=cols)
    data = np.column_stack([table.column(c).to_numpy() for c in cols])
    n = data.shape[0] // ROWS_PER_FRAME
    arr = data.reshape(n, ROWS_PER_FRAME, len(cols)).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if landmarks is not None:
        arr = arr[:, landmarks, :]
    T = arr.shape[0]
    if T > MAX_SEQ_LEN:
        arr = arr[np.linspace(0, T - 1, MAX_SEQ_LEN).astype(int)]
        T = MAX_SEQ_LEN
    return arr.reshape(T, -1), T


def evaluate_run(run_dir, checkpoint: str = R.CKPT_BEST, verbose: bool = True) -> dict:
    """Canonical per-class evaluation of one registry run; returns the summary dict.

    Side effects (all inside the run folder): assets/per_class_accuracy.{csv,png},
    assets/eval_summary.json, assets/val_predictions.npz, and meta.json promoted
    to eval_status="canonical". Safe to re-run — everything is overwritten.
    """
    run_dir = Path(run_dir)
    assert (run_dir / "meta.json").is_file(), f"not a registry run folder: {run_dir}"
    device = torch.device("cuda")

    def log(*a):
        if verbose:
            print(*a, flush=True)

    data_dir = gislr_dir()
    sign2idx = load_label_map(data_dir)
    idx2sign = {v: k for k, v in sign2idx.items()}
    _, val_split = get_canonical_split(data_dir, sign2idx)
    log(f"val split: {len(val_split)} videos")

    ckpt = torch.load(run_dir / checkpoint, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "gru")
    coords = ckpt.get("coords", "xyz")
    landmarks = np.asarray(ckpt["landmarks"]) if ckpt.get("landmarks") is not None else None
    model = build_model(arch, ckpt["feature_dim"], len(sign2idx), ckpt["hyp"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log(f"checkpoint: arch={arch} coords={coords} feature_dim={ckpt['feature_dim']} "
        f"best_val_acc={ckpt['best_val_acc']:.4f}")

    paths = [data_dir / p for p in val_split["path"]]
    labels_all = val_split["label"].to_numpy()
    preds_all = np.zeros(len(val_split), dtype=np.int64)

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex, torch.no_grad():
        for b0 in range(0, len(paths), BATCH):
            chunk = list(ex.map(lambda p: load_video(p, landmarks, coords), paths[b0:b0 + BATCH]))
            order = np.argsort([-t for _, t in chunk])
            lengths = torch.tensor([chunk[i][1] for i in order])
            padded = torch.zeros(len(chunk), int(lengths[0]), chunk[0][0].shape[1])
            for j, i in enumerate(order):
                padded[j, : chunk[i][1]] = torch.from_numpy(chunk[i][0])
            logits = model(padded.to(device), lengths)
            pred = logits.argmax(-1).cpu().numpy()
            inv = np.empty_like(order); inv[order] = np.arange(len(order))
            preds_all[b0:b0 + len(chunk)] = pred[inv]
            if (b0 // BATCH) % 10 == 0:
                log(f"  {b0 + len(chunk)}/{len(paths)}  ({time.time() - t0:.0f}s)")

    correct = preds_all == labels_all
    overall = correct.mean()
    df = pd.DataFrame({"label": labels_all, "correct": correct})
    per_class = (df.groupby("label")["correct"].agg(["mean", "count"])
                 .rename(columns={"mean": "accuracy", "count": "n_val"}))
    per_class["sign"] = per_class.index.map(idx2sign)
    per_class = per_class[["sign", "accuracy", "n_val"]].sort_values("accuracy")
    macro = per_class["accuracy"].mean()

    assets = run_dir / "assets"
    assets.mkdir(exist_ok=True)
    per_class.to_csv(assets / "per_class_accuracy.csv", index_label="label")
    # raw predictions: everything downstream (confusion matrices, confused-pair
    # analysis) derives from these, so no consumer needs to re-run inference
    np.savez_compressed(assets / "val_predictions.npz",
                        labels=labels_all.astype(np.int16),
                        preds=preds_all.astype(np.int16))

    # no matplotlib.use() here — evaluate_run is imported by the evaluation
    # notebook, where forcing Agg would kill inline figures; the CLI sets it
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].hist(per_class["accuracy"], bins=25, color="tab:blue", edgecolor="white")
    axes[0].axvline(overall, color="black", ls="--", label=f"overall {overall:.3f}")
    axes[0].axvline(macro, color="tab:red", ls=":", label=f"macro {macro:.3f}")
    axes[0].set_xlabel("per-class accuracy"); axes[0].set_ylabel("# classes")
    axes[0].set_title("Distribution of per-class accuracy (250 signs)"); axes[0].legend()
    worst = per_class.head(15)
    axes[1].barh(worst["sign"], worst["accuracy"], color="tab:red")
    axes[1].set_title("15 worst classes"); axes[1].set_xlabel("accuracy")
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(assets / "per_class_accuracy.png", dpi=110)

    summary = {
        "overall_accuracy": float(overall),
        "macro_accuracy": float(macro),
        "n_val": int(len(val_split)),
        "worst5": per_class.head(5)[["sign", "accuracy"]].values.tolist(),
        "best5": per_class.tail(5)[["sign", "accuracy"]].values.tolist(),
        "n_classes_below_50pct": int((per_class["accuracy"] < 0.5).sum()),
        "median_class_accuracy": float(per_class["accuracy"].median()),
    }
    (assets / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    plt.close(fig)
    log(json.dumps(summary, indent=2))

    # promote the canonical numbers into meta.json (the record
    # build_model_index.py aggregates into data/models/index.csv)
    meta = R.load_meta(run_dir)
    meta["metrics"].update({
        "eval_status": "canonical",
        "overall_accuracy": summary["overall_accuracy"],
        "macro_accuracy": summary["macro_accuracy"],
        "median_class_accuracy": summary["median_class_accuracy"],
        "n_classes_below_50pct": summary["n_classes_below_50pct"],
    })
    R.write_meta(run_dir, meta)
    R.register_assets(run_dir,
                      per_class_csv="assets/per_class_accuracy.csv",
                      per_class_png="assets/per_class_accuracy.png",
                      eval_summary="assets/eval_summary.json",
                      val_predictions="assets/val_predictions.npz")
    log(f"updated {run_dir / 'meta.json'} (eval_status=canonical) — rebuild the "
        f"index with modules/scripts/build_model_index.py")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="registry run folder (src/data/models/<run_id>/)")
    ap.add_argument("--checkpoint", default=R.CKPT_BEST,
                    help=f"checkpoint file inside the run folder (default {R.CKPT_BEST})")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")  # headless CLI; the notebook path keeps its backend
    evaluate_run(args.run_dir, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
