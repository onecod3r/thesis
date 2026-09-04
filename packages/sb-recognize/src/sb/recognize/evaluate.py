"""Canonical per-class evaluation of a registry run on the val split.

Handles every architecture in sb.recognize.architectures.ARCHS (gru, lstm,
bilstm, cnn1d, conv1d_transformer) by dispatching on the checkpoint's "arch" key; coordinate
channels follow its "coords" key ("xyz" or "xy"), landmark selection its
"landmarks" key. The model classes are imported from sb.recognize.architectures — the same
definitions the notebooks train — so state_dicts can never drift.

Reproduces the canonical split (stratified 10%, seed 42, 9,448 videos) and the
dataset preprocessing (NaN->0, uniform subsample to MAX_SEQ_LEN frames),
straight from the raw parquet files — no feature cache needed.

Usage (any CWD — installed as a console script):

    sb-evaluate <run_dir> [--checkpoint best.pt]

<run_dir> is a registry folder (registry/runs/<run_id>/). Writes
assets/per_class_accuracy.{csv,png} + assets/eval_summary.json +
assets/val_predictions.npz, promotes meta.json metrics to
eval_status="canonical", and registers the new assets — rebuild the index
afterwards with `sb-index`.

**The checkpoint is usually not on this disk.** Weights are pushed to Kaggle and
pruned locally, so this fetches the run's `best.pt` through `kagglehub` on
demand and verifies its sha256 against the manifest before using it. `--no-fetch`
turns that off and prints where the file is instead.

val_predictions.npz (labels, preds, and the top-5 ranked alternatives with
their probabilities, over the canonical val split in split order) is what makes
confusion matrices cheap: the evaluation notebook builds every matrix from these
files instead of re-running inference. The top-5 arrays are what let the plateau
diagnosis ask whether a wrong answer was *nearly* right (TODO §7.1). Runs
evaluated before 2026-09-04 have only `labels`/`preds`; re-run `sb-evaluate` to
backfill the rest.

Importable as well as runnable — the evaluation notebook calls

    from sb.recognize.evaluate import evaluate_run
    summary = evaluate_run(run_dir)
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from sb.mlops import registry as R
from sb.mlops import run as P
from sb.mlops.artifacts import ensure_local
from sb.recognize.architectures import build_model
from sb.recognize.data import MAX_SEQ_LEN, ROWS_PER_FRAME
from sb.recognize.sources import get_source

BATCH = 256
TOPK = 5  # ranked alternatives kept per sample (see the topk_* arrays below)


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


def load_video_firstplace(path, landmarks, coords, max_len, diff_mode):
    """Val-time loading for runs trained through the 1st-place feature pipeline
    (``sb.recognize.features.firstplace_v1``): NaNs preserved, reference-point normalization,
    lag-1/lag-2 differences, no augmentation.

    Scoring such a run with :func:`load_video` would feed it the wrong feature
    space entirely (2 channels of un-normalized coordinates instead of 6
    normalized ones) and silently report near-chance accuracy — hence the
    dispatch rather than a shared default.
    """
    from sb.recognize.features import firstplace_v1 as FP

    arr = FP.load_video_raw(Path(path), landmarks, coords)
    arr = FP.drop_empty_frames(arr)[:max_len]
    ref_idx = int(np.searchsorted(landmarks, FP.REF_LANDMARK))
    assert landmarks[ref_idx] == FP.REF_LANDMARK, (
        f"reference landmark {FP.REF_LANDMARK} missing from this run's subset")
    feats = FP.preprocess(arr, ref_idx, max_len, diff_mode)
    return feats, feats.shape[0]


def evaluate_run(run_dir, checkpoint: str = R.CKPT_BEST, verbose: bool = True,
                 fetch: bool = True) -> dict:
    """Canonical per-class evaluation of one registry run; returns the summary dict.

    Side effects (all inside the run folder): assets/per_class_accuracy.{csv,png},
    assets/eval_summary.json, assets/val_predictions.npz, and meta.json promoted
    to eval_status="canonical". Safe to re-run — everything is overwritten.

    The checkpoint is downloaded from the artifact remote when it is not on this
    disk, which is the normal state — weights live on Kaggle. ``fetch=False``
    turns that off and reports where the file is instead.
    """
    run_dir = Path(run_dir)
    assert (run_dir / "meta.json").is_file(), f"not a registry run folder: {run_dir}"
    device = torch.device("cuda")

    def log(*a):
        if verbose:
            print(*a, flush=True)

    # the run record says which dataset it was trained on; the split and the
    # sample reader come from that source, not from a hardcoded GISLR (TODO §9.5)
    meta = R.load_meta(run_dir)
    source = get_source(meta.get("dataset", "gislr"))
    data_dir = source.resolve_dir()
    sign2idx = source.label_map(data_dir)
    idx2sign = {v: k for k, v in sign2idx.items()}
    _, val_split = source.canonical_split(data_dir, sign2idx)
    log(f"dataset: {source.name} · val split: {len(val_split)} videos")

    # Checkpoints normally live on Kaggle, not on this disk, so an absent file
    # is the expected case rather than a broken run: fetch it (sha256-verified
    # against the manifest) instead of making the caller go and do it.
    ckpt_path = ensure_local(run_dir, checkpoint, fetch=fetch, verbose=verbose)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "gru")
    coords = ckpt.get("coords", "xyz")
    landmarks = np.asarray(ckpt["landmarks"]) if ckpt.get("landmarks") is not None else None
    model = build_model(arch, ckpt["feature_dim"], len(sign2idx), ckpt["hyp"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log(f"checkpoint: arch={arch} coords={coords} feature_dim={ckpt['feature_dim']} "
        f"best_val_acc={ckpt['best_val_acc']:.4f}")

    # which preprocessing produced this run's inputs — the split and the metric
    # stay canonical either way, only the feature construction differs
    if ckpt.get("features") == "firstplace":
        max_len = ckpt["hyp"].get("max_seq_len", 384)
        diff_mode = ckpt.get("diff_mode", "forward")
        log(f"features: 1st-place pipeline (max_len={max_len}, diff_mode={diff_mode})")

        def load_one(p):
            return load_video_firstplace(p, landmarks, coords, max_len, diff_mode)
    else:
        def load_one(p):
            return load_video(p, landmarks, coords)

    paths = [source.sample_path(data_dir, row) for _, row in val_split.iterrows()]
    labels_all = val_split["label"].to_numpy()
    preds_all = np.zeros(len(val_split), dtype=np.int64)
    # Top-K, not just argmax: "is the right answer ranked second?" separates a
    # model that lacks the information from one that has it and mis-ranks it —
    # the question the plateau diagnosis turns on (TODO §7.1). Storing K=5 costs
    # ~200 KB per run; full logits would be 9.4 MB and this asset is committed.
    topk_idx = np.zeros((len(val_split), TOPK), dtype=np.int16)
    topk_prob = np.zeros((len(val_split), TOPK), dtype=np.float32)

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex, torch.no_grad():
        for b0 in range(0, len(paths), BATCH):
            chunk = list(ex.map(load_one, paths[b0:b0 + BATCH]))
            order = np.argsort([-t for _, t in chunk])
            lengths = torch.tensor([chunk[i][1] for i in order])
            padded = torch.zeros(len(chunk), int(lengths[0]), chunk[0][0].shape[1])
            for j, i in enumerate(order):
                padded[j, : chunk[i][1]] = torch.from_numpy(chunk[i][0])
            logits = model(padded.to(device), lengths)
            probs = torch.softmax(logits.float(), dim=-1)
            tp, ti = probs.topk(TOPK, dim=-1)
            pred = logits.argmax(-1).cpu().numpy()
            inv = np.empty_like(order); inv[order] = np.arange(len(order))
            preds_all[b0:b0 + len(chunk)] = pred[inv]
            topk_idx[b0:b0 + len(chunk)] = ti.cpu().numpy()[inv]
            topk_prob[b0:b0 + len(chunk)] = tp.cpu().numpy()[inv]
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
                        topk_idx=topk_idx, topk_prob=topk_prob,
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
        # The canonical number is produced HERE, not in training, so the eval
        # gets its own provenance (TODO §9.1). It lives in the summary asset
        # rather than in meta.json["provenance"], which belongs to the run that
        # trained the weights.
        "provenance": P.build(
            dataset=source.name,
            kaggle_ref=source.kaggle_ref,
            manifest=source.manifest,
            data_dir=data_dir,
            feature_pipeline=(P.PIPELINE_FIRSTPLACE
                              if ckpt.get("features") == "firstplace"
                              else P.PIPELINE_BASE),
            n_videos=int(len(val_split)),
        ),
    }
    (assets / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    plt.close(fig)
    log(json.dumps(summary, indent=2))

    # promote the canonical numbers into meta.json (the record
    # build_model_index.py aggregates into data/models/index.csv). Re-read
    # rather than reusing the copy from the top: this evaluation can take
    # minutes and the training driver may have rewritten meta.json since.
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
    ap.add_argument("--no-fetch", dest="fetch", action="store_false",
                    help="do not download the checkpoint if it is missing; "
                         "report where it is instead")
    ap.add_argument("--checkpoint", default=R.CKPT_BEST,
                    help=f"checkpoint file inside the run folder (default {R.CKPT_BEST})")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")  # headless CLI; the notebook path keeps its backend
    evaluate_run(args.run_dir, checkpoint=args.checkpoint, fetch=args.fetch)


if __name__ == "__main__":
    main()
