"""Training driver for the 1st-place port (TODO §4.2).

Why this is not ``modules.model.train``: that driver implements regime
``v2-plateau-300`` — ReduceLROnPlateau + early stopping on a val-accuracy
plateau, one fixed feature tensor per subset, no augmentation. The 1st-place
recipe is a different regime in every one of those respects (fixed-length cosine
one-cycle, no early stop, per-sample augmentation, AWP, Lookahead), and folding
both into one function would mean a driver full of branches where the existing
one is a straight line. Everything downstream is shared unchanged: the same
canonical split, the same registry layout, the same ``meta.json`` schema, so a
run from here lands on the same leaderboard as every other run.

    from modules.model.train_fp import train_firstplace
    run_dir = train_firstplace()          # config-driven, one run per subset

Faithfulness to the reference, and the three places this deviates:

1. **Padding to the batch max**, not to a fixed 384 frames (``features.collate_fn``)
   — numerically identical given masking, ~10x less wasted compute.
2. **Masked BatchNorm** (``architectures.MaskedBatchNorm1d``) — the reference
   lets padded frames into the batch statistics; with variable-length batches
   that would make normalization depend on how a batch was bucketed.
3. **Best checkpoint by val accuracy**, not val loss — accuracy is the registry's
   comparable metric (`metrics.train_val_acc`) and what the canonical eval
   reproduces. Val loss is still recorded in ``assets/history.json``.

Deviations 1-2 should if anything help; 3 only changes which epoch is kept.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modules.dataset.landmark.subsets import get_subset
from modules.model import data as D
from modules.model import features as F
from modules.model import registry as R
from modules.model.architectures import ARCHS, build_model
from modules.model.optim import AWP, Lookahead, cosine_one_cycle
from modules.model.train import _atomic_write_json, atomic_torch_save
from modules.paths import SRC_DIR

DEFAULT_CONFIG = SRC_DIR / "config" / "gislr.firstplace.json"

REQUIRED_HYP_KEYS = (
    "batch_size", "lr", "hidden_size", "num_layers", "dropout", "kernel_size",
    "num_heads", "expand", "late_dropout", "late_dropout_start_epoch",
    "weight_decay", "epochs", "warmup_epochs", "lr_min_ratio", "grad_clip",
    "label_smoothing", "awp_delta", "awp_start_epoch", "lookahead_k",
    "lookahead_alpha", "num_workers",
)


def load_fp_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    """Read + validate the 1st-place training config.

    Same principle as ``modules.model.config``: no notebook cell owns a
    hyperparameter, and a missing/typo'd key fails here rather than an hour into
    a run.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"1st-place training config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    for key in ("schema_version", "dataset", "architecture", "regime", "source",
                "coords", "subsets", "features", "hyp"):
        assert key in raw, f"{path}: missing top-level key {key!r}"
    assert raw["schema_version"] == 1, f"{path}: unsupported schema_version"
    assert raw["architecture"] in ARCHS, f"{path}: unknown architecture"
    assert raw["coords"] == "xy", (
        f"{path}: the 1st-place pipeline is xy-only (its Preprocess drops z "
        "after normalizing); coords must be 'xy'")
    assert raw["subsets"], f"{path}: `subsets` is empty"

    missing = [k for k in REQUIRED_HYP_KEYS if k not in raw["hyp"]]
    assert not missing, f"{path}: hyp block missing {missing}"
    assert raw["features"]["diff_mode"] in ("forward", "backward"), (
        f"{path}: features.diff_mode must be 'forward' or 'backward'")
    return raw


def _build_meta(*, run_dir, cfg, subset, feature_dim, n_params, n_classes, hyp,
                history, best_val_acc, epochs_done, finished, wall_time_min, notes):
    arch = cfg["architecture"]
    spec = ARCHS[arch]
    best_epoch = (int(np.argmax(history["val_acc"])) + 1) if history["val_acc"] else None
    return {
        "schema_version": R.SCHEMA_VERSION,
        "run_id": int(run_dir.name),
        "created": datetime.fromtimestamp(int(run_dir.name)).isoformat(),
        "dataset": cfg["dataset"],
        "architecture": arch,
        "model_name": spec.model_name,
        "streaming": spec.streaming,
        "subset": subset.name,
        "coords": cfg["coords"],
        "n_landmarks": len(subset),
        "feature_dim": int(feature_dim),
        "n_classes": int(n_classes),
        "n_params": int(n_params),
        "split": {
            "strategy": "stratified 90/10",
            "random_state": D.SEED,
            "n_val": D.N_VAL,
        },
        "training": {
            "regime": cfg["regime"],
            "source": cfg["source"],
            "epoch_cap": hyp["epochs"],
            "epochs_trained": epochs_done,
            "best_epoch": best_epoch,
            "early_stopped": False,  # fixed-length cosine — no early stopping
            "finished": finished,
            "wall_time_min": round(wall_time_min, 1),
        },
        "hyperparameters": {
            **hyp,
            "seed": D.SEED,
            "max_seq_len": cfg["features"]["max_len"],
            "loss": f"CE + label smoothing {hyp['label_smoothing']}",
            "precision": "AMP",
            # what makes this run's INPUT different from every other registry
            # run — without these three the comparison is unattributable
            "features": "firstplace(norm+lag1+lag2)",
            "diff_mode": cfg["features"]["diff_mode"],
            "augment": cfg["features"]["augment"],
        },
        "metrics": {
            "train_val_acc": round(float(best_val_acc), 4),
            "eval_status": "pending",
            "overall_accuracy": None,
            "macro_accuracy": None,
            "median_class_accuracy": None,
            "n_classes_below_50pct": None,
        },
        "checkpoints": {"best": R.CKPT_BEST, "last": R.CKPT_LAST},
        "assets": {
            "landmarks": "assets/landmarks.npy",
            "history": "assets/history.json",
        },
        "submission": dict(R.SUBMISSION_DEFAULT),
        "notes": notes,
    }


def _is_finished(last_ckpt: Path) -> bool:
    ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    return ck.get("finished", ck["epoch"] + 1 >= ck["hyp"]["epochs"])


def _evaluate(model, loader, criterion, device, bar, phase):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    with torch.no_grad():
        for b, (feats, lengths, labels) in enumerate(loader):
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(feats, lengths)
                loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)
            if b % 10 == 0 or b == n_batches - 1:
                bar.set_postfix_str(
                    f"{phase} {b + 1}/{n_batches} · loss {total_loss / total:.4f} "
                    f"· acc {correct / total:.4f}", refresh=True)
    return total_loss / total, correct / total


def train_firstplace(config: dict | None = None, subsets: list[str] | None = None,
                     data_dir: Path | None = None) -> dict[str, Path]:
    """Train the 1st-place port for every configured subset. Returns
    {subset_name: run_dir}. Reads the config from disk on every call, so the
    notebook cell can be re-run alone after editing it."""
    cfg = config or load_fp_config()
    names = subsets if subsets is not None else cfg["subsets"]
    hyp = dict(cfg["hyp"])

    print(f"{cfg['architecture']} · regime {cfg['regime']} · coords {cfg['coords']} "
          f"· subsets {names}")
    print(f"  features: max_len={cfg['features']['max_len']} "
          f"diff_mode={cfg['features']['diff_mode']} "
          f"augment={cfg['features']['augment']}")
    print(f"  AWP delta={hyp['awp_delta']} from epoch {hyp['awp_start_epoch']} "
          f"· Lookahead k={hyp['lookahead_k']} · label smoothing "
          f"{hyp['label_smoothing']}")

    return {name: train_firstplace_run(cfg, name, hyp, data_dir=data_dir)
            for name in names}


def train_firstplace_run(cfg: dict, subset_name: str, hyp: dict,
                         data_dir: Path | None = None) -> Path:
    """One registry run of the 1st-place recipe. Auto-resumes an interrupted run
    in place; never reuses a finished one."""
    assert torch.cuda.is_available(), (
        "training requires the CUDA build of torch (uv sync)")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    from modules.paths import gislr_dir

    arch = cfg["architecture"]
    coords = cfg["coords"]
    fcfg = cfg["features"]
    data_dir = data_dir or gislr_dir()
    sign2idx = D.load_label_map(data_dir)
    subset = get_subset(subset_name)
    tag = D.subset_tag(subset_name, coords)
    feature_dim = len(subset) * F.CHANNELS_PER_LANDMARK

    train_split, val_split = D.get_canonical_split(data_dir, sign2idx)
    tr_data, tr_off = F.build_nan_cache(train_split, "train", subset, coords, data_dir)
    va_data, va_off = F.build_nan_cache(val_split, "val", subset, coords, data_dir)

    torch.manual_seed(D.SEED)
    np.random.seed(D.SEED)
    n_workers = int(hyp["num_workers"])
    train_ds = F.FirstPlaceDataset(
        train_split, tr_data, tr_off, subset, augment_data=fcfg["augment"],
        max_len=fcfg["max_len"], diff_mode=fcfg["diff_mode"], seed=D.SEED,
        mmap=n_workers > 0)
    val_ds = F.FirstPlaceDataset(
        val_split, va_data, va_off, subset, augment_data=False,
        max_len=fcfg["max_len"], diff_mode=fcfg["diff_mode"], seed=D.SEED,
        mmap=n_workers > 0)
    # length-bucketed batching: GISLR lengths are heavily skewed (median 22,
    # p99 219), so randomly composed batches pad to the tail and waste ~6x the
    # compute. See features.LengthBucketedBatchSampler.
    train_sampler = F.LengthBucketedBatchSampler(
        F.cached_lengths(tr_off), hyp["batch_size"], shuffle=True,
        drop_last=True, seed=D.SEED)
    val_sampler = F.LengthBucketedBatchSampler(
        F.cached_lengths(va_off), hyp["batch_size"], shuffle=False,
        drop_last=False, seed=D.SEED)
    # num_workers > 0 overlaps the numpy augmentation with GPU compute. Safe
    # here because the dataset class lives in an importable module rather than
    # __main__ (the same reason the POPSIGN pool runs in a kernel, TODO 2.3) —
    # but it is opt-in, since Windows spawn is the fragile path in this repo.
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=F.collate_fn,
        num_workers=n_workers, persistent_workers=n_workers > 0,
        prefetch_factor=4 if n_workers > 0 else None)
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, collate_fn=F.collate_fn,
        num_workers=n_workers, persistent_workers=n_workers > 0,
        prefetch_factor=4 if n_workers > 0 else None)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * hyp["epochs"]
    # LateDropout and AWP both switch on partway through training; the config
    # says "epoch", the implementations count optimizer steps
    build_hyp = {**hyp,
                 "late_dropout_start_step": hyp["late_dropout_start_epoch"] * steps_per_epoch}
    model = build_model(arch, feature_dim, len(sign2idx), build_hyp).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.RAdam(model.parameters(), lr=hyp["lr"],
                                  weight_decay=hyp["weight_decay"],
                                  decoupled_weight_decay=True)
    lookahead = Lookahead(optimizer, k=hyp["lookahead_k"],
                          alpha=hyp["lookahead_alpha"])
    scheduler = cosine_one_cycle(optimizer, total_steps,
                                 warmup_steps=hyp["warmup_epochs"] * steps_per_epoch,
                                 lr_min_ratio=hyp["lr_min_ratio"])
    awp = AWP(model, delta=hyp["awp_delta"],
              start_step=hyp["awp_start_epoch"] * steps_per_epoch)
    criterion = nn.CrossEntropyLoss(label_smoothing=hyp["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda")

    run_dir = R.resolve_run_dir(f"{cfg['dataset']}_{arch}_{tag}", _is_finished)
    np.save(run_dir / "assets" / "landmarks.npy", subset.array)

    last, best = run_dir / R.CKPT_LAST, run_dir / R.CKPT_BEST
    start_epoch, best_val_acc, wall_min, global_step = 0, 0.0, 0.0, 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        lookahead.load_state_dict(ck["lookahead_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = ck["epoch"] + 1
        best_val_acc, history = ck["best_val_acc"], ck["history"]
        wall_min = ck.get("wall_time_min", 0.0)
        global_step = ck.get("global_step", start_epoch * steps_per_epoch)

    meta_kw = dict(run_dir=run_dir, cfg=cfg, subset=subset, feature_dim=feature_dim,
                   n_params=n_params, n_classes=len(sign2idx), hyp=hyp,
                   notes=f"{subset_name} · 1st-place port (TODO §4.2) · "
                         f"regime {cfg['regime']}.")

    bar = tqdm(total=hyp["epochs"], initial=start_epoch, dynamic_ncols=True,
               desc=f"{cfg['dataset']}/{arch}/{tag} · run {run_dir.name}")
    bar.write(f"{tag}: {n_params / 1e6:.2f}M params · feature_dim {feature_dim} "
              f"· {steps_per_epoch} steps/epoch")
    if start_epoch:
        bar.write(f"{tag}: resumed at epoch {start_epoch}, best {best_val_acc:.4f}")

    t0 = time.time()
    for epoch in range(start_epoch, hyp["epochs"]):
        train_ds.set_epoch(epoch)       # fresh, reproducible augmentation
        train_sampler.set_epoch(epoch)  # ...and a fresh bucketing/shuffle
        model.train()
        total_loss, correct, total, n_awp = 0.0, 0, 0, 0
        n_batches = len(train_loader)
        for b, (feats, lengths, labels) in enumerate(train_loader):
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(feats, lengths)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()

            # AWP: re-take the gradient at an adversarially perturbed point.
            # Safe with the scaler because the perturbation is scale-invariant
            # (modules.model.optim.AWP), so no second unscale_ is needed.
            if awp.active(global_step) and awp.perturb():
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    adv_loss = criterion(model(feats, lengths), labels)
                scaler.scale(adv_loss).backward()
                awp.restore()
                n_awp += 1

            scaler.unscale_(optimizer)
            if hyp["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hyp["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            lookahead.sync()  # slow-weight pull, every k completed steps
            scheduler.step()  # per-batch cosine, as in the reference
            global_step += 1

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)
            if b % 10 == 0 or b == n_batches - 1:
                bar.set_postfix_str(
                    f"ep{epoch + 1} train {b + 1}/{n_batches} · "
                    f"loss {total_loss / total:.4f} · acc {correct / total:.4f}"
                    + (f" · awp {n_awp}" if n_awp else ""), refresh=True)
        tr_loss, tr_acc = total_loss / total, correct / total

        val_loss, val_acc = _evaluate(model, val_loader, criterion, device, bar,
                                      f"ep{epoch + 1} val")
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)
        finished = epoch + 1 >= hyp["epochs"]
        wall_now = wall_min + (time.time() - t0) / 60

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "lookahead_state": lookahead.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "history": history,
            "sign2idx": sign2idx,
            "hyp": {**build_hyp, "seed": D.SEED,
                    "max_seq_len": fcfg["max_len"]},
            "feature_dim": feature_dim,
            "landmarks": subset.array.tolist(),
            "subset_name": subset_name,
            "coords": coords,
            "arch": arch,
            "training_regime": cfg["regime"],
            # tells modules/scripts/eval_gru.py to score this run through the
            # 1st-place preprocessing rather than the default one
            "features": "firstplace",
            "diff_mode": fcfg["diff_mode"],
            "finished": finished,
            "wall_time_min": wall_now,
            "global_step": global_step,
        }
        atomic_torch_save(state, last)
        if is_best:
            atomic_torch_save(state, best)
        _atomic_write_json(run_dir / "assets" / "history.json", history)
        R.write_meta(run_dir, _build_meta(
            **meta_kw, history=history, best_val_acc=best_val_acc,
            epochs_done=epoch + 1, finished=finished, wall_time_min=wall_now))

        bar.set_postfix_str(
            f"tr {tr_loss:.3f}/{tr_acc:.4f} · val {val_loss:.3f}/{val_acc:.4f} "
            f"· best {best_val_acc:.4f}{' *' if is_best else ''} "
            f"· lr {history['lr'][-1]:.2e}")
        bar.update(1)
    bar.close()
    print(f"{tag}: DONE best_val_acc={best_val_acc:.4f} run_dir={run_dir}")
    return run_dir
