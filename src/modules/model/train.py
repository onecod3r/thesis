"""The unified training driver behind every gislr.1.model.*.ipynb notebook.

One call = one registry run:

    from modules.model import train_run
    run_dir = train_run(arch="gru", subset_name="ME_126", coords="xyz",
                        hyp=HYP, regime="v2-plateau-300",
                        source="gislr.1.model.gru.ipynb")

- **Single progress bar.** The whole run reports through ONE tqdm bar
  (total = epoch cap): batch-level progress, losses/accuracies, LR, best-so-far
  and the plateau counter all live in the same bar's description/postfix —
  no nested per-epoch bars, no per-epoch print spam.
- **Auto-resume.** ``last.pt`` is saved every epoch (atomically) with the
  optimizer/scheduler/early-stopping state; an interrupted run resumes in
  place via its pointer file, a FINISHED run is never reused (fresh
  epoch-seconds folder per training start).
- **meta.json always current.** The run record is (re)written every epoch, so
  even an interrupted run is indexed correctly; canonical-eval fields written
  by modules/scripts/eval_gru.py survive rewrites (registry.write_meta).
- Early stopping on val-accuracy plateau + ReduceLROnPlateau watching the same
  signal (regime v2-plateau-300, TODO §3.2).
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
from modules.model import registry as R
from modules.model.architectures import ARCHS, build_model


def atomic_torch_save(state: dict, path: Path) -> None:
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _atomic_write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _is_finished(last_ckpt: Path) -> bool:
    ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    return ck.get("finished", ck["epoch"] + 1 >= ck["hyp"]["epochs"])


def _run_epoch(
    model,
    loader,
    criterion,
    device,
    bar,
    phase,
    optimizer=None,
    scaler=None,
    grad_clip=5.0,
):
    """One pass over ``loader``; reports through ``bar``'s postfix only —
    never creates its own progress bar."""
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for b, (feats, lengths, labels) in enumerate(loader):
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(feats, lengths)
                loss = criterion(logits, labels)
            if train_mode:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)
            if b % 10 == 0 or b == n_batches - 1:
                bar.set_postfix_str(
                    f"{phase} {b + 1}/{n_batches} · loss {total_loss / total:.4f} "
                    f"· acc {correct / total:.4f}",
                    refresh=True,
                )
    return total_loss / total, correct / total


def _build_meta(
    *,
    run_dir,
    dataset,
    arch,
    subset,
    coords,
    feature_dim,
    n_params,
    n_classes,
    hyp,
    regime,
    source,
    history,
    best_val_acc,
    epochs_done,
    early_stopped,
    finished,
    wall_time_min,
    notes,
):
    spec = ARCHS[arch]
    best_epoch = (
        (int(np.argmax(history["val_acc"])) + 1) if history["val_acc"] else None
    )
    return {
        "schema_version": R.SCHEMA_VERSION,
        "run_id": int(run_dir.name),
        "created": datetime.fromtimestamp(int(run_dir.name)).isoformat(),
        "dataset": dataset,
        "architecture": arch,
        "model_name": spec.model_name,
        "streaming": spec.streaming,
        "subset": subset.name,
        "coords": coords,
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
            "regime": regime,
            "source": source,
            "epoch_cap": hyp["epochs"],
            "epochs_trained": epochs_done,
            "best_epoch": best_epoch,
            "early_stopped": early_stopped,
            "finished": finished,
            "wall_time_min": round(wall_time_min, 1),
        },
        "hyperparameters": {
            **hyp,
            "seed": D.SEED,
            "max_seq_len": D.MAX_SEQ_LEN,
            "num_workers": 0,
            "loss": "CE + label smoothing 0.1",
            "precision": "AMP",
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
        # never submitted by the training loop; write_meta preserves whatever
        # the evaluation notebook later records here
        "submission": dict(R.SUBMISSION_DEFAULT),
        "notes": notes,
    }


def train_from_config(
    arch: str,
    config=None,
    subsets: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Path]:
    """Train every subset for one architecture, using the shared training config.

    This is what each architecture section of `gislr.1.models.training.ipynb`
    calls, so a section is a two-liner and no notebook cell owns hyperparameters
    — `src/config/gislr.training.json` does (see modules.model.config).

    Reads the config from disk on every call, so a section can be re-run alone
    after editing the config and never depends on another cell's live state.
    Returns {subset_name: run_dir}.
    """
    from modules.model.config import load_config

    cfg = config or load_config()
    if not cfg.enabled(arch):
        print(f"{arch}: disabled in {cfg.path.name} — nothing to train")
        return {}

    hyp = cfg.hyp_for(arch)
    coords = cfg.coords_for(arch)
    overrides = cfg.overrides_for(arch)
    names = subsets if subsets is not None else cfg.subsets_for(arch)

    print(f"{arch} · regime {cfg.regime} · coords {coords} · subsets {names}")
    print(f"  hyperparameters from {cfg.path.name}"
          + (f" · OVERRIDES: {overrides}" if overrides else " (shared, no overrides)"))

    run_dirs = {}
    for name in names:
        run_dirs[name] = train_run(
            arch=arch, subset_name=name, coords=coords, hyp=hyp,
            regime=cfg.regime, source=cfg.source, dataset=cfg.dataset,
            notes=notes if notes is not None else cfg.notes_for(arch)
            or f"{name} · {arch} · regime {cfg.regime}.")
    return run_dirs


def train_run(
    *,
    arch: str,
    subset_name: str,
    hyp: dict,
    regime: str,
    source: str,
    coords: str = "xyz",
    dataset: str = "gislr",
    data_dir: Path | None = None,
    notes: str = "",
) -> Path:
    """Train one registry run; returns its run folder. Builds missing feature
    caches, auto-resumes an interrupted run (early-stopping counter and
    LR-scheduler state included), never reuses a finished one."""
    assert torch.cuda.is_available(), (
        "training requires the CUDA build of torch (uv sync)"
    )
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    from modules.paths import gislr_dir

    data_dir = data_dir or gislr_dir()
    sign2idx = D.load_label_map(data_dir)
    subset = get_subset(subset_name)
    tag = D.subset_tag(subset_name, coords)
    feature_dim = len(subset) * len(coords)

    train_split, val_split = D.get_canonical_split(data_dir, sign2idx)
    tr_data, tr_off = D.build_subset_cache(
        train_split, "train", subset, coords, data_dir
    )
    va_data, va_off = D.build_subset_cache(val_split, "val", subset, coords, data_dir)

    torch.manual_seed(D.SEED)
    np.random.seed(D.SEED)
    train_ds = D.SubsetArrayDataset(train_split, tr_data, tr_off, feature_dim)
    val_ds = D.SubsetArrayDataset(val_split, va_data, va_off, feature_dim)
    g = torch.Generator()
    g.manual_seed(D.SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=hyp["batch_size"],
        shuffle=True,
        collate_fn=D.collate_fn,
        num_workers=0,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=hyp["batch_size"],
        shuffle=False,
        collate_fn=D.collate_fn,
        num_workers=0,
    )

    model = build_model(arch, feature_dim, len(sign2idx), hyp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hyp["lr"], weight_decay=hyp["weight_decay"]
    )
    # plateau-coupled LR schedule: halves the LR when val acc stalls; the early
    # stop below watches the same signal with a longer patience, so the LR
    # gets lr_patience epochs to rescue a plateau before the run ends
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=hyp["lr_factor"], patience=hyp["lr_patience"]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")

    run_dir = R.resolve_run_dir(f"{dataset}_{arch}_{tag}", _is_finished)
    np.save(run_dir / "assets" / "landmarks.npy", subset.array)

    last, best = run_dir / R.CKPT_LAST, run_dir / R.CKPT_BEST
    start_epoch, best_val_acc, epochs_since_gain, wall_min = 0, 0.0, 0, 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch, best_val_acc, history = (
            ck["epoch"] + 1,
            ck["best_val_acc"],
            ck["history"],
        )
        epochs_since_gain = ck.get("epochs_since_gain", 0)
        wall_min = ck.get("wall_time_min", 0.0)

    meta_kw = dict(
        run_dir=run_dir,
        dataset=dataset,
        arch=arch,
        subset=subset,
        coords=coords,
        feature_dim=feature_dim,
        n_params=n_params,
        n_classes=len(sign2idx),
        hyp=hyp,
        regime=regime,
        source=source,
        notes=notes,
    )

    bar = tqdm(
        total=hyp["epochs"],
        initial=start_epoch,
        dynamic_ncols=True,
        desc=f"{dataset}/{arch}/{tag} · run {run_dir.name}",
    )
    if start_epoch:
        bar.write(
            f"{tag}: resumed at epoch {start_epoch}, best {best_val_acc:.4f}, "
            f"plateau {epochs_since_gain}/{hyp['es_patience']}"
        )
    t0 = time.time()
    early_stop = False
    for epoch in range(start_epoch, hyp["epochs"]):
        tr_loss, tr_acc = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            bar,
            f"ep{epoch + 1} train",
            optimizer,
            scaler,
            hyp["grad_clip"],
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, device, bar, f"ep{epoch + 1} val"
        )
        scheduler.step(val_acc)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])  # one point per epoch
        is_best = val_acc > best_val_acc
        # plateau detection: only a val-acc gain > es_min_delta resets the counter
        # (is_best still saves the best checkpoint on ANY improvement)
        epochs_since_gain = (
            0 if val_acc > best_val_acc + hyp["es_min_delta"] else epochs_since_gain + 1
        )
        best_val_acc = max(best_val_acc, val_acc)
        early_stop = epochs_since_gain >= hyp["es_patience"]
        finished = early_stop or (epoch + 1 >= hyp["epochs"])
        wall_now = wall_min + (time.time() - t0) / 60

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "history": history,
            "sign2idx": sign2idx,
            "hyp": {
                **hyp,
                "seed": D.SEED,
                "max_seq_len": D.MAX_SEQ_LEN,
                "num_workers": 0,
            },
            "feature_dim": feature_dim,
            "landmarks": subset.array.tolist(),
            "subset_name": subset_name,
            "coords": coords,
            "arch": arch,
            "training_regime": regime,
            "epochs_since_gain": epochs_since_gain,
            "finished": finished,
            "wall_time_min": wall_now,
        }
        atomic_torch_save(state, last)  # every epoch -> resume-safe
        if is_best:
            atomic_torch_save(state, best)
        # history as a plain asset too: the evaluation notebook plots learning
        # curves for runs whose (gitignored) checkpoints aren't on this machine
        _atomic_write_json(run_dir / "assets" / "history.json", history)
        R.write_meta(
            run_dir,
            _build_meta(
                **meta_kw,
                history=history,
                best_val_acc=best_val_acc,
                epochs_done=epoch + 1,
                early_stopped=early_stop,
                finished=finished,
                wall_time_min=wall_now,
            ),
        )

        bar.set_postfix_str(
            f"tr {tr_loss:.3f}/{tr_acc:.4f} · val {val_loss:.3f}/{val_acc:.4f} "
            f"· best {best_val_acc:.4f}{' *' if is_best else ''} "
            f"· lr {history['lr'][-1]:.1e} "
            f"· plateau {epochs_since_gain}/{hyp['es_patience']}"
        )
        bar.update(1)
        if early_stop:
            bar.write(
                f"{tag}: EARLY STOP at epoch {epoch + 1} — no val-acc gain "
                f"> {hyp['es_min_delta']} for {hyp['es_patience']} epochs"
            )
            break
    bar.close()
    print(f"{tag}: DONE best_val_acc={best_val_acc:.4f} run_dir={run_dir}")
    return run_dir
