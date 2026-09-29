"""Training driver for the sign-vs-gap detector (TODO §17).

One call = one run of ``configs/gislr.gapdetect.json``:

    from sb.recognize.gapdetect.train import train_gap
    run_dir = train_gap("A")

**Deliberately not a registry run** (``registry/runs/<id>/meta.json``,
schema v4): that schema's fields (``n_classes``, ``subset``, canonical
isolated accuracy, ...) describe a gloss classifier, and GapGRU is not one --
see the note above :class:`~sb.recognize.architectures.GapGRU`. Runs live
under ``experiments/recognition/gapdetect_runs/<run_id>/`` instead, with
their own small ``meta.json``, so `sb-docs`'s generated registry docs are
never asked to describe a model shape they don't cover.

Same run mechanics as the rest of the repo where they *do* apply: one
progress bar, ``last.pt`` every epoch (atomic) with auto-resume, ``best.pt``
on improvement, early stopping, a ``history.json``.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sb.core.paths import EXPERIMENTS_DIR, ROOT_DIR, gislr_gapcorpus_dir
from sb.recognize.architectures import GapGRU
from sb.recognize.gapdetect.data import GapCorpus, collate
from sb.recognize.train import _atomic_write_json, atomic_torch_save

CONFIG_PATH = EXPERIMENTS_DIR / "recognition" / "configs" / "gislr.gapdetect.json"
RUNS_DIR = EXPERIMENTS_DIR / "recognition" / "gapdetect_runs"
CKPT_LAST, CKPT_BEST = "last.pt", "best.pt"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_hyp(cfg: dict, run_name: str) -> dict:
    run = cfg["runs"][run_name]
    over = run.get("overrides", {})
    unknown = sorted(set(over) - set(cfg["shared"]))
    if unknown:
        raise KeyError(f"run {run_name!r} overrides keys not in `shared`: {unknown}")
    return {**cfg["shared"], **over}


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def frame_metrics(logits: torch.Tensor, gap: torch.Tensor, mask: torch.Tensor) -> dict:
    """Frame-level precision/recall/F1 of the **gap** class at threshold 0.5,
    plus plain accuracy -- gap frames are usually the minority class (a sign
    typically outlasts a transition), so accuracy alone would hide a model
    that just always predicts "sign"."""
    pred = (torch.sigmoid(logits) >= 0.5) & mask
    truth = (gap >= 0.5) & mask
    tp = (pred & truth).sum().item()
    fp = (pred & ~truth).sum().item()
    fn = (~pred & truth).sum().item()
    tn = (~pred & ~truth & mask).sum().item()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"acc": (tp + tn) / max(mask.sum().item(), 1), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9)}


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    tot_loss = n_batches = 0.0
    agg = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for batch in loader:
        x, gap, mask = batch["x"].to(device), batch["gap"].to(device), batch["mask"].to(device)
        with torch.amp.autocast(device.type):
            logits = model.forward_all(x)
        pos_weight = ((~gap.bool() & mask).sum() / (gap.bool() & mask).sum().clamp_min(1)).clamp(max=20.0)
        loss = F.binary_cross_entropy_with_logits(logits.float(), gap, weight=mask.float(),
                                                   pos_weight=pos_weight, reduction="sum") / mask.sum().clamp_min(1)
        tot_loss += float(loss)
        n_batches += 1
        m = frame_metrics(logits, gap, mask)
        for k in agg:
            agg[k] += m[k]
    return {"loss": tot_loss / max(n_batches, 1), **{k: v / max(n_batches, 1) for k, v in agg.items()}}


def train_gap(run_name: str, config_path: Path | str = CONFIG_PATH, *,
             corpus_dir: Path | str | None = None, smoke: int = 0) -> Path | dict:
    """Train one run of the gap-detect config; returns its run folder.

    ``smoke=N`` runs N training batches + a validation pass on up to 4
    batches, writes nothing, and returns the metrics -- a wiring check, not
    training (same contract as ``sb.recognize.continuous.train.train_continuous``).
    ``corpus_dir`` overrides ``gislr_gapcorpus_dir()`` (e.g. a smoke-scale
    local build, before the Kaggle dataset exists).
    """
    assert torch.cuda.is_available() or smoke, "training requires the CUDA build of torch (uv sync)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(config_path)
    hyp = run_hyp(cfg, run_name)
    seed = cfg["seed"]
    root = Path(corpus_dir) if corpus_dir else gislr_gapcorpus_dir(cfg["corpus_version"])

    torch.manual_seed(seed)
    train_ds, val_ds = GapCorpus(root, "train"), GapCorpus(root, "test")
    train_loader = DataLoader(train_ds, batch_size=hyp["batch_size"], shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=hyp["batch_size"], shuffle=False,
                            collate_fn=collate, num_workers=0)

    model = GapGRU(hyp["hidden_size"], hyp["num_layers"], hyp["dropout"],
                   hyp["jitter_window"], hyp["dist_window"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyp["lr"], weight_decay=hyp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=hyp["lr_factor"], patience=hyp["lr_patience"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    def step_batch(batch, train: bool):
        x, gap, mask = (batch[k].to(device, non_blocking=True) for k in ("x", "gap", "mask"))
        with torch.amp.autocast(device.type):
            logits = model.forward_all(x)
        pos_weight = ((~gap.bool() & mask).sum() / (gap.bool() & mask).sum().clamp_min(1)
                      ).clamp(max=hyp["pos_weight_cap"])
        loss = F.binary_cross_entropy_with_logits(logits.float(), gap, weight=mask.float(),
                                                   pos_weight=pos_weight, reduction="sum") / mask.sum().clamp_min(1)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        return loss.item()

    if smoke:
        model.train()
        losses, val_batches = [], []
        for i, batch in enumerate(train_loader):
            if i >= smoke:
                break
            losses.append(step_batch(batch, True))
        for i, batch in enumerate(val_loader):
            if i >= 4:
                break
            val_batches.append(batch)
        val = evaluate(model, val_batches, device) if val_batches else {}
        return {"train_losses": losses, "val": val, "n_train_sequences": len(train_ds),
                "n_val_sequences": len(val_ds), "n_params": n_params, "corpus_dir": str(root)}

    run_id = int(time.time())
    run_dir = RUNS_DIR / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    last, best = run_dir / CKPT_LAST, run_dir / CKPT_BEST
    start_epoch, best_val, since_gain = 0, 0.0, 0
    history: dict[str, list] = {k: [] for k in
                                ("train_loss", "val_loss", "val_acc", "val_precision", "val_recall", "val_f1", "lr")}
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch, best_val, history = ck["epoch"] + 1, ck["best_val_f1"], ck["history"]
        since_gain = ck.get("epochs_since_gain", 0)

    build_info_path = root / "build_info.json"
    corpus_build_info = json.loads(build_info_path.read_text(encoding="utf-8")) if build_info_path.is_file() else {}
    meta = {
        "run_id": run_id, "run_name": run_name, "arch": "gap_gru", "n_params": n_params,
        "hyp": hyp, "seed": seed, "corpus_dir": str(root), "corpus_build_info": corpus_build_info,
        "signbridge_commit": _git_commit(), "config_path": str(config_path),
    }
    _atomic_write_json(run_dir / "meta.json", meta)

    bar = tqdm(total=hyp["epochs"], initial=start_epoch, dynamic_ncols=True,
              desc=f"gapdetect/{run_name} · run {run_id}")
    if start_epoch:
        bar.write(f"{run_name}: resumed at epoch {start_epoch}, best f1 {best_val:.4f}")
    for epoch in range(start_epoch, hyp["epochs"]):
        model.train()
        tl = n = 0.0
        for k, batch in enumerate(train_loader):
            loss = step_batch(batch, True)
            tl, n = tl + loss, n + 1
            if k % 20 == 0:
                bar.set_postfix_str(f"ep{epoch + 1} train {k + 1}/{len(train_loader)} · loss {tl / n:.4f}",
                                    refresh=True)
        val = evaluate(model, val_loader, device)
        scheduler.step(val["f1"])
        for key, v in (("train_loss", tl / max(n, 1)), ("val_loss", val["loss"]), ("val_acc", val["acc"]),
                       ("val_precision", val["precision"]), ("val_recall", val["recall"]), ("val_f1", val["f1"]),
                       ("lr", optimizer.param_groups[0]["lr"])):
            history[key].append(v)
        is_best = val["f1"] > best_val
        since_gain = 0 if val["f1"] > best_val + hyp["es_min_delta"] else since_gain + 1
        best_val = max(best_val, val["f1"])
        early_stop = since_gain >= hyp["es_patience"]
        finished = early_stop or epoch + 1 >= hyp["epochs"]
        state = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict(), "best_val_f1": best_val, "history": history,
                 "hyp": hyp, "epochs_since_gain": since_gain, "finished": finished, "run_name": run_name}
        atomic_torch_save(state, last)
        if is_best:
            atomic_torch_save(state, best)
        _atomic_write_json(run_dir / "history.json", history)
        bar.set_postfix_str(f"val f1 {val['f1']:.4f} · prec {val['precision']:.3f} · rec {val['recall']:.3f} · "
                            f"acc {val['acc']:.4f} · best {best_val:.4f}{' *' if is_best else ''} · "
                            f"plateau {since_gain}/{hyp['es_patience']}")
        bar.update(1)
        if early_stop:
            bar.write(f"{run_name}: EARLY STOP at epoch {epoch + 1}")
            break
    bar.close()
    print(f"{run_name}: DONE best val gap-F1 {best_val:.4f} run_dir={run_dir}")
    return run_dir
