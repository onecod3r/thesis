"""Training driver for continuous-signing models (TODO §12.3).

One call = one registry run of ``configs/gislr.continuous.json``:

    from sb.recognize.continuous.train import train_continuous
    run_dir = train_continuous("C1")

Same run mechanics as :func:`sb.recognize.train.train_run`: one progress
bar, ``last.pt`` every epoch (atomic) with auto-resume through a pointer
file, ``best.pt`` on improvement, ``meta.json`` rewritten every epoch with
the same schema. The checkpoint carries the same keys ``sb-evaluate`` reads,
so the canonical isolated evaluation works on these runs as-is.

What differs is the unit of training: every epoch is a fresh random
partition of the ``train.csv`` clips into composed streams
(:mod:`sb.recognize.continuous.data`), trained with per-frame targets:

- ``loss: "frame"`` -- cross-entropy on every frame over glosses + null
  (sign frames weighted from ``ramp_start`` up to 1 across the sign, since a
  sign's first frames are ambiguous; null frames at ``null_weight``), plus
  ``boundary_weight`` x BCE on the sign-boundary head (positives weighted by
  the batch's negative/positive ratio).
- ``loss: "ctc"`` -- CTC over the gloss sequence with null as the blank; no
  alignment used; the boundary head is not trained.

Checkpoint selection and early stopping watch **segment accuracy** on
validation streams built once from a sign-stratified 5% of ``train.csv``:
the gloss with the highest non-null-weighted mean probability over each
true segment. ``test.csv`` (and GISLR-Sentences, built from it) is never used
for selection.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from sb.core.paths import EXPERIMENTS_DIR
from sb.core.subsets import get_subset
from sb.mlops import registry as R
from sb.mlops import run as P
from sb.recognize import data as D
from sb.recognize.architectures import ARCHS, ContinuousRNN, build_model
from sb.recognize.continuous import data as CD
from sb.recognize.continuous import phono as CPH
from sb.recognize.features import phono130_v1
from sb.recognize.features import cache
from sb.recognize.sources import get_source
from sb.recognize.train import _atomic_write_json, _build_meta, _is_finished, atomic_torch_save

CONFIG_PATH = EXPERIMENTS_DIR / "recognition" / "configs" / "gislr.continuous.json"
BOUNDARY_TOLERANCE = 3  # frames, for the validation boundary F1


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pointer_key(cfg: dict, run_name: str) -> str:
    """The registry pointer-file key of one run -- how notebooks find it from disk."""
    return f"{cfg['dataset']}_{cfg['runs'][run_name]['arch']}_{D.subset_tag(cfg['subset'], cfg['coords'])}_{run_name}"


def run_dir_for(run_name: str, config_path: Path | str = CONFIG_PATH) -> Path | None:
    """The run folder a config run currently points at (None before it has
    been started)."""
    return R.pointer_run_dir(pointer_key(load_config(config_path), run_name))


def load_run(run_dir: Path, device) -> tuple[ContinuousRNN, dict]:
    """``best.pt`` of a continuous run (fetched from the artifact remote if it
    is not on disk) -> ``(model in eval mode, checkpoint dict)``."""
    from sb.mlops.artifacts import ensure_local

    ck = torch.load(ensure_local(run_dir, R.CKPT_BEST), map_location=device, weights_only=False)
    model = cast(ContinuousRNN, build_model(ck["arch"], ck["feature_dim"], len(ck["sign2idx"]), ck["hyp"]))
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval(), ck


def run_hyp(cfg: dict, run_name: str) -> dict:
    """``shared`` + the run's ``overrides``; an override naming a key that is
    not in ``shared`` is an error (the repo-wide all-else-equal rule)."""
    run = cfg["runs"][run_name]
    over = run.get("overrides", {})
    unknown = sorted(set(over) - set(cfg["shared"]))
    if unknown:
        raise KeyError(f"run {run_name!r} overrides keys not in `shared`: {unknown}")
    return {**cfg["shared"], **over}


# ---------------------------------------------------------------------------
# losses and validation
# ---------------------------------------------------------------------------

def _log_probs(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """log-softmax with masked (-inf) classes made finite, and the
    allowed-class mask. Keeps label smoothing off the masked rows."""
    logits = logits.float()
    allowed = torch.isfinite(logits[(0,) * (logits.dim() - 1)])
    return F.log_softmax(logits.masked_fill(~allowed, -1e4), dim=-1), allowed


def frame_loss(gloss_logits, boundary_logits, batch, hyp) -> tuple[torch.Tensor, dict]:
    logp, allowed = _log_probs(gloss_logits)
    y, w, mask = batch["y"], batch["w"], batch["mask"]
    nll = -logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    smooth = -(logp * allowed).sum(-1) / allowed.sum()
    eps = hyp["label_smoothing"]
    ce = (1 - eps) * nll + eps * smooth
    wm = w * mask
    ce_loss = (ce * wm).sum() / wm.sum().clamp_min(1)
    b = batch["b"]
    pos = (b * mask).sum()
    neg = ((1 - b) * mask).sum()
    pos_weight = (neg / pos.clamp_min(1)).clamp(max=20.0)
    bce = F.binary_cross_entropy_with_logits(boundary_logits.float(), b, pos_weight=pos_weight, reduction="none")
    b_loss = (bce * mask).sum() / mask.sum().clamp_min(1)
    loss = ce_loss + hyp["boundary_weight"] * b_loss
    return loss, {"ce": ce_loss.item(), "boundary": b_loss.item()}


def ctc_loss(gloss_logits, batch, null_index: int) -> tuple[torch.Tensor, dict]:
    logp, _ = _log_probs(gloss_logits)  # (B, T, C+1)
    targets = torch.cat([torch.as_tensor(lab) for lab in batch["labels"]]).to(logp.device)
    tlen = torch.tensor([len(lab) for lab in batch["labels"]])
    loss = F.ctc_loss(logp.transpose(0, 1), targets, batch["lengths"], tlen,
                      blank=null_index, reduction="mean", zero_infinity=True)
    return loss, {"ctc": loss.item()}


def _boundary_f1(probs: np.ndarray, ends: np.ndarray, tol: int) -> tuple[int, int, int]:
    """(true positives, predicted, actual) for upward crossings of 0.5
    against true sign-end frames, greedy matching within ``tol``."""
    above = probs >= 0.5
    pred = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    used, tp = set(), 0
    for e in ends:
        cand = [p for p in pred if abs(p - e) <= tol and p not in used]
        if cand:
            used.add(min(cand, key=lambda p: abs(p - e)))
            tp += 1
    return tp, len(pred), len(ends)


@torch.no_grad()
def evaluate_val(model, val_batches, hyp, null_index: int, device) -> dict:
    model.eval()
    tot_loss = n_batches = 0.0
    fc = fn = 0
    sc = sn = 0
    tp = npred = nact = 0
    for batch in val_batches:
        bt = {k: v.to(device) if torch.is_tensor(v) and k != "lengths" else v for k, v in batch.items()}
        with torch.amp.autocast("cuda"):
            gl, bl = model.forward_frames(bt["x"])
        loss, _ = (ctc_loss(gl, bt, null_index) if hyp["loss"] == "ctc"
                   else frame_loss(gl, bl, bt, hyp))
        tot_loss += float(loss)
        n_batches += 1
        probs = torch.softmax(gl.float(), -1).cpu().numpy()
        bprob = torch.sigmoid(bl.float()).cpu().numpy()
        y = batch["y"].numpy()
        m = batch["mask"].numpy()
        fc += int(((probs.argmax(-1) == y) & m).sum())
        fn += int(m.sum())
        for i, (seg, labs) in enumerate(zip(batch["segments"], batch["labels"])):
            for (s, e), lab in zip(seg, labs):
                p = probs[i, s:e]
                score = ((1 - p[:, null_index:null_index + 1]) * p[:, :null_index]).sum(0)
                sc += int(score.argmax() == lab)
                sn += 1
            t, pr, ac = _boundary_f1(bprob[i, : int(batch["lengths"][i])], seg[:, 1] - 1, BOUNDARY_TOLERANCE)
            tp, npred, nact = tp + t, npred + pr, nact + ac
    prec, rec = tp / max(npred, 1), tp / max(nact, 1)
    return {"loss": tot_loss / max(n_batches, 1), "frame_acc": fc / max(fn, 1),
            "seg_acc": sc / max(sn, 1), "boundary_f1": 2 * prec * rec / max(prec + rec, 1e-9)}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _streams(seq_batch, bank, labels, null_index, lay, ccfg, rng, compose=CD.compose):
    out = []
    for seq in seq_batch:
        s = compose([bank.clip(c) for c in seq], [int(labels[c]) for c in seq],
                    null_index, lay, ccfg, rng)
        w = np.where(s["y"] == null_index, ccfg["null_weight"], 1.0).astype(np.float32)
        for a, b in s["segments"]:
            w[a:b] = np.linspace(ccfg["ramp_start"], 1.0, b - a, dtype=np.float32)
        s["w"] = w
        out.append(s)
    batch = CD.collate(out)
    T = batch["x"].shape[1]
    wpad = np.zeros((len(out), T), np.float32)
    for i, s in enumerate(out):
        wpad[i, : len(s["w"])] = s["w"]
    batch["w"] = torch.from_numpy(wpad)
    return batch


def train_continuous(run_name: str, config_path: Path | str = CONFIG_PATH, *,
                     smoke: int = 0) -> Path | dict:
    """Train one run of the continuous config; returns its run folder.

    ``smoke=N`` runs N training batches and a validation pass over 2 batches,
    writes nothing (no registry folder, no pointer) and returns the metrics --
    a wiring check, not training.
    """
    assert torch.cuda.is_available(), "training requires the CUDA build of torch (uv sync)"
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    cfg = load_config(config_path)
    run = cfg["runs"][run_name]
    hyp = {**run_hyp(cfg, run_name), "landmark_subset": cfg["subset"]}  # for front-end models (gru_continuous_phono)
    arch, seed = run["arch"], cfg["seed"]
    ccfg = {**cfg["composer"], "null_weight": hyp["null_weight"], "ramp_start": hyp["ramp_start"]}

    ds = get_source(cfg["dataset"])
    data_dir = ds.resolve_dir()
    sign2idx = ds.label_map(data_dir)
    n_classes = len(sign2idx)
    null_index = n_classes
    subset = get_subset(cfg["subset"])
    coords = cfg["coords"]
    feature_dim = len(subset) * len(coords)
    tag = D.subset_tag(cfg["subset"], coords)
    train_split, _ = ds.canonical_split(data_dir, sign2idx)

    # phonology arms read the phono130_v1 cache and compose streams in feature space
    phono = ARCHS[arch].pipeline == phono130_v1.PIPELINE
    compose = CPH.compose if phono else CD.compose
    if phono:
        full = get_subset("FULL_543")
        dp, op = phono130_v1.build_cache(train_split, "train", full, "xyz", data_dir)
        feature_dim = phono130_v1.feature_dim()
        bank = CD.ClipBank(np.load(dp).reshape(-1, feature_dim), np.load(op))
        pipeline, cache_key = phono130_v1.PIPELINE, phono130_v1.cache_key(full, "xyz", data_dir)
    else:
        bar0 = tqdm(total=len(train_split), desc="clip bank (train.csv)", leave=False)
        bank = CD.ClipBank.build(train_split, "train", subset, coords, data_dir,
                                 progress=lambda d, t: bar0.update(d - bar0.n))
        bar0.close()
        pipeline, cache_key = CD.PIPELINE, cache.cache_key(CD.ClipBank.inputs(subset, coords, data_dir))
    labels = train_split["label"].to_numpy()
    participants = train_split["participant_id"].to_numpy()
    lengths = np.diff(bank.offsets)
    tr_idx, va_idx = CD.train_val_indices(train_split, cfg["val_fraction"], seed)
    held = CD.holdout_glosses(train_split["sign"], run.get("n_holdout_glosses", 0), seed)
    if held:
        keep = ~train_split["sign"].isin(held).to_numpy()
        tr_idx, va_idx = tr_idx[keep[tr_idx]], va_idx[keep[va_idx]]
    held_idx = sorted(sign2idx[g] for g in held)
    lay = None if phono else CD.Layout.of(subset.array, coords)

    vrng = np.random.default_rng([seed, 1])
    val_seqs = CD.epoch_sequences(va_idx, participants, lengths, ccfg, vrng)
    val_batches = [_streams(b, bank, labels, null_index, lay, ccfg, vrng, compose)
                   for b in CD.batches(val_seqs, lengths, hyp["batch_size"], vrng)]

    torch.manual_seed(seed)
    model = cast(ContinuousRNN, build_model(arch, feature_dim, n_classes, hyp).to(device))
    if held_idx:
        model.head.class_mask[held_idx] = False
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyp["lr"], weight_decay=hyp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=hyp["lr_factor"], patience=hyp["lr_patience"])
    scaler = torch.amp.GradScaler("cuda")

    def step_batch(batch, train: bool):
        bt = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) and k != "lengths" else v
              for k, v in batch.items()}
        with torch.amp.autocast("cuda"):
            gl, bl = model.forward_frames(bt["x"])
        loss, parts = (ctc_loss(gl, bt, null_index) if hyp["loss"] == "ctc"
                       else frame_loss(gl, bl, bt, hyp))
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        return loss.item(), parts

    if smoke:
        model.train()
        rng = np.random.default_rng([seed, 0])
        seqs = CD.epoch_sequences(tr_idx, participants, lengths, ccfg, rng)
        losses = [step_batch(_streams(b, bank, labels, null_index, lay, ccfg, rng, compose), True)
                  for b in CD.batches(seqs, lengths, hyp["batch_size"], rng)[:smoke]]
        val = evaluate_val(model, val_batches[:2], hyp, null_index, device)
        return {"train_losses": losses, "val": val, "n_train_clips": len(tr_idx),
                "n_val_clips": len(va_idx), "n_train_sequences": len(seqs),
                "n_val_sequences": len(val_seqs), "held_out": held, "n_params": n_params}

    prov = P.build(dataset=cfg["dataset"], data_dir=data_dir, config_path=config_path, config_obj=cfg,
                   feature_pipeline=pipeline, feature_cache_key=cache_key,
                   kaggle_ref=ds.kaggle_ref, manifest=ds.manifest, n_videos=len(train_split))
    P.warn_if_dirty(prov, label=f"{cfg['dataset']}/{arch}/{tag}/{run_name}")

    run_dir = R.resolve_run_dir(pointer_key(cfg, run_name), _is_finished)
    np.save(run_dir / "assets" / "landmarks.npy", subset.array)
    last, best = run_dir / R.CKPT_LAST, run_dir / R.CKPT_BEST
    start_epoch, best_val, since_gain, wall_min = 0, 0.0, 0, 0.0
    history: dict[str, list] = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc",
                                                "val_frame_acc", "val_boundary_f1", "lr")}
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch, best_val, history = ck["epoch"] + 1, ck["best_val_acc"], ck["history"]
        since_gain, wall_min = ck.get("epochs_since_gain", 0), ck.get("wall_time_min", 0.0)

    notes = (f"{run_name}: {run.get('notes', '')} Continuous streams composed from train.csv "
             f"({len(tr_idx)} clips; {len(va_idx)} held out for selection, seed {seed}); "
             f"held-out glosses: {held or 'none'}. Selection metric = segment accuracy on "
             f"validation streams, not the canonical isolated split.")
    meta_kw = dict(run_dir=run_dir, dataset=cfg["dataset"], arch=arch, subset=subset, coords=coords,
                   feature_dim=feature_dim, n_params=n_params, n_classes=n_classes, hyp=hyp,
                   provenance=prov, regime=cfg["regime"], source=cfg["source"], notes=notes)

    bar = tqdm(total=hyp["epochs"], initial=start_epoch, dynamic_ncols=True,
               desc=f"{cfg['dataset']}/{arch}/{tag}/{run_name} · run {run_dir.name}")
    if start_epoch:
        bar.write(f"{run_name}: resumed at epoch {start_epoch}, best {best_val:.4f}")
    t0 = time.time()
    for epoch in range(start_epoch, hyp["epochs"]):
        model.train()
        rng = np.random.default_rng([seed, 0, epoch])  # epoch-seeded: a resume redraws the same epoch
        seqs = CD.epoch_sequences(tr_idx, participants, lengths, ccfg, rng)
        plan = CD.batches(seqs, lengths, hyp["batch_size"], rng)
        tl = n = 0.0
        for k, b in enumerate(plan):
            loss, parts = step_batch(_streams(b, bank, labels, null_index, lay, ccfg, rng, compose), True)
            tl, n = tl + loss, n + 1
            if k % 20 == 0 or k == len(plan) - 1:
                bar.set_postfix_str(f"ep{epoch + 1} train {k + 1}/{len(plan)} · loss {tl / n:.4f} · "
                                    + " · ".join(f"{a} {v:.3f}" for a, v in parts.items()), refresh=True)
        val = evaluate_val(model, val_batches, hyp, null_index, device)
        scheduler.step(val["seg_acc"])
        for key, v in (("train_loss", tl / max(n, 1)), ("train_acc", float("nan")),
                       ("val_loss", val["loss"]), ("val_acc", val["seg_acc"]),
                       ("val_frame_acc", val["frame_acc"]), ("val_boundary_f1", val["boundary_f1"]),
                       ("lr", optimizer.param_groups[0]["lr"])):
            history[key].append(v)
        is_best = val["seg_acc"] > best_val
        since_gain = 0 if val["seg_acc"] > best_val + hyp["es_min_delta"] else since_gain + 1
        best_val = max(best_val, val["seg_acc"])
        early_stop = since_gain >= hyp["es_patience"]
        finished = early_stop or epoch + 1 >= hyp["epochs"]
        wall_now = wall_min + (time.time() - t0) / 60
        state = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val, "history": history, "sign2idx": sign2idx,
            "hyp": {**hyp, "seed": seed, "max_seq_len": None, "num_workers": 0},
            "feature_dim": feature_dim, "features": pipeline, "landmarks": subset.array.tolist(),
            "subset_name": cfg["subset"], "coords": coords, "arch": arch,
            "training_regime": cfg["regime"], "epochs_since_gain": since_gain,
            "finished": finished, "wall_time_min": wall_now,
            "run_name": run_name, "null_index": null_index, "held_out_glosses": held,
            "composer": cfg["composer"],
        }
        atomic_torch_save(state, last)
        if is_best:
            atomic_torch_save(state, best)
        _atomic_write_json(run_dir / "assets" / "history.json", history)
        meta = _build_meta(**meta_kw, history=history, best_val_acc=best_val, epochs_done=epoch + 1,
                           early_stopped=early_stop, finished=finished, wall_time_min=wall_now)
        meta["hyperparameters"].update({
            "seed": seed, "max_seq_len": None,
            "loss": ("CTC over glosses, blank = null" if hyp["loss"] == "ctc" else
                     "per-frame CE over glosses+null (label smoothing, ramped sign weights) "
                     "+ boundary BCE"),
            "composer": cfg["composer"], "val_fraction": cfg["val_fraction"],
            "held_out_glosses": held,
        })
        R.write_meta(run_dir, meta)
        bar.set_postfix_str(f"val seg {val['seg_acc']:.4f} · frame {val['frame_acc']:.4f} · "
                            f"bF1 {val['boundary_f1']:.3f} · best {best_val:.4f}{' *' if is_best else ''} · "
                            f"lr {history['lr'][-1]:.1e} · plateau {since_gain}/{hyp['es_patience']}")
        bar.update(1)
        if early_stop:
            bar.write(f"{run_name}: EARLY STOP at epoch {epoch + 1}")
            break
    bar.close()
    print(f"{run_name}: DONE best val segment accuracy {best_val:.4f} run_dir={run_dir}")
    return run_dir
