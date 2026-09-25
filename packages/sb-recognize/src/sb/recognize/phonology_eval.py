"""Which inputs a phonology front-end model relies on (TODO §3.9): permutation
importance per feature group, on the canonical validation split.

A trained ``gru_phono`` / ``gru_phono_raw`` / ``bilstm_phono`` run is loaded
from the registry, and the output of its :class:`PhonologyFrontend` is
intercepted with a forward hook. For one feature group at a time (e.g.
``r_handshape``, ``l_location``, ``raw_face``; ``PhonologyFrontend.groups``),
those columns are replaced by the same columns of **another clip in the
batch** (time index clipped to that clip's length), and the drop in
validation accuracy is recorded. A large drop means the model depends on that
group; ~0 means it is redundant given the rest. Model-agnostic, no training,
no gradient.

Groups are per hand as labeled by MediaPipe (``r_`` / ``l_``), not
dominant/non-dominant, because the model sees them that way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sb.core.subsets import get_subset
from sb.mlops import registry as R
from sb.recognize.architectures import PhonologyFrontend, build_model, frontend_of
from sb.recognize.features import base_v1 as FEAT
from sb.recognize.sources import get_source


def load_phono_run(run_dir: Path | str, device) -> tuple[torch.nn.Module, dict]:
    """``best.pt`` of a front-end run (fetched if absent) -> ``(model in eval mode, checkpoint)``."""
    from sb.mlops.artifacts import ensure_local

    ck = torch.load(ensure_local(Path(run_dir), R.CKPT_BEST), map_location=device, weights_only=False)
    ds = get_source("gislr")
    n_classes = len(ds.label_map(ds.resolve_dir()))
    model = build_model(ck["arch"], ck["feature_dim"], n_classes, ck["hyp"])
    model.load_state_dict(ck["model_state"])
    frontend_of(model)  # ValueError if the run has no phonology front-end
    return model.to(device).eval(), ck


def val_loader(ck: dict, batch_size: int = 1024) -> DataLoader:
    """The canonical validation split through the ``base_v1`` cache the run trained on."""
    ds = get_source("gislr")
    data_dir = ds.resolve_dir()
    sign2idx = ds.label_map(data_dir)
    _, val_split = ds.canonical_split(data_dir, sign2idx)
    subset = get_subset(ck["hyp"]["landmark_subset"])
    data, off = FEAT.build_cache(val_split, "val", subset, ck["coords"], data_dir)
    dset = FEAT.SubsetArrayDataset(val_split, data, off, ck["feature_dim"])
    return DataLoader(dset, batch_size=batch_size, shuffle=False, collate_fn=FEAT.collate_fn, num_workers=0)


@torch.no_grad()
def feature_group_importance(model, loader, n_repeats: int = 3, seed: int = 0, progress=None) -> pd.DataFrame:
    """Validation accuracy with each front-end feature group permuted across
    clips -> one row per group: ``acc``, ``drop`` (baseline − permuted, mean
    over ``n_repeats``), ``drop_std``, ``n_features``. The first row is the
    unpermuted baseline."""
    device = next(model.parameters()).device
    fe: PhonologyFrontend = frontend_of(model)
    batches = [(x.to(device), lengths, y.to(device)) for x, lengths, y in loader]
    state: dict = {"cols": None, "perm": None, "src_t": None}

    def hook(_m, _inp, out):
        if state["cols"] is None:
            return out
        donor = out[state["perm"][:, None], state["src_t"]]  # (B, T, D): another clip, same time index
        out = out.clone()
        out[..., state["cols"]] = donor[..., state["cols"]]
        return out

    handle = fe.register_forward_hook(hook)
    rng = torch.Generator().manual_seed(seed)

    def accuracy(cols=None) -> float:
        correct = total = 0
        for x, lengths, y in batches:
            if cols is not None:
                B, T = x.shape[:2]
                perm = torch.randperm(B, generator=rng)
                t = torch.arange(T)[None, :].expand(B, T)
                src_t = torch.minimum(t, (lengths[perm] - 1)[:, None])  # clip to the donor's length
                state.update(cols=torch.as_tensor(cols, device=device), perm=perm.to(device), src_t=src_t.to(device))
            logits = model(x, lengths)
            state["cols"] = None
            correct += int((logits.argmax(-1) == y).sum())
            total += len(y)
        return correct / total

    try:
        rows = [{"group": "(none)", "n_features": 0, "acc": accuracy(), "drop": 0.0, "drop_std": 0.0}]
        base = rows[0]["acc"]
        for i, (name, cols) in enumerate(fe.groups.items()):
            accs = [accuracy(cols) for _ in range(n_repeats)]
            rows.append({"group": name, "n_features": len(cols), "acc": float(np.mean(accs)),
                         "drop": base - float(np.mean(accs)), "drop_std": float(np.std(accs))})
            if progress is not None:
                progress(i + 1, len(fe.groups))
    finally:
        handle.remove()
    return pd.DataFrame(rows)
