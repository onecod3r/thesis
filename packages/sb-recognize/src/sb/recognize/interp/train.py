"""Generic, resumable training loop for the landmark-importance DNN/LSTM/GRU
(``gislr.1.models.landmark-importance.ipynb``, TODO §3).

Deliberately separate from ``sb.recognize.train`` (the canonical
config-driven, fixed-canonical-split, registry-writing driver): this track
trains on a rotating k-fold split, not the fixed split alone, and its runs are
not registry entries (see ``sb.recognize.interp``'s module docstring for why).
The per-epoch mechanics mirror ``sb.recognize.train`` on purpose — AdamW +
``ReduceLROnPlateau`` + early-stopping-on-plateau — so results are comparable
in kind, just not registry-comparable.

``LandmarkDNN`` has no recurrent state, so it is trained and evaluated at
**frame** granularity (every valid frame gets the video's label; loss and
frame-accuracy are length-masked) with a separate **video**-level accuracy
computed by averaging per-frame softmax probabilities over the sequence and
taking the argmax — that second number is what is comparable to the RNNs'
video-level accuracy. ``LandmarkRNN`` (GRU/LSTM) trains and evaluates exactly
like ``sb.recognize.architectures.StreamingGRU``: one label per sequence, read
out at the last valid frame.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from sb.recognize.features import base_v1 as FEAT
from sb.recognize.interp.features import FEATURE_DIM


def load_split_arrays(data_path: Path, off_path: Path, df) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one split's flat feature cache ONCE — every k-fold's train/val
    subset (and the final full-train fit) shares this in memory via
    :class:`FoldArrayDataset` instead of each fold re-reading/duplicating it."""
    data = np.load(data_path)
    offsets = np.load(off_path)
    labels = df["label"].to_numpy()
    assert len(labels) == len(offsets) - 1, "cache/split mismatch"
    return data, offsets, labels


class FoldArrayDataset(Dataset):
    """Same flat-cache read as ``base_v1.SubsetArrayDataset``, indexed by an
    explicit ``row_indices`` array so many k-fold subsets of one split can
    share one already-loaded ``(data, offsets, labels)`` triple.

    ``return_row=True`` additionally yields the original row index — needed
    for any caller that must scatter results back into a full-split array,
    since ``collate_fn`` sorts each batch by length and a non-shuffled
    ``DataLoader`` therefore does NOT yield rows in ``row_indices`` order.
    """

    def __init__(self, data, offsets, labels, row_indices, feature_dim: int,
                 max_seq_len: int = FEAT.MAX_SEQ_LEN, return_row: bool = False):
        self.data = data
        self.offsets = offsets
        self.labels = labels
        self.row_indices = np.asarray(row_indices)
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len
        self.return_row = return_row

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, i):
        row = int(self.row_indices[i])
        d = self.feature_dim
        arr = self.data[self.offsets[row] * d: self.offsets[row + 1] * d].reshape(-1, d)
        T = arr.shape[0]
        if T > self.max_seq_len:
            arr = arr[np.linspace(0, T - 1, self.max_seq_len).astype(int)]
            T = self.max_seq_len
        item = torch.from_numpy(np.ascontiguousarray(arr)), T, int(self.labels[row])
        return (*item, row) if self.return_row else item


def collate_with_row(batch):
    """Same padding/sort as ``base_v1.collate_fn``, plus the row-index column."""
    batch.sort(key=lambda x: x[1], reverse=True)
    feats, lengths, labels, rows = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    rows = torch.tensor(rows, dtype=torch.long)
    padded = torch.zeros(len(feats), int(lengths[0]), feats[0].shape[1])
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = f
    return padded, lengths, labels, rows


def make_fold_loader(
    data, offsets, labels, row_indices, batch_size: int, shuffle: bool, seed: int = 42,
    feature_dim: int = FEATURE_DIM,
) -> DataLoader:
    """``feature_dim`` defaults to ``landmark_interp_v1``'s 5,442 — pass a
    pipeline's own ``FEATURE_DIM`` (e.g. ``features_curated.FEATURE_DIM``,
    922) to load a different cache through the same loader.

    ``pin_memory=True`` + the ``non_blocking=True`` transfers in
    ``run_epoch_dnn``/``run_epoch_rnn`` let the H2D copy overlap with GPU
    compute — the same pair ``sb.recognize.train``'s production driver
    already uses. Neither changes a single computed value (CUDA's
    stream-ordering guarantees the copy finishes before it's read); this is
    a GPU-utilization fix, not a training-behavior change, so it's safe even
    for a notebook whose results are already reported."""
    ds = FoldArrayDataset(data, offsets, labels, row_indices, feature_dim=feature_dim)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=FEAT.collate_fn,
        num_workers=0, pin_memory=True, generator=g if shuffle else None,
    )


def make_row_tracked_loader(
    data, offsets, labels, row_indices, batch_size: int, feature_dim: int = FEATURE_DIM,
) -> DataLoader:
    """Never shuffled — used wherever a prediction must be scattered back into
    a full-split array by its original row index (OOF predictions, the final
    test-set pass). ``feature_dim`` as in :func:`make_fold_loader`."""
    ds = FoldArrayDataset(data, offsets, labels, row_indices, feature_dim=feature_dim, return_row=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_with_row,
                      num_workers=0, pin_memory=True)


def _frame_mask(lengths: torch.Tensor, T: int, device) -> torch.Tensor:
    return torch.arange(T, device=device)[None, :] < lengths.to(device)[:, None]


def run_epoch_dnn(model, loader, criterion, device, optimizer=None, grad_clip: float = 5.0):
    """One pass for :class:`~sb.recognize.interp.models.LandmarkDNN`.
    Returns ``(frame_loss, frame_acc, video_acc)``."""
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, frame_correct, frame_total = 0.0, 0, 0
    video_correct, video_total = 0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for feats, lengths, labels in loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B, T, _ = feats.shape
            mask = _frame_mask(lengths, T, device)  # (B, T)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
            logits = model(feats)  # (B, T, C)
            per_frame_loss = criterion(
                logits.reshape(B * T, -1), labels.repeat_interleave(T)
            ).reshape(B, T)
            loss = (per_frame_loss * mask).sum() / mask.sum()
            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            preds = logits.argmax(-1)
            frame_correct += ((preds == labels[:, None]) & mask).sum().item()
            frame_total += mask.sum().item()
            probs = torch.softmax(logits, dim=-1) * mask.unsqueeze(-1)
            video_pred = (probs.sum(1) / mask.sum(1, keepdim=True)).argmax(-1)
            video_correct += (video_pred == labels).sum().item()
            video_total += B
    return total_loss / frame_total, frame_correct / frame_total, video_correct / video_total


def run_epoch_rnn(model, loader, criterion, device, optimizer=None, scaler=None, grad_clip: float = 5.0):
    """One pass for :class:`~sb.recognize.interp.models.LandmarkRNN`.
    Returns ``(loss, video_acc)``."""
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = device.type == "cuda"
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for feats, lengths, labels in loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(feats, lengths)
                loss = criterion(logits, labels)
            if train_mode:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_fold(
    *, arch: str, model_fn, train_loader, val_loader, hyp: dict, device, ckpt_path: Path,
    bar=None, bar_prefix: str = "",
) -> tuple[torch.nn.Module, dict, float, dict | None]:
    """Train one (architecture, fold) unit to plateau/epoch-cap, resuming from
    ``ckpt_path`` if it exists. Returns ``(model, history, best_val_acc,
    best_state_dict)`` — ``model`` holds the LAST epoch's weights;
    ``best_state_dict`` is what every downstream use (OOF predictions, final
    landmark-importance analysis) should load.

    ``bar``: an optional caller-owned ``tqdm`` (one bar per architecture,
    spanning every fold — never created here) updated once per epoch via
    ``bar.update(1)``/``set_postfix_str``, matching the single-progress-bar
    convention the rest of the repo's training driver uses.
    """
    model = model_fn().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyp["lr"], weight_decay=hyp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=hyp["lr_factor"], patience=hyp["lr_patience"]
    )
    criterion_dnn = nn.CrossEntropyLoss(label_smoothing=0.1, reduction="none")
    criterion_rnn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # -1.0, not 0.0: a val accuracy of exactly 0.0 on epoch 1 (plausible with a
    # tiny/synthetic dataset) must still count as an improvement, or best_state
    # is never set and stays None forever.
    start_epoch, best_val_acc, epochs_since_gain = 0, -1.0, 0
    history: dict = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state = None
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = ck["epoch"] + 1
        best_val_acc = ck["best_val_acc"]
        history = ck["history"]
        epochs_since_gain = ck["epochs_since_gain"]
        best_state = ck["best_state"]
        if bar is not None:
            bar.update(start_epoch)
        if ck.get("finished"):
            return model, history, best_val_acc, best_state

    for epoch in range(start_epoch, hyp["epochs"]):
        if arch == "dnn":
            tr_loss, _, tr_acc = run_epoch_dnn(
                model, train_loader, criterion_dnn, device, optimizer, hyp["grad_clip"]
            )
            val_loss, _, val_acc = run_epoch_dnn(model, val_loader, criterion_dnn, device)
        else:
            tr_loss, tr_acc = run_epoch_rnn(
                model, train_loader, criterion_rnn, device, optimizer, scaler, hyp["grad_clip"]
            )
            val_loss, val_acc = run_epoch_rnn(model, val_loader, criterion_rnn, device)
        scheduler.step(val_acc)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        is_best = val_acc > best_val_acc
        epochs_since_gain = 0 if val_acc > best_val_acc + hyp["es_min_delta"] else epochs_since_gain + 1
        best_val_acc = max(best_val_acc, val_acc)
        if is_best:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        finished = epochs_since_gain >= hyp["es_patience"] or (epoch + 1 >= hyp["epochs"])

        state = dict(
            epoch=epoch, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(), best_val_acc=best_val_acc, history=history,
            epochs_since_gain=epochs_since_gain, best_state=best_state, finished=finished,
        )
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        tmp.replace(ckpt_path)
        if bar is not None:
            bar.set_postfix_str(
                f"{bar_prefix} ep{epoch + 1} · tr {tr_loss:.3f}/{tr_acc:.4f} "
                f"· val {val_loss:.3f}/{val_acc:.4f} · best {best_val_acc:.4f} "
                f"· plateau {epochs_since_gain}/{hyp['es_patience']}"
            )
            bar.update(1)
        if finished:
            break
    return model, history, best_val_acc, best_state


@torch.no_grad()
def predict_probs_indexed(
    model, arch: str, loader: DataLoader, device, desc: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Video-level predicted class-probability matrix, true labels, and the
    ORIGINAL row index each prediction belongs to — pass a loader built by
    :func:`make_row_tracked_loader` so results can be scattered back into a
    full-split array regardless of collate_fn's per-batch length-sort.

    ``desc``: pass a label to show a ``tqdm`` bar over batches (inference is
    fast enough that this is opt-in, not automatic — the callers that want it
    ask for it explicitly)."""
    model.eval()
    all_rows, all_probs, all_labels = [], [], []
    batches = tqdm(loader, desc=desc, leave=False) if desc else loader
    for feats, lengths, labels, rows in batches:
        feats = feats.to(device, non_blocking=True)
        if arch == "dnn":
            B, T, _ = feats.shape
            mask = _frame_mask(lengths, T, device)
            logits = model(feats)
            probs = torch.softmax(logits, dim=-1) * mask.unsqueeze(-1)
            probs = probs.sum(1) / mask.sum(1, keepdim=True)
        else:
            logits = model(feats, lengths)
            probs = torch.softmax(logits, dim=-1)
        all_rows.append(rows.numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_rows), np.concatenate(all_probs), np.concatenate(all_labels)
