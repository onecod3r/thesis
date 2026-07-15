"""Train StreamingGRU on the ME-126 landmark subset (GISLR).

ME-126 = the 1st-place Kaggle subset (LIP 40 + LHAND 21 + RHAND 21 + NOSE 4 +
REYE 16 + LEYE 16 = 118) + upper-body pose {11,12,13,14,15,16,23,24}
(shoulders, elbows, wrists, hips) motivated by the motion-energy analysis
(docs/2026-07-15.md). Channels stay xyz so the ONLY variable changed vs the
20260713-213000 baseline is the landmark subset. All hyperparameters mirror
that baseline checkpoint's hyp: batch 192, lr 3e-4, hidden 256, 2 layers,
dropout 0.3, wd 1e-4, 60 epochs, max_seq_len 128, grad_clip 5, OneCycleLR,
CE + label smoothing 0.1, AMP.

Run with CWD = src/. Resumable: re-run to continue from gru_latest.pt.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

# ---------------- landmark subset ----------------
ROWS_PER_FRAME = 543
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17,
       314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88,
       178, 87, 14, 317, 402, 318, 324, 308]
NOSE = [1, 2, 98, 327]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POSE_UPPER = [489 + i for i in (11, 12, 13, 14, 15, 16, 23, 24)]  # shoulders/elbows/wrists/hips
LANDMARKS = np.array(sorted(LIP + NOSE + REYE + LEYE + LHAND + POSE_UPPER + RHAND))
assert len(LANDMARKS) == 126 == len(set(LANDMARKS))
N_LM = len(LANDMARKS)
FEATURE_DIM = N_LM * 3

HYP = dict(batch_size=192, lr=3e-4, hidden_size=256, num_layers=2, dropout=0.3,
           weight_decay=1e-4, epochs=60, max_seq_len=128, grad_clip=5.0, seed=42)

SUBSET_NAME = "me126"
CACHE_DIR = Path("cache")
RESUME_DIR_FILE = CACHE_DIR / f"{SUBSET_NAME}_current_run.txt"

torch.manual_seed(HYP["seed"])
np.random.seed(HYP["seed"])
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")

# ---------------- data ----------------
DATA_DIR = Path(kagglehub.competition_download("asl-signs"))
sign2idx = json.loads((DATA_DIR / "sign_to_prediction_index_map.json").read_text())
NUM_CLASSES = len(sign2idx)
train_df = pd.read_csv(DATA_DIR / "train.csv")
train_df["label"] = train_df["sign"].map(sign2idx)
train_split, val_split = train_test_split(train_df, test_size=0.1,
                                          stratify=train_df["sign"], random_state=42)
train_split = train_split.reset_index(drop=True)
val_split = val_split.reset_index(drop=True)
print(f"train {len(train_split)} / val {len(val_split)}", flush=True)


def load_subset(path):
    table = pq.read_table(path, columns=["x", "y", "z"])
    data = np.column_stack([table.column(c).to_numpy() for c in ("x", "y", "z")])
    n = data.shape[0] // ROWS_PER_FRAME
    arr = data.reshape(n, ROWS_PER_FRAME, 3)[:, LANDMARKS, :].astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_cache(df, prefix):
    data_path = CACHE_DIR / f"{prefix}_{SUBSET_NAME}_data.npy"
    off_path = CACHE_DIR / f"{prefix}_{SUBSET_NAME}_offsets.npy"
    if data_path.exists() and off_path.exists():
        print(f"{prefix}: cache exists, skipping", flush=True)
        return data_path, off_path
    t0 = time.time()
    paths = [DATA_DIR / p for p in df["path"]]
    chunks, offsets = [], [0]
    with ThreadPoolExecutor(12) as ex:
        for i, arr in enumerate(ex.map(load_subset, paths)):
            chunks.append(arr.reshape(-1))
            offsets.append(offsets[-1] + arr.shape[0])
            if i % 5000 == 0:
                print(f"  {prefix} {i}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
    flat = np.concatenate(chunks)
    np.save(data_path, flat)
    np.save(off_path, np.array(offsets, dtype=np.int64))
    print(f"{prefix}: cached {len(df)} videos, {flat.nbytes/1e9:.2f} GB "
          f"({time.time()-t0:.0f}s)", flush=True)
    return data_path, off_path


class SubsetDataset(Dataset):
    """In-RAM flat array + offsets; uniform subsample >max_seq_len (as baseline)."""

    def __init__(self, df, data_path, off_path, max_seq_len):
        self.labels = df["label"].to_numpy()
        self.data = np.load(data_path)          # fully in RAM (~5 GB train)
        self.offsets = np.load(off_path)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        stride = FEATURE_DIM
        arr = self.data[self.offsets[i]*stride:self.offsets[i+1]*stride].reshape(-1, stride)
        T = arr.shape[0]
        if T > self.max_seq_len:
            arr = arr[np.linspace(0, T - 1, self.max_seq_len).astype(int)]
            T = self.max_seq_len
        return torch.from_numpy(np.ascontiguousarray(arr)), T, int(self.labels[i])


def collate_fn(batch):
    batch.sort(key=lambda x: x[1], reverse=True)
    feats, lengths, labels = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    padded = torch.zeros(len(feats), int(lengths[0]), feats[0].shape[1])
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = f
    return padded, lengths, labels


# ---------------- model (identical to baseline) ----------------
class StreamingGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0, bidirectional=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout),
                                  nn.Linear(hidden_size, num_classes))

    def forward(self, x, lengths):
        x = self.input_norm(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1)).to(out.device)
        return self.head(out.gather(1, idx).squeeze(1))


def main():
    tr_data, tr_off = build_cache(train_split, "train")
    va_data, va_off = build_cache(val_split, "val")

    train_ds = SubsetDataset(train_split, tr_data, tr_off, HYP["max_seq_len"])
    val_ds = SubsetDataset(val_split, va_data, va_off, HYP["max_seq_len"])
    g = torch.Generator(); g.manual_seed(HYP["seed"])
    train_loader = DataLoader(train_ds, batch_size=HYP["batch_size"], shuffle=True,
                              collate_fn=collate_fn, num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=HYP["batch_size"], shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    model = StreamingGRU(FEATURE_DIM, HYP["hidden_size"], HYP["num_layers"],
                         NUM_CLASSES, HYP["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input dim {FEATURE_DIM} · params {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=HYP["lr"],
                                  weight_decay=HYP["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=HYP["lr"], epochs=HYP["epochs"],
        steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")

    # run dir — reuse if resuming, else new timestamp
    if RESUME_DIR_FILE.exists() and (Path(RESUME_DIR_FILE.read_text().strip()) / "gru_latest.pt").exists():
        run_dir = Path(RESUME_DIR_FILE.read_text().strip())
    else:
        run_dir = Path("models/gislr/gru") / datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        RESUME_DIR_FILE.write_text(str(run_dir))
    (run_dir / "cache").mkdir(exist_ok=True)
    (run_dir / "assets").mkdir(exist_ok=True)
    np.save(run_dir / "cache" / "landmarks.npy", LANDMARKS)
    print(f"run dir: {run_dir}", flush=True)

    latest, best = run_dir / "gru_latest.pt", run_dir / "gru_best.pt"
    start_epoch, best_val_acc = 0, 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    if latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch, best_val_acc, history = ck["epoch"] + 1, ck["best_val_acc"], ck["history"]
        print(f"resumed at epoch {start_epoch}, best {best_val_acc:.4f}", flush=True)

    def run_epoch(loader, train_mode):
        model.train() if train_mode else model.eval()
        total_loss, correct, total, lrs = 0.0, 0, 0, []
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            for feats, lengths, labels in loader:
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), HYP["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    lrs.append(scheduler.get_last_lr()[0])
                total_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / total, correct / total, lrs

    t_start = time.time()
    for epoch in range(start_epoch, HYP["epochs"]):
        tr_loss, tr_acc, lrs = run_epoch(train_loader, True)
        val_loss, val_acc, _ = run_epoch(val_loader, False)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        history["lr"].extend(lrs)
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        state = {"epoch": epoch, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict(),
                 "best_val_acc": best_val_acc, "history": history,
                 "sign2idx": sign2idx, "hyp": {**HYP, "num_workers": 0},
                 "feature_dim": FEATURE_DIM, "landmarks": LANDMARKS.tolist(),
                 "subset_name": SUBSET_NAME}
        torch.save(state, latest)
        if is_best:
            torch.save(state, best)
        print(f"epoch {epoch+1:03d}/{HYP['epochs']} | tr_loss {tr_loss:.4f} "
              f"acc {tr_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}"
              f"{' *BEST*' if is_best else ''} | {(time.time()-t_start)/60:.1f} min",
              flush=True)

    # learning curves (same 3-panel layout as baseline)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    axes[2].plot(history["lr"]); axes[2].set_title("LR schedule")
    fig.tight_layout()
    fig.savefig(run_dir / "assets" / "learning_loss_accuracy.png", dpi=110)
    print(f"DONE best_val_acc={best_val_acc:.4f} run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
