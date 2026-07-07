"""
Usage:
    python train.py --model lstm --train_csv data/train.csv --data_root data/ --epochs 30
    python train.py --model spectrogram --train_csv data/train.csv --data_root data/ --epochs 30

--model options: lstm, bilstm, gru, cnn1d, transformer, spectrogram
"""
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from gislr_dataset import GISLRDataset
from models import MODEL_REGISTRY


def build_model(name, feature_dim, num_classes, max_len, device):
    if name == "transformer":
        model = MODEL_REGISTRY[name](feature_dim, num_classes, max_len=max_len)
    else:
        model = MODEL_REGISTRY[name](feature_dim, num_classes)
    return model.to(device)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sign_map_json", default=None)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--use_z", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = GISLRDataset(
        train_csv_path=args.train_csv,
        data_root=args.data_root,
        sign_map_json=args.sign_map_json,
        max_len=args.max_len,
        use_z=args.use_z,
    )

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, dataset.feature_dim, dataset.num_classes,
                         args.max_len, device)
    print(f"Model: {args.model} | feature_dim={dataset.feature_dim} "
          f"num_classes={dataset.num_classes} | params={sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        elapsed = time.time() - start

        print(f"[{args.model}] epoch {epoch}/{args.epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model}_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_map": dataset.label_map,
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)

    print(f"Best val_acc for {args.model}: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
