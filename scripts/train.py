"""
Usage:
    python train.py --model lstm --epochs 30
    python train.py --model spectrogram --epochs 30

Dataset path resolves automatically via kagglehub's cached competition
download (asl-signs) unless you override --train_csv/--data_root explicitly.

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
from data_config import get_dataset_paths


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


def train_model(args):
    """Runs full training for one model config. Returns a dict with
    everything a benchmarking step needs: the trained model, dataset,
    val_loader, per-epoch history (including LR), and checkpoint path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.model}] using device: {device}")

    dataset = GISLRDataset(
        train_csv_path=args.train_csv,
        data_root=args.data_root,
        sign_map_json=args.sign_map_json,
        max_len=args.max_len,
        use_z=args.use_z,
    )

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, dataset.feature_dim, dataset.num_classes,
                         args.max_len, device)
    print(f"[{args.model}] feature_dim={dataset.feature_dim} "
          f"num_classes={dataset.num_classes} "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model}_best.pt")
    best_val_acc = 0.0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"[{args.model}] epoch {epoch}/{args.epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"lr={current_lr:.2e} ({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_map": dataset.label_map,
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)

    print(f"[{args.model}] best val_acc: {best_val_acc:.4f}")

    # Reload best checkpoint's weights before handing the model back, so
    # any benchmarking step evaluates the best epoch, not just the last one.
    best_state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])

    return {
        "model": model,
        "dataset": dataset,
        "val_loader": val_loader,
        "history": history,
        "best_val_acc": best_val_acc,
        "checkpoint_path": ckpt_path,
        "device": device,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--sign_map_json", default=None)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--use_z", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser


def resolve_paths(args):
    """Fills in --train_csv/--data_root/--sign_map_json from kagglehub's
    cached download if they weren't passed explicitly."""
    if args.train_csv is None or args.data_root is None:
        data_root, train_csv, sign_map_json = get_dataset_paths()
        args.data_root = args.data_root or data_root
        args.train_csv = args.train_csv or train_csv
        args.sign_map_json = args.sign_map_json or sign_map_json
    return args


def main():
    args = build_arg_parser().parse_args()
    args = resolve_paths(args)

    result = train_model(args)

    # Run the full benchmark suite (confusion matrix, precision, saliency,
    # learning curve) right after training, matching what run_all.py does.
    from benchmark import run_full_benchmark
    run_full_benchmark(args.model, result, args)


if __name__ == "__main__":
    main()
