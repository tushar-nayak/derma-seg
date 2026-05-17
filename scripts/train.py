import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import MedicalDataset
from src.data.lumiere_dataset import LumiereSliceDataset, build_lumiere_splits, discover_lumiere_samples
from src.models import get_model
from src.utils.losses import CombinedSegmentationLoss
from src.utils.metrics import dice_coeff, iou_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_logits(outputs):
    if isinstance(outputs, (list, tuple)):
        return outputs[-1]
    return outputs


def make_png_loaders(args):
    dataset = MedicalDataset(args.data_dir, image_size=args.image_size)
    train_size = int(round(0.8 * len(dataset)))
    val_size = max(1, len(dataset) - train_size)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    test_dataset = val_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def make_lumiere_loaders(args):
    train_samples, val_samples, test_samples = build_lumiere_splits(
        args.data_root,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        positive_only=args.positive_only,
    )

    train_dataset = LumiereSliceDataset(train_samples, image_size=args.image_size, augment=True)
    val_dataset = LumiereSliceDataset(val_samples, image_size=args.image_size, augment=False)
    test_dataset = LumiereSliceDataset(test_samples, image_size=args.image_size, augment=False)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = unwrap_logits(model(images))
            loss = criterion(outputs, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        outputs = unwrap_logits(model(images))

        total_loss += criterion(outputs, masks).item()
        total_dice += dice_coeff(outputs, masks).item()
        total_iou += iou_score(outputs, masks).item()

    denom = max(1, len(loader))
    return {
        "loss": total_loss / denom,
        "dice": total_dice / denom,
        "iou": total_iou / denom,
    }


def save_metrics(path, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dataset == "lumiere":
        train_loader, val_loader, test_loader = make_lumiere_loaders(args)
    elif args.dataset == "png":
        train_loader, val_loader, test_loader = make_png_loaders(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    model = get_model(args.model, n_channels=1, n_classes=2, img_size=args.image_size).to(device)
    criterion = CombinedSegmentationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" and args.amp else None

    run_dir = Path(args.output_dir) / args.dataset / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    best_val_dice = -1.0
    history = []
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        torch.save({"model_state": model.state_dict(), "args": vars(args), "epoch": epoch}, last_path)

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save({"model_state": model.state_dict(), "args": vars(args), "epoch": epoch}, best_path)
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"Test | loss={test_metrics['loss']:.4f} | "
        f"dice={test_metrics['dice']:.4f} | iou={test_metrics['iou']:.4f}"
    )

    results = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "history": history,
        "best_val_dice": best_val_dice,
        "test": test_metrics,
    }
    save_metrics(run_dir / "metrics.json", results)
    print(f"Saved checkpoint and metrics to {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lumiere", choices=["lumiere", "png"])
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/sofa/host_dir/hub/glioblastoma-evolution/data/lumiere",
        help="Root of the original LUMIERE dataset.",
    )
    parser.add_argument("--data_dir", type=str, default="data/lumiere_slices")
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--positive_only", action="store_true", default=True)
    parser.add_argument("--no_positive_only", action="store_false", dest="positive_only")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
