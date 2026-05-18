import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.isic_dataset import ISICSegmentationDataset, _official_dir_pairs
from src.models import get_model
from src.utils.metrics import iou_score


def unwrap_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    if isinstance(outputs, (list, tuple)):
        return outputs[-1]
    return outputs


def normalize_batch(images, pretrained):
    if not pretrained or images.shape[1] != 3:
        return images
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def overlay_mask(image, mask, color):
    image = image.copy()
    alpha = 0.45
    image[mask > 0] = (1 - alpha) * image[mask > 0] + alpha * np.array(color, dtype=np.float32)
    return np.clip(image, 0.0, 1.0)


def export_panels(args):
    data_dir = Path(args.data_dir)
    pairs = _official_dir_pairs(data_dir)
    test_dataset = ISICSegmentationDataset(
        data_dir,
        image_size=args.image_size,
        augment=False,
        images_dir=pairs["test"][0],
        masks_dir=pairs["test"][1],
    )
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    model = get_model(
        train_args.get("model", args.model),
        n_channels=3,
        n_classes=2,
        img_size=train_args.get("image_size", args.image_size),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    with torch.no_grad():
        for index, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)
            logits = unwrap_logits(model(normalize_batch(images, pretrained=args.pretrained)))
            pred = logits.argmax(dim=1)

            score = iou_score(logits, masks).item()
            if score < args.min_iou:
                continue

            image_np = images[0].cpu().permute(1, 2, 0).numpy()
            mask_np = masks[0].cpu().numpy()
            pred_np = pred[0].cpu().numpy()

            gt_overlay = overlay_mask(image_np, mask_np, color=[0.0, 1.0, 0.0])
            pred_overlay = overlay_mask(image_np, pred_np, color=[1.0, 0.2, 0.2])

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(image_np)
            axes[0].set_title("Input")
            axes[1].imshow(gt_overlay)
            axes[1].set_title("Ground Truth")
            axes[2].imshow(pred_overlay)
            axes[2].set_title(f"Prediction\nIoU={score:.3f}")
            for axis in axes:
                axis.axis("off")
            fig.tight_layout()

            out_path = output_dir / f"isic_deeplabv3_sample_{index:03d}.png"
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)

            exported += 1
            if exported >= args.num_samples:
                break

    if exported == 0:
        raise RuntimeError("No qualitative panels were exported. Lower --min_iou or inspect the checkpoint.")

    print(f"Exported {exported} qualitative panels to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/isic2018")
    parser.add_argument("--checkpoint", default="runs/isic/deeplabv3/best.pt")
    parser.add_argument("--output_dir", default="results/figures")
    parser.add_argument("--model", default="deeplabv3")
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--min_iou", type=float, default=0.75)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    return parser.parse_args()


if __name__ == "__main__":
    export_panels(parse_args())
