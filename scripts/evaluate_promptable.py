import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.lumiere_dataset import build_lumiere_splits, load_lumiere_slice
from src.models.promptable_sam import MedSAMPromptable, PromptableSAM
from src.utils.metrics import dice_coeff, iou_score
from src.utils.prompting import grayscale_to_rgb, mask_to_box


def evaluate_promptable(args):
    _, _, test_samples = build_lumiere_splits(
        args.data_root,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        positive_only=args.positive_only,
    )

    if args.model == "sam":
        predictor = PromptableSAM(args.checkpoint, model_type=args.model_type, device=args.device)
    elif args.model == "medsam":
        predictor = MedSAMPromptable(args.checkpoint, model_type=args.model_type, device=args.device)
    else:
        raise ValueError(f"Unknown promptable model: {args.model}")

    dice_scores = []
    iou_scores = []
    per_sample = []

    for sample in tqdm(test_samples, desc="promptable-eval"):
        image, mask = load_lumiere_slice(sample, image_size=args.image_size)
        image_rgb = grayscale_to_rgb(np.clip(image * 255.0, 0, 255).astype(np.uint8))
        box = mask_to_box(mask, pad=args.box_pad)

        pred_mask, score = predictor.predict_with_box(image_rgb, box)
        pred_tensor = torch.from_numpy(pred_mask.astype(np.int64)).unsqueeze(0).unsqueeze(0)
        target_tensor = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0)

        sample_dice = float(dice_coeff(pred_tensor.float(), target_tensor))
        sample_iou = float(iou_score(pred_tensor.float(), target_tensor))
        dice_scores.append(sample_dice)
        iou_scores.append(sample_iou)
        per_sample.append(
            {
                "patient_id": sample.patient_id,
                "week_id": sample.week_id,
                "slice_idx": sample.slice_idx,
                "prompt_score": score,
                "dice": sample_dice,
                "iou": sample_iou,
            }
        )

    results = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "mean_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "mean_iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "samples": per_sample,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_promptable_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved promptable metrics to {out_path}")
    print(f"Mean Dice: {results['mean_dice']:.4f}")
    print(f"Mean IoU: {results['mean_iou']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sam", choices=["sam", "medsam"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/sofa/host_dir/hub/glioblastoma-evolution/data/lumiere",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--positive_only", action="store_true", default=True)
    parser.add_argument("--no_positive_only", action="store_false", dest="positive_only")
    parser.add_argument("--box_pad", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="runs/promptable")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_promptable(parse_args())
