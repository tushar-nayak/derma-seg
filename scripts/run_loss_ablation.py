import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="isic")
    parser.add_argument("--data_dir", default="data/isic2018")
    parser.add_argument("--model", default="deeplabv3")
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--focal_tversky_gamma", type=float, default=1.33)
    return parser.parse_args()


def run_command(args, loss_name, gamma=None):
    command = [
        sys.executable,
        "scripts/train.py",
        "--dataset",
        args.dataset,
        "--data_dir",
        args.data_dir,
        "--model",
        args.model,
        "--image_size",
        str(args.image_size),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--num_workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--loss",
        loss_name,
        "--tversky_alpha",
        "0.3",
        "--tversky_beta",
        "0.7",
    ]
    if args.pretrained:
        command.append("--pretrained")
    if gamma is not None:
        command.extend(["--focal_tversky_gamma", str(gamma)])

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    run_command(args, "ce_dice")
    run_command(args, "ce_tversky")
    run_command(args, "ce_focal_tversky", gamma=args.focal_tversky_gamma)


if __name__ == "__main__":
    main()
