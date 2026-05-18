import argparse
import json
from pathlib import Path


def load_metrics(run_dir):
    metrics = {}
    for metrics_path in sorted(run_dir.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics[payload["model"]] = payload
    return metrics


def best_history_value(history, key):
    values = [epoch.get(key) for epoch in history if key in epoch]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values)


def format_value(value):
    if value is None:
        return "TBD"
    return f"{value:.4f}"


def write_markdown(metrics, output_path):
    lines = [
        "# ISIC Benchmark Results",
        "",
        "| Model | Best Val Dice | Best Val IoU | Best Val TJ | Test Dice | Test IoU | Test TJ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model_name, payload in metrics.items():
        history = payload.get("history", [])
        test = payload.get("test", {})
        lines.append(
            "| {model} | {val_dice} | {val_iou} | {val_tj} | {test_dice} | {test_iou} | {test_tj} |".format(
                model=model_name,
                val_dice=format_value(best_history_value(history, "val_dice")),
                val_iou=format_value(best_history_value(history, "val_iou")),
                val_tj=format_value(best_history_value(history, "val_threshold_jaccard")),
                test_dice=format_value(test.get("dice")),
                test_iou=format_value(test.get("iou")),
                test_tj=format_value(test.get("threshold_jaccard")),
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="runs/isic")
    parser.add_argument("--output", default="results/isic_benchmark.md")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = load_metrics(Path(args.run_dir))
    if not metrics:
        raise FileNotFoundError(f"No metrics.json files found under {args.run_dir}")
    write_markdown(metrics, Path(args.output))
    print(f"Saved markdown summary to {args.output}")


if __name__ == "__main__":
    main()
