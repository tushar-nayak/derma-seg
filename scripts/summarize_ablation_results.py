import argparse
import json
from pathlib import Path


def format_metric(value):
    if value is None:
        return "TBD"
    return f"{value:.4f}"


def best_history_value(history, key):
    values = [epoch.get(key) for epoch in history if epoch.get(key) is not None]
    if not values:
        return None
    return max(values)


def collect_runs(root):
    rows = []
    for metrics_path in sorted(Path(root).glob("**/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        history = payload.get("history", [])
        test = payload.get("test", {})
        row = {
            "run_name": metrics_path.parent.name,
            "model": payload.get("model", "unknown"),
            "loss": payload.get("loss", "unknown"),
            "lr": payload.get("history", [{}])[0].get("lr") if history else None,
            "use_boundary_head": payload.get("use_boundary_head", True),
            "use_uncertainty_head": payload.get("use_uncertainty_head", True),
            "best_val_dice": best_history_value(history, "val_dice"),
            "best_val_iou": best_history_value(history, "val_iou"),
            "best_val_tj": best_history_value(history, "val_threshold_jaccard"),
            "test_dice": test.get("dice"),
            "test_iou": test.get("iou"),
            "test_tj": test.get("threshold_jaccard"),
            "metrics_path": str(metrics_path),
        }
        rows.append(row)
    return rows


def write_markdown(rows, output_path, title):
    lines = [
        f"# {title}",
        "",
        "| Run | Model | Loss | LR | Boundary | Uncertainty | Best Val Dice | Best Val IoU | Best Val TJ | Test Dice | Test IoU | Test TJ |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| {run_name} | {model} | {loss} | {lr} | {boundary} | {uncertainty} | {best_val_dice} | {best_val_iou} | {best_val_tj} | {test_dice} | {test_iou} | {test_tj} |".format(
                run_name=row["run_name"],
                model=row["model"],
                loss=row["loss"],
                lr=format_metric(row["lr"]),
                boundary="on" if row["use_boundary_head"] else "off",
                uncertainty="on" if row["use_uncertainty_head"] else "off",
                best_val_dice=format_metric(row["best_val_dice"]),
                best_val_iou=format_metric(row["best_val_iou"]),
                best_val_tj=format_metric(row["best_val_tj"]),
                test_dice=format_metric(row["test_dice"]),
                test_iou=format_metric(row["test_iou"]),
                test_tj=format_metric(row["test_tj"]),
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--markdown_output", required=True)
    parser.add_argument("--json_output", required=True)
    parser.add_argument("--title", default="Ablation Summary")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_runs(args.root)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {args.root}")
    write_markdown(rows, Path(args.markdown_output), args.title)
    write_json(rows, Path(args.json_output))
    print(f"Saved ablation summaries under {Path(args.markdown_output).parent}")


if __name__ == "__main__":
    main()
