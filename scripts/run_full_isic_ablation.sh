#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-isic}"
DATA_DIR="${DATA_DIR:-data/isic2018}"
IMAGE_SIZE="${IMAGE_SIZE:-320}"
BATCH_SIZE="${BATCH_SIZE:-12}"
EPOCHS="${EPOCHS:-25}"
PATIENCE="${PATIENCE:-7}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-3e-4}"
LR_VALUES="${LR_VALUES:-1e-4 3e-4 1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-42}"
FOCAL_TVERSKY_GAMMA="${FOCAL_TVERSKY_GAMMA:-1.33}"
PRETRAINED_FLAG="${PRETRAINED_FLAG:---pretrained}"

BASE_RUN_ROOT="${BASE_RUN_ROOT:-runs/ablations}"
BASELINE_ROOT="$BASE_RUN_ROOT/deeplabv3_loss_ablation"
NOVEL_ROOT="$BASE_RUN_ROOT/ba_deeplabv3_loss_ablation"
BASELINE_LR_ROOT="$BASE_RUN_ROOT/deeplabv3_lr_ablation"
NOVEL_LR_ROOT="$BASE_RUN_ROOT/ba_deeplabv3_lr_ablation"
RESULT_ROOT="${RESULT_ROOT:-results/ablations}"

mkdir -p "$BASELINE_ROOT" "$NOVEL_ROOT" "$BASELINE_LR_ROOT" "$NOVEL_LR_ROOT" "$RESULT_ROOT"

run_train() {
  local model="$1"
  local loss_name="$2"
  local output_dir="$3"
  local lr_value="${4:-$LR}"
  shift 4

  echo "Running model=$model loss=$loss_name lr=$lr_value output_dir=$output_dir"
  "$PYTHON_BIN" scripts/train.py \
    --dataset "$DATASET" \
    --data_dir "$DATA_DIR" \
    --model "$model" \
    --image_size "$IMAGE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --num_workers "$NUM_WORKERS" \
    --lr "$lr_value" \
    --weight_decay "$WEIGHT_DECAY" \
    --seed "$SEED" \
    --loss "$loss_name" \
    --tversky_alpha 0.3 \
    --tversky_beta 0.7 \
    --focal_tversky_gamma "$FOCAL_TVERSKY_GAMMA" \
    --output_dir "$output_dir" \
    "$@" \
    $PRETRAINED_FLAG
}

run_baseline_ablation() {
  run_train "deeplabv3" "ce_dice" "$BASELINE_ROOT/ce_dice" "$LR"
  run_train "deeplabv3" "ce_tversky" "$BASELINE_ROOT/ce_tversky" "$LR"
  run_train "deeplabv3" "ce_focal_tversky" "$BASELINE_ROOT/ce_focal_tversky" "$LR"

  "$PYTHON_BIN" scripts/summarize_ablation_results.py \
    --root "$BASELINE_ROOT" \
    --markdown_output "$RESULT_ROOT/deeplabv3_loss_ablation.md" \
    --json_output "$RESULT_ROOT/deeplabv3_loss_ablation.json" \
    --title "DeepLabV3 Loss Ablation"
}

run_novel_ablation() {
  run_train "ba_deeplabv3" "ce_dice" "$NOVEL_ROOT/ce_dice" "$LR"
  run_train "ba_deeplabv3" "ce_tversky" "$NOVEL_ROOT/ce_tversky" "$LR"
  run_train "ba_deeplabv3" "ce_focal_tversky" "$NOVEL_ROOT/ce_focal_tversky" "$LR"

  "$PYTHON_BIN" scripts/summarize_ablation_results.py \
    --root "$NOVEL_ROOT" \
    --markdown_output "$RESULT_ROOT/ba_deeplabv3_loss_ablation.md" \
    --json_output "$RESULT_ROOT/ba_deeplabv3_loss_ablation.json" \
    --title "BA-DeepLabV3 Loss Ablation"
}

run_lr_sweep() {
  local model="$1"
  local loss_name="$2"
  local root_dir="$3"
  local summary_name="$4"

  for lr_value in $LR_VALUES; do
    local run_name
    run_name="$(printf 'lr_%s' "$lr_value" | tr '.' 'p')"
    run_train "$model" "$loss_name" "$root_dir/$run_name" "$lr_value"
  done

  "$PYTHON_BIN" scripts/summarize_ablation_results.py \
    --root "$root_dir" \
    --markdown_output "$RESULT_ROOT/${summary_name}.md" \
    --json_output "$RESULT_ROOT/${summary_name}.json" \
    --title "${model} Learning-Rate Ablation"
}

write_combined_summary() {
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

result_root = Path("results/ablations")
paths = [
    result_root / "deeplabv3_loss_ablation.json",
    result_root / "ba_deeplabv3_loss_ablation.json",
    result_root / "deeplabv3_lr_ablation.json",
    result_root / "ba_deeplabv3_lr_ablation.json",
]
rows = []
for path in paths:
    if path.exists():
        rows.extend(json.loads(path.read_text(encoding="utf-8")))

rows.sort(key=lambda item: (item["model"], item["loss"], item.get("lr") or 0.0, item["run_name"]))

md_lines = [
    "# ISIC Combined Ablation Summary",
    "",
    "| Run | Model | Loss | LR | Best Val Dice | Best Val IoU | Best Val TJ | Test Dice | Test IoU | Test TJ |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for row in rows:
    md_lines.append(
        "| {run_name} | {model} | {loss} | {lr:.4f} | {best_val_dice:.4f} | {best_val_iou:.4f} | {best_val_tj:.4f} | {test_dice:.4f} | {test_iou:.4f} | {test_tj:.4f} |".format(
            run_name=row["run_name"],
            model=row["model"],
            loss=row["loss"],
            lr=row.get("lr") or 0.0,
            best_val_dice=row["best_val_dice"] or 0.0,
            best_val_iou=row["best_val_iou"] or 0.0,
            best_val_tj=row["best_val_tj"] or 0.0,
            test_dice=row["test_dice"] or 0.0,
            test_iou=row["test_iou"] or 0.0,
            test_tj=row["test_tj"] or 0.0,
        )
    )

(result_root / "combined_isic_ablation.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
(result_root / "combined_isic_ablation.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print("Saved combined ablation summaries to results/ablations/")
PY
}

run_baseline_ablation
run_novel_ablation
run_lr_sweep "deeplabv3" "ce_tversky" "$BASELINE_LR_ROOT" "deeplabv3_lr_ablation"
run_lr_sweep "ba_deeplabv3" "ce_tversky" "$NOVEL_LR_ROOT" "ba_deeplabv3_lr_ablation"
write_combined_summary

echo "All ablations completed. Summaries are under $RESULT_ROOT"
