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
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-42}"
LOSS_NAME="${LOSS_NAME:-ce_tversky}"
FOCAL_TVERSKY_GAMMA="${FOCAL_TVERSKY_GAMMA:-1.33}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-runs/ablations}"
RUN_ROOT="$BASE_RUN_ROOT/ba_component_ablation"
RESULT_ROOT="${RESULT_ROOT:-results/ablations}"

mkdir -p "$RUN_ROOT" "$RESULT_ROOT"

run_component() {
  local run_name="$1"
  shift

  echo "Running component ablation: $run_name"
  "$PYTHON_BIN" scripts/train.py \
    --dataset "$DATASET" \
    --data_dir "$DATA_DIR" \
    --model ba_deeplabv3 \
    --image_size "$IMAGE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --num_workers "$NUM_WORKERS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --seed "$SEED" \
    --loss "$LOSS_NAME" \
    --tversky_alpha 0.3 \
    --tversky_beta 0.7 \
    --focal_tversky_gamma "$FOCAL_TVERSKY_GAMMA" \
    --output_dir "$RUN_ROOT/$run_name" \
    --pretrained \
    "$@"
}

run_component "full"
run_component "no_uncertainty" --no_uncertainty_head --uncertainty_weight 0.0
run_component "no_boundary" --no_boundary_head --boundary_weight 0.0 --consistency_weight 0.0
run_component "no_boundary_no_uncertainty" --no_boundary_head --no_uncertainty_head --boundary_weight 0.0 --uncertainty_weight 0.0 --consistency_weight 0.0

"$PYTHON_BIN" scripts/summarize_ablation_results.py \
  --root "$RUN_ROOT" \
  --markdown_output "$RESULT_ROOT/ba_component_ablation.md" \
  --json_output "$RESULT_ROOT/ba_component_ablation.json" \
  --title "BA-DeepLabV3 Component Ablation"

echo "Saved component ablation summaries under $RESULT_ROOT"
