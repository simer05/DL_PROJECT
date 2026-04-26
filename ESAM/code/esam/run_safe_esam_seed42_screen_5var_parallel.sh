#!/usr/bin/env bash
set -euo pipefail

# Corrected Safe-ESAM++ seed42 full-data screen (5 variants)
# Uses base config n_epochs/patience/batch_size unless compact flag is explicitly passed.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

set +u
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lpb
set -u

DATA_ROOT="/mnt/ssd/users/prithvi/deepLearning/data/tabred_preprocessed"
OUT_ROOT="/mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw"
RES_ROOT="/mnt/ssd/users/prithvi/deepLearning/results/esam"
LOG_ROOT="/mnt/ssd/users/prithvi/deepLearning/logs/esam"
mkdir -p "$OUT_ROOT" "$RES_ROOT" "$LOG_ROOT"

SEED=42
TRAIN_FRAC=1.00
COMPACT="${COMPACT:-0}"

run_one_dataset() {
  local gpu="$1"
  local ds="$2"
  export CUDA_VISIBLE_DEVICES="$gpu"

  local base_cfg
  base_cfg=$(DS="$ds" python - <<'PY'
import json, os
from pathlib import Path
cfg = json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(cfg['base_configs'][os.environ['DS']])
PY
)

  local budget_args=()
  if [[ "$COMPACT" == "1" ]]; then
    budget_args=(--n-epochs 20 --patience 4 --batch-size-override 2048 --compact-screen)
  fi

  # 1) baseline
  python esam/run_experiment_esam.py \
    --dataset "$ds" --base-config "$base_cfg" --data-root "$DATA_ROOT" --output-root "$OUT_ROOT" \
    --seed "$SEED" --rho 0.0 --train-fraction "$TRAIN_FRAC" --train-subsample-seed 42 --run-tag baseline \
    "${budget_args[@]}"

  # 2) full_esam_rho_0.0001
  python esam/run_experiment_esam.py \
    --dataset "$ds" --base-config "$base_cfg" --data-root "$DATA_ROOT" --output-root "$OUT_ROOT" \
    --seed "$SEED" --rho 0.0001 --train-fraction "$TRAIN_FRAC" --train-subsample-seed 42 --run-tag full_esam_rho_0.0001 \
    --esam-start-epoch 0 --esam-end-epoch -1 "${budget_args[@]}"

  # 3) full_esam_rho_0.00025
  python esam/run_experiment_esam.py \
    --dataset "$ds" --base-config "$base_cfg" --data-root "$DATA_ROOT" --output-root "$OUT_ROOT" \
    --seed "$SEED" --rho 0.00025 --train-fraction "$TRAIN_FRAC" --train-subsample-seed 42 --run-tag full_esam_rho_0.00025 \
    --esam-start-epoch 0 --esam-end-epoch -1 "${budget_args[@]}"

  # 4) sched25_esam_rho_0.0005
  python esam/run_experiment_esam.py \
    --dataset "$ds" --base-config "$base_cfg" --data-root "$DATA_ROOT" --output-root "$OUT_ROOT" \
    --seed "$SEED" --rho 0.0005 --train-fraction "$TRAIN_FRAC" --train-subsample-seed 42 --run-tag sched25_esam_rho_0.0005 \
    --esam-start-epoch 0 --esam-end-fraction 0.25 "${budget_args[@]}"

  # 5) sched50_esam_rho_0.0005
  python esam/run_experiment_esam.py \
    --dataset "$ds" --base-config "$base_cfg" --data-root "$DATA_ROOT" --output-root "$OUT_ROOT" \
    --seed "$SEED" --rho 0.0005 --train-fraction "$TRAIN_FRAC" --train-subsample-seed 42 --run-tag sched50_esam_rho_0.0005 \
    --esam-start-epoch 0 --esam-end-fraction 0.50 "${budget_args[@]}"
}

# This script expects caller to choose datasets after audits/readiness.
# Example:
#   DATASETS_GPU0="homesite-insurance cooking-time" DATASETS_GPU1="sberbank-housing delivery-eta" bash ...

DATASETS_GPU0="${DATASETS_GPU0-homesite-insurance cooking-time}"
DATASETS_GPU1="${DATASETS_GPU1-sberbank-housing}"

LOG0="$LOG_ROOT/safe_esam_seed42_gpu0_$(date +%Y%m%d_%H%M%S).log"
LOG1="$LOG_ROOT/safe_esam_seed42_gpu1_$(date +%Y%m%d_%H%M%S).log"

(
  for ds in $DATASETS_GPU0; do
    [[ "$ds" == "NONE" || -z "$ds" ]] && continue
    run_one_dataset 0 "$ds"
  done
) > "$LOG0" 2>&1 &
P0=$!

(
  for ds in $DATASETS_GPU1; do
    [[ "$ds" == "NONE" || -z "$ds" ]] && continue
    run_one_dataset 1 "$ds"
  done
) > "$LOG1" 2>&1 &
P1=$!

echo "started P0=$P0 P1=$P1"
echo "log0=$LOG0"
echo "log1=$LOG1"
wait "$P0"
wait "$P1"

echo "[done] seed42 full-data 5-variant screen"
