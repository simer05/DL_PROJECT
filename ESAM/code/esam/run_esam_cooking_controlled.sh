#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lpb" ]]; then
  set +u
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate lpb
  set -u
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

DATA_ROOT="/mnt/ssd/users/prithvi/deepLearning/data/tabred_preprocessed"
RAW_ROOT="/mnt/ssd/users/prithvi/deepLearning/outputs/esam/cooking_controlled_quick/raw"
RES_ROOT="/mnt/ssd/users/prithvi/deepLearning/results/esam"
LOG_ROOT="/mnt/ssd/users/prithvi/deepLearning/logs/esam"
mkdir -p "$RAW_ROOT" "$RES_ROOT" "$LOG_ROOT"

# Compact-quality profile for ~3-4 hour turnaround.
DATASET="cooking-time"
FRACTIONS=(0.10 0.20)
SEEDS=(42 43 44)
RHOS=(0.0 0.0025)
N_EPOCHS=24
PATIENCE=4
BATCH_OVERRIDE=2048
START_EPOCH=0
RUN_TAG="quick_3h"

BASE_CFG=$(python - <<'PY'
import json
from pathlib import Path
cfg=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(cfg['base_configs']['cooking-time'])
PY
)

echo "[cooking-controlled-quick] dataset=$DATASET"
echo "[cooking-controlled-quick] fractions=${FRACTIONS[*]}"
echo "[cooking-controlled-quick] seeds=${SEEDS[*]}"
echo "[cooking-controlled-quick] rhos=${RHOS[*]}"
echo "[cooking-controlled-quick] n_epochs=$N_EPOCHS patience=$PATIENCE batch_override=$BATCH_OVERRIDE"
echo "[cooking-controlled-quick] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for frac in "${FRACTIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for rho in "${RHOS[@]}"; do
      echo "[run] frac=$frac seed=$seed rho=$rho"
      python esam/run_experiment_esam.py \
        --dataset "$DATASET" \
        --base-config "$BASE_CFG" \
        --data-root "$DATA_ROOT" \
        --output-root "$RAW_ROOT" \
        --seed "$seed" \
        --rho "$rho" \
        --n-epochs "$N_EPOCHS" \
        --patience "$PATIENCE" \
        --batch-size-override "$BATCH_OVERRIDE" \
        --train-fraction "$frac" \
        --train-subsample-seed 42 \
        --esam-start-epoch "$START_EPOCH" \
        --run-tag "$RUN_TAG" \
        --max-retries 1
    done
  done
done

python esam/build_cooking_controlled_tables.py \
  --run-root "$RAW_ROOT" \
  --results-root "$RES_ROOT" \
  --dataset "$DATASET"

echo "[done] cooking controlled quick block complete"
