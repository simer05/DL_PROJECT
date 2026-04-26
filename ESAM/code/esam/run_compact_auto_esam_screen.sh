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
RAW_ROOT="/mnt/ssd/users/prithvi/deepLearning/outputs/esam/compact_auto_screen/raw"
RES_ROOT="/mnt/ssd/users/prithvi/deepLearning/results/esam"
mkdir -p "$RAW_ROOT" "$RES_ROOT" /mnt/ssd/users/prithvi/deepLearning/logs/esam

SEED=42
FRACTIONS=(0.10)
if [[ "${INCLUDE_FULL_FRACTION:-0}" == "1" ]]; then
  FRACTIONS+=(1.00)
fi
N_EPOCHS="${AUTO_ESAM_EPOCHS:-20}"
PATIENCE="${AUTO_ESAM_PATIENCE:-4}"
BATCH_OVERRIDE="${AUTO_ESAM_BATCH_OVERRIDE:-2048}"

DATASETS=(homesite-insurance cooking-time sberbank-housing)

for ds in "${DATASETS[@]}"; do
  if [[ ! -f "$DATA_ROOT/$ds/info.json" ]]; then
    echo "[skip] dataset not ready: $ds"
    continue
  fi

  BASE_CFG=$(python - <<PY
import json
from pathlib import Path
cfg=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(cfg['base_configs']['$ds'])
PY
)

  echo "[dataset] $ds base_cfg=$BASE_CFG"
  for frac in "${FRACTIONS[@]}"; do
    echo "[block] dataset=$ds frac=$frac seed=$SEED"

    # 1) baseline
    python esam/run_experiment_esam.py \
      --dataset "$ds" --base-config "$BASE_CFG" --data-root "$DATA_ROOT" --output-root "$RAW_ROOT" \
      --seed "$SEED" --rho 0.0 --n-epochs "$N_EPOCHS" --patience "$PATIENCE" --batch-size-override "$BATCH_OVERRIDE" \
      --train-fraction "$frac" --train-subsample-seed 42 --esam-start-epoch 0 --run-tag baseline --max-retries 1

    # 2-5) full ESAM
    for rho in 0.0001 0.00025 0.0005 0.001; do
      tag="esam_full_rho_${rho}"
      python esam/run_experiment_esam.py \
        --dataset "$ds" --base-config "$BASE_CFG" --data-root "$DATA_ROOT" --output-root "$RAW_ROOT" \
        --seed "$SEED" --rho "$rho" --n-epochs "$N_EPOCHS" --patience "$PATIENCE" --batch-size-override "$BATCH_OVERRIDE" \
        --train-fraction "$frac" --train-subsample-seed 42 --esam-start-epoch 0 --esam-end-epoch -1 --run-tag "$tag" --max-retries 1
    done

    # 6) scheduled 25%
    python esam/run_experiment_esam.py \
      --dataset "$ds" --base-config "$BASE_CFG" --data-root "$DATA_ROOT" --output-root "$RAW_ROOT" \
      --seed "$SEED" --rho 0.0005 --n-epochs "$N_EPOCHS" --patience "$PATIENCE" --batch-size-override "$BATCH_OVERRIDE" \
      --train-fraction "$frac" --train-subsample-seed 42 --esam-start-epoch 0 --esam-end-fraction 0.25 \
      --run-tag esam_sched25_rho_0.0005 --max-retries 1

    # 7) scheduled 50%
    python esam/run_experiment_esam.py \
      --dataset "$ds" --base-config "$BASE_CFG" --data-root "$DATA_ROOT" --output-root "$RAW_ROOT" \
      --seed "$SEED" --rho 0.0005 --n-epochs "$N_EPOCHS" --patience "$PATIENCE" --batch-size-override "$BATCH_OVERRIDE" \
      --train-fraction "$frac" --train-subsample-seed 42 --esam-start-epoch 0 --esam-end-fraction 0.50 \
      --run-tag esam_sched50_rho_0.0005 --max-retries 1

  done
done

python esam/build_compact_auto_esam_report.py \
  --run-root "$RAW_ROOT" \
  --results-root "$RES_ROOT"

echo "[done] compact auto-esam screen completed"
