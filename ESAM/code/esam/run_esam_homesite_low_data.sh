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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

DATA_ROOT="/mnt/ssd/users/prithvi/deepLearning/data/tabred_preprocessed"
RAW_ROOT="/mnt/ssd/users/prithvi/deepLearning/outputs/esam/low_data/raw"
RES_ROOT="/mnt/ssd/users/prithvi/deepLearning/results/esam"
mkdir -p "$RAW_ROOT" "$RES_ROOT" /mnt/ssd/users/prithvi/deepLearning/logs/esam

DATASET="homesite-insurance"
FRACTIONS=(0.05 0.10 0.20 0.50 1.00)
SEEDS=(42 43 44)
RHOS=(0.0 0.0025 0.005 0.01)
N_EPOCHS=80
PATIENCE=12
WARMUP_START=5

BASE_CFG=$(python - <<'PY'
import json
from pathlib import Path
cfg=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(cfg['base_configs']['homesite-insurance'])
PY
)

echo "[low-data] dataset=$DATASET"
echo "[low-data] fractions=${FRACTIONS[*]}"
echo "[low-data] seeds=${SEEDS[*]}"
echo "[low-data] rhos=${RHOS[*]}"
echo "[low-data] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# First pass: start_epoch=0 refined rho sweep.
for frac in "${FRACTIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for rho in "${RHOS[@]}"; do
      echo "[run:start0] frac=$frac seed=$seed rho=$rho"
      python esam/run_experiment_esam.py \
        --dataset "$DATASET" \
        --base-config "$BASE_CFG" \
        --data-root "$DATA_ROOT" \
        --output-root "$RAW_ROOT" \
        --seed "$seed" \
        --rho "$rho" \
        --n-epochs "$N_EPOCHS" \
        --patience "$PATIENCE" \
        --train-fraction "$frac" \
        --train-subsample-seed 42 \
        --esam-start-epoch 0 \
        --max-retries 1
    done
  done
done

# Validation-only best rho among nonzero rhos from first pass.
BEST_RHO=$(python - <<'PY'
import json
from pathlib import Path
import numpy as np
base=Path('/mnt/ssd/users/prithvi/deepLearning/outputs/esam/low_data/raw/homesite-insurance')
vals={}
for rp in base.glob('trainfrac_*/seed*/start_0/rho_*/report.json'):
    rho=float(rp.parent.name.replace('rho_','').replace('m','-').replace('p','.'))
    if rho<=0:
        continue
    d=json.loads(rp.read_text())
    vals.setdefault(rho,[]).append(float(d['metrics']['val']['score']))
if not vals:
    raise SystemExit('no nonzero rho reports found')
best=max(vals.items(), key=lambda kv: float(np.mean(kv[1])))[0]
print(best)
PY
)

echo "[selection] best_rho_by_validation=$BEST_RHO"

# Warmup variants only for selected rho.
for frac in "${FRACTIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "[run:warmup] frac=$frac seed=$seed rho=$BEST_RHO start=$WARMUP_START"
    python esam/run_experiment_esam.py \
      --dataset "$DATASET" \
      --base-config "$BASE_CFG" \
      --data-root "$DATA_ROOT" \
      --output-root "$RAW_ROOT" \
      --seed "$seed" \
      --rho "$BEST_RHO" \
      --n-epochs "$N_EPOCHS" \
      --patience "$PATIENCE" \
      --train-fraction "$frac" \
      --train-subsample-seed 42 \
      --esam-start-epoch "$WARMUP_START" \
      --max-retries 1
  done
done

python esam/build_low_data_tables.py \
  --run-root "$RAW_ROOT" \
  --results-root "$RES_ROOT" \
  --dataset "$DATASET" \
  --warmup-epoch "$WARMUP_START"

echo "[done] low-data homesite block complete"
