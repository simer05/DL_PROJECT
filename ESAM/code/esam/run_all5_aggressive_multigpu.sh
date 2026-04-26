#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

set +u
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lpb
set -u

export PYTHONUNBUFFERED=1
DATA_ROOT="/mnt/ssd/users/prithvi/deepLearning/data/tabred_preprocessed"
RAW_ROOT="/mnt/ssd/users/prithvi/deepLearning/outputs/esam/all5_aggressive/raw"
RES_ROOT="/mnt/ssd/users/prithvi/deepLearning/results/esam"
LOG_ROOT="/mnt/ssd/users/prithvi/deepLearning/logs/esam"
mkdir -p "$RAW_ROOT" "$RES_ROOT" "$LOG_ROOT"

SEED="${SEED:-42}"
FRACTION="${FRACTION:-1.00}"
N_EPOCHS="${N_EPOCHS:-20}"
PATIENCE="${PATIENCE:-4}"
BATCH_OVERRIDE="${BATCH_OVERRIDE:-2048}"

run_block() {
  local gpu="$1"
  shift
  local datasets=("$@")
  export CUDA_VISIBLE_DEVICES="$gpu"

  for ds in "${datasets[@]}"; do
    if [[ ! -f "$DATA_ROOT/$ds/info.json" ]]; then
      echo "[skip][$ds] not ready"
      continue
    fi

    base_cfg=$(DS="$ds" python - <<'PY2'
import json, os
from pathlib import Path
cfg = json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(cfg['base_configs'][os.environ['DS']])
PY2
)

    echo "[gpu=$gpu][dataset=$ds] base_cfg=$base_cfg"

    python esam/run_experiment_esam.py \
      --dataset "$ds" --base-config "$base_cfg" --data-root "$DATA_ROOT" --output-root "$RAW_ROOT" \
      --seed "$SEED" --rho 0.0 --n-epochs "$N_EPOCHS" --patience "$PATIENCE" --batch-size-override "$BATCH_OVERRIDE" \
      --train-fraction "$FRACTION" --train-subsample-seed 42 --esam-start-epoch 0 --run-tag baseline --max-retries 1

    for rho in 0.0001 0.00025 0.0005 0.001; do
      tag="esam_full_rho_${rho}"
      python esam/run_experiment_esam.py \
        --dataset "$ds" --base-config "$base_cfg" --data-root "$DATA_ROOT" --output-root "$RAW_ROOT" \
        --seed "$SEED" --rho "$rho" --n-epochs "$N_EPOCHS" --patience "$PATIENCE" --batch-size-override "$BATCH_OVERRIDE" \
        --train-fraction "$FRACTION" --train-subsample-seed 42 --esam-start-epoch 0 --esam-end-epoch -1 --run-tag "$tag" --max-retries 1
    done

    python esam/run_experiment_esam.py \
      --dataset "$ds" --base-config "$base_cfg" --data-root "$DATA_ROOT" --output-root "$RAW_ROOT" \
      --seed "$SEED" --rho 0.0005 --n-epochs "$N_EPOCHS" --patience "$PATIENCE" --batch-size-override "$BATCH_OVERRIDE" \
      --train-fraction "$FRACTION" --train-subsample-seed 42 --esam-start-epoch 0 --esam-end-fraction 0.25 \
      --run-tag esam_sched25_rho_0.0005 --max-retries 1

    python esam/run_experiment_esam.py \
      --dataset "$ds" --base-config "$base_cfg" --data-root "$DATA_ROOT" --output-root "$RAW_ROOT" \
      --seed "$SEED" --rho 0.0005 --n-epochs "$N_EPOCHS" --patience "$PATIENCE" --batch-size-override "$BATCH_OVERRIDE" \
      --train-fraction "$FRACTION" --train-subsample-seed 42 --esam-start-epoch 0 --esam-end-fraction 0.50 \
      --run-tag esam_sched50_rho_0.0005 --max-retries 1
  done
}

run_block 0 homesite-insurance ecom-offers cooking-time > "$LOG_ROOT/all5_aggressive_gpu0_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID0=$!
run_block 1 sberbank-housing delivery-eta > "$LOG_ROOT/all5_aggressive_gpu1_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID1=$!

echo "started PID0=$PID0 PID1=$PID1"
wait "$PID0"
wait "$PID1"

python esam/build_compact_auto_esam_report.py --run-root "$RAW_ROOT" --results-root "$RES_ROOT"
echo "[done] all5 aggressive screen"
