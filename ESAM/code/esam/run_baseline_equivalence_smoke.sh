#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

set +u
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lpb
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA_ROOT="/mnt/ssd/users/prithvi/deepLearning/data/tabred_preprocessed"
OUT_ROOT="/mnt/ssd/users/prithvi/deepLearning/outputs/esam/baseline_equivalence_smoke/raw"
LOG_ROOT="/mnt/ssd/users/prithvi/deepLearning/logs/esam"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

DATASETS="${DATASETS:-homesite-insurance cooking-time sberbank-housing ecom-offers delivery-eta}"
SEED="${SEED:-42}"
TRAIN_FRAC="${TRAIN_FRAC:-1.00}"

for ds in $DATASETS; do
  if [[ ! -f "$DATA_ROOT/$ds/info.json" ]]; then
    echo "[skip] $ds not ready"
    continue
  fi
  base_cfg=$(DS="$ds" python - <<'PY'
import json, os
from pathlib import Path
cfg = json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(cfg['base_configs'][os.environ['DS']])
PY
)

  python esam/run_experiment_esam.py \
    --dataset "$ds" --base-config "$base_cfg" --data-root "$DATA_ROOT" --output-root "$OUT_ROOT" \
    --seed "$SEED" --rho 0.0 --train-fraction "$TRAIN_FRAC" --train-subsample-seed 42 --run-tag baseline_smoke

done

echo "[done] baseline smoke"
