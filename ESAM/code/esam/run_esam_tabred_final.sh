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

mkdir -p /mnt/ssd/users/prithvi/deepLearning/{outputs/esam,results/esam,logs/esam,checkpoints/esam,artifacts/esam,data/tabred_preprocessed}

DATA_CFG="esam/configs/esam_dataset_config.json"
EXP_CFG="esam/configs/esam_experiment_config.json"

python esam/check_tabred_ready_esam.py --config "$DATA_CFG" --strict

DATASETS=$(python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(','.join(c['final']['datasets']))
PY
)
SEEDS=$(python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(','.join(map(str, c['final']['seeds'])))
PY
)
SELECTED_RHO=${SELECTED_RHO:-$(python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(c['final']['default_selected_rho'])
PY
)}

PRE_ROOT=$(python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('esam/configs/esam_dataset_config.json').read_text())
print(c['preprocessed_root'])
PY
)

RAW_ROOT="/mnt/ssd/users/prithvi/deepLearning/outputs/esam/final/raw"
mkdir -p "$RAW_ROOT"

IFS=',' read -r -a DSETS <<< "$DATASETS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"

for ds in "${DSETS[@]}"; do
  base_cfg=$(python - <<PY
import json
from pathlib import Path
cfg=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(cfg['base_configs']['$ds'])
PY
)
  for seed in "${SEED_ARR[@]}"; do
    python esam/run_experiment_esam.py --dataset "$ds" --base-config "$base_cfg" --data-root "$PRE_ROOT" --output-root "$RAW_ROOT" --seed "$seed" --rho 0.0 --n-epochs 80 --patience 12
    python esam/run_experiment_esam.py --dataset "$ds" --base-config "$base_cfg" --data-root "$PRE_ROOT" --output-root "$RAW_ROOT" --seed "$seed" --rho "$SELECTED_RHO" --n-epochs 80 --patience 12
  done
done

python esam/prepare_selection_esam.py --run-root "$RAW_ROOT" --datasets "$DATASETS" --seeds "$SEEDS" --rhos "0.0,$SELECTED_RHO" --result /mnt/ssd/users/prithvi/deepLearning/results/esam/esam_final_selection.json
python esam/build_esam_report.py --run-root "$RAW_ROOT" --selection /mnt/ssd/users/prithvi/deepLearning/results/esam/esam_final_selection.json --out-root /mnt/ssd/users/prithvi/deepLearning/results/esam --tag final

echo "[done] final pipeline"
