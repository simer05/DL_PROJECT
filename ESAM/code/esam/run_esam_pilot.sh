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

mkdir -p /mnt/ssd/users/prithvi/deepLearning/{outputs/esam,results/esam,logs/esam,checkpoints/esam,artifacts/esam}

PRE_ROOT=$(python - <<'PY'
import json
from pathlib import Path
c = json.loads(Path('esam/configs/esam_dataset_config.json').read_text())
print(c['preprocessed_root'])
PY
)

DATASETS=$(python - <<'PY'
import json
from pathlib import Path
exp = json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
pre = Path(json.loads(Path('esam/configs/esam_dataset_config.json').read_text())['preprocessed_root'])
ready = []
for d in exp['pilot']['datasets']:
    p = pre / d
    needed = ['info.json', 'Y_train.npy', 'Y_val.npy', 'Y_test.npy']
    if all((p / f).exists() for f in needed):
        ready.append(d)
print(','.join(ready))
PY
)

SEEDS=$(python - <<'PY'
import json
from pathlib import Path
c = json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(','.join(map(str, c['pilot']['seeds'])))
PY
)

RHOS=$(python - <<'PY'
import json
from pathlib import Path
c = json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(','.join(map(str, c['pilot']['rhos'])))
PY
)

N_EPOCHS=$(python - <<'PY'
import json
from pathlib import Path
c = json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(c['pilot']['n_epochs'])
PY
)

PATIENCE=$(python - <<'PY'
import json
from pathlib import Path
c = json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(c['pilot']['patience'])
PY
)

RAW_ROOT="/mnt/ssd/users/prithvi/deepLearning/outputs/esam/pilot/raw"
mkdir -p "$RAW_ROOT"

if [[ -z "$DATASETS" ]]; then
  echo "[error] no ready pilot datasets found in $PRE_ROOT"
  exit 2
fi

echo "[pilot] datasets=$DATASETS"
echo "[pilot] seeds=$SEEDS"
echo "[pilot] rhos=$RHOS"
echo "[pilot] n_epochs=$N_EPOCHS patience=$PATIENCE"
echo "[pilot] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

IFS=',' read -r -a DSETS <<< "$DATASETS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
IFS=',' read -r -a RHO_ARR <<< "$RHOS"

for ds in "${DSETS[@]}"; do
  base_cfg=$(python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(cfg['base_configs']['$ds'])
PY
)
  for seed in "${SEED_ARR[@]}"; do
    for rho in "${RHO_ARR[@]}"; do
      echo "[run] dataset=$ds seed=$seed rho=$rho"
      python esam/run_experiment_esam.py \
        --dataset "$ds" \
        --base-config "$base_cfg" \
        --data-root "$PRE_ROOT" \
        --output-root "$RAW_ROOT" \
        --seed "$seed" \
        --rho "$rho" \
        --n-epochs "$N_EPOCHS" \
        --patience "$PATIENCE" \
        --max-retries 1
    done
  done
done

python esam/prepare_selection_esam.py \
  --run-root "$RAW_ROOT" \
  --datasets "$DATASETS" \
  --seeds "$SEEDS" \
  --rhos "$RHOS" \
  --result /mnt/ssd/users/prithvi/deepLearning/results/esam/esam_pilot_selection.json

python esam/build_esam_report.py \
  --run-root "$RAW_ROOT" \
  --selection /mnt/ssd/users/prithvi/deepLearning/results/esam/esam_pilot_selection.json \
  --out-root /mnt/ssd/users/prithvi/deepLearning/results/esam \
  --tag pilot

echo "[done] pilot pipeline"
