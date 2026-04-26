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

CFG="esam/configs/esam_experiment_config.json"
DATA_CFG="esam/configs/esam_dataset_config.json"
DRY_RUN="false"
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN="true"
fi

mkdir -p /mnt/ssd/users/prithvi/deepLearning/{outputs/esam,results/esam,logs/esam,checkpoints/esam,artifacts/esam,data/tabred_preprocessed}

python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print('[cfg] smoke datasets', c['smoke']['datasets'])
print('[cfg] smoke rhos', c['smoke']['rhos'])
PY

python esam/preprocess_tabred_to_tabm.py --config "$DATA_CFG" --datasets homesite-insurance,ecom-offers --force
python esam/check_tabred_ready_esam.py --config "$DATA_CFG" --strict --datasets homesite-insurance,ecom-offers

SMOKE_DATASETS=$(python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(','.join(c['smoke']['datasets']))
PY
)
SEED=$(python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(c['smoke']['seed'])
PY
)
RHOS=$(python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(','.join(map(str, c['smoke']['rhos'])))
PY
)
N_EPOCHS=$(python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(c['smoke']['n_epochs'])
PY
)
PATIENCE=$(python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(c['smoke']['patience'])
PY
)
BATCH_OVERRIDE=$(python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(c['smoke'].get('batch_size_override', 0))
PY
)

PRE_ROOT=$(python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('esam/configs/esam_dataset_config.json').read_text())
print(c['preprocessed_root'])
PY
)

RAW_ROOT="/mnt/ssd/users/prithvi/deepLearning/outputs/esam/smoke/raw"
mkdir -p "$RAW_ROOT"

IFS=',' read -r -a DSETS <<< "$SMOKE_DATASETS"
IFS=',' read -r -a RHO_ARR <<< "$RHOS"

for ds in "${DSETS[@]}"; do
  ready_dir="$PRE_ROOT/$ds"
  if [[ ! -f "$ready_dir/info.json" ]]; then
    echo "[skip] $ds not ready at $ready_dir"
    continue
  fi
  base_cfg=$(python - <<PY
import json
from pathlib import Path
cfg=json.loads(Path('esam/configs/esam_experiment_config.json').read_text())
print(cfg['base_configs']['$ds'])
PY
)

  for rho in "${RHO_ARR[@]}"; do
    cmd=(python esam/run_experiment_esam.py
      --dataset "$ds"
      --base-config "$base_cfg"
      --data-root "$PRE_ROOT"
      --output-root "$RAW_ROOT"
      --seed "$SEED"
      --rho "$rho"
      --n-epochs "$N_EPOCHS"
      --patience "$PATIENCE"
      --batch-size-override "$BATCH_OVERRIDE"
      --max-retries 1)

    if [[ "$DRY_RUN" == "true" ]]; then
      cmd+=(--dry-run)
    fi

    echo "[exec] ${cmd[*]}"
    "${cmd[@]}"
  done
done

python esam/prepare_selection_esam.py \
  --run-root "$RAW_ROOT" \
  --datasets "$SMOKE_DATASETS" \
  --seeds "$SEED" \
  --rhos "$RHOS" \
  --result /mnt/ssd/users/prithvi/deepLearning/results/esam/esam_smoke_selection.json

python esam/build_esam_report.py \
  --run-root "$RAW_ROOT" \
  --selection /mnt/ssd/users/prithvi/deepLearning/results/esam/esam_smoke_selection.json \
  --out-root /mnt/ssd/users/prithvi/deepLearning/results/esam \
  --tag smoke

echo "[done] smoke pipeline"
