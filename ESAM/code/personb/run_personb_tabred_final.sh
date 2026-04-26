#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PREFIX="${PREFIX:-personb_tabred_final}"
DATA_CFG="${DATA_CFG:-personb/tabred_dataset_config.json}"
NCL_SPACE="${NCL_SPACE:-logits}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
LAMBDAS_CSV="${LAMBDAS_CSV:-0,5e-4,1e-3,2e-3}"
N_EPOCHS="${N_EPOCHS:-80}"
PATIENCE="${PATIENCE:-12}"
MAX_RETRIES="${MAX_RETRIES:-2}"
NCL_WARMUP_EPOCHS="${NCL_WARMUP_EPOCHS:-10}"
TIE_BREAK="${TIE_BREAK:-val_cross_entropy}"

RAW_ROOT="outputs/${PREFIX}/raw"
mkdir -p "$RAW_ROOT" results job_logs

python personb/check_tabred_ready.py --config "$DATA_CFG" --strict

DATA_ROOT=$(python - <<PY
import json
from pathlib import Path
cfg=json.loads(Path("$DATA_CFG").read_text())
print(cfg["data_root"])
PY
)

mapfile -t DATASET_ROWS < <(python - <<PY
import json
from pathlib import Path
cfg=json.loads(Path("$DATA_CFG").read_text())
for key,meta in cfg["datasets"].items():
    print(f"{key}\t{meta['dataset_dir']}\t{meta['base_config']}")
PY
)

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a LAMBDAS <<< "$LAMBDAS_CSV"

for row in "${DATASET_ROWS[@]}"; do
  key="$(echo "$row" | awk -F'\t' '{print $1}')"
  dataset_dir="$(echo "$row" | awk -F'\t' '{print $2}')"
  base_cfg="$(echo "$row" | awk -F'\t' '{print $3}')"

  for seed in "${SEEDS[@]}"; do
    for lam in "${LAMBDAS[@]}"; do
      python personb/run_experiment.py \
        --dataset "$key" \
        --base-config "$base_cfg" \
        --dataset-dir "$dataset_dir" \
        --data-root "$DATA_ROOT" \
        --output-root "$RAW_ROOT" \
        --seed "$seed" \
        --lambda-ncl "$lam" \
        --ncl-warmup-epochs "$NCL_WARMUP_EPOCHS" \
        --ncl-space "$NCL_SPACE" \
        --share-training-batches \
        --n-epochs "$N_EPOCHS" \
        --patience "$PATIENCE" \
        --max-retries "$MAX_RETRIES"
    done
  done
done

DATASETS_CSV=$(python - <<PY
import json
from pathlib import Path
cfg=json.loads(Path("$DATA_CFG").read_text())
print(",".join(cfg["datasets"].keys()))
PY
)

python personb/prepare_selection_final.py \
  --run-root "$RAW_ROOT" \
  --result-prefix "$PREFIX" \
  --datasets "$DATASETS_CSV" \
  --seeds "$SEEDS_CSV" \
  --lambdas "$LAMBDAS_CSV" \
  --ncl-space "$NCL_SPACE" \
  --ncl-warmup-epochs "$NCL_WARMUP_EPOCHS" \
  --require-share-training-batches \
  --tie-break "$TIE_BREAK"

bash personb/run_selected_eval_final.sh "$PREFIX"
python personb/build_final_report_final.py --prefix "$PREFIX"

touch "results/${PREFIX}_done.flag"
