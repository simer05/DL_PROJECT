#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

BASE_PREFIX="${BASE_PREFIX:-personb_dev_space_ablation}"
DATA_CFG="${DATA_CFG:-personb/dev_dataset_config.json}"
SPACES_CSV="${SPACES_CSV:-logits,probs,hybrid}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
LAMBDAS_CSV="${LAMBDAS_CSV:-0,5e-4,1e-3,2e-3}"
N_EPOCHS="${N_EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
MAX_RETRIES="${MAX_RETRIES:-2}"
NCL_WARMUP_EPOCHS="${NCL_WARMUP_EPOCHS:-10}"

mkdir -p "results" "outputs/${BASE_PREFIX}" "job_logs"
python personb/check_tabred_ready.py --config "$DATA_CFG"

DATA_ROOT=$(python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("$DATA_CFG").read_text())
print(cfg["data_root"])
PY
)

mapfile -t DATASET_ROWS < <(python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("$DATA_CFG").read_text())
for key, meta in cfg["datasets"].items():
    print(f"{key}\t{meta['dataset_dir']}\t{meta['base_config']}")
PY
)

DATASETS_CSV=$(python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("$DATA_CFG").read_text())
print(",".join(cfg["datasets"].keys()))
PY
)

IFS=',' read -r -a SPACES <<< "$SPACES_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a LAMBDAS <<< "$LAMBDAS_CSV"

for space in "${SPACES[@]}"; do
  space_trimmed="$(echo "$space" | xargs)"
  prefix="${BASE_PREFIX}_${space_trimmed}"
  raw_root="outputs/${BASE_PREFIX}/raw_${space_trimmed}"
  mkdir -p "$raw_root"

  echo "[space] ${space_trimmed}"
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
          --output-root "$raw_root" \
          --seed "$seed" \
          --lambda-ncl "$lam" \
          --ncl-warmup-epochs "$NCL_WARMUP_EPOCHS" \
          --ncl-space "$space_trimmed" \
          --share-training-batches \
          --n-epochs "$N_EPOCHS" \
          --patience "$PATIENCE" \
          --max-retries "$MAX_RETRIES"
      done
    done
  done

  for tie_break in none val_cross_entropy; do
    sel_prefix="${prefix}_${tie_break}"
    python personb/prepare_selection_final.py \
      --run-root "$raw_root" \
      --result-prefix "$sel_prefix" \
      --datasets "$DATASETS_CSV" \
      --seeds "$SEEDS_CSV" \
      --lambdas "$LAMBDAS_CSV" \
      --ncl-space "$space_trimmed" \
      --ncl-warmup-epochs "$NCL_WARMUP_EPOCHS" \
      --require-share-training-batches \
      --tie-break "$tie_break"

    bash personb/run_selected_eval_final.sh "$sel_prefix"
    python personb/build_final_report_final.py \
      --prefix "$sel_prefix" \
      --convention-share-training-batches
  done
done

python personb/summarize_dev_space_ablation.py \
  --base-prefix "$BASE_PREFIX" \
  --spaces "$SPACES_CSV" \
  --tie-breaks "none,val_cross_entropy" \
  --output-md "results/${BASE_PREFIX}_summary.md" \
  --output-csv "results/${BASE_PREFIX}_summary.csv" \
  --output-json "results/${BASE_PREFIX}_summary.json"

touch "results/${BASE_PREFIX}_done.flag"
