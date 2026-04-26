#!/bin/bash
set -eo pipefail

cd /workspace/tabm
mkdir -p /workspace/logs
python3 tools/generate_rla_followup_configs.py >/workspace/logs/rla_followup_generate.log 2>&1

cd /workspace/tabm/paper
export PYTHONPATH="$PWD:$PYTHONPATH"

N_GPUS=${N_GPUS:-8}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/workspace/logs/rla_followup_${STAMP}
mkdir -p "$LOG_DIR"
QUEUE="$LOG_DIR/queue.txt"

pop_task() {
  {
    flock -x 200
    if [ -s "$QUEUE" ]; then
      head -n 1 "$QUEUE"
      tail -n +2 "$QUEUE" > "$QUEUE.tmp"
      mv "$QUEUE.tmp" "$QUEUE"
    fi
  } 200>"$QUEUE.lock"
}

run_workers() {
  local phase=$1
  for gpu in $(seq 0 $((N_GPUS - 1))); do
    (
      while true; do
        task=$(pop_task)
        [ -n "$task" ] || break
        IFS='|' read -r eval_dir seed tag <<< "$task"
        echo "[$phase][GPU $gpu] START $eval_dir seed=$seed"
        CUDA_VISIBLE_DEVICES=$gpu python3 ../tools/run_single_seed.py "$eval_dir" "$seed" > "$LOG_DIR/${tag}.log" 2> "$LOG_DIR/${tag}.err"
        echo "[$phase][GPU $gpu] DONE  $eval_dir seed=$seed"
      done
    ) &
  done
  wait
}

echo "PHASE 1: RLA follow-up seed0 smoke"
: > "$QUEUE"
find exp/rla -mindepth 2 -maxdepth 2 -type d -name 'rla_followup_*-evaluation' | sort | while read -r eval_dir; do
  variant=$(basename "$eval_dir" -evaluation)
  dataset=$(basename "$(dirname "$eval_dir")")
  printf '%s|0|smoke__%s__%s__s0\n' "$eval_dir" "$dataset" "$variant" >> "$QUEUE"
done
run_workers phase1

echo "PHASE 1 aggregate/select"
cd /workspace/tabm
python3 tools/aggregate_rla_results.py
python3 tools/aggregate_rla_claims.py
python3 tools/aggregate_rla_claims.py --variant-prefix rla_followup_ --out-name rla_followup_claim_table.csv

echo "PHASE 2: rerun validation-selected follow-up winners at seeds 0,1,2"
cd /workspace/tabm/paper
: > "$QUEUE"
python3 - <<'PY' > "$QUEUE"
import csv
from pathlib import Path
path = Path('/workspace/tabm/paper/exp/rla/_aggregated/rla_followup_claim_table.csv')
if path.exists():
    for row in csv.DictReader(path.open()):
        variant_mode = row['selected_config']
        variant = variant_mode.split(':', 1)[0]
        if not variant.startswith('rla_followup_'):
            continue
        if row['validation_score_delta'] in ('', 'None'):
            continue
        if float(row['validation_score_delta']) <= 0.0:
            continue
        dataset = row['dataset']
        for seed in (0, 1, 2):
            eval_dir = f'exp/rla/{dataset}/{variant}-evaluation'
            print(f'{eval_dir}|{seed}|final__{dataset}__{variant}__s{seed}')
PY
if [ -s "$QUEUE" ]; then
  run_workers phase2
else
  echo "No validation-positive follow-up winners to rerun."
fi

echo "FINAL aggregate"
cd /workspace/tabm
python3 tools/aggregate_rla_results.py
python3 tools/aggregate_rla_claims.py
python3 tools/aggregate_rla_claims.py --variant-prefix rla_followup_ --out-name rla_followup_claim_table.csv
cp paper/exp/rla/_aggregated/rla_claim_table.csv paper/exp/rla/_aggregated/rla_claim_table_final.csv
cp paper/exp/rla/_aggregated/rla_followup_claim_table.csv paper/exp/rla/_aggregated/rla_followup_claim_table_final.csv
echo "DONE $(date) logs=$LOG_DIR"
