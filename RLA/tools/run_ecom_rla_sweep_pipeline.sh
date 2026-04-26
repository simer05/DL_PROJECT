#!/bin/bash
set -eo pipefail

cd /workspace/tabm
mkdir -p /workspace/logs
python3 tools/generate_ecom_rla_sweep_configs.py >/workspace/logs/ecom_rla_sweep_generate.log 2>&1

cd /workspace/tabm/paper
export PYTHONPATH="$PWD:$PYTHONPATH"

N_GPUS=${N_GPUS:-8}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/workspace/logs/ecom_rla_sweep_${STAMP}
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

echo "PHASE 1: ecom-offers RLA seed0 smoke"
: > "$QUEUE"
find exp/rla/ecom-offers -mindepth 1 -maxdepth 1 -type d -name 'rla_ecom_sweep_*-evaluation' \
  ! -name '*-best-head-evaluation' ! -name '*-greedy-heads-evaluation' | sort | while read -r eval_dir; do
  variant=$(basename "$eval_dir" -evaluation)
  printf '%s|0|smoke__ecom-offers__%s__s0\n' "$eval_dir" "$variant" >> "$QUEUE"
done
run_workers phase1

echo "PHASE 1 aggregate/select"
cd /workspace/tabm
python3 tools/aggregate_ecom_rla_sweep.py --top-k 2

echo "PHASE 2: rerun top 2 configs at seeds 0,1,2"
cd /workspace/tabm/paper
: > "$QUEUE"
python3 - <<'PY' > "$QUEUE"
import csv
from pathlib import Path
path = Path('/workspace/tabm/paper/exp/rla/_aggregated/ecom_rla_sweep_selected.csv')
for row in csv.DictReader(path.open()):
    variant = row['selected_config'].split(':', 1)[0]
    for seed in (0, 1, 2):
        eval_dir = f'exp/rla/ecom-offers/{variant}-evaluation'
        print(f'{eval_dir}|{seed}|final__ecom-offers__{variant}__s{seed}')
PY
run_workers phase2

echo "FINAL aggregate"
cd /workspace/tabm
python3 tools/aggregate_ecom_rla_sweep.py --top-k 2
echo "DONE $(date) logs=$LOG_DIR"
