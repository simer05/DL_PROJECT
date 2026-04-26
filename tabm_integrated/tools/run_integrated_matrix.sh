#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER="$ROOT/paper"
PYTHON="${PYTHON:-/workspace/.venvs/tabm_integrated/bin/python}"
N_GPUS="${N_GPUS:-16}"
SEEDS="${SEEDS:-0}"
VARIANTS="${VARIANTS:-baseline_plr rla_only esam_only mfb_only cf_fisd_only all_four_combined}"
DATASETS="sberbank-housing ecom-offers homesite-insurance cooking-time delivery-eta"
LOG_ROOT="$PAPER/exp/integrated/_logs"
QUEUE="$PAPER/exp/integrated/_queue_${SEEDS// /_}.txt"
mkdir -p "$LOG_ROOT"
"$PYTHON" "$ROOT/tools/generate_integrated_configs.py"
: > "$QUEUE"
for seed in $SEEDS; do
  for dataset in $DATASETS; do
    for variant in $VARIANTS; do
      cfg="$PAPER/exp/integrated/$dataset/$variant-evaluation/$seed.toml"
      out="$PAPER/exp/integrated/$dataset/$variant-evaluation/$seed"
      if [[ -f "$out/DONE" && -f "$out/report.json" ]] && "$PYTHON" - "$out/report.json" <<'PYCHK'
import json, sys
r=json.load(open(sys.argv[1]))
sys.exit(1 if r.get('failure') else 0)
PYCHK
      then
        echo "reuse $dataset $variant $seed"
      else
        echo "$cfg|$out|$dataset|$variant|$seed" >> "$QUEUE"
      fi
    done
  done
done
worker() {
  gpu="$1"
  while true; do
    line=""
    exec 9<>"$QUEUE.lock"
    flock 9
    if [[ -s "$QUEUE" ]]; then
      line="$(head -n 1 "$QUEUE")"
      tail -n +2 "$QUEUE" > "$QUEUE.tmp"
      mv "$QUEUE.tmp" "$QUEUE"
    fi
    flock -u 9
    [[ -z "$line" ]] && break
    IFS='|' read -r cfg out dataset variant seed <<< "$line"
    mkdir -p "$(dirname "$out")" "$LOG_ROOT/$dataset/$variant"
    log="$LOG_ROOT/$dataset/$variant/seed${seed}.gpu${gpu}.log"
    echo "START $(date -Iseconds) gpu=$gpu dataset=$dataset variant=$variant seed=$seed" | tee "$log"
    if ! (cd "$PAPER" && CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$PAPER/bin/run_integrated.py" "$cfg" "$out" --force) >> "$log" 2>&1; then
      echo "FAIL dataset=$dataset variant=$variant seed=$seed log=$log" | tee -a "$log"
      touch "$PAPER/exp/integrated/FAILED"
      exit 1
    fi
    if [[ ! -f "$out/DONE" || ! -f "$out/report.json" ]]; then
      echo "FAIL missing DONE/report dataset=$dataset variant=$variant seed=$seed log=$log" | tee -a "$log"
      touch "$PAPER/exp/integrated/FAILED"
      exit 1
    fi
    if ! "$PYTHON" - "$out/report.json" <<'PYCHK'
import json, sys
r=json.load(open(sys.argv[1]))
sys.exit(1 if r.get('failure') else 0)
PYCHK
    then
      echo "FAIL failure block dataset=$dataset variant=$variant seed=$seed log=$log" | tee -a "$log"
      touch "$PAPER/exp/integrated/FAILED"
      exit 1
    fi
    echo "DONE $(date -Iseconds) gpu=$gpu dataset=$dataset variant=$variant seed=$seed" | tee -a "$log"
  done
}
rm -f "$PAPER/exp/integrated/FAILED"
pids=()
for gpu in $(seq 0 $((N_GPUS - 1))); do worker "$gpu" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
"$PYTHON" "$ROOT/tools/aggregate_integrated_results.py" || status=1
exit "$status"
