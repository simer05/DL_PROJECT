#!/usr/bin/env bash
set -euo pipefail

PAPER_DIR="${PAPER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../paper" && pwd)}"
MANIFEST="${1:-${MANIFEST:-}}"
if [[ -z "$MANIFEST" ]]; then
  echo "usage: $0 paper/exp/integrated/manifest_*.txt" >&2
  exit 2
fi
cd "$PAPER_DIR"
if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  exit 2
fi

PYTHON="${PYTHON:-/workspace/.venvs/tabm_integrated/bin/python}"
N_GPUS="${N_GPUS:-16}"
FORCE="${FORCE:-0}"
LOG_ROOT="exp/integrated/_logs/$(basename "$MANIFEST" .txt)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
QUEUE="exp/integrated/_queue_$(basename "$MANIFEST" .txt)_$$.txt"
LOCK="$QUEUE.lock"
cp "$MANIFEST" "$QUEUE"
: > "$LOCK"
USAGE_CSV="$LOG_ROOT/gpu_usage.csv"
echo "gpu,config,started_at" > "$USAGE_CSV"

cleanup_artifacts() {
  local out="$1"
  rm -f "$out/checkpoint.pt" "$out/checkpoint_best.pt" "$out/predictions.npz" "$out/summary.json"
  rm -f "$out"/events.out.tfevents.* 2>/dev/null || true
}

check_done() {
  local out="$1"
  "$PYTHON" - "$out" <<'PY'
import json, sys
from pathlib import Path
out=Path(sys.argv[1])
report=out/'report.json'
done=out/'DONE'
if not done.exists() or not report.exists():
    raise SystemExit(1)
payload=json.loads(report.read_text())
if payload.get('failure'):
    raise SystemExit(2)
PY
}

worker() {
  local gpu="$1"
  export CUDA_VISIBLE_DEVICES="$gpu"
  while true; do
    local cfg=""
    {
      flock 9
      if [[ -s "$QUEUE" ]]; then
        cfg="$(head -n 1 "$QUEUE")"
        tail -n +2 "$QUEUE" > "$QUEUE.tmp"
        mv "$QUEUE.tmp" "$QUEUE"
      fi
    } 9>"$LOCK"
    [[ -n "$cfg" ]] || break
    local out="${cfg%.toml}"
    local safe
    safe="$(echo "$cfg" | tr '/ ' '__')"
    local log="$LOG_ROOT/gpu${gpu}_${safe}.out"
    echo "$gpu,$cfg,$(date -Is)" >> "$USAGE_CSV"
    if [[ "$FORCE" != "1" ]] && check_done "$out" >/dev/null 2>&1; then
      echo "SKIP $cfg" | tee -a "$log"
      cleanup_artifacts "$out"
      continue
    fi
    mkdir -p "$out"
    echo "RUN gpu=$gpu cfg=$cfg out=$out" | tee "$log"
    set +e
    "$PYTHON" bin/run_integrated.py "$cfg" --output "$out" --force >> "$log" 2>&1
    rc=$?
    set -e
    cleanup_artifacts "$out"
    if [[ $rc -ne 0 ]]; then
      echo "FAILED rc=$rc cfg=$cfg log=$log" | tee -a "$LOG_ROOT/FAILED"
      return $rc
    fi
    if ! check_done "$out" >/dev/null 2>&1; then
      echo "FAILED missing DONE/report or failure block cfg=$cfg log=$log" | tee -a "$LOG_ROOT/FAILED"
      return 1
    fi
    echo "DONE $cfg" | tee -a "$log"
  done
}

pids=()
for ((gpu=0; gpu<N_GPUS; gpu++)); do
  worker "$gpu" &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done
rm -f "$QUEUE" "$LOCK" "$QUEUE.tmp"
if [[ $rc -ne 0 ]]; then
  echo "matrix failed; see $LOG_ROOT/FAILED" >&2
  exit $rc
fi
"$PYTHON" ../tools/aggregate_integrated_results.py --stage wave --manifest "$MANIFEST" || true
used_gpus=$(tail -n +2 "$USAGE_CSV" | cut -d, -f1 | sort -n | uniq | tr '\n' ' ')
echo "used_gpus: $used_gpus"
echo "logs: $LOG_ROOT"
