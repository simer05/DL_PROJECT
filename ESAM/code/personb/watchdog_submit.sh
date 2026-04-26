#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/users/ntu/prithvi2/DeepLearning/TabM/paper
PBS_SCRIPT="$ROOT/personb/personb_ncl_singlejob.pbs"
LOG="$ROOT/job_logs/watchdog.log"

mkdir -p "$ROOT/job_logs"

while true; do
  cd "$ROOT"

  if [ -f results/personb_ncl_results.csv ] && [ "$(wc -l < results/personb_ncl_results.csv)" -gt 1 ]; then
    echo "[$(date)] results present, watchdog exiting" >> "$LOG"
    exit 0
  fi

  active=$(qstat -u prithvi2 2>/dev/null | awk 'NR>5 && $4=="tabm_ncl_b" && ($10=="Q" || $10=="R") {c++} END {print c+0}')
  if [ "$active" -eq 0 ]; then
    jobid=$(qsub "$PBS_SCRIPT")
    echo "[$(date)] submitted $jobid" >> "$LOG"
  fi

  sleep 300
done
