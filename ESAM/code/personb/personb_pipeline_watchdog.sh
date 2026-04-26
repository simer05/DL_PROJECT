#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/users/ntu/prithvi2/DeepLearning/TabM/paper
LOG="$ROOT/job_logs/personb_pipeline_watchdog.log"
RIG="$ROOT/personb/personb_ncl_rigorous.pbs"
POST="$ROOT/personb/personb_ncl_post.pbs"
EXPECTED_REPORTS=38

mkdir -p "$ROOT/job_logs"

while true; do
  cd "$ROOT"

  if [ -f results/personb_post_done.flag ] && [ -f results/personb_ncl_final_report.md ]; then
    echo "[$(date)] pipeline complete, watchdog exiting" >> "$LOG"
    exit 0
  fi

  active=$(qstat -u prithvi2 2>/dev/null | awk 'NR>5 && (index($4,"tabm_ncl_")==1) && ($10=="Q" || $10=="R") {c++} END {print c+0}')
  if [ "$active" -gt 0 ]; then
    sleep 300
    continue
  fi

  n_reports=$(python3 - <<'PY'
from pathlib import Path
print(len(list(Path('outputs/personb_ncl_rigorous').glob('*/seed*/lambda_*/report.json'))))
PY
)

  if [ "$n_reports" -lt "$EXPECTED_REPORTS" ]; then
    jid=$(qsub "$RIG")
    echo "[$(date)] submitted rigorous job $jid (reports=$n_reports/$EXPECTED_REPORTS)" >> "$LOG"
    sleep 300
    continue
  fi

  if [ ! -f results/personb_post_done.flag ]; then
    jid=$(qsub "$POST")
    echo "[$(date)] submitted post job $jid" >> "$LOG"
    sleep 300
    continue
  fi

  sleep 300
done
