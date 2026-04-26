#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results
bash personb/run_selected_eval.sh
python personb/build_final_report.py
touch results/personb_post_done.flag
