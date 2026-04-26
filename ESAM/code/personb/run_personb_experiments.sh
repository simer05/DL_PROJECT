#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/personb_ncl job_logs results personb

# Full lambda sweep (seed 0) on all 3 datasets.
LAMBDAS=(0 1e-4 5e-4 1e-3 5e-3 1e-2)
DATASETS=(adult california covtype2)

for ds in "${DATASETS[@]}"; do
  for lam in "${LAMBDAS[@]}"; do
    python personb/run_experiment.py --dataset "$ds" --seed 0 --lambda-ncl "$lam" --ncl-warmup-epochs 10 --n-epochs 80 --patience 12
  done
done

# Second seed for baseline and best-priority lambda candidate.
for ds in "${DATASETS[@]}"; do
  python personb/run_experiment.py --dataset "$ds" --seed 1 --lambda-ncl 0 --ncl-warmup-epochs 10 --n-epochs 80 --patience 12
  python personb/run_experiment.py --dataset "$ds" --seed 1 --lambda-ncl 1e-3 --ncl-warmup-epochs 10 --n-epochs 80 --patience 12
done

python personb/summarize_results.py --outputs outputs/personb_ncl --out-csv results/personb_ncl_results.csv --out-md results/personb_ncl_results.md
