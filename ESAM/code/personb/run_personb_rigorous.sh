#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/personb_ncl_rigorous job_logs results personb

DATASETS=(adult california covtype2)
SEEDS=(0 1 2)
LAMBDAS=(0 5e-4 1e-3 2e-3)

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for lam in "${LAMBDAS[@]}"; do
      python personb/run_experiment.py \
        --dataset "$ds" \
        --seed "$seed" \
        --lambda-ncl "$lam" \
        --ncl-warmup-epochs 10 \
        --n-epochs 80 \
        --patience 12 \
        --max-retries 2
    done
  done
done

# One compact method ablation: warmup (adult, seed2, lambda=1e-3)
python personb/run_experiment.py --dataset adult --seed 2 --lambda-ncl 1e-3 --ncl-warmup-epochs 0  --n-epochs 80 --patience 12 --max-retries 2 --tag warmup0
python personb/run_experiment.py --dataset adult --seed 2 --lambda-ncl 1e-3 --ncl-warmup-epochs 20 --n-epochs 80 --patience 12 --max-retries 2 --tag warmup20
