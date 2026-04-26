#!/bin/bash
# Run RLA sweep in parallel across N_GPUS local GPUs.
#
# Usage (on the rented vast.ai box):
#   cd /workspace/tabm/paper
#   bash ../tools/run_vast_parallel.sh
#
# Distributes (dataset, variant) jobs across CUDA_VISIBLE_DEVICES 0..N-1.
# Each job runs N_SEEDS seeds sequentially via paper/bin/evaluate.py.

set -eo pipefail

cd "$(dirname "$0")/../paper"
export PYTHONPATH="$PWD:$PYTHONPATH"

N_GPUS=${N_GPUS:-8}
N_SEEDS=${N_SEEDS:-3}
LOG_DIR=${LOG_DIR:-/workspace/logs}
mkdir -p "$LOG_DIR"

DATASETS=(
    sberbank-housing
    ecom-offers
    homesite-insurance
    cooking-time
    delivery-eta
)

# Final matched-PLR matrix per Codex P0 plan: 7 variants × 5 datasets = 35 cells.
VARIANTS=(
    baseline_plr
    baseline_plr_fp32
    rla_plr_first_r2_basepreserve
    rla_plr_first_r2_basepreserve_fp32
    rla_plr_first_r4_basepreserve_fp32
    rla_plr_uniform_r4_basepreserve_fp32
    rla_plr_uniform_r8_basepreserve_fp32
)

JOBS=()
MISSING=()
for ds in "${DATASETS[@]}"; do
    for var in "${VARIANTS[@]}"; do
        if [[ -f "exp/rla/$ds/$var-evaluation/0.toml" ]]; then
            JOBS+=("$ds:$var")
        else
            MISSING+=("$ds:$var")
        fi
    done
done
if (( ${#MISSING[@]} > 0 )); then
    echo "FATAL: missing ${#MISSING[@]} templates:" >&2
    for m in "${MISSING[@]}"; do echo "  - exp/rla/$m-evaluation/0.toml" >&2; done
    echo "Refusing to launch a partial sweep. Regenerate configs first." >&2
    exit 2
fi

echo "Total jobs: ${#JOBS[@]} on $N_GPUS GPUs (N_SEEDS=$N_SEEDS)"
echo "Logs in $LOG_DIR/"
echo "Started: $(date)"

gpu_idx=0
for spec in "${JOBS[@]}"; do
    IFS=: read ds var <<< "$spec"
    g=$((gpu_idx % N_GPUS))
    log="$LOG_DIR/${ds}__${var}.log"
    err="$LOG_DIR/${ds}__${var}.err"
    CUDA_VISIBLE_DEVICES=$g \
        nohup python3 bin/evaluate.py "exp/rla/$ds/$var-evaluation" \
            --function bin.model.main \
            --n_seeds "$N_SEEDS" \
            --force \
            > "$log" 2> "$err" &
    echo "[GPU $g] launched $ds/$var (pid $!)"
    gpu_idx=$((gpu_idx + 1))
    # Throttle to N_GPUS concurrent jobs.
    if (( $(jobs -r -p | wc -l) >= N_GPUS )); then
        wait -n
    fi
done
wait
echo "Finished: $(date)"
