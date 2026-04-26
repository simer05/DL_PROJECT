#!/bin/bash
# Submit RLA sweep jobs to NSCC PBS scheduler.
#
# Usage from the repo root on NSCC:
#   bash pbs/submit_rla.sh smoke         # smoke test (1 dataset, 4 variants, 3 seeds each)
#   bash pbs/submit_rla.sh full          # full sweep (5 datasets x 10 variants)
#   bash pbs/submit_rla.sh dataset homesite-insurance   # all variants on one dataset
#
# Per Section 12 of the RLA spec, every job is routed through PBS with
# `-P personal-abhipray` and the `normal` queue.

set -eo pipefail

cd "$(dirname "$0")/.."
mkdir -p "$HOME/tabmpp/logs"

DATASETS_ALL=(
    homesite-insurance
    ecom-offers
    sberbank-housing
    cooking-time
    delivery-eta
)

VARIANTS_SMOKE=(
    baseline
    rla_first_r2
    rla_first_r4
    rla_uniform_r2
)

VARIANTS_FULL=(
    baseline
    rla_first_r1 rla_first_r2 rla_first_r4 rla_first_r8
    rla_uniform_r1 rla_uniform_r2 rla_uniform_r4 rla_uniform_r8
    rla_additive_first_r4
)

DATASETS_V2=(
    sberbank-housing
    ecom-offers
    homesite-insurance
    cooking-time
    delivery-eta
)

# v2: PLR + mini-PLR baselines with base-preserving init.
VARIANTS_V2=(
    baseline_plr
    rla_plr_first_r2_basepreserve
    rla_plr_first_r4_basepreserve
    rla_plr_uniform_r4_basepreserve
    rla_plr_uniform_r8_basepreserve
    baseline_mini_plr
    rla_mini_plr_first_r2_basepreserve
    rla_mini_plr_first_r4_basepreserve
)

submit_one() {
    local ds="$1" var="$2" nseeds="${3:-3}"
    local logdir="$HOME/tabmpp/logs"
    local tag="rla_${ds}_${var}"
    qsub \
        -v "DATASET=$ds,VARIANT=$var,N_SEEDS=$nseeds" \
        -N "$tag" \
        -o "$logdir/${tag}.log" \
        -e "$logdir/${tag}.err" \
        pbs/run_rla.pbs
}

mode="${1:-smoke}"
case "$mode" in
    smoke)
        ds="homesite-insurance"
        for v in "${VARIANTS_SMOKE[@]}"; do
            submit_one "$ds" "$v" 3
        done
        ;;
    full)
        for ds in "${DATASETS_ALL[@]}"; do
            for v in "${VARIANTS_FULL[@]}"; do
                submit_one "$ds" "$v" 3
            done
        done
        ;;
    dataset)
        ds="${2:?dataset name required}"
        for v in "${VARIANTS_FULL[@]}"; do
            submit_one "$ds" "$v" 3
        done
        ;;
    v2_smoke)
        # 2-seed smoke pass: PLR + mini-PLR families on 3 datasets,
        # base-preserving init.
        for ds in "${DATASETS_V2[@]}"; do
            for v in "${VARIANTS_V2[@]}"; do
                submit_one "$ds" "$v" 2
            done
        done
        ;;
    v2)
        # Full 3-seed v2 pass.
        for ds in "${DATASETS_V2[@]}"; do
            for v in "${VARIANTS_V2[@]}"; do
                submit_one "$ds" "$v" 3
            done
        done
        ;;
    expand)
        # Expand a single (dataset, variant) to N seeds.
        ds="${2:?dataset required}"
        v="${3:?variant required}"
        nseeds="${4:-5}"
        submit_one "$ds" "$v" "$nseeds"
        ;;
    *)
        echo "Unknown mode: $mode (smoke|full|dataset <name>)"
        exit 1
        ;;
esac
