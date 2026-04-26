#!/bin/bash
# Submit the P0 sweep on NSCC using qsub + `-W depend=afterany:<prev>` chaining
# so we don't spam the queue.
#
# Usage:  pbs/submit_sweep.sh [--smoke]
#
#   --smoke : submit only (sberbank-housing, sls_default) with 1 seed.

set -eo pipefail

ROOT="$HOME/tabmpp/tabm"
cd "$ROOT"

DATASETS=(homesite-insurance ecom-offers sberbank-housing cooking-time delivery-eta)
VARIANTS=(baseline sls_default sls_gate_off sls_lambda_0p01 sls_lambda_1p0)
N_SEEDS=3

if [[ "${1:-}" == "--smoke" ]]; then
    DATASETS=(sberbank-housing)
    VARIANTS=(sls_default)
    N_SEEDS=1
fi

PREV_JID=""
N=0
for ds in "${DATASETS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        dep_arg=()
        if [[ -n "$PREV_JID" ]]; then
            dep_arg=(-W "depend=afterany:${PREV_JID}")
        fi
        JID=$(qsub \
            -v "DATASET=${ds},VARIANT=${variant},N_SEEDS=${N_SEEDS}" \
            "${dep_arg[@]}" \
            pbs/run_variant.pbs)
        N=$((N + 1))
        echo "[$N] submitted ${ds}/${variant} as ${JID}${PREV_JID:+ (depends on ${PREV_JID})}"
        PREV_JID="$JID"
    done
done
echo "done: ${N} jobs queued"
