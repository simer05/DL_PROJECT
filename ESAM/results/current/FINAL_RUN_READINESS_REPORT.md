# Final Run Readiness Report
- generated_at: 2026-04-26T16:39:45
- branch: `feature/tabm-esam`
- commit: `28e47ae301c92ec37787dde1ce923a0793f405b4`

| Area | Status | Ready? | Requires Rerun? | Notes |
|---|---|---|---|---|
| Canonical data | FAIL | no | yes | Canonical artifacts/split-default/sha256 not verified |
| Five datasets ready | FAIL | no | yes | ecom and delivery blocked under strict canonical policy |
| Ecom validity | FAIL | no | yes | canonical audit blocked; prior leakage risk |
| Delivery validity | FAIL | no | yes | canonical audit blocked |
| Code surface | FAIL | no | yes | shared core files modified beyond strict allowance |
| Baseline equivalence | FAIL | no | yes | full-model bit-identical test not complete |
| Config generation | FAIL | no | yes | generator/run layout not finalized |
| report.json metadata | FAIL | no | yes | git_commit/amp_dtype missing |
| Run layout | FAIL | no | yes | not using paper/exp/esam strict layout |
| Inference modes | FAIL | no | yes | 3-mode eval missing for strict final |
| 3-seed plan | PASS | yes | yes | plan defined; runs pending |
| Aggregator | FAIL | no | yes | strict aggregator not finalized |

## Final Decision
**C. NOT READY — DATA BLOCKER**

## Launch Command
- Not provided because readiness is NOT READY.

## Estimated strict run count and ETA once unblocked
- Train runs: 75 (5 datasets x 5 variants x 3 seeds)
- Additional eval-mode passes: mean/best-head/greedy-heads per variant/seed
- Expected ETA: to be recomputed after canonical data is installed and one benchmark run is profiled.
