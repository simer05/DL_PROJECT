# MANIFESTO Compliance Fix Plan
- generated_at: 2026-04-26T16:39:45
- branch: `feature/tabm-esam`
- commit: `28e47ae301c92ec37787dde1ce923a0793f405b4`

| Area | Current Status | Fix Required | Requires Rerun? | Priority |
|---|---|---|---|---|
| Canonical TabReD data/splits | FAIL_NEEDS_DATA_RERUN | Acquire pinned canonical .npy + split-default idx + manifest sha256 | yes | P0 |
| Custom preprocessing split policies | FAIL_NEEDS_DATA_RERUN | Stop using custom stratified/group_hash for final; use canonical split-default | yes | P0 |
| paper/lib/data.py modified | FAIL_NEEDS_FIX | Constrain to safe NaN check or revert to baseline-compatible minimal patch | possibly | P1 |
| Run layout under outputs/esam | FAIL_NEEDS_EXPERIMENT_RERUN | Use paper/exp/esam/<dataset>/<variant>-evaluation/<seed> layout | yes | P0 |
| Inference modes mean/best/greedy | FAIL_NEEDS_EXPERIMENT_RERUN | Run evaluation modes for each completed training run | yes | P1 |
| 3-seed minimum | FAIL_NEEDS_EXPERIMENT_RERUN | Run seeds 0/1/2 (or approved 42/43/44) | yes | P0 |
| report.json missing git_commit/amp_dtype | FAIL_NEEDS_FIX | Add metadata writing in model report block | yes (new runs) | P1 |
| ecom-offers leakage risk | FAIL_NEEDS_DATA_RERUN | Re-audit on canonical split only; block claims until clean | yes | P0 |
| delivery-eta conditional validity | FAIL_NEEDS_DATA_RERUN | Canonical audit + sanity baseline under canonical pipeline | yes | P0 |
| Off-state equivalence test | FAIL_NEEDS_FIX | Add full-model deterministic off-state equivalence test | no | P1 |
| Run dirs require 0.toml/report.json/DONE | FAIL_NEEDS_EXPERIMENT_RERUN | Use strict run-dir contract + DONE marker | yes | P0 |
| Aggregator under paper/exp/esam/_aggregated | FAIL_NEEDS_FIX | Add compliant aggregator outputs and filters | no | P1 |
| NSCC/PBS rules | NOT_APPLICABLE_ARGCLUSTER | Ignored per instruction | no | N/A |
