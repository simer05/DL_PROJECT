# ESAM Next Phase Plan

## Scope
This plan follows the requested order and executes **Phase 1 first** (Cooking-time controlled block), then pauses for summary before larger follow-up runs.

## Reuse vs New Runs

### Reuse
- Homesite low-data completed results (75 runs) from:
  - `/mnt/ssd/users/prithvi/deepLearning/outputs/esam/low_data/raw/homesite-insurance/...`
- ESAM implementation and low-data pipeline utilities:
  - `esam/run_experiment_esam.py`
  - `esam/build_low_data_tables.py`
- Data integrity outputs:
  - `/mnt/ssd/users/prithvi/deepLearning/results/esam/ECOM_DATA_INTEGRITY_AUDIT.md`
  - `/mnt/ssd/users/prithvi/deepLearning/results/esam/dataset_integrity_summary.csv`
- Cooking-time preprocessed data (READY):
  - `/mnt/ssd/users/prithvi/deepLearning/data/tabred_preprocessed/cooking-time`

### New Runs to Launch (Phase 1)
Dataset: `cooking-time`
- Fractions: `0.10, 0.20, 0.50, 1.00`
- Seeds: `42, 43, 44`
- Methods:
  - Baseline (`rho=0.0`)
  - ESAM (`rho=0.0025`, `start_epoch=0`)
- Total new runs: `4 x 3 x 2 = 24`

## Scripts to Add/Modify
- Add `esam/run_esam_cooking_controlled.sh`
  - sequential launcher, one GPU only
  - runs the 24 Phase-1 jobs
- Add `esam/build_cooking_controlled_tables.py`
  - creates:
    - `cooking_time_controlled_results.csv`
    - `cooking_time_paired_deltas.csv`
    - `COOKING_TIME_ESAM_REPORT.md`

No baseline architecture/objective changes. No test-based tuning.

## Expected Output Files (Phase 1)
- `/mnt/ssd/users/prithvi/deepLearning/results/esam/cooking_time_controlled_results.csv`
- `/mnt/ssd/users/prithvi/deepLearning/results/esam/cooking_time_paired_deltas.csv`
- `/mnt/ssd/users/prithvi/deepLearning/results/esam/COOKING_TIME_ESAM_REPORT.md`

## Estimated Runtime (Phase 1)
- Expected per-run runtime: medium (dataset is large but model is compact).
- Estimated wall-clock for 24 sequential runs: ~2 to 5 hours (depends on early stopping and epoch durations).

## Post-Phase-1 Stop Point
After Phase 1 finishes:
- report completion status
- provide summary table and paired deltas
- state whether Cooking-time strengthens or weakens the ESAM low-data claim
- recommend next action
