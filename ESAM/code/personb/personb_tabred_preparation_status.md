# Person B TabReD Preparation Status

## Scope and objective
This update prepares the Person B pipeline (TabM + explicit NCL on rank-1 members) for immediate final TabReD benchmarking once the dataset is rsynced.

## What was found
- Root workspace: `/home/users/ntu/prithvi2/DeepLearning`
- Repo: `/home/users/ntu/prithvi2/DeepLearning/TabM`
- Paper code root: `/home/users/ntu/prithvi2/DeepLearning/TabM/paper`
- Scheduler detected: PBS (`qsub`, `qstat`)
- Existing Person B stack already present under `paper/personb/`:
  - training launcher and PBS scripts
  - validation-based selection and post-eval pipeline
  - reporting artifacts in `paper/results/`
- Dataset loader is file-based via `lib/data.py::build_dataset(path=...)` and expects dataset directories containing `info.json`, `Y_*.npy`, and feature arrays.

## Current Person B pipeline map
- Core model/training entry: `paper/bin/model.py`
- Person B train wrapper: `paper/personb/run_experiment.py`
- Selection logic: `paper/personb/prepare_selection.py` (dev-rigorous legacy), and new `prepare_selection_final.py` (final-ready)
- Eval and diagnostics: `paper/personb/evaluate_checkpoint.py`
- Final summary/report: `paper/personb/build_final_report.py` (legacy) and new `build_final_report_final.py`

## Key technical locations
- Per-member logits interface (shape `(B, K, D_OUT)`): `paper/bin/model.py`, `Model.forward`
- Added integration-safe helper: `Model.get_member_logits(...)`
- First BatchEnsemble layer `r` parameter definition: `paper/lib/deep.py`, `LinearEfficientEnsemble.r`
- Added integration-safe helper: `Model.get_first_batchensemble_r()`
- `share_training_batches` usage:
  - model arg and runtime path in `paper/bin/model.py`
  - final launcher now forces team convention `share_training_batches=true`

## Changes made in this preparation
1. Team-convention alignment
- `personb/run_experiment.py` now supports final-convention controls and keeps override ability:
  - `--share-training-batches/--no-share-training-batches`
  - default set to `true` for launchers
  - supports `--base-config`, `--data-root`, `--dataset-dir`, `--output-root`

2. TabReD dataset readiness layer
- Added central dataset config:
  - `personb/tabred_dataset_config.json`
- Added readiness checker:
  - `personb/check_tabred_ready.py`
  - strict mode exits non-zero if any required dataset/files are missing

3. Final launch-ready pipeline scripts
- Added final sequential launcher:
  - `personb/run_personb_tabred_final.sh`
- Added final PBS job script:
  - `personb/personb_tabred_final.pbs`
- Added final selection script (validation-only + optional calibration tie-break):
  - `personb/prepare_selection_final.py`
- Added final eval runner:
  - `personb/run_selected_eval_final.sh`
- Added final report/summary builder:
  - `personb/build_final_report_final.py`

4. Compact within-scope improvement path
- NCL penalty-space now supports:
  - `logits`
  - `probs`
  - `hybrid` (50/50 logits + probs)
- Implemented in `paper/bin/model.py` (`ncl_space` now includes `hybrid`)
- Added space-ablation launcher:
  - `personb/run_personb_tabred_space_ablation.sh`
- Selection supports calibration-aware tie-break for classification:
  - `--tie-break val_cross_entropy`

5. Integration compatibility
- Kept model output contract unchanged: per-member outputs stay `(B, K, D_OUT)`
- Did not modify or overwrite BE `r`; only exposed accessor helper
- No architecture refactor beyond minimal loss-level extension

## Dev-only sanity checks completed
- Small smoke checks executed on `adult` only with `n_epochs=1`:
  - baseline and NCL runs under `share_training_batches=true`
  - verified `ncl_space=hybrid` training path works
  - verified selection/eval/report scripts on smoke outputs
- Smoke artifacts generated under:
  - `outputs/personb_dev_smoke*`
  - `results/personb_dev_smoke_hybrid_*`

## Final team convention status
- Convention encoded in final launch path:
  - `share_training_batches=true`
  - fixed seeds = `0,1,2`
  - lambda grid = `0, 5e-4, 1e-3, 2e-3`
  - lambda selected by validation only
- Final report builder records convention and checks selected-run consistency.

## Blockers
- Only blocker is data availability:
  - TabReD dataset directories are currently missing at configured root.

## Readiness verdict
- Pipeline is launch-ready for final TabReD benchmarking once rsync provides data at configured paths.
