# Person B TabReD Launch Guide

## Expected dataset config
Default config file:
- `personb/tabred_dataset_config.json`

Default data root currently set to:
- `/home/users/ntu/prithvi2/DeepLearning/TabM/paper/data`

Final dataset keys configured:
- `sberbank` -> `sberbank-housing`
- `ecom-offers` -> `ecom-offers`
- `homesite` -> `homesite-insurance`
- `cooking-time` -> `cooking-time`
- `delivery-eta` -> `delivery-eta`

Update only `data_root` (or the per-dataset dirs if needed) after rsync.

## Readiness check
Run:
```bash
cd /home/users/ntu/prithvi2/DeepLearning/TabM/paper
python personb/check_tabred_ready.py --config personb/tabred_dataset_config.json --strict
```

- `READY` for all 5 datasets is required before final launch.
- If missing, the script prints exact missing paths/files.

## Final launch (single job, sequential)
Submit:
```bash
cd /home/users/ntu/prithvi2/DeepLearning/TabM/paper
qsub personb/personb_tabred_final.pbs
```

This runs:
1. readiness check
2. raw training grid (5 datasets x 3 seeds x 4 lambdas)
3. validation-only lambda selection
4. baseline vs selected eval artifacts (clean/corruptions)
5. final summary/report generation

## Direct non-PBS launch (if needed)
```bash
cd /home/users/ntu/prithvi2/DeepLearning/TabM/paper
bash personb/run_personb_tabred_final.sh
```

## Main configurable knobs (env vars)
Set before launch if needed:
- `PREFIX` (default `personb_tabred_final`)
- `NCL_SPACE` (`logits`, `probs`, `hybrid`; default `logits`)
- `SEEDS_CSV` (default `0,1,2`)
- `LAMBDAS_CSV` (default `0,5e-4,1e-3,2e-3`)
- `TIE_BREAK` (`none` or `val_cross_entropy`, default `val_cross_entropy`)

Example:
```bash
PREFIX=personb_tabred_final_logits NCL_SPACE=logits qsub personb/personb_tabred_final.pbs
```

## Penalty-space ablation launcher
For logits/probs/hybrid runs with isolated prefixes:
```bash
cd /home/users/ntu/prithvi2/DeepLearning/TabM/paper
bash personb/run_personb_tabred_space_ablation.sh
```

This avoids collisions by using different `PREFIX` per space.

## Expected output locations
Raw runs:
- `outputs/<PREFIX>/raw/<dataset>/seed<seed>/lambda_<...>/`

Selection:
- `results/<PREFIX>_selected_runs.json`

Eval artifacts:
- `results/<PREFIX>_eval/`

Final summaries:
- `results/<PREFIX>_summary.csv`
- `results/<PREFIX>_report.md`
- `results/<PREFIX>_manifest.md`
- `results/<PREFIX>_done.flag`

## Selection and reporting rules
- Best lambda selected per dataset+seed by **validation score only**
- Optional classification tie-break: lower validation cross-entropy
- Test metric is reported only after selection
- Final report aggregates mean ± std across fixed seeds

## Team convention enforced
- `share_training_batches=true` is forced in final launcher
- 3 fixed seeds per config
- dev datasets remain dev-only; TabReD is final benchmark
