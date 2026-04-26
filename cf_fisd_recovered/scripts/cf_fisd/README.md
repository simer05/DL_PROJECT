# CF-FISD operations playbook

NSCC PBS, account `personal-simerjit`, queue `normal`, ≤5 concurrent.
All compute through `qsub`; never on the login node.

## One-time setup (4 PBS jobs, ≤5 min wall each)

1. Sync repo to NSCC:
   ```bash
   bash scripts/cf_fisd/deploy.sh
   ```

2. Install paper-aligned deps inside simerjit's venv:
   ```bash
   ssh nscc 'qsub /home/users/ntu/simerjit/tabm_proj/tabm_fork/scripts/cf_fisd/nscc_setup_env.pbs'
   ```

3. Convert TabReD legacy layout to paper split-files layout and symlink:
   ```bash
   ssh nscc 'qsub /home/users/ntu/simerjit/tabm_proj/tabm_fork/scripts/cf_fisd/nscc_prepare_data.pbs'
   ```

4. Verify data integrity (sha256, canonical row counts, Rules 87–89):
   ```bash
   ssh nscc 'cd /home/users/ntu/simerjit/tabm_proj/tabm_fork/paper && \
     /home/users/ntu/simerjit/tabm_proj/code/venv/bin/python \
     scripts/cf_fisd/verify_data_integrity.py --data-root data --write'
   ```
   This must run on a compute node (interactive PBS), not the login node.

## Train teachers (5 jobs, ~30 min wall)

```bash
NSCC_PW=... python scripts/cf_fisd/throttled_submit.py --kind teacher
```

Outputs land at `paper/exp/cf_fisd/_teachers/tabred/<dataset>/{xgb,lgbm,cat}.npy`.

## Baseline sanity (25 jobs, ~30 min wall)

```bash
NSCC_PW=... python scripts/cf_fisd/throttled_submit.py \
  --kind student --variants baseline_plr --n-seeds 5
```

After completion, aggregate:
```bash
NSCC_PW=... python scripts/cf_fisd/aggregate.py \
  --variants baseline_plr --n-seeds 5
```

**Gate**: `paper/exp/cf_fisd/_aggregated/wide.csv` rows for `baseline_plr` must
land within ±1σ of the paper's TabM† 15-seed numbers (cached in
`D:/TabM_PROJ/CFRD/_paper_reports/`). If not, debug before continuing.

## CF-FISD λ-sweep (75 jobs, ~75 min wall)

```bash
NSCC_PW=... python scripts/cf_fisd/throttled_submit.py \
  --kind student \
  --variants hetero_raw_lam0.05 hetero_raw_lam0.1 hetero_raw_lam0.2 \
  --n-seeds 5
```

Aggregate, pick winning λ by val-best score (Rule 47).

## Headline 15-seed rerun (~150 min wall)

Regenerate configs with 15 seeds for the chosen λ and `baseline_plr`:
```bash
cd paper && python tools/generate_cf_fisd_configs.py --n-seeds 15 \
  --lambdas <chosen_lambda>
```
Then `deploy.sh` and re-run the throttled submit with `--n-seeds 15`.

## Outputs

- Per-seed `report.json` under `paper/exp/cf_fisd/tabred/<ds>/<variant>-evaluation/<seed>/`
- Auto-generated head-selection siblings: `<variant>-best-head-evaluation/`,
  `<variant>-greedy-heads-evaluation/` (Rule 46)
- Aggregated CSV/markdown at `paper/exp/cf_fisd/_aggregated/`
