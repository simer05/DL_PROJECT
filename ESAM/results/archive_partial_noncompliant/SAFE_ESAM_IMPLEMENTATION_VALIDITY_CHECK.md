# SAFE ESAM Implementation Validity Check

- generated_at: 2026-04-26T13:34:17
- run_root: `/mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw`

| Area | Status | Severity | Action Required |
|---|---|---|---|
| share_training_batches config | PASS | OK | No forced override in current path; keep base config value per dataset. |
| training budget | PASS | OK | Current run uses full/base config; keep COMPACT=0 for final screen. |
| ESAM perturb/restore | PASS | OK | Adapter-only perturbation and restore-before-step verified in code. |
| validation-only selection | PASS | OK | Use prepare_selection_esam.py with margin and baseline fallback. |
| parser layout | PASS | OK | Dry-run parser works on current nested layout. |
| preprocessing split | PASS | WARN | Split policies present; ecom still leakage-risk despite group_hash. |
| NaN feature check | PASS | OK | lib/data.py now checks NaNs on x_num/x_bin arrays directly. |
| ecom validity | FAIL | CRITICAL | Run rows included, but mark diagnostic-only if failed. |
| delivery validity | PASS | WARN | Keep conditional until full variant behavior verified in current screen. |

## Final Decision
**B. CONTINUE BUT MARK SOME DATASETS DIAGNOSTIC-ONLY**

## 1. Config / Training Behavior Check
- Forced `share_training_batches` override found in current path: `False`
- Weak compact override observed in current active runs: `False`

| dataset | variant | seed | train_fraction | n_epochs | patience | batch_size | share_training_batches | compact_screen | base_config | config_file |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| cooking-time | baseline | 42 | 1.0 | -1 | 16 | 1024 | False | None |  | /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/cooking-time/trainfrac_1/seed42/start_0/rho_0p0/baseline/config.generated.toml |
| homesite-insurance | baseline | 42 | 1.0 | -1 | 16 | 1024 | False | False |  | /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/homesite-insurance/trainfrac_1/seed42/start_0/rho_0p0/baseline/config.generated.toml |
| homesite-insurance | full_esam_rho_0.0001 | 42 | 1.0 | -1 | 16 | 1024 | False | None |  | /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/homesite-insurance/trainfrac_1/seed42/start_0/rho_0p0001/full_esam_rho_0.0001/config.generated.toml |
| homesite-insurance | full_esam_rho_0.00025 | 42 | 1.0 | -1 | 16 | 1024 | False | None |  | /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/homesite-insurance/trainfrac_1/seed42/start_0/rho_0p00025/full_esam_rho_0.00025/config.generated.toml |
| homesite-insurance | sched25_esam_rho_0.0005 | 42 | 1.0 | -1 | 16 | 1024 | False | None |  | /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/homesite-insurance/trainfrac_1/seed42/start_0/rho_0p0005/sched25_esam_rho_0.0005/config.generated.toml |
| homesite-insurance | sched50_esam_rho_0.0005 | 42 | 1.0 | -1 | 16 | 1024 | False | None |  | /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/homesite-insurance/trainfrac_1/seed42/start_0/rho_0p0005/sched50_esam_rho_0.0005/config.generated.toml |

## 2. ESAM Implementation Check
- baseline_path_gated_by_use_esam: `True`
- adapter_only_selection: `True`
- shared_backbone_not_perturbed: `True`
- restore_before_step: `True`
- double_zero_grad_present: `True`
- scheduled_esam_end_epoch: `True`
- esam_diagnostics_saved: `True`
- adapter_parameter_names (example union): ['backbone.blocks.0.0.r', 'backbone.blocks.0.0.s', 'backbone.blocks.0.0.bias', 'backbone.blocks.1.0.r', 'backbone.blocks.1.0.s', 'backbone.blocks.1.0.bias']
- number_of_adapter_parameters_perturbed(max observed): 6
- fallback_mode_observed_any: False
- nan_or_inf_in_esam_diagnostics_any: False

## 3. Selection / Parser Check
- parser_discovered_rows: 6
- parser_selected_rows: 2
- parser_fields_ok: True
- validation_only_protocol: True
- uses_test_for_selection: False
- dryrun_csv: `/mnt/ssd/users/prithvi/deepLearning/results/esam/SAFE_ESAM_SELECTION_DRYRUN_CHECK.csv`

## 4. Preprocessing / Split Check (All 5 Datasets)
### homesite-insurance
- task_type: binclass
- score: accuracy
- metric_direction: higher_is_better_score
- target_column: QuoteConversion_Flag
- split_policy: stratified
- fit_scope: train_only
- dup_overlap: {'train_val': 0, 'train_test': 0, 'val_test': 0}
- target_like_columns: []
- id_like_columns: []
- train_n: 182527
- val_n: 39113
- test_n: 39113
- train_mean: 0.18751198452831636
- val_mean: 0.1875079896709534
- test_mean: 0.1875079896709534
- train_std: 0.3903219698999389
- val_std: 0.39031877162200523
- test_std: 0.39031877162200523
- nan_report: {'Y_train': False, 'Y_val': False, 'Y_test': False, 'X_num_train': False, 'X_num_val': False, 'X_num_test': False, 'X_bin_train': False, 'X_bin_val': False, 'X_bin_test': False, 'X_cat_train': False, 'X_cat_val': False, 'X_cat_test': False}
### cooking-time
- task_type: regression
- score: rmse
- metric_direction: higher_is_better_score
- target_column: cooking_time_minutes
- split_policy: stratified_quantile
- fit_scope: train_only
- dup_overlap: {'train_val': 0, 'train_test': 0, 'val_test': 0}
- target_like_columns: []
- id_like_columns: []
- train_n: 8959749
- val_n: 1919946
- test_n: 1919947
- train_mean: 15.77555054859359
- val_mean: 15.775313316104292
- test_mean: 15.772426036106479
- train_std: 10.674268199307337
- val_std: 10.667179501910182
- test_std: 10.66669643909415
- nan_report: {'Y_train': False, 'Y_val': False, 'Y_test': False, 'X_num_train': False, 'X_num_val': False, 'X_num_test': False, 'X_bin_train': False, 'X_bin_val': False, 'X_bin_test': False}
### sberbank-housing
- task_type: regression
- score: rmse
- metric_direction: higher_is_better_score
- target_column: price_doc
- split_policy: stratified_quantile
- fit_scope: train_only
- dup_overlap: {'train_val': 223, 'train_test': 211, 'val_test': 61}
- target_like_columns: []
- id_like_columns: ['ID_metro', 'ID_railroad_station_walk', 'ID_railroad_station_avto', 'ID_big_road1', 'ID_big_road2', 'ID_railroad_terminal', 'ID_bus_terminal']
- train_n: 21329
- val_n: 4571
- test_n: 4571
- train_mean: 7106147.788691453
- val_mean: 7157661.760446291
- test_mean: 7167208.456792824
- train_std: 4678382.741013362
- val_std: 5067543.276497892
- test_std: 4949669.283548686
- nan_report: {'Y_train': False, 'Y_val': False, 'Y_test': False, 'X_num_train': False, 'X_num_val': False, 'X_num_test': False, 'X_bin_train': False, 'X_bin_val': False, 'X_bin_test': False, 'X_cat_train': False, 'X_cat_val': False, 'X_cat_test': False}
### ecom-offers
- task_type: binclass
- score: accuracy
- metric_direction: higher_is_better_score
- target_column: repeater
- split_policy: group_hash
- fit_scope: train_only
- dup_overlap: {'train_val': 1, 'train_test': 0, 'val_test': 1}
- target_like_columns: []
- id_like_columns: []
- train_n: 112040
- val_n: 24009
- test_n: 24008
- train_mean: 0.2403695108889682
- val_mean: 0.26615019367737097
- test_mean: 0.4214011996001333
- train_std: 0.4273078622304607
- val_std: 0.44194373859448327
- test_std: 0.49378358475723144
- nan_report: {'Y_train': False, 'Y_val': False, 'Y_test': False, 'X_num_train': False, 'X_num_val': False, 'X_num_test': False}
### delivery-eta
- task_type: regression
- score: rmse
- metric_direction: higher_is_better_score
- target_column: delivery_eta_minutes
- split_policy: stratified_quantile
- fit_scope: train_only
- dup_overlap: {'train_val': 0, 'train_test': 0, 'val_test': 0}
- target_like_columns: []
- id_like_columns: []
- train_n: 11930830
- val_n: 2556606
- test_n: 2556607
- train_mean: 7.53334459814688
- val_mean: 7.5302026277526455
- test_mean: 7.533577818544665
- train_std: 6.24055013635461
- val_std: 6.230166802128451
- test_std: 6.241760422431994
- nan_report: {'Y_train': False, 'Y_val': False, 'Y_test': False, 'X_num_train': False, 'X_num_val': False, 'X_num_test': False, 'X_bin_train': False, 'X_bin_val': False, 'X_bin_test': False}

## 5. Ecom-offers Specific Check
- audit_passed: False
- include_in_scientific_claims: False
- reason: Leakage-risk remains suspicious (logistic val/test=1.0).
- source: `/mnt/ssd/users/prithvi/deepLearning/results/esam/ECOM_FIXED_AUDIT.md`

## 6. Delivery-eta Specific Check
- audit_passed: True
- include_in_scientific_claims: True
- reason: Target/metric pipeline numerically valid; include only after variant behavior verified.
- delivery_variants_completed_in_current_screen: 0
- variant_scores_differ_so_far: None
- source: `/mnt/ssd/users/prithvi/deepLearning/results/esam/DELIVERY_FIXED_AUDIT.md`

## 7. Live Run Sanity Check
- completed_run_count: 6
- active_run_count: 3
- failed_run_count (heuristic): 0
- skipped_run_count (from log markers): 1
- best_validation_variant_per_completed_dataset: {'cooking-time': {'variant': 'baseline', 'val_score': -7.407313532057829, 'test_score': -7.4172973650391025}, 'homesite-insurance': {'variant': 'full_esam_rho_0.0001', 'val_score': 0.9227858457762323, 'test_score': 0.9251418929283632}}
- active_processes:
```text
539731 bash esam/run_safe_esam_seed42_screen_5var_parallel.sh
539745 bash esam/run_safe_esam_seed42_screen_5var_parallel.sh
540661 python esam/run_experiment_esam.py --dataset cooking-time --base-config exp/tabm/tabred/cooking-time/0-evaluation/0.toml --data-root /mnt/ssd/users/prithvi/deepLearning/data/tabred_preprocessed --output-root /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw --seed 42 --rho 0.0 --train-fraction 1.00 --train-subsample-seed 42 --run-tag baseline
540662 python bin/model.py /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/cooking-time/trainfrac_1/seed42/start_0/rho_0p0/baseline/config.generated.toml --output /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/cooking-time/trainfrac_1/seed42/start_0/rho_0p0/baseline --force
543280 /bin/sh -c pgrep -af 'run_experiment_esam.py|bin/model.py|run_safe_esam_seed42_screen_5var_parallel.sh' || true
```
- logs root: `/mnt/ssd/users/prithvi/deepLearning/logs/esam`