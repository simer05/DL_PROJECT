# ESAM Read-Only Implementation Audit

- generated_at: 2026-04-26T13:43:00
- branch: `feature/tabm-esam`
- commit: `28e47ae301c92ec37787dde1ce923a0793f405b4`

| Area | Status | Severity | Evidence | Action Needed |
|---|---|---|---|---|
| share_training_batches override | PASS | OK | esam/run_experiment_esam.py:222-243,292 | No forced True found in active path. |
| training budget | PASS | OK | esam/run_safe_esam_seed42_screen_5var_parallel.sh:39-42 + run_meta | Keep COMPACT=0 for final runs. |
| baseline path unchanged | PASS | OK | bin/model.py:874-880 | Ensure ESAM stays gated by use_esam and rho. |
| ESAM perturb/restore | PASS | OK | bin/model.py:541-571,899-915,932-940 | Monitor diagnostics for runtime anomalies. |
| scheduled ESAM | PASS | OK | bin/model.py:878-880 + run_experiment_esam.py:253-261 | No action. |
| validation-only selection | PASS | WARN | esam/prepare_selection_esam.py:95-115 | Signed-score assumption should be documented in final report. |
| parser/output layout | PASS | WARN | esam/prepare_selection_esam.py:20-77 | Populate log_path if full provenance is required. |
| preprocessing split | PASS | WARN | esam/preprocess_tabred_to_tabm.py:90-151,296-312 | ecom remains leakage-risk despite group_hash split. |
| train-only preprocessing | PASS | OK | esam/preprocess_tabred_to_tabm.py:188-215,348-353 | No action. |
| NaN feature check | PASS | OK | lib/data.py:185-194 | No action. |
| ecom validity | FAIL | CRITICAL | results/esam/ECOM_FIXED_AUDIT.md | Keep ecom diagnostic-only unless leakage root cause fixed. |
| delivery validity | PASS | WARN | results/esam/DELIVERY_FIXED_AUDIT.md + run rows | Keep conditional until delivery variants complete. |
| GPU/run safety | PASS | WARN | esam/run_safe_esam_seed42_screen_5var_parallel.sh:28,85-105 | Avoid duplicate launcher invocations on same dataset list. |

## Final Recommendation
**B. SAFE TO CONTINUE, BUT SOME DATASETS MUST BE DIAGNOSTIC-ONLY**

## Section 1: Git and Active Run Snapshot
- git status changed files: 15
```text
M bin/ensemble.py
 M bin/evaluate.py
 M bin/go.py
 M bin/model.py
 M bin/tune.py
 M lib/__init__.py
 M lib/data.py
 M lib/deep.py
 M lib/env.py
 M lib/metrics.py
 M lib/util.py
?? esam/
?? outputs/
?? personb/
?? results/
```
```text
539731 bash esam/run_safe_esam_seed42_screen_5var_parallel.sh
539745 bash esam/run_safe_esam_seed42_screen_5var_parallel.sh
540661 python esam/run_experiment_esam.py --dataset cooking-time --base-config exp/tabm/tabred/cooking-time/0-evaluation/0.toml --data-root /mnt/ssd/users/prithvi/deepLearning/data/tabred_preprocessed --output-root /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw --seed 42 --rho 0.0 --train-fraction 1.00 --train-subsample-seed 42 --run-tag baseline
540662 python bin/model.py /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/cooking-time/trainfrac_1/seed42/start_0/rho_0p0/baseline/config.generated.toml --output /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/cooking-time/trainfrac_1/seed42/start_0/rho_0p0/baseline --force
544519 /bin/sh -c pgrep -af 'run_experiment_esam.py|bin/model.py|run_safe_esam_seed42_screen_5var_parallel.sh' || true
```
```text
0, GPU-e449d029-bccd-d6b9-2d17-230455d68706, 652 MiB, 97887 MiB
1, GPU-4d483cfa-cf16-2bc6-b6b2-dfab0cd69b1c, 21817 MiB, 97887 MiB
2, GPU-f18c4a59-a337-be2f-1db0-ee7d39dd2efd, 36460 MiB, 97887 MiB
```
```text
GPU-e449d029-bccd-d6b9-2d17-230455d68706, 412375, python, 552 MiB
GPU-4d483cfa-cf16-2bc6-b6b2-dfab0cd69b1c, 540662, python, 21794 MiB
GPU-f18c4a59-a337-be2f-1db0-ee7d39dd2efd, 412375, python, 36436 MiB
```

## Section 2: Config Override Audit
```text
esam/build_baseline_smoke_report.py:20:        'share_training_batches': meta.get('share_training_batches', None),
esam/run_experiment_esam.py:292:        'share_training_batches': (cfg.get('model') or {}).get('share_training_batches'),
esam/run_experiment_esam.py.bak:297:        'share_training_batches': (cfg.get('model') or {}).get('share_training_batches'),
bin/model.py:119:        share_training_batches: bool = DEFAULT_SHARE_TRAINING_BATCHES,
bin/model.py:127:                share_training_batches
bin/model.py:243:        self.share_training_batches = share_training_batches
bin/model.py:259:            if self.share_training_batches or not self.training:
bin/model.py:454:    share_training_batches = config['model'].get(
bin/model.py:455:        'share_training_batches', DEFAULT_SHARE_TRAINING_BATCHES
bin/model.py:507:                if share_training_batches
bin/model.py:843:            if share_training_batches
```
- forced share_training_batches=True found: False
- compact rows in completed outputs: 0
- weak budget rows (20/4/2048): 0
| dataset | variant | seed | train_fraction | n_epochs | patience | batch_size | share_training_batches | compact_screen | config_file |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| cooking-time | baseline | 42 | None | -1 | 16 | 1024 | False | False | /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/cooking-time/trainfrac_1/seed42/start_0/rho_0p0/baseline/config.generated.toml |
| homesite-insurance | baseline | 42 | 1.0 | -1 | 16 | 1024 | False | False | /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/homesite-insurance/trainfrac_1/seed42/start_0/rho_0p0/baseline/config.generated.toml |
| homesite-insurance | full_esam_rho_0.0001 | 42 | None | -1 | 16 | 1024 | False | False | /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/homesite-insurance/trainfrac_1/seed42/start_0/rho_0p0001/full_esam_rho_0.0001/config.generated.toml |
| homesite-insurance | full_esam_rho_0.00025 | 42 | None | -1 | 16 | 1024 | False | False | /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/homesite-insurance/trainfrac_1/seed42/start_0/rho_0p00025/full_esam_rho_0.00025/config.generated.toml |
| homesite-insurance | sched25_esam_rho_0.0005 | 42 | None | -1 | 16 | 1024 | False | False | /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/homesite-insurance/trainfrac_1/seed42/start_0/rho_0p0005/sched25_esam_rho_0.0005/config.generated.toml |
| homesite-insurance | sched50_esam_rho_0.0005 | 42 | None | -1 | 16 | 1024 | False | False | /mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_full_seed42/raw/homesite-insurance/trainfrac_1/seed42/start_0/rho_0p0005/sched50_esam_rho_0.0005/config.generated.toml |

## Section 3: ESAM Implementation Audit
- baseline_gate_use_esam: True
- adapter_only_param_selection: True
- perturbation_target_adapter_params_only: True
- restore_before_optimizer_step: True
- double_zero_grad: True
- scheduled_end_epoch_logic: True
- diagnostics_logging: True
- adapter parameter names observed: ['backbone.blocks.0.0.bias', 'backbone.blocks.0.0.r', 'backbone.blocks.0.0.s', 'backbone.blocks.1.0.bias', 'backbone.blocks.1.0.r', 'backbone.blocks.1.0.s']

## Section 4: Validation-Only Selection Audit
- baseline_candidate_included: True
- validation_score_used: True
- margin_applied: True
- test_not_in_selection_branch: True
- metric_direction_explicit: True
- metric_direction_risk_note: True

## Section 5: Output Layout / Parser Audit
- layout_ok: True
- extract_fields_ok: True
- log_path_missing_risk: True

## Section 6: Preprocessing / Split Audit
### homesite-insurance
- ready: True
- split_policy: stratified
- task_type: binclass
- target_column: QuoteConversion_Flag
- fit_scope: train_only
- leakage_checks: {'duplicate_feature_hash_overlap': {'train_val': 0, 'train_test': 0, 'val_test': 0}, 'target_like_columns': [], 'id_like_columns': []}
- include_recommendation: include
### cooking-time
- ready: True
- split_policy: stratified_quantile
- task_type: regression
- target_column: cooking_time_minutes
- fit_scope: train_only
- leakage_checks: {'duplicate_feature_hash_overlap': {'train_val': 0, 'train_test': 0, 'val_test': 0}, 'target_like_columns': [], 'id_like_columns': []}
- include_recommendation: include
### sberbank-housing
- ready: True
- split_policy: stratified_quantile
- task_type: regression
- target_column: price_doc
- fit_scope: train_only
- leakage_checks: {'duplicate_feature_hash_overlap': {'train_val': 223, 'train_test': 211, 'val_test': 61}, 'target_like_columns': [], 'id_like_columns': ['ID_metro', 'ID_railroad_station_walk', 'ID_railroad_station_avto', 'ID_big_road1', 'ID_big_road2', 'ID_railroad_terminal', 'ID_bus_terminal']}
- include_recommendation: include
### ecom-offers
- ready: True
- split_policy: group_hash
- task_type: binclass
- target_column: repeater
- fit_scope: train_only
- leakage_checks: {'duplicate_feature_hash_overlap': {'train_val': 1, 'train_test': 0, 'val_test': 1}, 'target_like_columns': [], 'id_like_columns': []}
- include_recommendation: diagnostic_only
### delivery-eta
- ready: True
- split_policy: stratified_quantile
- task_type: regression
- target_column: delivery_eta_minutes
- fit_scope: train_only
- leakage_checks: {'duplicate_feature_hash_overlap': {'train_val': 0, 'train_test': 0, 'val_test': 0}, 'target_like_columns': [], 'id_like_columns': []}
- include_recommendation: conditional

## Section 7: Feature NaN Validation Audit
- feature_nan_check_fixed: True
- evidence: lib/data.py:185-194

## Section 8: Ecom-offers Validity Audit
- audit_passed: False
- include_in_scientific_claims: False
- reason: Logistic baseline val/test=1.0 in audit -> leakage-risk suspicious.

## Section 9: Delivery-eta Validity Audit
- audit_passed: conditional
- include_in_scientific_claims: conditional
- reason: Numerically valid audit, but current final-screen delivery rows not complete yet.
- completed delivery rows in current run root: 0

## Section 10: GPU / Parallel Run Safety Audit
- gpu2_avoided_in_launcher: True
- CUDA_VISIBLE_DEVICES_set: True
- launcher_waits_workers: True
- output_collision_risk_if_duplicate_launcher: True

## Section 11: Final Audit Summary
- recommendation: **B. SAFE TO CONTINUE, BUT SOME DATASETS MUST BE DIAGNOSTIC-ONLY**
- completed run rows discovered: 6
- rows by dataset: {'homesite-insurance': 5, 'cooking-time': 1}