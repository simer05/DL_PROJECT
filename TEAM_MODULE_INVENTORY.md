# Team module inventory

## RLA
- Source files: `RLA/paper/lib/deep.py`, `RLA/paper/bin/model.py`, `RLA/paper/tests/test_rla.py`.
- Integrated destination files: `tabm_integrated/paper/lib/deep.py`, `tabm_integrated/paper/bin/run_integrated.py`.
- Configs: official baseline configs plus `model.rla_rank`, `model.rla_first_only`, `model.rla_additive`, `model.rla_init`, `model.rla_base_preserve_noise`, `rla_adapter_lr_multiplier`, `rla_extra_paths_freeze_fraction`.
- Scripts: `tabm_integrated/tools/generate_integrated_configs.py`, `tabm_integrated/tools/run_integrated_matrix.sh`, `tabm_integrated/tools/aggregate_integrated_results.py`.
- Notebooks: none required for RLA.
- Result artifacts: `tabm_integrated/paper/exp/integrated/**/rla*`, `tabm_integrated/paper/exp/final_integrated_summary.csv`, `tabm_integrated/paper/exp/final_integrated_audit.csv`.
- Reproduction command: `N_GPUS=16 tabm_integrated/tools/run_integrated_matrix.sh exp/integrated/manifest_sweeps.txt` followed by final selection and final matrix commands.
- Assumptions: RLA rank-r adapters are used only in TabM-style BatchEnsemble layers.
- Known limitations: CF-FISD alignment with RLA combined variants regularizes the first RLA input path (`R[:, :, 0]`).

## ESAM
- Source files: `ESAM/code/bin/model.py`, `ESAM/code/lib/*`.
- Integrated destination files: `tabm_integrated/paper/bin/run_integrated.py`, `tabm_integrated/paper/lib/*`.
- Configs: `use_esam`, `esam_rho`, `esam_eps`, `esam_adapter_only`, `esam_memberwise`, warmup/start/end epoch flags, diagnostics flags.
- Scripts: same integrated generator, queue launcher, and aggregator.
- Notebooks: none required for ESAM.
- Result artifacts: `tabm_integrated/paper/exp/integrated/**/esam*` and final CSV/report artifacts.
- Reproduction command: generate configs, then run the smoke/sweep/final manifests with `N_GPUS=16`.
- Assumptions: adapter-only ESAM perturbs member-specific TabM adapters, including RLA `R/S` when present.
- Known limitations: diagnostics are disabled by default to keep committed artifacts compact.

## MFB
- Source files: `MFB/code/ramp_ncl_mfb_tc5_end_to_end.ipynb`.
- Integrated destination files: `tabm_integrated/paper/bin/run_integrated.py`.
- Configs: `model.mfb.enabled`, `mask_mode`, `mask_granularity`, `keep_rate`, `inverted_scaling`, `use_soft_mask`, `mask_strength`, `anchor_fraction`, `warmup_epochs`, `mask_seed`.
- Scripts: same integrated generator, queue launcher, and aggregator.
- Notebooks: source notebook above; integrated runner ports the legacy member-fixed feature-group MFB path used by the requested keep-rate sweep.
- Result artifacts: `tabm_integrated/paper/exp/integrated/**/mfb*` and final CSV/report artifacts.
- Reproduction command: run the smoke/sweep/final manifests with `N_GPUS=16`.
- Assumptions: requested MFB sweep uses the notebook's legacy hard member-fixed feature-group mask, not the RAMP router ablations.
- Known limitations: RAMP router/fidelity variants from the notebook are not part of the requested keep-rate sweep.

## CF-FISD
- Source files: `cf_fisd_recovered/paper/lib/cf_fisd.py`, `cf_fisd_recovered/paper/bin/cf_fisd_teachers.py`, `cf_fisd_recovered/paper/tools/generate_cf_fisd_configs.py`, teacher importance arrays under `cf_fisd_recovered/paper/exp/cf_fisd/_teachers/tabred/`.
- Integrated destination files: `tabm_integrated/paper/lib/cf_fisd.py`, `tabm_integrated/paper/tools/cf_fisd_teachers.py`, `tabm_integrated/paper/bin/run_integrated.py`.
- Configs: `cf_fisd.lambda`, `cf_fisd.variant`, `cf_fisd.dataset_name`, `cf_fisd.teacher_dir`, `cf_fisd.teacher_names`, optional member groups.
- Scripts: same integrated generator, queue launcher, and aggregator.
- Notebooks: none required for CF-FISD.
- Result artifacts: `tabm_integrated/paper/exp/integrated/**/cf_fisd*` and final CSV/report artifacts.
- Reproduction command: run generated manifests after verifying teacher arrays exist.
- Assumptions: bundled teammate teacher importance arrays are reused for the five TabReD datasets.
- Known limitations: if teacher arrays are regenerated, final configs should point to the regenerated teacher directory.

## End-to-end reproduction
1. `cd tabm_integrated/paper && /workspace/.venvs/tabm_integrated/bin/python ../tools/generate_integrated_configs.py --stage initial --clean`
2. `N_GPUS=16 tabm_integrated/tools/run_integrated_matrix.sh exp/integrated/manifest_baseline_fidelity.txt`
3. `N_GPUS=16 tabm_integrated/tools/run_integrated_matrix.sh exp/integrated/manifest_smoke.txt`
4. `N_GPUS=16 tabm_integrated/tools/run_integrated_matrix.sh exp/integrated/manifest_sweeps.txt`
5. `/workspace/.venvs/tabm_integrated/bin/python tabm_integrated/tools/aggregate_integrated_results.py --stage select-final`
6. `N_GPUS=16 tabm_integrated/tools/run_integrated_matrix.sh exp/integrated/manifest_final.txt`
7. `/workspace/.venvs/tabm_integrated/bin/python tabm_integrated/tools/aggregate_integrated_results.py --stage final`
