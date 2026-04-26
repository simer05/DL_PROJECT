# ESAM-TabM Pipeline

This folder contains the new active ESAM experiment flow for TabM.

Scope:
- Replace Person-B/NCL as active experiment pipeline.
- Keep existing repo/data recoverable.
- Reuse existing run/eval/report structure where possible.

Primary paths:
- Raw data: `/mnt/ssd/users/prithvi/deepLearning/data/tabred`
- Preprocessed data: `/mnt/ssd/users/prithvi/deepLearning/data/tabred_preprocessed`
- Outputs: `/mnt/ssd/users/prithvi/deepLearning/outputs/esam`
- Results: `/mnt/ssd/users/prithvi/deepLearning/results/esam`
- Logs: `/mnt/ssd/users/prithvi/deepLearning/logs/esam`
- Checkpoints: `/mnt/ssd/users/prithvi/deepLearning/checkpoints/esam`
- Artifacts: `/mnt/ssd/users/prithvi/deepLearning/artifacts/esam`

Main scripts:
- `preprocess_tabred_to_tabm.py`
- `check_tabred_ready_esam.py`
- `run_experiment_esam.py`
- `prepare_selection_esam.py`
- `run_esam_smoke.sh`
- `run_esam_tabred_final.sh`
- `esam_tabred_final.pbs`
- `build_esam_report.py`

Important:
- Baseline behavior remains unchanged when `use_esam=false`.
- ESAM perturbation is adapter-only by default.
