# ESAM Launch Guide

## Setup
```bash
cd /mnt/ssd/users/prithvi/deepLearning/TabM/paper
mkdir -p /mnt/ssd/users/prithvi/deepLearning/{outputs/esam,results/esam,logs/esam,checkpoints/esam,artifacts/esam,data/tabred_preprocessed}
```

## Preprocess + readiness
```bash
python esam/preprocess_tabred_to_tabm.py --config esam/configs/esam_dataset_config.json --datasets homesite-insurance,ecom-offers --force
python esam/check_tabred_ready_esam.py --config esam/configs/esam_dataset_config.json --strict --datasets homesite-insurance,ecom-offers
```

## Smoke run
```bash
bash esam/run_esam_smoke.sh
```

Dry-run:
```bash
bash esam/run_esam_smoke.sh --dry-run
```

## Final launch (PBS)
```bash
qsub esam/esam_tabred_final.pbs
```

## Monitoring
```bash
qstat -u prithvi
tail -f /mnt/ssd/users/prithvi/deepLearning/logs/esam/<logfile>
nvidia-smi
du -sh /mnt/ssd/users/prithvi/deepLearning/outputs/esam
du -sh /mnt/ssd/users/prithvi/deepLearning/checkpoints/esam
```

## Failure recovery
- Preprocessing failure: inspect `/mnt/ssd/users/prithvi/deepLearning/artifacts/esam/schema_debug_<dataset>.json`.
- CUDA OOM: reduce batch size first.
- NaNs with ESAM: reduce rho (`0.05 -> 0.01 -> 0.005`).
- Too much overhead: increase `--esam-diagnostics-every`.
- If memberwise perturbation unavailable, fallback adapter-only global mode is logged in diagnostics.
- If dataset too heavy, skip delivery-eta initially and continue with ready datasets.

## Reproducibility metadata
Each launcher logs:
- branch + commit
- hostname/date
- Python and torch versions
- CUDA visibility
