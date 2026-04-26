# Stop And Archive Report
- timestamp: 20260426_163752
- branch: feature/tabm-esam
- commit: 28e47ae301c92ec37787dde1ce923a0793f405b4
- archive_location: /mnt/ssd/users/prithvi/deepLearning/archive/esam_partial_noncompliant_20260426_163752
- reason: manifesto non-compliance and dataset validity cleanup

## Active Processes Found Before Stop
- run_experiment_esam.py (PID 557462)
- bin/model.py (PID 557463)
- launcher helper process (PID 566564)

## Processes Stopped
- launcher: run_safe_esam_seed42_screen_5var_parallel.sh
- wrapper: run_experiment_esam.py
- trainer: bin/model.py (safe_esam_all5_full_seed42)

## Interrupted Runs
- any run dir missing DONE marker was marked with INCOMPLETE before archiving.

## Notes
- no output was deleted; partial state moved to archive.
