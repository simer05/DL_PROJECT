# MFB Task

This folder contains the current single-notebook implementation for the MFB track built on top of TabM.

## Scope

- Task: MFB / RAMP++-NCL-MFB follow-up
- Base model: TabM
- Datasets: `homesite`, `sberbank`, `ecom-offers`, `cooking-time`, `delivery-eta`
- Packaging rule: one main `.ipynb` notebook plus one `requirements.txt`
- Cluster rule: execution is done through Slurm jobs only

## Folder Layout

- `../code/ramp_ncl_mfb_tc5_end_to_end.ipynb`
  - Canonical notebook currently running on TC2
- `../code/requirements.txt`
  - Python dependencies for the notebook bundle
- `../code/run_setup.slurm`
  - Creates the TC2 virtual environment and installs dependencies
- `../code/run_full.slurm`
  - Launches the main notebook execution on TC2
- `../code/run_resume.slurm`
  - Launches the single resume pass if the main job fails or times out
- `../results`
  - Placeholder for final result tables and reports after the TC2 run completes

## Current Status

The code in `../code` is the latest compliant TC2 bundle for the MFB task. It is the same bundle currently being executed on TC2 for the fresh compliant rerun.

## Notes

- The notebook is the primary submission artifact.
- Final results are intentionally not committed yet because the TC2 run is still in progress.
