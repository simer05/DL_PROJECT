# ESAM Repo Inspection

## Workspace
- Repo root: `/mnt/ssd/users/prithvi/deepLearning/TabM`
- Paper workdir: `/mnt/ssd/users/prithvi/deepLearning/TabM/paper`
- Active branch: `feature/tabm-esam`
- Base status on branch creation inherited existing local changes from prior Person-B work.

## Training entrypoint
- Main training script: `paper/bin/model.py`
- CLI entry: `python bin/model.py <config.toml> --output <dir> --force`
- Full train loop and evaluation are inside `main()` in `paper/bin/model.py`.

## Model and adapter locations
- Model class: `paper/bin/model.py:84` (`class Model`)
- Per-member forward output shape: `(B, K, D_OUT)` for ensemble models.
- Per-member accessor: `Model.get_member_logits(...)` in `paper/bin/model.py`.
- First BE adapter accessor: `Model.get_first_batchensemble_r()` in `paper/bin/model.py`.

BatchEnsemble-style adapter implementation:
- `paper/lib/deep.py:159` (`class LinearEfficientEnsemble`)
- Adapter parameters in this layer:
  - `r` with shape `(k, in_features)` when enabled
  - `s` with shape `(k, out_features)` when enabled
  - `bias` can be `(k, out_features)` when `ensemble_bias=True`
  - shared weight matrix `weight` with shape `(out_features, in_features)`

## Existing Person-B/NCL modifications discovered
- NCL integration is embedded in `paper/bin/model.py` (`use_ncl`, `lambda_ncl`, `ncl_space`, warmup).
- Person-B wrappers and selectors are under `paper/personb/`.
- Existing run/eval/report artifacts are under `paper/outputs/` and `paper/results/`.

## Dataset format expected by TabM runner
From `paper/lib/data.py`:
- Required dataset directory files:
  - `info.json`
  - `Y_train.npy`, `Y_val.npy`, `Y_test.npy`
- Optional feature blocks:
  - `X_num_{train,val,test}.npy`
  - `X_cat_{train,val,test}.npy`
  - `X_bin_{train,val,test}.npy`
- Loader behavior:
  - Reads arrays and task metadata from `info.json`.
  - Uses train/val/test files directly.

## Raw TabReD state vs expected format
Current raw data in `/mnt/ssd/users/prithvi/deepLearning/data/tabred` is Kaggle-style files (zip/gz/parquet) and not directly loadable by `lib.data.load_data`.

## Existing preprocessing utility check
- Found `paper/tools/prepare_tabred.py`, but it expects already-processed TabReD source containing files such as:
  - `info.json`, `X_*.npy`, `Y.npy`, `split-default/*_idx.npy`
- This does not match the currently downloaded raw Kaggle files, so a conversion/preprocessing stage is required first.

## Per-member ESAM feasibility
- True per-member signals are available in training because `apply_model` returns per-member predictions (`y_pred`) before member averaging.
- Adapter parameters are identifiable via module type `LinearEfficientEnsemble` and parameter names (`r`, `s`, and memberwise `bias`).

## ESAM implementation feasibility verdict
- **Primary mode feasible**: adapter-only ESAM with memberwise perturbation normalization (using adapter tensors with leading `k` dimension).
- **Fallback mode feasible**: adapter-only global normalization when member dimension cannot be inferred for a parameter.

## Notes for implementation boundaries
- Keep baseline behavior unchanged when ESAM is disabled.
- Do not perturb shared `weight` matrices in `LinearEfficientEnsemble` during SAM perturbation pass.
- Continue using existing training objective as-is for final optimization pass.
