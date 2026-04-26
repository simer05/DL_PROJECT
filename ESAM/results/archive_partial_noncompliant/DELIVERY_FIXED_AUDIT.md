# DELIVERY Fixed Audit

- task_type: regression
- score: rmse
- metric_direction: higher_is_better_score
- target_column: delivery_eta_minutes
- train: n=11930830, min=0.000000, max=170.516663, mean=7.533345, std=6.240550, nan=0
- val: n=2556606, min=0.016667, max=169.283340, mean=7.530203, std=6.230167, nan=0
- test: n=2556607, min=0.000000, max=161.199997, mean=7.533578, std=6.241760, nan=0

## Constant baseline (mean train target)
- val_rmse: 6.230168
- test_rmse: 6.241760
- verdict: basic target pipeline numerically valid; full training metrics still require verification.