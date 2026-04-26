# ECOM Fixed Audit

- task_type: binclass
- score: accuracy
- split_policy: group_hash
- leakage_checks: {
  "duplicate_feature_hash_overlap": {
    "train_val": 1,
    "train_test": 0,
    "val_test": 1
  },
  "target_like_columns": [],
  "id_like_columns": []
}

## Majority baseline
- class: 0
- val_acc: 0.733850
- test_acc: 0.578599

## Logistic baseline (train-only fit)
- val_acc: 1.000000
- test_acc: 1.000000
- verdict: leakage-risk remains suspicious, exclude from final scientific claims.