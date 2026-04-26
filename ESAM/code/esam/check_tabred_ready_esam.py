#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _shape(path: Path) -> tuple[int, ...]:
    return tuple(np.load(path, mmap_mode='r').shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('esam/configs/esam_dataset_config.json'))
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--out-csv', type=Path, default=Path('/mnt/ssd/users/prithvi/deepLearning/results/esam/DATASET_READINESS_FIXED.csv'))
    parser.add_argument('--out-md', type=Path, default=Path('/mnt/ssd/users/prithvi/deepLearning/results/esam/DATASET_READINESS_FIXED.md'))
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    pre_root = Path(cfg['preprocessed_root']).expanduser().resolve()
    datasets = cfg['datasets']
    selected = [x.strip() for x in args.datasets.split(',') if x.strip()]
    keys = selected if selected else [k for k, v in datasets.items() if v.get('enabled', True)]

    rows = []
    failed = []

    for name in keys:
        dpath = pre_root / name
        info_p = dpath / 'info.json'
        required = [dpath / 'Y_train.npy', dpath / 'Y_val.npy', dpath / 'Y_test.npy']
        ok = True
        info = {}

        if not info_p.exists() or not all(p.exists() for p in required):
            ok = False
        else:
            info = json.loads(info_p.read_text())

        if ok:
            ytr = np.load(required[0], mmap_mode='r')
            yva = np.load(required[1], mmap_mode='r')
            yte = np.load(required[2], mmap_mode='r')
            train, val, test = int(len(ytr)), int(len(yva)), int(len(yte))
            target_shape = str(tuple(ytr.shape))
        else:
            train = val = test = 0
            target_shape = '-'

        xn = dpath / 'X_num_train.npy'
        xc = dpath / 'X_cat_train.npy'
        xb = dpath / 'X_bin_train.npy'
        num_dim = (_shape(xn)[1] if xn.exists() and len(_shape(xn)) > 1 else 0)
        cat_dim = (_shape(xc)[1] if xc.exists() and len(_shape(xc)) > 1 else 0)
        bin_dim = (_shape(xb)[1] if xb.exists() and len(_shape(xb)) > 1 else 0)

        leakage = ((info.get('leakage_checks') or {}).get('duplicate_feature_hash_overlap') or {}) if info else {}
        leak_sum = int(leakage.get('train_val', 0)) + int(leakage.get('train_test', 0)) + int(leakage.get('val_test', 0))
        leakage_status = 'ok' if leak_sum == 0 else f'risk:{leak_sum}'

        include_claims = 'yes'
        if name == 'ecom-offers' and leak_sum > 0:
            include_claims = 'no'
        if name == 'delivery-eta':
            include_claims = 'conditional'

        row = {
            'dataset': name,
            'ready': 'READY' if ok else 'MISSING',
            'task_type': info.get('task_type', 'unknown') if info else 'unknown',
            'metric': info.get('score', 'unknown') if info else 'unknown',
            'metric_direction': info.get('metric_direction', 'unknown') if info else 'unknown',
            'train_rows': train,
            'val_rows': val,
            'test_rows': test,
            'num_dim': int(num_dim),
            'cat_dim': int(cat_dim),
            'bin_dim': int(bin_dim),
            'target_shape': target_shape,
            'missing_values': 'check_in_preprocessing_manifest',
            'leakage_risk_status': leakage_status,
            'include_in_final_claims': include_claims,
        }
        rows.append(row)
        if not ok:
            failed.append(name)

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.write_text(df.to_csv(index=False))
    args.out_md.write_text('# Dataset Readiness (Fixed)\n\n' + df.to_markdown(index=False))

    print(df.to_string(index=False))
    print(f'[saved] {args.out_csv}')
    print(f'[saved] {args.out_md}')

    if args.strict and failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
