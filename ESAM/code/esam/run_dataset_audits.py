#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None

PRE = Path('/mnt/ssd/users/prithvi/deepLearning/data/tabred_preprocessed')
RES = Path('/mnt/ssd/users/prithvi/deepLearning/results/esam')
RES.mkdir(parents=True, exist_ok=True)


def load_split(ds: str):
    p = PRE / ds
    info = json.loads((p / 'info.json').read_text())
    y = {k: np.load(p / f'Y_{k}.npy') for k in ['train', 'val', 'test']}
    x_num = {k: np.load(p / f'X_num_{k}.npy') for k in ['train', 'val', 'test']} if (p / 'X_num_train.npy').exists() else None
    x_cat = {k: np.load(p / f'X_cat_{k}.npy') for k in ['train', 'val', 'test']} if (p / 'X_cat_train.npy').exists() else None
    x_bin = {k: np.load(p / f'X_bin_{k}.npy') for k in ['train', 'val', 'test']} if (p / 'X_bin_train.npy').exists() else None
    return info, y, x_num, x_cat, x_bin


def hstack_features(x_num, x_cat, x_bin, part):
    xs = []
    if x_num is not None:
        xs.append(x_num[part].astype(np.float32))
    if x_bin is not None:
        xs.append(x_bin[part].astype(np.float32))
    if x_cat is not None:
        xs.append(x_cat[part].astype(np.float32))
    if not xs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.hstack(xs)


def ecom_audit() -> None:
    ds = 'ecom-offers'
    info, y, x_num, x_cat, x_bin = load_split(ds)
    md = ['# ECOM Fixed Audit', '']
    md.append(f"- task_type: {info.get('task_type')}")
    md.append(f"- score: {info.get('score')}")
    md.append(f"- split_policy: {info.get('split_policy')}")
    md.append(f"- leakage_checks: {json.dumps(info.get('leakage_checks', {}), indent=2)}")

    ytr = y['train']
    maj = np.bincount(ytr.astype(int)).argmax()
    maj_acc_val = float((y['val'] == maj).mean())
    maj_acc_test = float((y['test'] == maj).mean())
    md.append('')
    md.append('## Majority baseline')
    md.append(f'- class: {int(maj)}')
    md.append(f'- val_acc: {maj_acc_val:.6f}')
    md.append(f'- test_acc: {maj_acc_test:.6f}')

    if LogisticRegression is not None:
        xtr = hstack_features(x_num, x_cat, x_bin, 'train')
        xva = hstack_features(x_num, x_cat, x_bin, 'val')
        xte = hstack_features(x_num, x_cat, x_bin, 'test')
        mdl = LogisticRegression(max_iter=200, n_jobs=1)
        try:
            mdl.fit(xtr, ytr)
            val = float((mdl.predict(xva) == y['val']).mean())
            test = float((mdl.predict(xte) == y['test']).mean())
            md.append('')
            md.append('## Logistic baseline (train-only fit)')
            md.append(f'- val_acc: {val:.6f}')
            md.append(f'- test_acc: {test:.6f}')
            if val > 0.995 and test > 0.995:
                md.append('- verdict: leakage-risk remains suspicious, exclude from final scientific claims.')
            else:
                md.append('- verdict: no trivial near-perfect classifier after fixes.')
        except Exception as e:
            md.append(f'- logistic_error: {e}')
    else:
        md.append('- sklearn logistic unavailable; only majority baseline computed.')

    (RES / 'ECOM_FIXED_AUDIT.md').write_text('\n'.join(md))


def delivery_audit() -> None:
    ds = 'delivery-eta'
    info, y, x_num, x_cat, x_bin = load_split(ds)
    md = ['# DELIVERY Fixed Audit', '']
    md.append(f"- task_type: {info.get('task_type')}")
    md.append(f"- score: {info.get('score')}")
    md.append(f"- metric_direction: {info.get('metric_direction')}")
    md.append(f"- target_column: {info.get('target_column')}")

    for part in ['train', 'val', 'test']:
        arr = y[part].astype(np.float64)
        md.append(f"- {part}: n={len(arr)}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}, std={arr.std():.6f}, nan={np.isnan(arr).sum()}")

    ytr = y['train'].astype(np.float64)
    const = float(np.mean(ytr))
    rmse = lambda yt, yp: float(np.sqrt(np.mean((yt.astype(np.float64)-yp)**2)))
    val_rmse = rmse(y['val'], np.full_like(y['val'], const, dtype=np.float64))
    test_rmse = rmse(y['test'], np.full_like(y['test'], const, dtype=np.float64))
    md.append('')
    md.append('## Constant baseline (mean train target)')
    md.append(f'- val_rmse: {val_rmse:.6f}')
    md.append(f'- test_rmse: {test_rmse:.6f}')

    if not np.isfinite(val_rmse) or not np.isfinite(test_rmse):
        md.append('- verdict: invalid metric/data behavior, exclude from final claims until fixed.')
    else:
        md.append('- verdict: basic target pipeline numerically valid; full training metrics still require verification.')

    (RES / 'DELIVERY_FIXED_AUDIT.md').write_text('\n'.join(md))


def main() -> None:
    ecom_audit()
    delivery_audit()
    print('[done] audits written')


if __name__ == '__main__':
    main()
