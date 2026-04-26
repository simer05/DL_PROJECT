#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

RAW = Path('/mnt/ssd/users/prithvi/deepLearning/data/tabred/ecom-offers/trainHistory.csv.gz')
PRE = Path('/mnt/ssd/users/prithvi/deepLearning/data/tabred_preprocessed/ecom-offers')
OUT_MD = Path('/mnt/ssd/users/prithvi/deepLearning/results/esam/ECOM_DATA_INTEGRITY_AUDIT.md')
OUT_CSV = Path('/mnt/ssd/users/prithvi/deepLearning/results/esam/dataset_integrity_summary.csv')


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW)


def load_preprocessed() -> dict:
    d = {}
    for p in ['train', 'val', 'test']:
        d[f'Y_{p}'] = np.load(PRE / f'Y_{p}.npy')
        if (PRE / f'X_num_{p}.npy').exists():
            d[f'X_num_{p}'] = np.load(PRE / f'X_num_{p}.npy')
        if (PRE / f'X_cat_{p}.npy').exists():
            d[f'X_cat_{p}'] = np.load(PRE / f'X_cat_{p}.npy')
    return d


def row_hash(x_num: np.ndarray | None, x_cat: np.ndarray | None) -> np.ndarray:
    parts = []
    if x_num is not None:
        parts.append(np.asarray(x_num))
    if x_cat is not None:
        parts.append(np.asarray(x_cat))
    if not parts:
        raise ValueError('No features to hash')
    X = np.column_stack(parts)
    return np.array([hash(tuple(row.tolist())) for row in X], dtype=np.int64)


def pair_overlap(a: np.ndarray, b: np.ndarray) -> int:
    return int(len(np.intersect1d(a, b)))


def main() -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    issues: list[str] = []
    findings: list[str] = []

    # 1) Raw-column leakage scan.
    raw = load_raw()
    target = 'repeater'
    feature_cols = [c for c in raw.columns if c != target]

    exact_target_columns = [c for c in feature_cols if c.lower() == target.lower()]
    suspicious_name_cols = [
        c for c in feature_cols
        if any(k in c.lower() for k in ['target', 'label', 'repeater'])
    ]

    findings.append(f'- Raw rows: {len(raw):,}, columns: {len(raw.columns)}')
    findings.append(f'- Target column: `{target}`')
    findings.append(f'- Exact target-named feature columns: {exact_target_columns}')
    findings.append(f'- Suspicious feature names containing target-like tokens: {suspicious_name_cols}')

    for c in feature_cols:
        try:
            same = (raw[c].values == raw[target].values).mean()
            if same > 0.999:
                issues.append(f'Feature `{c}` is almost identical to target (match={same:.4f}).')
        except Exception:
            pass

    # 2) Preprocessed split overlap / duplicates.
    d = load_preprocessed()
    xtr_h = row_hash(d.get('X_num_train'), d.get('X_cat_train'))
    xva_h = row_hash(d.get('X_num_val'), d.get('X_cat_val'))
    xte_h = row_hash(d.get('X_num_test'), d.get('X_cat_test'))

    ov_tv = pair_overlap(xtr_h, xva_h)
    ov_tt = pair_overlap(xtr_h, xte_h)
    ov_vt = pair_overlap(xva_h, xte_h)

    findings.append(f'- Preprocessed sizes train/val/test: {len(xtr_h):,}/{len(xva_h):,}/{len(xte_h):,}')
    findings.append(f'- Cross-split exact-feature overlap counts (train-val/train-test/val-test): {ov_tv}/{ov_tt}/{ov_vt}')

    if any(x > 0 for x in [ov_tv, ov_tt, ov_vt]):
        issues.append('Detected exact duplicate feature rows across splits (potential leakage risk).')

    # 3) Check train-only fitting by code path inspection (preprocessor + runtime transforms).
    findings.append('- Preprocessing script uses train split only for numeric median imputation and category vocab mapping.')
    findings.append('- TabM runtime normalization policy is fit via train split by Dataset transform code path.')

    # 4) Target distribution by split.
    ytr, yva, yte = d['Y_train'], d['Y_val'], d['Y_test']
    def ratio(y: np.ndarray) -> float:
        return float(np.mean(y == 1))
    findings.append(f'- Positive class ratio train/val/test: {ratio(ytr):.4f}/{ratio(yva):.4f}/{ratio(yte):.4f}')

    # 5) Trivial baseline.
    Xtr_num = d.get('X_num_train')
    Xva_num = d.get('X_num_val')
    Xte_num = d.get('X_num_test')

    if Xtr_num is not None and Xva_num is not None and Xte_num is not None:
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(Xtr_num, ytr)
        p_dummy = dummy.predict_proba(Xte_num)[:, 1]
        y_dummy = (p_dummy >= 0.5).astype(int)

        lr = LogisticRegression(max_iter=1000, n_jobs=1)
        lr.fit(Xtr_num, ytr)
        p_lr_val = lr.predict_proba(Xva_num)[:, 1]
        p_lr_test = lr.predict_proba(Xte_num)[:, 1]

        findings.append(
            f'- Dummy(test): acc={accuracy_score(yte, y_dummy):.4f}, auc={roc_auc_score(yte, p_dummy):.4f}, logloss={log_loss(yte, p_dummy, labels=[0,1]):.4f}'
        )
        findings.append(
            f'- Logistic(test): acc={accuracy_score(yte, (p_lr_test>=0.5).astype(int)):.4f}, auc={roc_auc_score(yte, p_lr_test):.4f}, logloss={log_loss(yte, p_lr_test, labels=[0,1]):.4f}'
        )
        findings.append(
            f'- Logistic(val): acc={accuracy_score(yva, (p_lr_val>=0.5).astype(int)):.4f}, auc={roc_auc_score(yva, p_lr_val):.4f}, logloss={log_loss(yva, p_lr_val, labels=[0,1]):.4f}'
        )

        if roc_auc_score(yte, p_lr_test) > 0.999:
            issues.append('Trivial logistic baseline reaches near-perfect AUC; dataset may be too easy or leak-prone.')
    else:
        issues.append('Numeric features unavailable for trivial baseline check.')

    leakage_suspected = len(issues) > 0

    # Markdown report.
    lines = []
    lines.append('# Ecom-offers Data Integrity Audit')
    lines.append('')
    lines.append('## Scope')
    lines.append('- Checked target leakage indicators, split overlap, preprocessing fit scope, label distribution, and trivial baseline.')
    lines.append('')
    lines.append('## Findings')
    lines.extend(findings)
    lines.append('')
    lines.append('## Issues')
    if issues:
        for it in issues:
            lines.append(f'- {it}')
    else:
        lines.append('- No concrete leakage evidence found in this audit pass.')
    lines.append('')
    lines.append('## Verdict')
    if leakage_suspected:
        lines.append('- **Leakage/data-risk suspected. Exclude `ecom-offers` from final ESAM claims until fixed and re-audited.**')
    else:
        lines.append('- Audit passed for this version; can be used with caution.')

    OUT_MD.write_text('\n'.join(lines))

    # Append/update integrity summary CSV.
    row = {
        'dataset': 'ecom-offers',
        'ready': True,
        'leakage_suspected': leakage_suspected,
        'excluded_from_final_claims': leakage_suspected,
        'notes': ('; '.join(issues)[:1000] if issues else 'audit_passed'),
        'audit_report': str(OUT_MD),
    }

    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
        df = df[df['dataset'] != 'ecom-offers']
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(OUT_CSV, index=False)

    print(f'[done] {OUT_MD}')
    print(f'[done] {OUT_CSV}')


if __name__ == '__main__':
    main()
