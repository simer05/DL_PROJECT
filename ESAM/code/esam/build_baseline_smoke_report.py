#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path('/mnt/ssd/users/prithvi/deepLearning/outputs/esam/baseline_equivalence_smoke/raw')
OUT_CSV=Path('/mnt/ssd/users/prithvi/deepLearning/results/esam/baseline_equivalence_smoke.csv')
OUT_MD=Path('/mnt/ssd/users/prithvi/deepLearning/results/esam/BASELINE_EQUIVALENCE_SMOKE_REPORT.md')

rows=[]
for rpath in ROOT.glob('**/report.json'):
    r=json.loads(rpath.read_text())
    meta_p=rpath.parent/'run_meta.json'
    meta=json.loads(meta_p.read_text()) if meta_p.exists() else {}
    rows.append({
        'dataset': meta.get('dataset', rpath.parts[-7] if len(rpath.parts)>=7 else 'unknown'),
        'seed': meta.get('seed', None),
        'train_fraction': meta.get('train_fraction', None),
        'share_training_batches': meta.get('share_training_batches', None),
        'n_epochs': meta.get('n_epochs_effective', None),
        'patience': meta.get('patience_effective', None),
        'batch_size': meta.get('batch_size_effective', None),
        'val_score': (r.get('metrics',{}).get('val',{}) or {}).get('score'),
        'test_score': (r.get('metrics',{}).get('test',{}) or {}).get('score'),
        'time': r.get('time'),
        'best_step': r.get('best_step'),
        'prediction_type': r.get('prediction_type'),
        'report_path': str(rpath),
    })

df=pd.DataFrame(rows).sort_values(['dataset']) if rows else pd.DataFrame()
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV,index=False)
md=['# Baseline Equivalence Smoke Report','']
if df.empty:
    md.append('No smoke runs found.')
else:
    md.append(df.to_markdown(index=False))
OUT_MD.write_text('\n'.join(md))
print('saved',OUT_CSV)
print('saved',OUT_MD)
