#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

set +u
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lpb
set -u

DATA_ROOT="/mnt/ssd/users/prithvi/deepLearning/data/tabred_preprocessed"
OUT_ROOT="/mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_confirmation/raw"
SEL_JSON="/mnt/ssd/users/prithvi/deepLearning/results/esam/safe_esam_selection.json"
mkdir -p "$OUT_ROOT"

python - <<'PY'
import json,subprocess,os
from pathlib import Path

root=Path('/mnt/ssd/users/prithvi/deepLearning/TabM/paper')
sel_path=Path('/mnt/ssd/users/prithvi/deepLearning/results/esam/safe_esam_selection.json')
out_root='/mnt/ssd/users/prithvi/deepLearning/outputs/esam/safe_esam_all5_confirmation/raw'
data_root='/mnt/ssd/users/prithvi/deepLearning/data/tabred_preprocessed'
sel=json.loads(sel_path.read_text())
rows=sel.get('selected_rows',[])

cfg = json.loads((root/'esam/configs/esam_experiment_config.json').read_text())
base_cfgs = cfg['base_configs']

for row in rows:
    if row.get('selected_reason') in {'baseline_best_val','baseline_fallback_margin'}:
        continue
    ds=row['dataset']
    tf=float(row.get('train_fraction',1.0))
    rho=float(row['rho'])
    schedule=row.get('schedule','full')
    base_cfg=base_cfgs[ds]

    for seed in [43,44]:
        subprocess.check_call([
            'python','esam/run_experiment_esam.py',
            '--dataset',ds,'--base-config',base_cfg,'--data-root',data_root,'--output-root',out_root,
            '--seed',str(seed),'--rho','0.0','--train-fraction',str(tf),'--run-tag','baseline_confirm'
        ], cwd=root)

        cmd=['python','esam/run_experiment_esam.py',
             '--dataset',ds,'--base-config',base_cfg,'--data-root',data_root,'--output-root',out_root,
             '--seed',str(seed),'--rho',str(rho),'--train-fraction',str(tf),'--run-tag',f'selected_confirm_{schedule}']
        if str(schedule).startswith('stop@'):
            stop_ep=int(str(schedule).split('@')[1])
            cmd += ['--esam-end-epoch',str(stop_ep)]
        subprocess.check_call(cmd, cwd=root)

print('[done] confirmation runs')
PY
