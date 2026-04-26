#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python personb/prepare_selection.py

python - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path('.')
sel = json.loads((root / 'results' / 'personb_selected_runs.json').read_text())
missing = sel.get('missing_expected_runs', [])
if missing:
    raise SystemExit(f"Missing expected runs: {len(missing)}. Run training first.")

cases = []
for row in sel['selected_by_val']:
    for model_name, run_dir in [('baseline', row['baseline_run']), ('selected', row['selected_run'])]:
        out_base = root / 'results' / 'personb_eval' / row['dataset'] / f"seed{row['seed']}" / model_name
        cases.append((run_dir, out_base / 'clean.json', 'none', 'mild'))
        for sev in ['mild', 'moderate']:
            cases.append((run_dir, out_base / f'mask_{sev}.json', 'mask', sev))
            cases.append((run_dir, out_base / f'noise_{sev}.json', 'noise', sev))

for run_dir, out_json, corr, sev in cases:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists():
        print('[skip]', out_json)
        continue
    cmd = [
        'python', 'personb/evaluate_checkpoint.py',
        '--paper-dir', '.',
        '--run-dir', run_dir,
        '--part', 'test',
        '--corruption', corr,
        '--severity', sev,
        '--output-json', str(out_json),
    ]
    print('[run]', ' '.join(cmd))
    subprocess.run(cmd, check=True)
PY
