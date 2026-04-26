#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <selection_json>"
  exit 1
fi

SEL="$1"
python - <<PY
import json, subprocess
from pathlib import Path
sel = json.loads(Path('$SEL').read_text())
for row in sel.get('rows', []):
    report = Path(row['report'])
    run_dir = report.parent
    cfg = run_dir / 'config.generated.toml'
    cmd = ['python', 'personb/evaluate_checkpoint.py', '--run-dir', str(run_dir)]
    print('[eval]', ' '.join(cmd))
    subprocess.run(cmd, check=True)
PY
