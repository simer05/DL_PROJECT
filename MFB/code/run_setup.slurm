#!/bin/bash
#SBATCH --job-name=rampncl-setup
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SCRIPT_DIR"

mkdir -p logs artifacts/executed_notebooks

module add python312

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python - <<'PY'
from pathlib import Path
required = [
    Path('ramp_ncl_mfb_tc5_end_to_end.ipynb'),
    Path('requirements.txt'),
]
missing = [str(x) for x in required if not x.exists()]
if missing:
    raise FileNotFoundError(f'Missing required bundle files: {missing}')
print('TC5 setup complete.')
PY
