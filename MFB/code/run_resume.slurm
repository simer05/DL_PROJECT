#!/bin/bash
#SBATCH --job-name=rampncl-resume
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=05:45:00
#SBATCH --output=logs/resume_%j.out
#SBATCH --error=logs/resume_%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SCRIPT_DIR"

mkdir -p logs artifacts/executed_notebooks

module add python312
source .venv/bin/activate

export PROJECT_ROOT="$SCRIPT_DIR"
export WORK_ROOT="$SCRIPT_DIR"
export ARTIFACT_ROOT="$SCRIPT_DIR/artifacts/ramp_ncl_mfb_tc5_compliant_v2"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8192}"
export FORCE_RERUN=0
export INSTALL_REQUIREMENTS_IN_NOTEBOOK=0

python -m jupyter nbconvert \
  --to notebook \
  --execute ramp_ncl_mfb_tc5_end_to_end.ipynb \
  --output-dir artifacts/executed_notebooks \
  --output ramp_ncl_mfb_tc5_resume_all.ipynb
