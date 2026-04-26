#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

BASE_PREFIX="${BASE_PREFIX:-personb_tabred_space_ablation}"
SPACES_CSV="${SPACES_CSV:-logits,probs,hybrid}"

IFS=',' read -r -a SPACES <<< "$SPACES_CSV"
for space in "${SPACES[@]}"; do
  space_trimmed="$(echo "$space" | xargs)"
  PREFIX="${BASE_PREFIX}_${space_trimmed}"
  echo "[run] PREFIX=$PREFIX NCL_SPACE=$space_trimmed"
  PREFIX="$PREFIX" NCL_SPACE="$space_trimmed" bash personb/run_personb_tabred_final.sh
  echo "[done] $PREFIX"
done
