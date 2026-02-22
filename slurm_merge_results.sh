#!/bin/bash
#SBATCH -J qml_merge
#SBATCH -A naiss2023-6-336
#SBATCH -p main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
VENV_PATH="${VENV_PATH:-$ROOT/quantum-venv/bin/activate}"
RUN_TAG="${RUN_TAG:-std_gru_none}"
METRICS_NAME="${METRICS_NAME:-lopo_fold_metrics.xlsx}"
GROUP_KEYS="${GROUP_KEYS:-model,mixer}"

cd "$ROOT"
mkdir -p logs

if [[ -f "$VENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
fi

python merge_lopo_runs.py \
  --runs_root "$ROOT/runs/jobarray/$RUN_TAG" \
  --metrics_name "$METRICS_NAME" \
  --group_keys "$GROUP_KEYS"

