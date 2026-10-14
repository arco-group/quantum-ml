#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:20:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-Data}"
VENV_PATH="${VENV_PATH:-$ROOT/quantum-venv/bin/activate}"

RUN_TAG="${RUN_TAG:-std_gru_none}"
EXP_MODE="${EXP_MODE:-standard}"
MODEL="${MODEL:-gru_mixer}"
MIXER="${MIXER:-none}"

L_WINDOW="${L_WINDOW:-240}"
WINDOW_STRIDE="${WINDOW_STRIDE:-1}"
HORIZONS="${HORIZONS:-15,30,60}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cuda}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=..."
  exit 1
fi

cd "$ROOT"
mkdir -p logs

if [[ -f "$VENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
fi

mapfile -t SUBJECTS < <(
  python - "$ROOT" "$DATA_DIR" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
data_dir_raw = Path(sys.argv[2])
data_dir = data_dir_raw if data_dir_raw.is_absolute() else (root / data_dir_raw)

regex = re.compile(r"bidmc_(\d+)_Numerics\.csv$")
subjects = []
for path in sorted(data_dir.glob("bidmc_*_Numerics.csv")):
    match = regex.match(path.name)
    if match:
        subjects.append(match.group(1))

for subject in sorted(subjects, key=int):
    print(subject)
PY
)

if [[ "${#SUBJECTS[@]}" -eq 0 ]]; then
  echo "No BIDMC Numerics subjects found in $DATA_DIR"
  exit 1
fi

IDX="$SLURM_ARRAY_TASK_ID"
if (( IDX < 0 || IDX >= ${#SUBJECTS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=$IDX is out of range [0, $(( ${#SUBJECTS[@]} - 1 ))]"
  exit 1
fi

TEST_SUBJECT="${SUBJECTS[$IDX]}"
OUT_DIR="$ROOT/runs/jobarray/$RUN_TAG/subj_${TEST_SUBJECT}"

CMD=(
  python train_lopo.py
  --data_dir "$DATA_DIR"
  --exp_mode "$EXP_MODE"
  --model "$MODEL"
  --mixer "$MIXER"
  --test_subject "$TEST_SUBJECT"
  --L "$L_WINDOW"
  --window_stride "$WINDOW_STRIDE"
  --horizons "$HORIZONS"
  --seed "$SEED"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --device "$DEVICE"
  --output_dir "$OUT_DIR"
)

if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a EXTRA_ARRAY <<< "$EXTRA_ARGS"
  CMD+=( "${EXTRA_ARRAY[@]}" )
fi

echo "Submitting fold for test_subject=$TEST_SUBJECT run_tag=$RUN_TAG"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
