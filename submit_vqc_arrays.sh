#!/usr/bin/env bash
# Submit the VQC (qubit, depth, entanglement) combinations as SLURM arrays.
# Run from the repository root. Set ARRAY_RANGE for the number of subjects,
# for example ARRAY_RANGE=0-52 for 53 subjects.

set -euo pipefail

NQ_GRID=(4 6 8 10)
DEPTH_GRID=(1 3 5)
ENT_GRID=(0 1)
ARRAY_RANGE="${ARRAY_RANGE:-0-52}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-8}"
SBATCH_ARGS="${SBATCH_ARGS:-}"
read -r -a SBATCH_EXTRA <<< "$SBATCH_ARGS"

for nq in "${NQ_GRID[@]}"; do
  for d in "${DEPTH_GRID[@]}"; do
    for e in "${ENT_GRID[@]}"; do
      tag="vqc_sweep/nq_${nq}_d_${d}_e_${e}"
      echo "Submitting ${tag}"
      jid=$(sbatch "${SBATCH_EXTRA[@]}" --array="${ARRAY_RANGE}%${ARRAY_CONCURRENCY}" \
        --export="ALL,RUN_TAG=${tag},EXP_MODE=standard,MODEL=gru_mixer,MIXER=vqc,DEVICE=cuda,EXTRA_ARGS=--n_qubits ${nq} --vqc_depth ${d} --vqc_entangle ${e} --vqc_backend auto --target_mode delta" \
        slurm_lopo_array.sh | awk '{print $4}')
      sbatch "${SBATCH_EXTRA[@]}" --dependency="afterok:${jid}" \
        --export="ALL,RUN_TAG=${tag},GROUP_KEYS=model,mixer" \
        slurm_merge_results.sh >/dev/null
    done
  done
done

echo "All jobs submitted. Check the queue with: squeue -u \"$USER\""
