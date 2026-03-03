#!/usr/bin/env bash
# Submit the 24 (nq, depth, entangle) VQC combinations as SLURM arrays on Alvis.
# Run this from the repo root: /mimer/NOBACKUP/groups/naiss2023-6-336/mtortora/quantum-ml

set -euo pipefail

NQ_GRID=(4 6 8 10)
DEPTH_GRID=(1 3 5)
ENT_GRID=(0 1)
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-8}"  # max parallel folds per combo

for nq in "${NQ_GRID[@]}"; do
  for d in "${DEPTH_GRID[@]}"; do
    for e in "${ENT_GRID[@]}"; do
      tag="vqc_sweep/nq_${nq}_d_${d}_e_${e}"
      echo "Submitting ${tag}"
      jid=$(sbatch --array=0-52%${ARRAY_CONCURRENCY} \
        --export=ALL,RUN_TAG=${tag},EXP_MODE=standard,MODEL=gru_mixer,MIXER=vqc,DEVICE=cuda,\
EXTRA_ARGS="--n_qubits ${nq} --vqc_depth ${d} --vqc_entangle ${e} --vqc_backend auto --target_mode delta" \
        slurm_lopo_array.sh | awk '{print $4}')
      sbatch --dependency=afterok:${jid} \
        --export=ALL,RUN_TAG=${tag},GROUP_KEYS=model,mixer \
        slurm_merge_results.sh >/dev/null
    done
  done
done

echo "Tutti i job sono stati sottomessi. Controlla con: squeue -u $USER"
