#!/usr/bin/env bash
#SBATCH -A naiss2025-5-662 -p alvis
#SBATCH -C NOGPU
#SBATCH --ntasks-per-node=1
#SBATCH -t 00-01:00:00
# Output files
#SBATCH --error=outputs/job_%J.err
#SBATCH --output=outputs/out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.tortora@unicampus.it

# Load modules
module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Activate venv
cd /mimer/NOBACKUP/groups/naiss2023-6-336/mtortora/quantum-ml/quantum-venv || exit

source bin/activate

# Executes the code
cd /mimer/NOBACKUP/groups/naiss2023-6-336/mtortora/quantum-ml || exit

# RUN YOUR PROGRAM
python train_lopo.py --model gru_mixer --mixer vqc --n_qubits 6 --vqc_depth 3 --vqc_entangle 1 --output_dir runs/VQC