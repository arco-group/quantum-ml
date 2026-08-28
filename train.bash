#!/usr/bin/env bash
#SBATCH -A naiss2025-5-662 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00-05:00:00
# Output files
#SBATCH --error=outputs/job_%J.err
#SBATCH --output=outputs/out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.iele@unicampus.it

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
#python train_lopo.py --model gru_mixer --mixer vqc --n_qubits 6 --vqc_depth 3 --vqc_entangle 1 --output_dir runs/VQC_gpu --device cpu --vqc_backend default.qubit

#: <<'COMMENTO'
python train_lopo.py \
  --model gru_mixer \
  --mixer vqc \
  --n_qubits 10 \
  --vqc_depth 5 \
  --vqc_entangle 1 \
  --target_mode delta \
  --horizons 15,30,60 \
  --L 240 \
  --window_stride 1 \
  --epochs 30 \
  --batch_size 128 \
  --lr 1e-3 \
  --output_dir runs_2/final_gru_vqc_n10_d5_e1 \
  --device cpu \
  --vqc_backend default.qubit
#COMMENTO

: <<'COMMENTO'
python train_lopo.py \
  --model gru_mixer \
  --mixer mlp \
  --exp_mode robustness \
  --perturb_type missingness \
  --missing_rates 0.0,0.1,0.3 \
  --perturb_target test \
  --mask_channel 1 \
  --output_dir runs/rob_miss_new
#COMMENTO