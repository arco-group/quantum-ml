# quantum-ml

Pipeline LOPO/LOSO su BIDMC Numerics (1 Hz) per forecasting multi-horizon con focus su ablation `GRU + mixer` (`none`, `mlp`, `vqc`).

## Struttura

```text
train_lopo.py
plot_curves.py
sweep_vqc.py
src/
  data.py
  splits.py
  baselines.py
  train.py
  eval.py
  utils.py
  models/
    gru_mixer.py
    transformer.py
    vqc.py
```

## Dipendenze

```bash
pip install numpy pandas torch matplotlib
pip install openpyxl  # necessario per output .xlsx
pip install pennylane  # obbligatorio solo per mixer=vqc
pip install pennylane-lightning  # consigliato (backend lightning.qubit)
# pip install pennylane-lightning-gpu  # opzionale per backend lightning.gpu
```

## Dataset atteso

File `Data/bidmc_XX_Numerics.csv` con colonne:

- `HR`
- `PULSE`
- `RESP`
- `SpO2`

Windowing:

- input `L=240` (default)
- target multi-horizon da `--horizons` (default `15,30,60`)
- `Y[i] = X[t + h_i]` con `t` ultimo timestamp della finestra input.

## Esecuzione standard (LOPO)

Baseline GRU:

```bash
python train_lopo.py --model gru_mixer --mixer none
```

Run singolo fold LOPO (utile per job array):

```bash
python train_lopo.py --model gru_mixer --mixer none --test_subject 01
```

Ablation non-quantum:

```bash
python train_lopo.py --model gru_mixer --mixer mlp
```

VQC con entanglement:

```bash
python train_lopo.py --model gru_mixer --mixer vqc --n_qubits 6 --vqc_depth 3 --vqc_entangle 1
```

VQC con tentativo backend GPU (se disponibile):

```bash
python train_lopo.py --model gru_mixer --mixer vqc --vqc_backend auto --vqc_vectorized 1 --device cuda
```

VQC forzato su `lightning.gpu` (errore se plugin non presente):

```bash
python train_lopo.py --model gru_mixer --mixer vqc --vqc_backend lightning.gpu --device cuda
```

VQC senza entanglement:

```bash
python train_lopo.py --model gru_mixer --mixer vqc --n_qubits 6 --vqc_depth 3 --vqc_entangle 0
```

Baselines classiche:

```bash
python train_lopo.py --model persistence
python train_lopo.py --model ridge
```

Output principali in `--output_dir`:

- `lopo_fold_metrics.xlsx`
- `lopo_aggregate.xlsx` (mean/std su soggetti test)
- `checkpoints/*.pt`
- `histories/*.xlsx`

## Data Efficiency

```bash
python train_lopo.py \
  --model gru_mixer \
  --exp_mode data_efficiency \
  --L 240 \
  --window_stride 1 \
  --horizons 15,30,60 \
  --seed 0 \
  --output_dir runs/de
```

Confronto automatico:

- `GRU + none`
- `GRU + vqc`

Output:

- `data_efficiency_fold_metrics.xlsx`
- `data_efficiency_summary.xlsx`

## Robustness

Missingness (test-only):

```bash
python train_lopo.py \
  --model gru_mixer \
  --exp_mode robustness \
  --perturb_type missingness \
  --missing_rates 0.0,0.1,0.3 \
  --perturb_target test \
  --mask_channel 1 \
  --output_dir runs/rob_miss
```

Noise (test-only):

```bash
python train_lopo.py \
  --model gru_mixer \
  --exp_mode robustness \
  --perturb_type noise \
  --noise_sigmas 0.0,0.01,0.05 \
  --output_dir runs/rob_noise
```

Output:

- `robustness_fold_metrics.xlsx`
- `robustness_summary.xlsx`

## Plot curve

```bash
python plot_curves.py \
  --data_eff_file runs/de/data_efficiency_summary.xlsx \
  --robustness_file runs/rob_miss/robustness_summary.xlsx \
  --metric macro_rmse \
  --output_dir runs/plots
```

Figure generate:

- `learning_curve.png`
- `robustness_curve.png`

## Sweep opzionale VQC (n_qubits, depth)

```bash
python sweep_vqc.py --n_qubits_grid 4,6,8 --depth_grid 1,3 --output_dir runs/vqc_sweep
```

Output:

- `runs/vqc_sweep/sweep_summary.xlsx`

## Note implementative

- Split LOPO by-subject: ogni fold usa 1 soggetto test e gli altri train/val.
- Val split by-subject (default `--val_ratio 0.1`).
- Standardizzazione sempre calcolata solo sui soggetti train del fold/configurazione corrente.
- `--nan_strategy`:
  - `ffill_bfill` (default)
  - `drop` (scarta finestre con NaN)
- `--window_stride`:
  - `1` (default): usa tutte le finestre
  - `>1`: sottocampiona le finestre (piu veloce, meno sample)
- `--test_subject`:
  - default vuoto: esegue tutti i fold LOPO
  - es. `01` o `1`: esegue solo il fold con quel soggetto in test
- `--vqc_backend`:
  - `auto` (default): prova `lightning.gpu` su host CUDA, altrimenti fallback CPU
  - `lightning.gpu`, `lightning.qubit`, `default.qubit`, `default.qubit.torch`
- `--vqc_vectorized`:
  - `1` (default): prova inferenza/training batched nel VQC, con fallback automatico a loop per-sample
  - `0`: forza modalità per-sample (più lenta ma utile per debug)
- Seed fisso con `--seed`.

## SLURM Job Array

Script inclusi in root:

- `slurm_lopo_array.sh`: lancia un fold per task array (usa `--test_subject`).
- `merge_lopo_runs.py`: unisce risultati per-soggetto in un unico `.xlsx`.
- `slurm_merge_results.sh`: wrapper SLURM per merge finale.

Rendi eseguibili gli script:

```bash
chmod +x slurm_lopo_array.sh slurm_merge_results.sh
```

Esempio submit baseline `GRU+none` su tutti i 53 soggetti:

```bash
sbatch --array=0-52%8 \
  --export=ALL,RUN_TAG=std_gru_none,EXP_MODE=standard,MODEL=gru_mixer,MIXER=none \
  slurm_lopo_array.sh
```

Esempio submit `GRU+VQC` CPU (`lightning.qubit`):

```bash
sbatch --array=0-52%8 \
  --export=ALL,RUN_TAG=std_gru_vqc,EXP_MODE=standard,MODEL=gru_mixer,MIXER=vqc,DEVICE=cpu,EXTRA_ARGS="--vqc_backend lightning.qubit --vqc_vectorized 1 --vqc_entangle 1" \
  slurm_lopo_array.sh
```

Merge finale risultati:

```bash
python merge_lopo_runs.py --runs_root runs/jobarray/std_gru_none
```

Submit merge come job dipendente:

```bash
jid=$(sbatch --array=0-52%8 --export=ALL,RUN_TAG=std_gru_none,EXP_MODE=standard,MODEL=gru_mixer,MIXER=none slurm_lopo_array.sh | awk '{print $4}')
sbatch --dependency=afterok:${jid} --export=ALL,RUN_TAG=std_gru_none slurm_merge_results.sh
```
