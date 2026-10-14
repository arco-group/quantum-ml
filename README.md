# Hybrid Quantum Neural Network for Multivariate Clinical Time Series Forecasting

[![arXiv](https://img.shields.io/badge/arXiv-2603.08072-b31b1b.svg)](https://arxiv.org/abs/2603.08072)

Code accompanying the paper *Hybrid Quantum Neural Network for Multivariate Clinical Time Series Forecasting*.

Irene Iele¹, Floriano Caprio¹, Paolo Soda¹˒², Matteo Tortora³

¹ University Campus Bio-Medico of Rome · ² Umeå University · ³ University of Genoa

![Overview of the proposed method](https://github.com/user-attachments/assets/6748f527-0eb3-402e-9541-3b490c7293d6)

## What this repository does

This project trains and evaluates forecasting models on the BIDMC PPG and Respiration dataset's Numerics recordings. Each recording is treated as one subject. The main evaluation uses leave-one-subject-out (LOSO/LOPO) cross-validation: one subject is held out for testing, and the remaining subjects are split into training and validation sets.

The default task uses four 1 Hz signals (`HR`, `PULSE`, `RESP`, and `SpO2`). A 240-step input window predicts the values 15, 30, and 60 steps ahead. Window length, stride, horizons, target representation, model, and training settings can be changed from the command line.

## Repository contents

- `train_lopo.py`: main entry point for standard, data-efficiency, and robustness experiments.
- `sweep_vqc.py`: sequential sweep over VQC qubit count, depth, and entanglement settings.
- `bidmc_download.py`: downloads CSV files from the BIDMC dataset index.
- `merge_lopo_runs.py`, `merge_found_reports.py`, `export_competitor_metrics.py`: combine or export experiment metrics.
- `src/`: data preparation, subject splits, models, training, baselines, and evaluation code.
- `slurm_*.sh`, `submit_vqc_arrays.sh`: optional SLURM job scripts. They contain generic resource settings; supply your cluster's account and partition when submitting.

Generated data and experiment outputs are not included in the repository.

## Requirements and installation

Python 3.10 or newer is recommended. Create an environment and install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default workflow writes Excel files, so `openpyxl` is included in the requirements. For CUDA, install a PyTorch build compatible with your system and CUDA driver.

Optional dependencies:

```bash
python -m pip install pennylane pennylane-lightning  # VQC mixer and CPU Lightning backend
python -m pip install xgboost                         # xgboost_direct baseline
```

Other PennyLane backends, such as `lightning.gpu`, require their corresponding PennyLane plugin and a compatible GPU environment.

## Data

Place files named `bidmc_<subject>_Numerics.csv` in `Data/`. The loader expects the columns `HR`, `PULSE`, `RESP`, and `SpO2`. Data files are not tracked by Git.

To download the CSV files from the default BIDMC index:

```bash
python bidmc_download.py --dest Data
```

The loader discovers subject IDs from filenames. By default it forward-fills and backward-fills missing values. Use `--nan_strategy drop` to discard windows containing missing values instead.

## Run an experiment

Run the default GRU baseline on CPU:

```bash
python train_lopo.py \
  --model gru_mixer \
  --mixer none \
  --device cpu \
  --output_dir runs/gru_baseline
```

The default `--device` is `cuda`; if CUDA is unavailable, the code falls back to CPU. To run only one test-subject fold, add `--test_subject 1`. To limit a run for a quick check, add `--max_folds 1`.

### Models

The `--model` option supports:

- Neural models: `gru_mixer`, `patchtst_tiny`, `dlinear`, `nhits_tiny`, `lstm_tiny`, `tcn`, and `transformer`.
- Baselines: `persistence`, `ridge`, `extratrees_direct`, `random_forest_direct`, `cnn1d_direct`, and `xgboost_direct`.

The `--mixer` option applies to `gru_mixer` and accepts `none`, `mlp`, or `vqc`. PennyLane is required for `--mixer vqc`.

Examples:

```bash
# GRU with an MLP mixer
python train_lopo.py --model gru_mixer --mixer mlp --device cpu

# GRU with a variational quantum circuit
python train_lopo.py --model gru_mixer --mixer vqc \
  --n_qubits 6 --vqc_depth 3 --vqc_entangle 1 --device cpu

# PatchTST and an optional XGBoost baseline
python train_lopo.py --model patchtst_tiny --device cpu
python train_lopo.py --model xgboost_direct --device cpu
```

Run `python train_lopo.py --help` for all model and experiment options.

### Forecast targets and evaluation

- `--L`: input window length (default `240`).
- `--window_stride`: distance between input windows (default `1`).
- `--horizons`: comma-separated forecast steps (default `15,30,60`).
- `--target_mode value`: predict future signal values directly (default).
- `--target_mode delta`: predict changes relative to the last input step; reported metrics are converted back to signal values.

The training and validation split is by subject. Feature normalization statistics are computed from training subjects only. The default validation fraction is `0.1`; `--seed` controls randomized splits and training. Results include macro MAE/RMSE and per-horizon, per-signal metrics.

## Experiment modes

### Data efficiency

Compare GRU mixers while varying the fraction of training subjects:

```bash
python train_lopo.py --exp_mode data_efficiency \
  --data_fractions 1.0,0.5,0.25 \
  --device cpu --output_dir runs/data_efficiency
```

The mode evaluates `gru_mixer` with `none`, `mlp`, and `vqc` mixers.

### Robustness

Evaluate the three GRU mixer variants under missingness or added noise. By default, perturbations affect the test split only.

```bash
python train_lopo.py --exp_mode robustness \
  --perturb_type missingness --missing_rates 0.0,0.1,0.3 \
  --device cpu --output_dir runs/robustness_missingness
```

Use `--perturb_type noise --noise_sigmas 0.0,0.01,0.05` for noise, or set `--perturb_target train_test` to perturb training, validation, and test data.

### VQC parameter sweep

Run a sequential sweep over qubit counts, circuit depths, and entanglement settings:

```bash
python sweep_vqc.py --data_dir Data --n_qubits_grid 4,6,8 \
  --depth_grid 1,3 --entangle_grid 0,1 --output_dir runs/vqc_sweep
```

This requires PennyLane. The sweep runs each configuration in sequence and writes a summary after the runs complete.

## Outputs

Each training run writes to `--output_dir`. Standard mode produces:

- `lopo_fold_metrics.xlsx`: metrics for each test-subject fold.
- `lopo_aggregate.xlsx`: mean and standard deviation across folds.
- `checkpoints/`: best model checkpoints.
- `histories/`: per-epoch training histories.

Data-efficiency and robustness modes write corresponding `*_fold_metrics.xlsx` and summary `.xlsx` files. The sweep writes `sweep_summary.xlsx` in its output directory. Excel output requires `openpyxl`.

Merge separate SLURM array outputs with:

```bash
python merge_lopo_runs.py \
  --runs_root runs/jobarray/my_run \
  --metrics_name lopo_fold_metrics.xlsx
```

Use `python merge_lopo_runs.py --help` and `python export_competitor_metrics.py --help` for their available options.

## SLURM (optional)

The job scripts assume you run them from the repository root and have created `logs/`. Set `DATA_DIR` if your data is elsewhere. If you created the `.venv` environment above, export its activation script before submitting jobs:

```bash
export VENV_PATH="$PWD/.venv/bin/activate"
```

For example, submit a GPU array for 53 subjects, with at most 8 folds running at once:

```bash
mkdir -p logs
sbatch --account YOUR_ACCOUNT --partition YOUR_PARTITION \
  --array=0-52%8 \
  --export=ALL,RUN_TAG=gru_baseline,EXP_MODE=standard,MODEL=gru_mixer,MIXER=none \
  slurm_lopo_array.sh
```

Use `slurm_lopo_array_cpu.sh` for CPU jobs. Set the array range to match the number of subject files available. The GPU script requests one GPU; edit its resource directives if your cluster uses different syntax or hardware.

`submit_vqc_arrays.sh` submits the 24 combinations in its configured grid and a merge job after each array. Configure the subject range and cluster options before using it:

```bash
mkdir -p logs
ARRAY_RANGE=0-52 ARRAY_CONCURRENCY=8 \
SBATCH_ARGS="--account YOUR_ACCOUNT --partition YOUR_PARTITION" \
  ./submit_vqc_arrays.sh
```

SLURM is optional; all experiments can also be run directly with `train_lopo.py`.
