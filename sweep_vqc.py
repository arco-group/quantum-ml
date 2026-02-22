#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.utils import ensure_dir, parse_int_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Small sweep for VQC n_qubits/depth")
    parser.add_argument("--data_dir", type=str, default="Data")
    parser.add_argument("--output_dir", type=str, default="runs/vqc_sweep")
    parser.add_argument("--n_qubits_grid", type=str, default="4,6,8")
    parser.add_argument("--depth_grid", type=str, default="1,3")
    parser.add_argument("--L", type=int, default=240)
    parser.add_argument("--horizons", type=str, default="15,30,60")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_folds", type=int, default=0)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    n_qubits_grid = parse_int_csv(args.n_qubits_grid)
    depth_grid = parse_int_csv(args.depth_grid)

    rows: list[dict] = []
    for n_qubits in n_qubits_grid:
        for depth in depth_grid:
            run_dir = output_dir / f"nq_{n_qubits}_d_{depth}"
            cmd = [
                sys.executable,
                "train_lopo.py",
                "--data_dir",
                args.data_dir,
                "--model",
                "gru_mixer",
                "--mixer",
                "vqc",
                "--n_qubits",
                str(n_qubits),
                "--vqc_depth",
                str(depth),
                "--vqc_entangle",
                "1",
                "--L",
                str(args.L),
                "--horizons",
                args.horizons,
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--output_dir",
                str(run_dir),
                "--max_folds",
                str(args.max_folds),
            ]
            subprocess.run(cmd, check=True)
            agg_path = run_dir / "lopo_aggregate.xlsx"
            if not agg_path.exists():
                continue
            agg = pd.read_excel(agg_path)
            if agg.empty:
                continue
            row = {
                "n_qubits": n_qubits,
                "vqc_depth": depth,
                "macro_mae_mean": float(agg["macro_mae_mean"].iloc[0]),
                "macro_mae_std": float(agg["macro_mae_std"].iloc[0]),
                "macro_rmse_mean": float(agg["macro_rmse_mean"].iloc[0]),
                "macro_rmse_std": float(agg["macro_rmse_std"].iloc[0]),
                "run_dir": str(run_dir),
            }
            rows.append(row)

    pd.DataFrame(rows).to_excel(output_dir / "sweep_summary.xlsx", index=False)


if __name__ == "__main__":
    main()
