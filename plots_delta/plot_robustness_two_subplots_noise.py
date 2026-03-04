#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


VARIANT_NAME = {
    "gru_none": "GRU",
    "gru_vqc": "GRU+VQC",
    "gru_mlp": "GRU+MLP",
}

VARIANT_COLOR = {
    "gru_none": "#1f77b4",
    "gru_vqc": "#d62728",
    "gru_mlp": "#2ca02c",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness-to-noise plot with two aligned subplots (MAE/RMSE)")
    parser.add_argument("--input_xlsx", required=True, type=str)
    parser.add_argument("--output_png", required=True, type=str)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    df = pd.read_excel(args.input_xlsx)
    needed = {"variant", "perturb_level", "macro_mae_mean", "macro_rmse_mean"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["perturb_level"] = df["perturb_level"].astype(float)
    perturb_type = str(df["perturb_type"].dropna().iloc[0]) if "perturb_type" in df.columns else "noise"
    if perturb_type != "noise":
        raise ValueError(
            f"This script is for noise only, but got perturb_type='{perturb_type}'. "
            "Use plot_robustness_two_subplots.py for missingness."
        )
    df["x_level"] = df["perturb_level"]
    x_label = "Noise σ"
    df = df.sort_values(["variant", "x_level"])

    fig, (ax_mae, ax_rmse) = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.4),
        sharex=True,
        gridspec_kw={"hspace": 0.10},
        constrained_layout=True,
    )

    handles = []
    labels = []
    for variant, g in df.groupby("variant", sort=False):
        x = g["x_level"].to_numpy()
        mae = g["macro_mae_mean"].to_numpy()
        rmse = g["macro_rmse_mean"].to_numpy()
        color = VARIANT_COLOR.get(str(variant), None)
        name = VARIANT_NAME.get(str(variant), str(variant))

        h1 = ax_mae.plot(x, mae, marker="o", linewidth=2.2, color=color, label=name)[0]
        ax_rmse.plot(x, rmse, marker="s", linewidth=2.2, color=color, label=name)
        handles.append(h1)
        labels.append(name)

    levels = sorted(df["x_level"].unique().tolist())
    ax_rmse.set_xticks(levels)
    ax_rmse.set_xlabel(x_label)

    ax_mae.set_ylabel("MAE ↓")
    ax_rmse.set_ylabel("RMSE ↓")
    ax_mae.grid(True, axis="both", alpha=0.25)
    ax_rmse.grid(True, axis="both", alpha=0.25)

    ax_mae.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.86),
        frameon=True,
        fontsize=9,
    )
    ax_rmse.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.86),
        frameon=True,
        fontsize=9,
    )

    if args.title.strip():
        fig.suptitle(args.title)

    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
