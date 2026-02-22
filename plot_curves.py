#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_dir


def _resolve_metric_cols(df: pd.DataFrame, metric: str) -> tuple[str, str | None]:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col in df.columns:
        return mean_col, std_col if std_col in df.columns else None
    if metric in df.columns:
        return metric, None
    raise ValueError(f"Metric '{metric}' not found in columns: {list(df.columns)}")


def _plot_data_efficiency(
    csv_path: Path,
    metric: str,
    output_dir: Path,
) -> None:
    df = pd.read_csv(csv_path)
    mean_col, std_col = _resolve_metric_cols(df, metric)
    group_col = "variant" if "variant" in df.columns else "model"
    if "train_fraction" not in df.columns:
        raise ValueError("Expected 'train_fraction' column in data efficiency CSV")

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in df.groupby(group_col):
        group = group.sort_values("train_fraction")
        x = group["train_fraction"].to_numpy()
        y = group[mean_col].to_numpy()
        ax.plot(x, y, marker="o", label=str(label))
        if std_col is not None:
            y_std = group[std_col].to_numpy()
            ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)

    ax.set_xlabel("Train Subject Fraction (p)")
    ax.set_ylabel(metric)
    ax.set_title("Learning Curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curve.png", dpi=150)
    plt.close(fig)


def _plot_robustness(
    csv_path: Path,
    metric: str,
    output_dir: Path,
) -> None:
    df = pd.read_csv(csv_path)
    mean_col, std_col = _resolve_metric_cols(df, metric)
    group_col = "variant" if "variant" in df.columns else "model"
    if "perturb_level" not in df.columns:
        raise ValueError("Expected 'perturb_level' column in robustness CSV")

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in df.groupby(group_col):
        group = group.sort_values("perturb_level")
        x = group["perturb_level"].to_numpy()
        y = group[mean_col].to_numpy()
        ax.plot(x, y, marker="o", label=str(label))
        if std_col is not None:
            y_std = group[std_col].to_numpy()
            ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)

    perturb_type = str(df["perturb_type"].iloc[0]) if "perturb_type" in df.columns else "perturbation"
    ax.set_xlabel(perturb_type)
    ax.set_ylabel(metric)
    ax.set_title("Robustness Curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning and robustness curves")
    parser.add_argument("--data_eff_csv", type=str, default="")
    parser.add_argument("--robustness_csv", type=str, default="")
    parser.add_argument("--metric", type=str, default="macro_rmse")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    if not args.data_eff_csv and not args.robustness_csv:
        raise ValueError("Provide at least one of --data_eff_csv or --robustness_csv")

    if args.data_eff_csv:
        _plot_data_efficiency(
            csv_path=Path(args.data_eff_csv),
            metric=args.metric,
            output_dir=output_dir,
        )
    if args.robustness_csv:
        _plot_robustness(
            csv_path=Path(args.robustness_csv),
            metric=args.metric,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()

