#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISPLAY_NAME_MAP = {
    "extratrees": "ExtraTrees",
    "cnn1d": "CNN 1D",
    "tcn": "TCN",
    "lstm": "LSTM",
    "patchtst": "PatchTST",
    "ours": "Ours",
}

MODEL_COLOR_MAP = {
    "extratrees": "#a6cee3",
    "cnn1d": "#fdbf6f",
    "tcn": "#b2df8a",
    "lstm": "#fb9a99",
    "patchtst": "#cab2d6",
    "ours": "#4c78a8",
}


def build_palette(name: str, n: int) -> list:
    cmap = plt.get_cmap(name)
    colors = list(cmap.colors) if hasattr(cmap, "colors") else [cmap(i / max(n - 1, 1)) for i in range(n)]
    return [colors[i % len(colors)] for i in range(n)]


def resolve_metrics_file(run_dir: Path) -> Path:
    for name in ("all_folds.xlsx", "lopo_fold_metrics.xlsx"):
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No metrics file found in {run_dir}")


def load_model_series(run_dir: Path, label: str, metric_col: str) -> pd.DataFrame:
    metrics_file = resolve_metrics_file(run_dir)
    df = pd.read_excel(metrics_file)
    required = {"test_subject", metric_col}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns {missing} in {metrics_file}")
    out = df[["test_subject", metric_col]].copy()
    out["test_subject"] = out["test_subject"].astype(str).str.zfill(2)
    out = out.rename(columns={metric_col: "metric"})
    out["model_label"] = label
    return out


def main() -> None:
    plt.rcParams["font.family"] = "Didot"

    parser = argparse.ArgumentParser(description="Scaled score plot from per-subject rankings")
    parser.add_argument("--metric", choices=["rmse", "mae"], default="rmse")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--model_run",
        action="append",
        required=True,
        help="Format: label=/abs/path/to/run_dir",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=6.0,
        help="Normalization divisor after average score: score_norm=(sum(score)/n_subjects)/scale_factor",
    )
    parser.add_argument("--palette", choices=["Pastel1", "Pastel2"], default="Pastel1")
    args = parser.parse_args()

    metric_col = f"macro_{args.metric}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    model_order: list[str] = []
    for item in args.model_run:
        if "=" not in item:
            raise ValueError(f"Invalid --model_run '{item}', expected label=/path")
        label, path = item.split("=", 1)
        label = label.strip()
        model_order.append(label)
        frames.append(load_model_series(Path(path), label=label, metric_col=metric_col))

    long_df = pd.concat(frames, ignore_index=True)

    counts = long_df.groupby("test_subject")["model_label"].nunique()
    common_subjects = counts[counts == len(model_order)].index
    long_df = long_df[long_df["test_subject"].isin(common_subjects)].copy()
    if long_df.empty:
        raise RuntimeError("No common subjects across selected model runs.")

    order_map = {m: i for i, m in enumerate(model_order)}
    long_df["order_idx"] = long_df["model_label"].map(order_map)
    long_df = long_df.sort_values(["test_subject", "metric", "order_idx"], kind="mergesort")
    long_df["rank_pos"] = long_df.groupby("test_subject").cumcount() + 1
    n_models = len(model_order)
    long_df["score"] = (n_models + 1) - long_df["rank_pos"]  # best -> n_models, worst -> 1

    n_subjects = int(long_df["test_subject"].nunique())
    summary = long_df.groupby("model_label", as_index=False)["score"].sum().rename(columns={"score": "score_sum"})
    summary["n_subjects"] = n_subjects
    summary["score_avg"] = summary["score_sum"] / n_subjects
    summary["score_scaled"] = summary["score_avg"] / float(args.scale_factor)
    summary["model_label"] = pd.Categorical(summary["model_label"], categories=model_order, ordered=True)
    summary = summary.sort_values("model_label").reset_index(drop=True)
    summary["display_label"] = summary["model_label"].astype(str).map(
        lambda x: DISPLAY_NAME_MAP.get(x.lower(), x)
    )

    summary_path = out_dir / f"model_score_scaled_summary_macro_{args.metric}.csv"
    summary.to_csv(summary_path, index=False)

    x = np.arange(len(summary))
    y = summary["score_scaled"].to_numpy(dtype=float)
    fallback = build_palette(args.palette, len(summary))
    colors = []
    for i, raw_label in enumerate(summary["model_label"].astype(str).tolist()):
        colors.append(MODEL_COLOR_MAP.get(raw_label.lower(), fallback[i]))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, y, color=colors)
    for i, val in enumerate(y):
        y_mid = val / 2.0
        ax.text(
            bars[i].get_x() + bars[i].get_width() / 2,
            y_mid,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color="black",
            fontweight="bold",
        )
    ax.set_xticks(x, summary["display_label"], rotation=0, ha="center")
    ax.set_ylabel(f"Normalized score = (sum(score)/{n_subjects})/{args.scale_factor:g}", fontsize=16)
    ax.set_title(f"Model Normalized Score (macro {args.metric.upper()}, no ties)")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    fig.tight_layout()

    fig_path = out_dir / f"model_score_scaled_macro_{args.metric}.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {fig_path}")
    print(f"Common subjects: {n_subjects}")


if __name__ == "__main__":
    main()
