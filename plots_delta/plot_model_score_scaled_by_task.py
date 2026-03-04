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


TASKS = ["HR", "PULSE", "RESP", "SpO2"]
HORIZONS = [15, 30, 60]


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


def load_model_df(run_dir: Path, label: str, metric: str) -> pd.DataFrame:
    metric_cols = [f"{metric}_h{h}_{t}" for t in TASKS for h in HORIZONS]
    metrics_file = resolve_metrics_file(run_dir)
    df = pd.read_excel(metrics_file)
    required = {"test_subject", *metric_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {metrics_file}")
    out = df[["test_subject", *metric_cols]].copy()
    out["test_subject"] = out["test_subject"].astype(str).str.zfill(2)
    out["model_label"] = label
    return out


def task_average(df: pd.DataFrame, metric: str, task: str) -> pd.Series:
    cols = [f"{metric}_h{h}_{task}" for h in HORIZONS]
    return df[cols].mean(axis=1)


def main() -> None:
    plt.rcParams["font.family"] = "Didot"

    parser = argparse.ArgumentParser(description="Normalized ranking score by task (4 subplots)")
    parser.add_argument("--metric", choices=["rmse", "mae"], default="rmse")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_run", action="append", required=True, help="label=/abs/path/to/run_dir")
    parser.add_argument("--scale_factor", type=float, default=6.0)
    parser.add_argument("--palette", choices=["Pastel1", "Pastel2"], default="Pastel1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_order: list[str] = []
    frames: list[pd.DataFrame] = []
    for item in args.model_run:
        if "=" not in item:
            raise ValueError(f"Invalid --model_run '{item}', expected label=/path")
        label, path = item.split("=", 1)
        label = label.strip()
        model_order.append(label)
        frames.append(load_model_df(Path(path), label=label, metric=args.metric))

    long_df = pd.concat(frames, ignore_index=True)
    counts = long_df.groupby("test_subject")["model_label"].nunique()
    common_subjects = counts[counts == len(model_order)].index
    long_df = long_df[long_df["test_subject"].isin(common_subjects)].copy()
    if long_df.empty:
        raise RuntimeError("No common subjects across selected model runs.")

    n_subjects = int(long_df["test_subject"].nunique())
    n_models = len(model_order)
    order_map = {m: i for i, m in enumerate(model_order)}

    all_summary_rows: list[dict[str, object]] = []
    task_to_summary: dict[str, pd.DataFrame] = {}
    for task in TASKS:
        task_df = long_df[["test_subject", "model_label"]].copy()
        task_df["metric_task"] = task_average(long_df, metric=args.metric, task=task)
        task_df["order_idx"] = task_df["model_label"].map(order_map)
        task_df = task_df.sort_values(["test_subject", "metric_task", "order_idx"], kind="mergesort")
        task_df["rank_pos"] = task_df.groupby("test_subject").cumcount() + 1
        task_df["score"] = (n_models + 1) - task_df["rank_pos"]  # best -> n_models, worst -> 1

        summary = (
            task_df.groupby("model_label", as_index=False)["score"]
            .sum()
            .rename(columns={"score": "score_sum"})
        )
        summary["score_avg"] = summary["score_sum"] / n_subjects
        summary["score_norm"] = summary["score_avg"] / float(args.scale_factor)
        summary["task"] = task
        summary["n_subjects"] = n_subjects
        summary["model_label"] = pd.Categorical(summary["model_label"], categories=model_order, ordered=True)
        summary = summary.sort_values("model_label").reset_index(drop=True)
        task_to_summary[task] = summary
        all_summary_rows.extend(summary.to_dict(orient="records"))

    summary_df = pd.DataFrame(all_summary_rows)
    summary_path = out_dir / f"model_score_scaled_by_task_summary_{args.metric}.csv"
    summary_df.to_csv(summary_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes = axes.ravel()
    x = np.arange(len(model_order))
    display_order = [DISPLAY_NAME_MAP.get(m.lower(), m) for m in model_order]
    fallback = build_palette(args.palette, len(model_order))
    colors = [MODEL_COLOR_MAP.get(m.lower(), fallback[i]) for i, m in enumerate(model_order)]
    for idx, (ax, task) in enumerate(zip(axes, TASKS)):
        s = task_to_summary[task]
        y = s["score_norm"].to_numpy(dtype=float)
        bars = ax.bar(x, y, color=colors)
        for i, val in enumerate(y):
            ax.text(
                bars[i].get_x() + bars[i].get_width() / 2,
                val / 2.0,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
                fontweight="bold",
            )
        ax.set_title(task)
        ax.set_xticks(x, display_order, rotation=0, ha="center")
        ax.set_ylim(0.0, 1.05)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.grid(axis="y", alpha=0.25)
        if idx % 2 == 0:  # left column only
            ax.set_ylabel("Normalized score", fontsize=14)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=11)

    fig.suptitle(f"Per-task normalized ranking score (macro {args.metric.upper()}, no ties)", y=0.98)
    fig.tight_layout()
    fig_path = out_dir / f"model_score_scaled_by_task_{args.metric}.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {fig_path}")
    print(f"Common subjects: {n_subjects}")


if __name__ == "__main__":
    main()
