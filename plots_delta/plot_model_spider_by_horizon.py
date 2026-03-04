#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZONS = [15, 30, 60]
TASKS = ["HR", "PULSE", "RESP", "SpO2"]

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


def resolve_metrics_file(run_dir: Path) -> Path:
    for name in ("all_folds.xlsx", "lopo_fold_metrics.xlsx"):
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No metrics file found in {run_dir}")


def load_model_df(run_dir: Path, label: str, metric: str) -> pd.DataFrame:
    metric_cols = [f"{metric}_h{h}_{t}" for h in HORIZONS for t in TASKS]
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


def horizon_average(df: pd.DataFrame, metric: str, horizon: int) -> pd.Series:
    cols = [f"{metric}_h{horizon}_{task}" for task in TASKS]
    return df[cols].mean(axis=1)


def main() -> None:
    plt.rcParams["font.family"] = "Didot"

    parser = argparse.ArgumentParser(description="Spider plot by horizon using normalized ranking scores")
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

    rows: list[dict[str, object]] = []
    for h in HORIZONS:
        tmp = long_df[["test_subject", "model_label"]].copy()
        tmp["metric_h"] = horizon_average(long_df, metric=args.metric, horizon=h)
        tmp["order_idx"] = tmp["model_label"].map(order_map)
        tmp = tmp.sort_values(["test_subject", "metric_h", "order_idx"], kind="mergesort")
        tmp["rank_pos"] = tmp.groupby("test_subject").cumcount() + 1
        tmp["score"] = (n_models + 1) - tmp["rank_pos"]  # best->n_models, worst->1

        summ = tmp.groupby("model_label", as_index=False)["score"].sum().rename(columns={"score": "score_sum"})
        summ["score_avg"] = summ["score_sum"] / n_subjects
        summ["score_norm"] = summ["score_avg"] / float(args.scale_factor)
        summ["horizon"] = f"h{h}"
        rows.extend(summ.to_dict(orient="records"))

    summary = pd.DataFrame(rows)
    summary["model_label"] = pd.Categorical(summary["model_label"], categories=model_order, ordered=True)
    summary = summary.sort_values(["model_label", "horizon"]).reset_index(drop=True)

    csv_path = out_dir / f"model_spider_by_horizon_summary_{args.metric}.csv"
    summary.to_csv(csv_path, index=False)

    categories = [f"h{h}" for h in HORIZONS]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for raw_label in model_order:
        disp = DISPLAY_NAME_MAP.get(raw_label.lower(), raw_label)
        color = MODEL_COLOR_MAP.get(raw_label.lower(), None)
        s = summary[summary["model_label"].astype(str) == raw_label]
        vals = [float(s[s["horizon"] == cat]["score_norm"].iloc[0]) for cat in categories]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=disp, color=color)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1], categories)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.2, 1.0, 5))
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0.2, 1.0, 5)])
    ax.set_title(f"Spider by Horizon (macro {args.metric.upper()}, normalized ranking)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10))
    fig.tight_layout()

    fig_path = out_dir / f"model_spider_by_horizon_{args.metric}.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    print(f"Saved summary: {csv_path}")
    print(f"Saved plot: {fig_path}")
    print(f"Common subjects: {n_subjects}")


if __name__ == "__main__":
    main()
