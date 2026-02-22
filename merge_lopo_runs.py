#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format '{suffix}' for {path}")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        df.to_excel(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported file format '{suffix}' for {path}")


def _aggregate(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in numeric_cols if c not in group_keys]
    if not numeric_cols:
        raise ValueError("No numeric columns available for aggregation")

    if group_keys:
        missing = [k for k in group_keys if k not in df.columns]
        if missing:
            raise ValueError(f"Missing group keys in dataframe: {missing}")
        grouped = df.groupby(group_keys, dropna=False, sort=True)[numeric_cols]
        agg = grouped.agg(["mean", "std"]).reset_index()
    else:
        agg_values = {}
        for col in numeric_cols:
            agg_values[f"{col}_mean"] = float(df[col].mean())
            agg_values[f"{col}_std"] = float(df[col].std(ddof=0))
        agg_values["count"] = int(df.shape[0])
        return pd.DataFrame([agg_values])

    agg.columns = [
        "_".join([piece for piece in col if piece]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in agg.columns
    ]
    if "count" not in agg.columns:
        counts = df.groupby(group_keys, dropna=False, sort=True).size().reset_index(name="count")
        agg = agg.merge(counts, on=group_keys, how="left")
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-subject LOPO outputs from job arrays")
    parser.add_argument("--runs_root", type=str, required=True)
    parser.add_argument("--metrics_name", type=str, default="lopo_fold_metrics.xlsx")
    parser.add_argument("--output_all", type=str, default="all_folds.xlsx")
    parser.add_argument("--output_agg", type=str, default="aggregate_mean_std.xlsx")
    parser.add_argument("--group_keys", type=str, default="model,mixer")
    parser.add_argument("--subject_glob", type=str, default="subj_*")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    group_keys = [k.strip() for k in args.group_keys.split(",") if k.strip()]

    files = sorted((runs_root / args.subject_glob).parent.glob(f"{args.subject_glob}/{args.metrics_name}"))
    if not files:
        raise FileNotFoundError(f"No files found: {runs_root}/{args.subject_glob}/{args.metrics_name}")

    frames: list[pd.DataFrame] = []
    for file in files:
        df = _read_table(file)
        if df.empty:
            continue
        df["source_run"] = str(file.parent.name)
        frames.append(df)

    if not frames:
        raise RuntimeError("All discovered metrics files are empty")

    merged = pd.concat(frames, ignore_index=True)
    merged_out = runs_root / args.output_all
    _write_table(merged, merged_out)

    aggregated = _aggregate(merged, group_keys=group_keys)
    agg_out = runs_root / args.output_agg
    _write_table(aggregated, agg_out)

    print(f"Saved merged file: {merged_out}")
    print(f"Saved aggregate file: {agg_out}")
    print(f"Loaded subject runs: {len(frames)}")


if __name__ == "__main__":
    main()

