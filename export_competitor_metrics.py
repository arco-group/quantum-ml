#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    out = df.copy()
    out.columns = [
        "_".join([str(piece) for piece in col if str(piece) != ""]).rstrip("_")
        for col in out.columns
    ]
    return out


def load_or_build_aggregate(run_dir: Path) -> pd.DataFrame | None:
    agg_path = run_dir / "aggregate_mean_std.xlsx"
    if agg_path.exists():
        agg = pd.read_excel(agg_path)
        if not agg.empty:
            return agg

    all_folds_path = run_dir / "all_folds.xlsx"
    if not all_folds_path.exists():
        return None

    folds = pd.read_excel(all_folds_path)
    if folds.empty:
        return None

    group_keys = [k for k in ["model", "mixer"] if k in folds.columns]
    numeric_cols = [
        c for c in folds.columns if pd.api.types.is_numeric_dtype(folds[c]) and c not in group_keys
    ]
    if not numeric_cols:
        return None

    if group_keys:
        agg = folds.groupby(group_keys, dropna=False, sort=True)[numeric_cols].agg(["mean", "std"])
        agg = agg.reset_index()
    else:
        row = {}
        for c in numeric_cols:
            row[f"{c}_mean"] = float(folds[c].mean())
            row[f"{c}_std"] = float(folds[c].std(ddof=0))
        agg = pd.DataFrame([row])

    return flatten_columns(agg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export competitor metrics into CSV/XLSX (aggregate and per-fold)."
    )
    parser.add_argument(
        "--jobarray_root",
        type=str,
        default="runs/jobarray",
        help="Root directory containing competitor run folders.",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default="std_cnn1d_direct_new,std_extratrees_direct_new,std_persistence_new,std_ridge_new,std_xgboost_direct_new",
        help="Comma-separated run folder names.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="competitors_full_metrics",
        help="Output prefix inside jobarray_root.",
    )
    args = parser.parse_args()

    root = Path(args.jobarray_root)
    run_names = [x.strip() for x in args.runs.split(",") if x.strip()]

    all_folds_rows: list[pd.DataFrame] = []
    aggregate_rows: list[pd.DataFrame] = []

    for run_name in run_names:
        run_dir = root / run_name
        if not run_dir.exists():
            continue

        all_folds_path = run_dir / "all_folds.xlsx"
        if all_folds_path.exists():
            folds = pd.read_excel(all_folds_path)
            if not folds.empty:
                folds = folds.copy()
                folds["run_tag"] = run_name
                all_folds_rows.append(folds)

        agg = load_or_build_aggregate(run_dir)
        if agg is not None and not agg.empty:
            agg = agg.copy()
            agg["run_tag"] = run_name
            aggregate_rows.append(agg)

    out_all = pd.concat(all_folds_rows, ignore_index=True) if all_folds_rows else pd.DataFrame()
    out_agg = pd.concat(aggregate_rows, ignore_index=True) if aggregate_rows else pd.DataFrame()

    all_csv = root / f"{args.output_prefix}_all_folds.csv"
    all_xlsx = root / f"{args.output_prefix}_all_folds.xlsx"
    agg_csv = root / f"{args.output_prefix}_aggregate.csv"
    agg_xlsx = root / f"{args.output_prefix}_aggregate.xlsx"

    out_all.to_csv(all_csv, index=False)
    out_all.to_excel(all_xlsx, index=False)
    out_agg.to_csv(agg_csv, index=False)
    out_agg.to_excel(agg_xlsx, index=False)

    print(f"Saved: {all_csv}")
    print(f"Saved: {all_xlsx}")
    print(f"Saved: {agg_csv}")
    print(f"Saved: {agg_xlsx}")
    print(f"Rows all_folds={len(out_all)}")
    print(f"Rows aggregate={len(out_agg)}")


if __name__ == "__main__":
    main()
