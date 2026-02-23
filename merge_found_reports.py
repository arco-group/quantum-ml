#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _discover_files(root_dir: Path, patterns: list[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        for path in root_dir.rglob(pattern):
            if path.is_file():
                found.add(path.resolve())
    return sorted(found)


def _load_and_stack(files: list[Path], root_dir: Path) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_excel(file_path)
        df["source_file"] = str(file_path)
        df["source_dir"] = str(file_path.parent)
        df["source_rel"] = str(file_path.relative_to(root_dir))
        chunks.append(df)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, axis=0, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search recursively for aggregate_*_mean_std.xlsx and all_folds*.xlsx, "
            "then produce two merged Excel files."
        )
    )
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument(
        "--agg_output_name",
        type=str,
        default="merged_aggregate_mean_std.xlsx",
    )
    parser.add_argument(
        "--folds_output_name",
        type=str,
        default="merged_all_folds.xlsx",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_files = _discover_files(
        root_dir=root_dir,
        patterns=["aggregate_*_mean_std.xlsx", "aggregate_mean_std.xlsx"],
    )
    all_folds_files = _discover_files(
        root_dir=root_dir,
        patterns=["all_folds*.xlsx"],
    )

    merged_aggregate = _load_and_stack(aggregate_files, root_dir=root_dir)
    merged_all_folds = _load_and_stack(all_folds_files, root_dir=root_dir)

    agg_out_path = output_dir / args.agg_output_name
    folds_out_path = output_dir / args.folds_output_name
    merged_aggregate.to_excel(agg_out_path, index=False)
    merged_all_folds.to_excel(folds_out_path, index=False)

    print(f"aggregate files found: {len(aggregate_files)}")
    print(f"all_folds files found: {len(all_folds_files)}")
    print(f"saved: {agg_out_path}")
    print(f"saved: {folds_out_path}")


if __name__ == "__main__":
    main()

