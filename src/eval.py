from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.data import FEATURES
from src.utils import ensure_dir


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: Sequence[int],
    features: Sequence[str] = FEATURES,
) -> dict[str, float]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch y_true={y_true.shape}, y_pred={y_pred.shape}")

    errors = y_pred - y_true
    abs_err = np.abs(errors)
    sq_err = errors**2
    mae_hf = abs_err.mean(axis=0)
    rmse_hf = np.sqrt(sq_err.mean(axis=0))

    out: dict[str, float] = {
        "macro_mae": float(mae_hf.mean()),
        "macro_rmse": float(rmse_hf.mean()),
    }

    for h_idx, horizon in enumerate(horizons):
        out[f"mae_h{horizon}"] = float(mae_hf[h_idx].mean())
        out[f"rmse_h{horizon}"] = float(rmse_hf[h_idx].mean())
        for f_idx, feat_name in enumerate(features):
            out[f"mae_h{horizon}_{feat_name}"] = float(mae_hf[h_idx, f_idx])
            out[f"rmse_h{horizon}_{feat_name}"] = float(rmse_hf[h_idx, f_idx])

    for f_idx, feat_name in enumerate(features):
        out[f"mae_{feat_name}"] = float(mae_hf[:, f_idx].mean())
        out[f"rmse_{feat_name}"] = float(rmse_hf[:, f_idx].mean())

    return out


def save_rows_table(rows: list[dict], path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    df = pd.DataFrame(rows)
    suffix = target.suffix.lower()
    if suffix == ".xlsx":
        df.to_excel(target, index=False)
        return
    if suffix == ".csv":
        df.to_csv(target, index=False)
        return
    raise ValueError(f"Unsupported table format '{suffix}'. Use .xlsx or .csv")


def save_rows_csv(rows: list[dict], path: str | Path) -> None:
    # Backward-compatible wrapper.
    save_rows_table(rows, path)


def aggregate_rows(
    rows: list[dict],
    group_keys: list[str] | None = None,
) -> list[dict]:
    if not rows:
        return []

    group_keys = group_keys or []
    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in numeric_cols if c not in group_keys]

    if not group_keys:
        groups = [((), df)]
    else:
        groups = list(df.groupby(group_keys, dropna=False, sort=True))

    summary: list[dict] = []
    for key, frame in groups:
        row: dict[str, float | str | int] = {}
        if group_keys:
            if len(group_keys) == 1:
                row[group_keys[0]] = key
            else:
                for idx, col in enumerate(group_keys):
                    row[col] = key[idx]
        for col in numeric_cols:
            row[f"{col}_mean"] = float(frame[col].mean())
            row[f"{col}_std"] = float(frame[col].std(ddof=0))
        row["count"] = int(frame.shape[0])
        summary.append(row)
    return summary
