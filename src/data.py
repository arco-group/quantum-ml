from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.splits import discover_subjects

FEATURES = ("HR", "PULSE", "RESP", "SpO2")


@dataclass(frozen=True)
class PerturbConfig:
    perturb_type: str
    level: float


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def load_subject_numerics(
    csv_path: str | Path,
    features: Sequence[str] = FEATURES,
    nan_strategy: str = "ffill_bfill",
) -> np.ndarray:
    frame = pd.read_csv(csv_path)
    frame.columns = [col.strip() for col in frame.columns]
    missing_cols = [col for col in features if col not in frame.columns]
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols} in {csv_path}")

    values = frame[list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    if nan_strategy == "ffill_bfill":
        values = pd.DataFrame(values).ffill().bfill().to_numpy(np.float32)
    elif nan_strategy == "drop":
        pass
    else:
        raise ValueError(f"Unsupported nan_strategy: {nan_strategy}")
    return values


def load_subject_series(
    data_dir: str | Path,
    features: Sequence[str] = FEATURES,
    nan_strategy: str = "ffill_bfill",
) -> dict[str, np.ndarray]:
    subject_series: dict[str, np.ndarray] = {}
    for subject in discover_subjects(data_dir):
        csv_path = Path(data_dir) / f"bidmc_{subject}_Numerics.csv"
        subject_series[subject] = load_subject_numerics(
            csv_path=csv_path,
            features=features,
            nan_strategy=nan_strategy,
        )
    return subject_series


def compute_feature_stats(
    subject_series: Mapping[str, np.ndarray], train_subjects: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    if not train_subjects:
        raise ValueError("At least one train subject is required to compute stats")
    stacked = np.concatenate([subject_series[s] for s in train_subjects], axis=0)
    mean = np.nanmean(stacked, axis=0).astype(np.float32)
    std = np.nanstd(stacked, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where((np.isfinite(std)) & (std > 1e-8), std, 1.0).astype(np.float32)
    return mean, std


def standardize_subject_series(
    subject_series: Mapping[str, np.ndarray], mean: np.ndarray, std: np.ndarray
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for subject, series in subject_series.items():
        out[subject] = ((series - mean) / std).astype(np.float32)
    return out


def _impute_missing_window(window: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "zero":
        return np.nan_to_num(window, nan=0.0).astype(np.float32)
    if strategy == "ffill_bfill":
        return pd.DataFrame(window).ffill().bfill().fillna(0.0).to_numpy(np.float32)
    raise ValueError(f"Unsupported missing_impute strategy: {strategy}")


def _apply_perturbation(
    x: np.ndarray,
    perturb: PerturbConfig | None,
    rng: np.random.Generator,
    mask_channel: bool,
    missing_impute: str,
) -> np.ndarray:
    n_steps, n_features = x.shape
    if perturb is None:
        if not mask_channel:
            return x.astype(np.float32)
        mask = np.zeros((n_steps, n_features), dtype=np.float32)
        return np.concatenate([x, mask], axis=1).astype(np.float32)

    if perturb.perturb_type == "missingness":
        prob = float(np.clip(perturb.level, 0.0, 1.0))
        mask = (rng.random(x.shape) < prob).astype(np.float32)
        corrupted = x.copy()
        corrupted[mask > 0] = np.nan
        imputed = _impute_missing_window(corrupted, strategy=missing_impute)
        if mask_channel:
            return np.concatenate([imputed, mask], axis=1).astype(np.float32)
        return imputed.astype(np.float32)

    if perturb.perturb_type == "noise":
        sigma = max(0.0, float(perturb.level))
        if sigma > 0:
            noise = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
            x = x + noise
        if not mask_channel:
            return x.astype(np.float32)
        mask = np.zeros((n_steps, n_features), dtype=np.float32)
        return np.concatenate([x, mask], axis=1).astype(np.float32)

    raise ValueError(f"Unsupported perturb_type: {perturb.perturb_type}")


def build_windows_for_subject(
    series: np.ndarray,
    l_window: int,
    horizons: Sequence[int],
    window_stride: int = 1,
    perturb: PerturbConfig | None = None,
    mask_channel: bool = False,
    missing_impute: str = "ffill_bfill",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    horizons = sorted(horizons)
    n_features = series.shape[1]
    input_features = n_features * 2 if mask_channel else n_features
    max_horizon = max(horizons)
    max_start = series.shape[0] - l_window - max_horizon + 1
    if window_stride <= 0:
        raise ValueError(f"window_stride must be > 0, got {window_stride}")

    if max_start <= 0:
        return (
            np.empty((0, l_window, input_features), dtype=np.float32),
            np.empty((0, len(horizons), n_features), dtype=np.float32),
        )

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for start in range(0, max_start, window_stride):
        x = series[start : start + l_window].astype(np.float32, copy=True)
        anchor = start + l_window - 1
        y = np.stack([series[anchor + h] for h in horizons], axis=0).astype(np.float32)
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        x = _apply_perturbation(
            x=x,
            perturb=perturb,
            rng=rng,
            mask_channel=mask_channel,
            missing_impute=missing_impute,
        )
        x_list.append(x)
        y_list.append(y)

    if not x_list:
        return (
            np.empty((0, l_window, input_features), dtype=np.float32),
            np.empty((0, len(horizons), n_features), dtype=np.float32),
        )
    return np.stack(x_list, axis=0), np.stack(y_list, axis=0)


def build_windows_for_subjects(
    subject_series: Mapping[str, np.ndarray],
    subjects: Sequence[str],
    l_window: int,
    horizons: Sequence[int],
    window_stride: int = 1,
    perturb: PerturbConfig | None = None,
    mask_channel: bool = False,
    missing_impute: str = "ffill_bfill",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    n_features = next(iter(subject_series.values())).shape[1]
    input_features = n_features * 2 if mask_channel else n_features
    if not subjects:
        return (
            np.empty((0, l_window, input_features), dtype=np.float32),
            np.empty((0, len(horizons), n_features), dtype=np.float32),
        )

    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    for idx, subject in enumerate(sorted(subjects, key=int)):
        local_seed = seed + int(subject) + (idx * 104729)
        x_sub, y_sub = build_windows_for_subject(
            series=subject_series[subject],
            l_window=l_window,
            horizons=horizons,
            window_stride=window_stride,
            perturb=perturb,
            mask_channel=mask_channel,
            missing_impute=missing_impute,
            seed=local_seed,
        )
        if x_sub.shape[0] == 0:
            continue
        x_blocks.append(x_sub)
        y_blocks.append(y_sub)

    if not x_blocks:
        return (
            np.empty((0, l_window, input_features), dtype=np.float32),
            np.empty((0, len(horizons), n_features), dtype=np.float32),
        )
    return np.concatenate(x_blocks, axis=0), np.concatenate(y_blocks, axis=0)


def prepare_split_data(
    subject_series: Mapping[str, np.ndarray],
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    test_subject: str,
    l_window: int,
    horizons: Sequence[int],
    window_stride: int = 1,
    train_perturb: PerturbConfig | None = None,
    val_perturb: PerturbConfig | None = None,
    test_perturb: PerturbConfig | None = None,
    mask_channel: bool = False,
    missing_impute: str = "ffill_bfill",
    seed: int = 0,
) -> dict[str, np.ndarray]:
    mean, std = compute_feature_stats(subject_series=subject_series, train_subjects=train_subjects)
    standardized = standardize_subject_series(subject_series, mean=mean, std=std)

    x_train, y_train = build_windows_for_subjects(
        subject_series=standardized,
        subjects=train_subjects,
        l_window=l_window,
        horizons=horizons,
        window_stride=window_stride,
        perturb=train_perturb,
        mask_channel=mask_channel,
        missing_impute=missing_impute,
        seed=seed + 11,
    )
    x_val, y_val = build_windows_for_subjects(
        subject_series=standardized,
        subjects=val_subjects,
        l_window=l_window,
        horizons=horizons,
        window_stride=window_stride,
        perturb=val_perturb,
        mask_channel=mask_channel,
        missing_impute=missing_impute,
        seed=seed + 23,
    )
    x_test, y_test = build_windows_for_subjects(
        subject_series=standardized,
        subjects=[test_subject],
        l_window=l_window,
        horizons=horizons,
        window_stride=window_stride,
        perturb=test_perturb,
        mask_channel=mask_channel,
        missing_impute=missing_impute,
        seed=seed + 37,
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "mean": mean,
        "std": std,
    }
