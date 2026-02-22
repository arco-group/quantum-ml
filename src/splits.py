from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_NUMERICS_RE = re.compile(r"bidmc_(\d+)_Numerics\.csv$")


def discover_subjects(data_dir: str | Path) -> list[str]:
    paths = sorted(Path(data_dir).glob("bidmc_*_Numerics.csv"))
    subjects: list[str] = []
    for path in paths:
        match = _NUMERICS_RE.match(path.name)
        if match:
            subjects.append(match.group(1))
    if not subjects:
        raise FileNotFoundError(f"No BIDMC Numerics files found in {data_dir}")
    return sorted(subjects, key=int)


def lopo_folds(subjects: list[str]) -> list[tuple[str, list[str]]]:
    folds: list[tuple[str, list[str]]] = []
    ordered = sorted(subjects, key=int)
    for test_subject in ordered:
        train_pool = [s for s in ordered if s != test_subject]
        folds.append((test_subject, train_pool))
    return folds


def split_train_val_subjects(
    train_pool: list[str], val_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    if not train_pool:
        raise ValueError("train_pool cannot be empty")

    ordered = sorted(train_pool, key=int)
    if len(ordered) == 1:
        return ordered, []

    n_val = max(1, int(round(len(ordered) * val_ratio)))
    n_val = min(n_val, len(ordered) - 1)

    rng = np.random.default_rng(seed)
    permutation = list(rng.permutation(ordered))
    val_subjects = sorted(permutation[:n_val], key=int)
    train_subjects = sorted(permutation[n_val:], key=int)
    return train_subjects, val_subjects


def sample_subject_fraction(subjects: list[str], fraction: float, seed: int) -> list[str]:
    if not subjects:
        return []
    if fraction >= 1.0:
        return sorted(subjects, key=int)
    if fraction <= 0.0:
        raise ValueError("fraction must be > 0")

    ordered = sorted(subjects, key=int)
    n_keep = max(1, int(round(len(ordered) * fraction)))
    n_keep = min(n_keep, len(ordered))
    rng = np.random.default_rng(seed)
    chosen = list(rng.choice(ordered, size=n_keep, replace=False))
    return sorted(chosen, key=int)

