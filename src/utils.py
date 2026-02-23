from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(output_dir: str | Path, name: str = "train") -> logging.Logger:
    ensure_dir(output_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(Path(output_dir) / f"{name}.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_int_csv(values: str) -> List[int]:
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def parse_float_csv(values: str) -> List[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def save_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def numeric_columns(rows: Iterable[dict], exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    names: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in exclude:
                continue
            if isinstance(value, (int, float, np.floating, np.integer)):
                names.add(key)
    return sorted(names)

