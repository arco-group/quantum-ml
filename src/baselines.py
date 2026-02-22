from __future__ import annotations

import numpy as np


class PersistenceBaseline:
    def __init__(self, n_features: int = 4, target_mode: str = "value") -> None:
        self.n_features = n_features
        self.target_mode = target_mode
        self.n_horizons: int | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "PersistenceBaseline":
        del x_train
        if self.target_mode not in {"value", "delta"}:
            raise ValueError(f"Unsupported target_mode for PersistenceBaseline: {self.target_mode}")
        self.n_horizons = y_train.shape[1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.n_horizons is None:
            raise RuntimeError("PersistenceBaseline must be fit before predict")
        if self.target_mode == "delta":
            return np.zeros((x.shape[0], self.n_horizons, self.n_features), dtype=np.float32)
        last_step = x[:, -1, : self.n_features]
        return np.repeat(last_step[:, None, :], repeats=self.n_horizons, axis=1)


class RidgePerHorizonBaseline:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.weights: list[np.ndarray] = []
        self.n_horizons: int | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "RidgePerHorizonBaseline":
        if x_train.shape[0] == 0:
            raise ValueError("Empty training set for ridge baseline")

        x_flat = x_train.reshape(x_train.shape[0], -1).astype(np.float64)
        x_aug = np.concatenate([x_flat, np.ones((x_flat.shape[0], 1))], axis=1)
        gram = x_aug.T @ x_aug
        reg = self.alpha * np.eye(gram.shape[0], dtype=np.float64)
        reg[-1, -1] = 0.0

        self.n_horizons = y_train.shape[1]
        self.weights = []
        for horizon_idx in range(self.n_horizons):
            y_h = y_train[:, horizon_idx, :].astype(np.float64)
            rhs = x_aug.T @ y_h
            try:
                w = np.linalg.solve(gram + reg, rhs)
            except np.linalg.LinAlgError:
                w = np.linalg.pinv(gram + reg) @ rhs
            self.weights.append(w.astype(np.float32))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.n_horizons is None or not self.weights:
            raise RuntimeError("RidgePerHorizonBaseline must be fit before predict")
        x_flat = x.reshape(x.shape[0], -1).astype(np.float32)
        x_aug = np.concatenate([x_flat, np.ones((x_flat.shape[0], 1), dtype=np.float32)], axis=1)
        outputs = [x_aug @ w for w in self.weights]
        return np.stack(outputs, axis=1).astype(np.float32)
