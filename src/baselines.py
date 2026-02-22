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


class XGBoostDirectBaseline:
    def __init__(
        self,
        n_estimators: int = 16,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        seed: int = 0,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.models: dict[tuple[int, int], object] = {}
        self.n_horizons: int | None = None
        self.n_features: int | None = None
        self._xgb = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "XGBoostDirectBaseline":
        try:
            import xgboost as xgb
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "xgboost is required for model='xgboost_direct'. Install with `pip install xgboost`."
            ) from exc

        if x_train.shape[0] == 0:
            raise ValueError("Empty training set for xgboost_direct baseline")

        self._xgb = xgb
        x_flat = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
        self.n_horizons = int(y_train.shape[1])
        self.n_features = int(y_train.shape[2])
        self.models = {}

        params = {
            "objective": "reg:squarederror",
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "eval_metric": "rmse",
            "seed": self.seed,
            "nthread": self.n_jobs,
        }
        dtrain = xgb.DMatrix(x_flat)
        for horizon_idx in range(self.n_horizons):
            for feature_idx in range(self.n_features):
                dtrain.set_label(y_train[:, horizon_idx, feature_idx].astype(np.float32))
                model = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=self.n_estimators,
                    verbose_eval=False,
                )
                self.models[(horizon_idx, feature_idx)] = model
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.n_horizons is None or self.n_features is None or not self.models:
            raise RuntimeError("XGBoostDirectBaseline must be fit before predict")
        if self._xgb is None:
            raise RuntimeError("xgboost module not initialized")

        x_flat = x.reshape(x.shape[0], -1).astype(np.float32)
        dtest = self._xgb.DMatrix(x_flat)
        preds = np.zeros((x.shape[0], self.n_horizons, self.n_features), dtype=np.float32)
        for horizon_idx in range(self.n_horizons):
            for feature_idx in range(self.n_features):
                preds[:, horizon_idx, feature_idx] = self.models[(horizon_idx, feature_idx)].predict(
                    dtest
                )
        return preds

    def effective_param_count(self) -> int:
        if self.n_horizons is None or self.n_features is None:
            return 0
        max_leaves = 2 ** max(self.max_depth, 1)
        return int(self.n_horizons * self.n_features * self.n_estimators * max_leaves)
