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


class ExtraTreesDirectBaseline:
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        seed: int = 0,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = None if max_depth is None or int(max_depth) <= 0 else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.models: dict[tuple[int, int], object] = {}
        self.n_horizons: int | None = None
        self.n_features: int | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "ExtraTreesDirectBaseline":
        try:
            from sklearn.ensemble import ExtraTreesRegressor
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "scikit-learn is required for model='extratrees_direct'. "
                "Install with `pip install scikit-learn`."
            ) from exc

        if x_train.shape[0] == 0:
            raise ValueError("Empty training set for extratrees_direct baseline")

        x_flat = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
        self.n_horizons = int(y_train.shape[1])
        self.n_features = int(y_train.shape[2])
        self.models = {}

        for horizon_idx in range(self.n_horizons):
            for feature_idx in range(self.n_features):
                model = ExtraTreesRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.seed + horizon_idx * 100 + feature_idx,
                    n_jobs=self.n_jobs,
                )
                model.fit(x_flat, y_train[:, horizon_idx, feature_idx].astype(np.float32))
                self.models[(horizon_idx, feature_idx)] = model
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.n_horizons is None or self.n_features is None or not self.models:
            raise RuntimeError("ExtraTreesDirectBaseline must be fit before predict")
        x_flat = x.reshape(x.shape[0], -1).astype(np.float32)
        preds = np.zeros((x.shape[0], self.n_horizons, self.n_features), dtype=np.float32)
        for horizon_idx in range(self.n_horizons):
            for feature_idx in range(self.n_features):
                preds[:, horizon_idx, feature_idx] = self.models[(horizon_idx, feature_idx)].predict(
                    x_flat
                )
        return preds

    def effective_param_count(self) -> int:
        if self.n_horizons is None or self.n_features is None:
            return 0
        max_nodes = (2 ** (self.max_depth + 1) - 1) if self.max_depth is not None else 1023
        return int(self.n_horizons * self.n_features * self.n_estimators * max_nodes)


class RandomForestDirectBaseline:
    def __init__(
        self,
        n_estimators: int = 4,
        max_depth: int | None = 3,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        seed: int = 0,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = None if max_depth is None or int(max_depth) <= 0 else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.models: dict[tuple[int, int], object] = {}
        self.n_horizons: int | None = None
        self.n_features: int | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "RandomForestDirectBaseline":
        try:
            from sklearn.ensemble import RandomForestRegressor
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "scikit-learn is required for model='random_forest_direct'. "
                "Install with `pip install scikit-learn`."
            ) from exc

        if x_train.shape[0] == 0:
            raise ValueError("Empty training set for random_forest_direct baseline")

        x_flat = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
        self.n_horizons = int(y_train.shape[1])
        self.n_features = int(y_train.shape[2])
        self.models = {}

        for horizon_idx in range(self.n_horizons):
            for feature_idx in range(self.n_features):
                model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.seed + horizon_idx * 100 + feature_idx,
                    n_jobs=self.n_jobs,
                )
                model.fit(x_flat, y_train[:, horizon_idx, feature_idx].astype(np.float32))
                self.models[(horizon_idx, feature_idx)] = model
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.n_horizons is None or self.n_features is None or not self.models:
            raise RuntimeError("RandomForestDirectBaseline must be fit before predict")
        x_flat = x.reshape(x.shape[0], -1).astype(np.float32)
        preds = np.zeros((x.shape[0], self.n_horizons, self.n_features), dtype=np.float32)
        for horizon_idx in range(self.n_horizons):
            for feature_idx in range(self.n_features):
                preds[:, horizon_idx, feature_idx] = self.models[(horizon_idx, feature_idx)].predict(
                    x_flat
                )
        return preds

    def effective_param_count(self) -> int:
        if self.n_horizons is None or self.n_features is None:
            return 0
        max_nodes = (2 ** (self.max_depth + 1) - 1) if self.max_depth is not None else 1023
        return int(self.n_horizons * self.n_features * self.n_estimators * max_nodes)


class _TinyCNN1DRegressor:
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        out_horizons: int,
        out_features: int,
        channels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        import torch
        from torch import nn

        if kernel_size <= 0:
            raise ValueError("kernel_size must be > 0")
        if channels <= 0:
            raise ValueError("channels must be > 0")
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")

        pad = kernel_size // 2
        self.model = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(p=max(0.0, float(dropout))),
            nn.Flatten(),
            nn.Linear(channels * seq_len, out_horizons * out_features),
        )
        self._torch = torch
        self._out_horizons = out_horizons
        self._out_features = out_features

    def forward(self, x: np.ndarray, device: str) -> np.ndarray:
        xt = self._torch.from_numpy(x.astype(np.float32)).to(device)
        xt = xt.transpose(1, 2)
        out = self.model(xt).view(-1, self._out_horizons, self._out_features)
        return out.detach().cpu().numpy()


class CNN1DDirectBaseline:
    def __init__(
        self,
        channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.seed = int(seed)
        self.device = device
        self._net: _TinyCNN1DRegressor | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "CNN1DDirectBaseline":
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise ImportError("torch is required for model='cnn1d_direct'.") from exc

        if x_train.shape[0] == 0:
            raise ValueError("Empty training set for cnn1d_direct baseline")

        torch.manual_seed(self.seed)
        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"

        n_samples, seq_len, input_dim = x_train.shape
        n_horizons = int(y_train.shape[1])
        n_features = int(y_train.shape[2])

        self._net = _TinyCNN1DRegressor(
            input_dim=input_dim,
            seq_len=seq_len,
            out_horizons=n_horizons,
            out_features=n_features,
            channels=self.channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )
        model = self._net.model.to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        x_t = torch.from_numpy(x_train.astype(np.float32))
        y_t = torch.from_numpy(y_train.astype(np.float32))

        for _ in range(max(self.epochs, 1)):
            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, max(self.batch_size, 1)):
                idx = perm[start : start + self.batch_size]
                xb = x_t[idx].to(device).transpose(1, 2)
                yb = y_t[idx].to(device)
                optimizer.zero_grad(set_to_none=True)
                preds = model(xb).view(-1, n_horizons, n_features)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("CNN1DDirectBaseline must be fit before predict")
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise ImportError("torch is required for model='cnn1d_direct'.") from exc

        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        self._net.model.eval()
        self._net.model.to(device)
        with torch.no_grad():
            return self._net.forward(x=x, device=device).astype(np.float32)

    def effective_param_count(self) -> int:
        if self._net is None:
            return 0
        return int(sum(p.numel() for p in self._net.model.parameters() if p.requires_grad))
