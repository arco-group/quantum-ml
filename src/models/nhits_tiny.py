from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class _NHitsBlock(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_dim: int,
        n_horizons: int,
        output_dim: int,
        pool_size: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0")
        if input_len <= 0:
            raise ValueError("input_len must be > 0")
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")

        self.input_len = int(input_len)
        self.input_dim = int(input_dim)
        self.n_horizons = int(n_horizons)
        self.output_dim = int(output_dim)
        self.pool_size = int(pool_size)

        pooled_len = (self.input_len + self.pool_size - 1) // self.pool_size
        in_features = pooled_len * self.input_dim
        out_features = self.input_len * self.input_dim + self.n_horizons * self.output_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, C)
        pooled = F.avg_pool1d(
            x.transpose(1, 2),
            kernel_size=self.pool_size,
            stride=self.pool_size,
            ceil_mode=True,
        )
        pooled = pooled.reshape(pooled.shape[0], -1)
        out = self.mlp(pooled)
        backcast_size = self.input_len * self.input_dim
        backcast = out[:, :backcast_size].view(-1, self.input_len, self.input_dim)
        forecast = out[:, backcast_size:].view(-1, self.n_horizons, self.output_dim)
        return backcast, forecast


class NHiTSTinyModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int],
        output_dim: int = 4,
        seq_len: int = 240,
        n_blocks: int = 2,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        pool_base: int = 2,
    ) -> None:
        super().__init__()
        if n_blocks <= 0:
            raise ValueError("n_blocks must be > 0")
        if pool_base <= 0:
            raise ValueError("pool_base must be > 0")
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        self.horizons = list(horizons)
        self.seq_len = int(seq_len)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        blocks = []
        for block_idx in range(int(n_blocks)):
            pool_size = pool_base ** block_idx
            pool_size = max(1, min(pool_size, self.seq_len))
            blocks.append(
                _NHitsBlock(
                    input_len=self.seq_len,
                    input_dim=self.input_dim,
                    n_horizons=len(self.horizons),
                    output_dim=self.output_dim,
                    pool_size=pool_size,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"Input length {x.shape[1]} does not match seq_len={self.seq_len} for NHiTSTinyModel"
            )
        residual = x
        forecast = None
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = block_forecast if forecast is None else forecast + block_forecast
        return forecast
