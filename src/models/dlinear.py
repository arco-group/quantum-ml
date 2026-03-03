from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class DLinearTinyModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int],
        output_dim: int = 4,
        seq_len: int = 240,
        ma_window: int = 25,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if ma_window <= 0:
            raise ValueError("ma_window must be > 0")

        self.horizons = list(horizons)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.seq_len = int(seq_len)
        self.ma_window = int(ma_window)

        weight_shape = (len(self.horizons), self.output_dim, self.input_dim, self.seq_len)
        self.weight_seasonal = nn.Parameter(0.01 * torch.randn(*weight_shape))
        self.weight_trend = nn.Parameter(0.01 * torch.randn(*weight_shape))
        self.bias_seasonal = nn.Parameter(torch.zeros(len(self.horizons), self.output_dim))
        self.bias_trend = nn.Parameter(torch.zeros(len(self.horizons), self.output_dim))

    def _moving_avg(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        x_t = x.transpose(1, 2)
        trend = F.avg_pool1d(
            x_t,
            kernel_size=self.ma_window,
            stride=1,
            padding=self.ma_window // 2,
            count_include_pad=False,
        )
        if trend.shape[-1] > x_t.shape[-1]:
            trend = trend[..., : x_t.shape[-1]]
        elif trend.shape[-1] < x_t.shape[-1]:
            pad = x_t.shape[-1] - trend.shape[-1]
            trend = F.pad(trend, (0, pad))
        return trend.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"Input length {x.shape[1]} does not match seq_len={self.seq_len} for DLinearTinyModel"
            )
        trend = self._moving_avg(x)
        seasonal = x - trend

        seasonal_out = torch.einsum("blc,hocl->bho", seasonal, self.weight_seasonal)
        trend_out = torch.einsum("blc,hocl->bho", trend, self.weight_trend)
        seasonal_out = seasonal_out + self.bias_seasonal[None, :, :]
        trend_out = trend_out + self.bias_trend[None, :, :]
        return seasonal_out + trend_out
