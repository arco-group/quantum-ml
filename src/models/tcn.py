from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class _TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            _Chomp1d(chomp_size=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            _Chomp1d(chomp_size=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TinyTCNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int],
        output_dim: int = 4,
        channels: int = 32,
        levels: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizons = list(horizons)
        blocks: list[nn.Module] = []
        in_channels = input_dim
        for level in range(levels):
            dilation = 2**level
            blocks.append(
                _TemporalBlock(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = channels
        self.tcn = nn.Sequential(*blocks)
        self.heads = nn.ModuleList([nn.Linear(channels, output_dim) for _ in self.horizons])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> (B, C, L)
        x_t = x.transpose(1, 2)
        features = self.tcn(x_t)
        z = features[:, :, -1]
        outputs = [head(z) for head in self.heads]
        return torch.stack(outputs, dim=1)

