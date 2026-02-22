from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class LSTMTinyModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int],
        output_dim: int = 4,
        hidden_dim: int = 56,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.horizons = list(horizons)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in self.horizons])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        z = h_n[-1]
        outputs = [head(z) for head in self.heads]
        return torch.stack(outputs, dim=1)

