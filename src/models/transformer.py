from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class TinyTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int],
        output_dim: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.horizons = list(horizons)
        self.max_len = max_len

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.heads = nn.ModuleList([nn.Linear(d_model, output_dim) for _ in self.horizons])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.max_len:
            raise ValueError(
                f"Input length {x.shape[1]} exceeds max_len={self.max_len} for TinyTransformerModel"
            )

        hidden = self.input_proj(x) + self.pos_embedding[:, : x.shape[1], :]
        encoded = self.encoder(hidden)
        z = encoded[:, -1, :]
        outputs = [head(z) for head in self.heads]
        return torch.stack(outputs, dim=1)

