from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class PatchTSTTinyModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int],
        output_dim: int = 4,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        max_patches: int = 64,
    ) -> None:
        super().__init__()
        if patch_len <= 0 or patch_stride <= 0:
            raise ValueError("patch_len and patch_stride must be > 0")
        if max_patches <= 0:
            raise ValueError("max_patches must be > 0")

        self.horizons = list(horizons)
        self.output_dim = int(output_dim)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.max_patches = int(max_patches)

        self.patch_embed = nn.Linear(self.patch_len * input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.heads = nn.ModuleList([nn.Linear(d_model, self.output_dim) for _ in self.horizons])

    def _make_patches(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.patch_len:
            raise ValueError(
                f"Input length {x.shape[1]} is shorter than patch_len={self.patch_len}"
            )
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        # patches: (B, n_patches, patch_len, C)
        patches = patches.contiguous().view(patches.shape[0], patches.shape[1], -1)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._make_patches(x)
        if patches.shape[1] > self.max_patches:
            raise ValueError(
                f"n_patches={patches.shape[1]} exceeds max_patches={self.max_patches}. "
                "Increase patch settings or max_patches."
            )

        tokens = self.patch_embed(patches)
        tokens = tokens + self.pos_embedding[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        outputs = [head(pooled) for head in self.heads]
        return torch.stack(outputs, dim=1)
