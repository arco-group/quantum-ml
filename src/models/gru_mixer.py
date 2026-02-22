from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.models.vqc import VQCLayer
from src.models.vqc_torchquantum import TorchQuantumVQCLayer


class GRUMixerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int],
        output_dim: int = 4,
        hidden_dim: int = 64,
        mixer: str = "none",
        mlp_dim: int | None = None,
        n_qubits: int = 6,
        vqc_depth: int = 3,
        vqc_entangle: bool = True,
        vqc_backend: str = "auto",
        vqc_prefer_gpu: bool = False,
        vqc_vectorized: bool = True,
        vqc_library: str = "pennylane",
    ) -> None:
        super().__init__()
        self.horizons = list(horizons)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.mixer = mixer
        self.vqc_library = vqc_library

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        head_in_dim = hidden_dim
        if mixer == "none":
            self.mixer_block = None
        elif mixer == "mlp":
            mlp_dim = mlp_dim or n_qubits
            bottleneck_dim = max(8, hidden_dim // 4)
            self.mixer_block = nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, mlp_dim),
                nn.ReLU(),
            )
            head_in_dim = hidden_dim + mlp_dim
        elif mixer == "vqc":
            self.to_qubits = nn.Linear(hidden_dim, n_qubits)
            if vqc_library == "pennylane":
                self.mixer_block = VQCLayer(
                    n_qubits=n_qubits,
                    depth=vqc_depth,
                    entangle=vqc_entangle,
                    device_name=vqc_backend,
                    prefer_gpu=vqc_prefer_gpu,
                    vectorized=vqc_vectorized,
                )
            elif vqc_library == "torchquantum":
                self.mixer_block = TorchQuantumVQCLayer(
                    n_qubits=n_qubits,
                    depth=vqc_depth,
                    entangle=vqc_entangle,
                )
            else:
                raise ValueError(f"Unsupported vqc_library: {vqc_library}")
            head_in_dim = hidden_dim + n_qubits
        else:
            raise ValueError(f"Unsupported mixer: {mixer}")

        self.heads = nn.ModuleList(
            [nn.Linear(head_in_dim, output_dim) for _ in self.horizons]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.encoder(x)
        z = h_n[-1]

        if self.mixer == "none":
            features = z
        elif self.mixer == "mlp":
            mixed = self.mixer_block(z)
            features = torch.cat([z, mixed], dim=-1)
        elif self.mixer == "vqc":
            angles = self.to_qubits(z)
            quantum_features = self.mixer_block(angles)
            features = torch.cat([z, quantum_features], dim=-1)
        else:  # pragma: no cover
            raise RuntimeError(f"Unexpected mixer value: {self.mixer}")

        outputs = [head(features) for head in self.heads]
        return torch.stack(outputs, dim=1)
