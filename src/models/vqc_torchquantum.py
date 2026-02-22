from __future__ import annotations

import torch
from torch import nn

try:
    import torchquantum as tq
    import torchquantum.functional as tqf

    _HAS_TORCHQUANTUM = True
    _TORCHQUANTUM_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only when torchquantum missing
    tq = None
    tqf = None
    _HAS_TORCHQUANTUM = False
    _TORCHQUANTUM_IMPORT_ERROR = exc


def torchquantum_available() -> bool:
    return _HAS_TORCHQUANTUM


class TorchQuantumVQCLayer(nn.Module):
    def __init__(
        self,
        n_qubits: int = 6,
        depth: int = 3,
        entangle: bool = True,
    ) -> None:
        super().__init__()
        if not _HAS_TORCHQUANTUM:
            raise ImportError(
                "torchquantum is required for vqc_library='torchquantum'. "
                "Install with `pip install torchquantum`."
            ) from _TORCHQUANTUM_IMPORT_ERROR

        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")

        self.n_qubits = n_qubits
        self.depth = depth
        self.entangle = bool(entangle)
        self.weights = nn.Parameter(0.01 * torch.randn(depth, n_qubits, 3))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if angles.ndim != 2 or angles.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected angles shape (batch, {self.n_qubits}), got {tuple(angles.shape)}"
            )

        batch_size = angles.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_qubits,
            bsz=batch_size,
            device=angles.device,
        )

        for wire in range(self.n_qubits):
            tqf.ry(qdev, wires=wire, params=angles[:, wire])

        for layer in range(self.depth):
            for wire in range(self.n_qubits):
                tqf.rx(qdev, wires=wire, params=self.weights[layer, wire, 0])
                tqf.ry(qdev, wires=wire, params=self.weights[layer, wire, 1])
                tqf.rz(qdev, wires=wire, params=self.weights[layer, wire, 2])
            if self.entangle and self.n_qubits > 1:
                for wire in range(self.n_qubits):
                    tqf.cnot(qdev, wires=[wire, (wire + 1) % self.n_qubits])

        out = self.measure(qdev)
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        return out[:, : self.n_qubits].to(dtype=angles.dtype, device=angles.device)

