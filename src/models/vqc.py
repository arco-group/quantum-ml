from __future__ import annotations

import torch
from torch import nn

try:
    import pennylane as qml
    _HAS_PENNYLANE = True
    _PENNYLANE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only when PennyLane missing
    qml = None
    _HAS_PENNYLANE = False
    _PENNYLANE_IMPORT_ERROR = exc

_GPU_BACKENDS = {"lightning.gpu"}


def pennylane_available() -> bool:
    return _HAS_PENNYLANE


def _can_build_device(device_name: str, n_qubits: int) -> bool:
    try:
        qml.device(device_name, wires=n_qubits)
        return True
    except Exception:
        return False


def resolve_vqc_backend(
    requested: str,
    prefer_gpu: bool,
    n_qubits: int,
) -> tuple[str, bool]:
    if not _HAS_PENNYLANE:
        raise ImportError(
            "PennyLane is required for mixer='vqc'. Install with `pip install pennylane`."
        ) from _PENNYLANE_IMPORT_ERROR

    requested = requested.strip()
    if requested != "auto":
        if not _can_build_device(requested, n_qubits=n_qubits):
            raise RuntimeError(
                f"Requested VQC backend '{requested}' is not available in this environment."
            )
        return requested, requested in _GPU_BACKENDS

    if prefer_gpu:
        candidates = ["lightning.gpu", "lightning.qubit", "default.qubit.torch", "default.qubit"]
    else:
        candidates = ["lightning.qubit", "default.qubit.torch", "default.qubit"]

    for device_name in candidates:
        if _can_build_device(device_name, n_qubits=n_qubits):
            return device_name, device_name in _GPU_BACKENDS

    raise RuntimeError(
        "No compatible PennyLane simulator backend found. "
        "Install PennyLane plugins or pass --vqc_backend explicitly."
    )


class VQCLayer(nn.Module):
    def __init__(
        self,
        n_qubits: int = 6,
        depth: int = 3,
        entangle: bool = True,
        device_name: str = "auto",
        prefer_gpu: bool = False,
    ) -> None:
        super().__init__()
        if not _HAS_PENNYLANE:
            raise ImportError(
                "PennyLane is required for mixer='vqc'. Install with `pip install pennylane`."
            ) from _PENNYLANE_IMPORT_ERROR

        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")

        self.n_qubits = n_qubits
        self.depth = depth
        self.entangle = bool(entangle)
        self.device_name, self.uses_gpu_backend = resolve_vqc_backend(
            requested=device_name,
            prefer_gpu=prefer_gpu,
            n_qubits=n_qubits,
        )
        self.weights = nn.Parameter(0.01 * torch.randn(depth, n_qubits, 3))
        self.q_device = qml.device(self.device_name, wires=n_qubits)

        n_qubits_local = self.n_qubits
        depth_local = self.depth
        entangle_local = self.entangle

        @qml.qnode(self.q_device, interface="torch", diff_method="backprop")
        def _circuit(angles: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            for wire in range(n_qubits_local):
                qml.RY(angles[wire], wires=wire)

            for layer in range(depth_local):
                for wire in range(n_qubits_local):
                    qml.Rot(
                        weights[layer, wire, 0],
                        weights[layer, wire, 1],
                        weights[layer, wire, 2],
                        wires=wire,
                    )
                if entangle_local and n_qubits_local > 1:
                    for wire in range(n_qubits_local):
                        qml.CNOT(wires=[wire, (wire + 1) % n_qubits_local])

            return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits_local)]

        self.circuit = _circuit

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if angles.ndim != 2 or angles.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected angles shape (batch, {self.n_qubits}), got {tuple(angles.shape)}"
            )

        batch_outputs = []
        for i in range(angles.shape[0]):
            sample_out = self.circuit(angles[i], self.weights)
            if isinstance(sample_out, (tuple, list)):
                sample_out = torch.stack(list(sample_out))
            batch_outputs.append(sample_out)
        return torch.stack(batch_outputs, dim=0).to(dtype=angles.dtype, device=angles.device)
