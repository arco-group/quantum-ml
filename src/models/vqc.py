from __future__ import annotations

import warnings

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


def _resolve_diff_method(device_name: str) -> str:
    # Lightning backends do not support torch backprop through the device.
    # Use adjoint differentiation instead.
    if device_name.startswith("lightning."):
        return "adjoint"
    return "backprop"


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
        vectorized: bool = True,
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
        self.diff_method = _resolve_diff_method(self.device_name)
        self.vectorized_requested = bool(vectorized)
        self.vectorized_enabled = bool(vectorized)
        self.weights = nn.Parameter(0.01 * torch.randn(depth, n_qubits, 3))
        self.q_device = qml.device(self.device_name, wires=n_qubits)

        n_qubits_local = self.n_qubits
        depth_local = self.depth
        entangle_local = self.entangle

        @qml.qnode(self.q_device, interface="torch", diff_method=self.diff_method)
        def _circuit(angles: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            is_batched = angles.ndim == 2
            for wire in range(n_qubits_local):
                angle_param = angles[:, wire] if is_batched else angles[wire]
                qml.RY(angle_param, wires=wire)

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
        if self.vectorized_requested and not self.uses_gpu_backend:
            self._probe_vectorized_support()

    def _format_single_output(self, sample_out: torch.Tensor | list | tuple) -> torch.Tensor:
        if isinstance(sample_out, (tuple, list)):
            sample_out = torch.stack([torch.as_tensor(x) for x in sample_out], dim=0)
        else:
            sample_out = torch.as_tensor(sample_out)
        if sample_out.ndim == 0:
            sample_out = sample_out.unsqueeze(0)
        if sample_out.ndim > 1:
            sample_out = sample_out.reshape(-1)
        if sample_out.shape[0] != self.n_qubits:
            raise RuntimeError(
                f"Unexpected single-sample VQC output shape {tuple(sample_out.shape)}"
            )
        return sample_out

    def _format_batched_output(
        self, batch_out: torch.Tensor | list | tuple, batch_size: int
    ) -> torch.Tensor:
        if isinstance(batch_out, (tuple, list)):
            stacked = torch.stack([torch.as_tensor(x) for x in batch_out], dim=-1)
            if stacked.ndim == 1:
                stacked = stacked.unsqueeze(0)
            if stacked.shape[0] == self.n_qubits and stacked.shape[-1] == batch_size:
                stacked = stacked.transpose(0, 1)
            if stacked.shape != (batch_size, self.n_qubits):
                stacked = stacked.reshape(batch_size, self.n_qubits)
            return stacked

        batch_out = torch.as_tensor(batch_out)
        if batch_out.ndim == 1:
            if batch_size == 1 and batch_out.shape[0] == self.n_qubits:
                return batch_out.unsqueeze(0)
            raise RuntimeError(f"Unexpected batched VQC output shape {tuple(batch_out.shape)}")
        if batch_out.ndim == 2:
            if batch_out.shape == (batch_size, self.n_qubits):
                return batch_out
            if batch_out.shape == (self.n_qubits, batch_size):
                return batch_out.transpose(0, 1)
        if batch_out.numel() == batch_size * self.n_qubits:
            return batch_out.reshape(batch_size, self.n_qubits)
        raise RuntimeError(f"Unexpected batched VQC output shape {tuple(batch_out.shape)}")

    def _run_vectorized(self, angles: torch.Tensor) -> torch.Tensor:
        raw = self.circuit(angles, self.weights)
        return self._format_batched_output(raw, batch_size=angles.shape[0])

    def _run_loop(self, angles: torch.Tensor) -> torch.Tensor:
        batch_outputs = []
        for i in range(angles.shape[0]):
            sample_out = self.circuit(angles[i], self.weights)
            sample_out = self._format_single_output(sample_out)
            batch_outputs.append(sample_out)
        return torch.stack(batch_outputs, dim=0)

    def _probe_vectorized_support(self) -> None:
        probe = torch.zeros(2, self.n_qubits, dtype=torch.float32)
        try:
            with torch.no_grad():
                _ = self._run_vectorized(probe)
        except Exception as exc:
            self.vectorized_enabled = False
            warnings.warn(
                (
                    f"VQC vectorized mode disabled for backend={self.device_name}, "
                    f"diff_method={self.diff_method}: {exc}"
                ),
                RuntimeWarning,
            )

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if angles.ndim != 2 or angles.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected angles shape (batch, {self.n_qubits}), got {tuple(angles.shape)}"
            )

        if self.vectorized_enabled:
            try:
                out = self._run_vectorized(angles)
            except Exception as exc:
                self.vectorized_enabled = False
                warnings.warn(
                    (
                        f"VQC runtime fallback to per-sample mode for backend={self.device_name}: {exc}"
                    ),
                    RuntimeWarning,
                )
                out = self._run_loop(angles)
        else:
            out = self._run_loop(angles)
        return out.to(dtype=angles.dtype, device=angles.device)
