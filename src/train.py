from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import WindowDataset
from src.utils import ensure_dir


def make_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader | None:
    if x.shape[0] == 0:
        return None
    dataset = WindowDataset(x=x, y=y)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _epoch_step(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        preds = model(x_batch)
        loss = criterion(preds, y_batch)

        if train_mode:
            loss.backward()
            optimizer.step()

        batch_size = x_batch.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
    lr: float,
    device: torch.device,
    output_dir: str | Path,
    run_name: str,
    patience: int = 10,
    logger: Any | None = None,
) -> tuple[Path, list[dict[str, float]]]:
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []

    checkpoints_dir = ensure_dir(Path(output_dir) / "checkpoints")
    checkpoint_path = checkpoints_dir / f"{run_name}.pt"

    best_val_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = _epoch_step(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        if val_loader is not None:
            with torch.no_grad():
                val_loss = _epoch_step(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    optimizer=None,
                    device=device,
                )
        else:
            val_loss = train_loss

        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        if logger is not None:
            logger.info(
                "run=%s epoch=%d train_loss=%.6f val_loss=%.6f",
                run_name,
                epoch,
                train_loss,
                val_loss,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                if logger is not None:
                    logger.info(
                        "Early stopping at epoch=%d (best_epoch=%d, best_val=%.6f)",
                        epoch,
                        best_epoch,
                        best_val_loss,
                    )
                break

    if not checkpoint_path.exists():
        torch.save({"epoch": epochs, "model_state_dict": model.state_dict()}, checkpoint_path)
    return checkpoint_path, history


def predict_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            out = model(x_batch).cpu().numpy()
            preds.append(out)
            targets.append(y_batch.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)

