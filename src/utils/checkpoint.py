"""
Checkpoint save/load helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.utils.io import ensure_dir


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    epoch: int | None = None,
    step: int | None = None,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint."""
    path = Path(path)
    ensure_dir(path)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "config": config or {},
        "extra": extra or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a training checkpoint and restore states where provided."""
    checkpoint = torch.load(Path(path), map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def load_model_weights(
    path: str | Path,
    model,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load model weights from either a full checkpoint or a raw state_dict file.
    """
    payload = torch.load(Path(path), map_location=map_location)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload

    model.load_state_dict(state_dict, strict=strict)
    return payload