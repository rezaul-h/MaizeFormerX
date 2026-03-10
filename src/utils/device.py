"""
Device and runtime helpers for CPU/GPU selection.
"""

from __future__ import annotations

import torch


def get_device(device_preference: str | None = None) -> torch.device:
    """
    Resolve the runtime device.

    Parameters
    ----------
    device_preference:
        One of {"cuda", "cpu", "auto", None}.
    """
    if device_preference in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if device_preference == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device_preference: {device_preference}")


def move_to_device(batch, device: torch.device):
    """
    Recursively move tensors inside a nested structure to a device.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    return batch


def get_autocast_dtype(use_fp16: bool = True) -> torch.dtype:
    """Return the autocast dtype for CUDA mixed precision."""
    return torch.float16 if use_fp16 else torch.bfloat16


def get_num_available_gpus() -> int:
    """Return the number of available CUDA devices."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0