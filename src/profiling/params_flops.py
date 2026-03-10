"""
Parameter and FLOP profiling utilities.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Count all parameters."""
    return sum(p.numel() for p in model.parameters())


def estimate_flops_with_thop(model: nn.Module, input_size: tuple[int, ...]) -> dict[str, Any]:
    """
    Estimate FLOPs and params using thop if available.
    """
    try:
        from thop import profile
    except ImportError:
        return {
            "flops": None,
            "params": count_total_parameters(model),
            "backend": "fallback_no_thop",
        }

    device = next(model.parameters()).device
    dummy = torch.randn(*input_size, device=device)
    model.eval()
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    return {
        "flops": float(flops),
        "params": float(params),
        "backend": "thop",
    }