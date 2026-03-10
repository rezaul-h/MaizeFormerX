"""
Regularization helpers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def l1_regularization(model: nn.Module) -> torch.Tensor:
    """Compute L1 penalty over all trainable parameters."""
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            penalty = penalty + param.abs().sum()
    return penalty


def l2_regularization(model: nn.Module) -> torch.Tensor:
    """Compute L2 penalty over all trainable parameters."""
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            penalty = penalty + param.pow(2).sum()
    return penalty


def apply_regularization(
    loss: torch.Tensor,
    model: nn.Module,
    l1_lambda: float = 0.0,
    l2_lambda: float = 0.0,
) -> torch.Tensor:
    """Add explicit L1/L2 penalties to the loss."""
    if l1_lambda > 0.0:
        loss = loss + l1_lambda * l1_regularization(model)
    if l2_lambda > 0.0:
        loss = loss + l2_lambda * l2_regularization(model)
    return loss