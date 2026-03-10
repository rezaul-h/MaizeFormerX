"""
Optimizer builders.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, optimizer_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    name = optimizer_cfg.get("name", "adam").lower()
    lr = float(optimizer_cfg.get("lr", 5e-5))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
            eps=float(optimizer_cfg.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
            eps=float(optimizer_cfg.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(optimizer_cfg.get("momentum", 0.9)),
            nesterov=bool(optimizer_cfg.get("nesterov", False)),
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {name}")