"""
Classification loss builders.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def build_classification_loss(
    criterion_cfg: dict[str, Any],
    class_weights: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Build a classification loss function from config.

    Supported
    ---------
    - cross_entropy
    """
    name = criterion_cfg.get("name", "cross_entropy").lower()

    if class_weights is not None and device is not None:
        class_weights = class_weights.to(device)

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(criterion_cfg.get("label_smoothing", 0.0)),
        )

    raise ValueError(f"Unsupported classification loss: {name}")