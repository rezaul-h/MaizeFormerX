"""
Normalization helpers.
"""

from __future__ import annotations

import torch.nn as nn


def get_norm_layer(name: str, dim: int) -> nn.Module:
    """
    Return a normalization layer by name.
    """
    name = name.lower()
    if name in {"layernorm", "ln"}:
        return nn.LayerNorm(dim)
    if name in {"batchnorm1d", "bn1d"}:
        return nn.BatchNorm1d(dim)
    raise ValueError(f"Unsupported norm layer: {name}")