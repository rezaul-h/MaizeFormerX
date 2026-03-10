"""
Classification heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common.layers import init_linear


class LinearHead(nn.Module):
    """
    Simple dropout + linear classification head.
    """

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            init_linear(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(x)
        return self.fc(x)


class MLPHead(nn.Module):
    """
    MLP classification head.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            init_linear(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)