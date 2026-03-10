"""
Classifier head for MaizeFormerX.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common.heads import LinearHead


class MaizeFormerXClassifier(nn.Module):
    """
    Pooling + classification head.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        pooling: str = "cls",
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pooling = pooling.lower()
        self.head = LinearHead(embed_dim, num_classes, dropout=head_dropout)

    def pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return x[:, 0]
        if self.pooling == "mean":
            return x.mean(dim=1)
        raise ValueError(f"Unsupported pooling mode: {self.pooling}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool_tokens(x)
        return self.head(pooled)