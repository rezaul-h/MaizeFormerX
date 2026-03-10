"""
Patch embedding modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common.layers import init_conv
from src.models.common.norms import get_norm_layer


class PatchEmbed(nn.Module):
    """
    Single-scale convolutional patch embedding.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        stride: int | None = None,
        in_channels: int = 3,
        embed_dim: int = 768,
        norm_layer: str | None = "layernorm",
        flatten: bool = True,
    ) -> None:
        super().__init__()
        stride = stride or patch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=0,
        )
        self.norm = get_norm_layer(norm_layer, embed_dim) if norm_layer is not None else nn.Identity()

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            init_conv(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # B, C, H, W
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B, N, C
            x = self.norm(x)
        return x