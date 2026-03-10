"""
Multi-scale patch embedding for MaizeFormerX.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common.layers import init_conv
from src.models.common.norms import get_norm_layer


class MultiScalePatchEmbed(nn.Module):
    """
    Multi-branch patch embedding with different patch sizes / strides.

    Returns a list of token tensors, one per scale:
        [B, N1, D1], [B, N2, D2], ...
    """

    def __init__(
        self,
        in_channels: int,
        patch_sizes: list[int],
        strides: list[int],
        embed_dims: list[int],
        norm_layer: str = "layernorm",
        flatten: bool = True,
        use_conv_projection: bool = True,
    ) -> None:
        super().__init__()

        if not (len(patch_sizes) == len(strides) == len(embed_dims)):
            raise ValueError("patch_sizes, strides, and embed_dims must have the same length.")

        self.flatten = flatten
        self.branches = nn.ModuleList()

        for patch_size, stride, embed_dim in zip(patch_sizes, strides, embed_dims):
            if use_conv_projection:
                proj = nn.Conv2d(
                    in_channels,
                    embed_dim,
                    kernel_size=patch_size,
                    stride=stride,
                    padding=0,
                    bias=False,
                )
            else:
                raise NotImplementedError("Only convolutional projection is currently supported.")

            norm = get_norm_layer(norm_layer, embed_dim)
            self.branches.append(
                nn.ModuleDict(
                    {
                        "proj": proj,
                        "norm": norm,
                    }
                )
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            init_conv(m)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []

        for branch in self.branches:
            feat = branch["proj"](x)  # B, C, H, W
            feat = feat.flatten(2).transpose(1, 2)  # B, N, C
            feat = branch["norm"](feat)
            outputs.append(feat)

        return outputs