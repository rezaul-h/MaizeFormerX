"""
Common neural network layers and initialization helpers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """
    Truncated normal initialization wrapper.
    """
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-2.0 * std, b=2.0 * std)


def init_linear(module: nn.Linear, std: float = 0.02) -> None:
    """Initialize a linear layer."""
    trunc_normal_(module.weight, std=std)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.0)


def init_conv(module: nn.Conv2d) -> None:
    """Kaiming initialization for convolution layers."""
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.0)


def init_layernorm(module: nn.LayerNorm) -> None:
    """Initialize LayerNorm."""
    nn.init.constant_(module.bias, 0.0)
    nn.init.constant_(module.weight, 1.0)


class DropPath(nn.Module):
    """
    Stochastic depth per sample.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerScale(nn.Module):
    """
    Learnable residual scaling.
    """

    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class ConvBNAct(nn.Module):
    """
    Standard Conv-BN-Activation block.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        bias: bool = False,
        act_layer: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_chans,
                out_chans,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_chans),
            act_layer(inplace=True) if act_layer in {nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU} else act_layer(),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            init_conv(m)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)