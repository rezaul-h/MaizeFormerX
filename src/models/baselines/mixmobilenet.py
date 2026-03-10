"""
Custom lightweight CNN baseline: MixMobileNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common.heads import MLPHead
from src.models.common.layers import ConvBNAct


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True) -> None:
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, kernel_size=3, stride=stride, groups=in_ch, act_layer=nn.Hardswish)
        self.pw = ConvBNAct(in_ch, out_ch, kernel_size=1, stride=1, groups=1, act_layer=nn.Hardswish)
        self.use_se = use_se
        if use_se:
            se_hidden = max(out_ch // 4, 8)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch, se_hidden, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(se_hidden, out_ch, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        if self.use_se:
            scale = self.se(x)
            x = x * scale
        return x


class MixMobileNet(nn.Module):
    """
    Compact CNN baseline for mobile-efficient benchmarking.
    """

    def __init__(
        self,
        num_classes: int,
        in_chans: int = 3,
        width_mult: float = 1.0,
        use_se: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        c3 = int(128 * width_mult)
        c4 = int(192 * width_mult)
        c5 = int(256 * width_mult)

        self.stem = ConvBNAct(in_chans, c1, kernel_size=3, stride=2, act_layer=nn.Hardswish)
        self.stage1 = DepthwiseSeparableBlock(c1, c2, stride=1, use_se=use_se)
        self.stage2 = DepthwiseSeparableBlock(c2, c3, stride=2, use_se=use_se)
        self.stage3 = DepthwiseSeparableBlock(c3, c4, stride=2, use_se=use_se)
        self.stage4 = DepthwiseSeparableBlock(c4, c5, stride=2, use_se=use_se)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = MLPHead(
            in_dim=c5,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            act_layer=nn.Hardswish,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.head(feats)