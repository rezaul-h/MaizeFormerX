"""
MobileNetV3 baseline wrapper.
"""

from __future__ import annotations

import timm
import torch.nn as nn


class MobileNetV3Baseline(nn.Module):
    def __init__(
        self,
        num_classes: int,
        variant: str = "mobilenetv3_large_100",
        pretrained: bool = False,
        in_chans: int = 3,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            variant,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=drop_rate,
        )

    def forward(self, x):
        return self.model(x)