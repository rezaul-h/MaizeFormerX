"""
ShuffleNetV2 baseline wrapper.
"""

from __future__ import annotations

import timm
import torch.nn as nn


class ShuffleNetV2Baseline(nn.Module):
    def __init__(
        self,
        num_classes: int,
        variant: str = "shufflenet_v2_x1_0",
        pretrained: bool = False,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            variant,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
        )

    def forward(self, x):
        return self.model(x)