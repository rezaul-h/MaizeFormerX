"""
SwinV2 baseline wrapper.
"""

from __future__ import annotations

import timm
import torch.nn as nn


class SwinV2Baseline(nn.Module):
    def __init__(
        self,
        num_classes: int,
        variant: str = "swinv2_tiny_window8_256",
        pretrained: bool = False,
        in_chans: int = 3,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            variant,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        return self.model(x)