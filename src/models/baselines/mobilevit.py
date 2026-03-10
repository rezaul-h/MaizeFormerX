"""
MobileViT baseline wrapper.
"""

from __future__ import annotations

import timm
import torch.nn as nn


class MobileViTBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int,
        variant: str = "mobilevit_xs",
        pretrained: bool = False,
        in_chans: int = 3,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.05,
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