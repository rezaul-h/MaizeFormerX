"""
Exponential Moving Average (EMA) model tracking.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    """
    Maintain an exponential moving average of model parameters.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()

        for key, value in ema_state.items():
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(model_state[key].detach(), alpha=1.0 - self.decay)
            else:
                value.copy_(model_state[key])

    def to(self, device: torch.device) -> "ModelEMA":
        self.ema_model.to(device)
        return self

    def state_dict(self):
        return {
            "decay": self.decay,
            "ema_model_state_dict": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state_dict) -> None:
        self.decay = state_dict["decay"]
        self.ema_model.load_state_dict(state_dict["ema_model_state_dict"])