"""
Scheduler builders.
"""

from __future__ import annotations

import math
from typing import Any

import torch


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup followed by cosine decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step <= self.warmup_steps and self.warmup_steps > 0:
            scale = step / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)

        return [
            self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: dict[str, Any],
    steps_per_epoch: int,
    max_epochs: int,
):
    """
    Build LR scheduler from config.

    Supported
    ---------
    - cosine
    - step
    - plateau
    - none
    """
    name = scheduler_cfg.get("name", "cosine").lower()

    if name in {"none", "null", "off"}:
        return None

    if name == "cosine":
        total_steps = int(max_epochs * steps_per_epoch)
        warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
        eta_min = float(scheduler_cfg.get("eta_min", 1e-6))
        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=eta_min,
        )

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 10)),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )

    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get("mode", "max"),
            factor=float(scheduler_cfg.get("factor", 0.1)),
            patience=int(scheduler_cfg.get("patience", 3)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )

    raise ValueError(f"Unsupported scheduler: {name}")