"""
Batch-wise accumulation of predictions and losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class MetricsAccumulator:
    losses: list[float] = field(default_factory=list)
    logits: list[np.ndarray] = field(default_factory=list)
    probabilities: list[np.ndarray] = field(default_factory=list)
    predictions: list[np.ndarray] = field(default_factory=list)
    targets: list[np.ndarray] = field(default_factory=list)

    def update(
        self,
        loss: float,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self.losses.append(float(loss))
        self.logits.append(logits.detach().cpu().numpy())
        self.probabilities.append(probabilities.detach().cpu().numpy())
        self.predictions.append(predictions.detach().cpu().numpy())
        self.targets.append(targets.detach().cpu().numpy())

    def compute(self) -> dict[str, np.ndarray | float]:
        avg_loss = float(np.mean(self.losses)) if self.losses else 0.0

        logits = np.concatenate(self.logits, axis=0) if self.logits else np.empty((0,))
        probabilities = np.concatenate(self.probabilities, axis=0) if self.probabilities else np.empty((0,))
        predictions = np.concatenate(self.predictions, axis=0) if self.predictions else np.empty((0,))
        targets = np.concatenate(self.targets, axis=0) if self.targets else np.empty((0,))

        return {
            "loss": avg_loss,
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "targets": targets,
        }

    def reset(self) -> None:
        self.losses.clear()
        self.logits.clear()
        self.probabilities.clear()
        self.predictions.clear()
        self.targets.clear()