"""
Evaluation loop.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.engine.metrics_accumulator import MetricsAccumulator
from src.metrics.classification import compute_classification_metrics
from src.metrics.confusion_matrix import compute_confusion_matrix
from src.metrics.per_class import compute_per_class_metrics
from src.utils.device import move_to_device


class Evaluator:
    """
    Generic evaluator for classification models.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        num_classes: int,
        class_names: list[str] | None = None,
        use_amp: bool = True,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        self.use_amp = use_amp and device.type == "cuda"

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        accumulator = MetricsAccumulator()

        autocast_enabled = self.use_amp
        for batch in dataloader:
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            inputs = move_to_device(inputs, self.device)
            targets = move_to_device(targets, self.device)

            with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            accumulator.update(
                loss=float(loss.item()),
                logits=logits,
                probabilities=probabilities,
                predictions=predictions,
                targets=targets,
            )

        outputs = accumulator.compute()
        metrics = compute_classification_metrics(
            targets=outputs["targets"],
            predictions=outputs["predictions"],
            probabilities=outputs["probabilities"],
            num_classes=self.num_classes,
        )
        metrics["loss"] = float(outputs["loss"])

        cm = compute_confusion_matrix(
            outputs["targets"],
            outputs["predictions"],
            labels=list(range(self.num_classes)),
        )
        per_class = compute_per_class_metrics(
            outputs["targets"],
            outputs["predictions"],
            class_names=self.class_names,
        )

        return {
            "metrics": metrics,
            "confusion_matrix": cm,
            "per_class_metrics": per_class,
            "logits": outputs["logits"],
            "probabilities": outputs["probabilities"],
            "predictions": outputs["predictions"],
            "targets": outputs["targets"],
        }