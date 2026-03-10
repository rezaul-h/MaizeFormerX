"""
Inference helper.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.device import move_to_device


class Inferencer:
    """
    Lightweight batched inference wrapper.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: list[str] | None = None,
        use_amp: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.class_names = class_names
        self.use_amp = use_amp and device.type == "cuda"

    @torch.no_grad()
    def predict_batch(self, inputs: torch.Tensor, topk: int = 3) -> dict:
        self.model.eval()
        inputs = move_to_device(inputs, self.device)

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            logits = self.model(inputs)

        probabilities = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=min(topk, probabilities.shape[1]), dim=1)

        result = {
            "logits": logits.detach().cpu(),
            "probabilities": probabilities.detach().cpu(),
            "top_probs": top_probs.detach().cpu(),
            "top_indices": top_indices.detach().cpu(),
        }

        if self.class_names is not None:
            top_labels = []
            for row in top_indices.detach().cpu().tolist():
                top_labels.append([self.class_names[idx] for idx in row])
            result["top_labels"] = top_labels

        return result