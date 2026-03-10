"""
Minimal Grad-CAM implementation for classification models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GradCAMResult:
    saliency: torch.Tensor
    logits: torch.Tensor
    predicted_indices: torch.Tensor
    target_indices: torch.Tensor


class GradCAM:
    """
    Generic Grad-CAM for CNN- or token-based models.

    Notes
    -----
    - For 4D feature maps: standard Grad-CAM.
    - For 3D token outputs [B, N, C]: token maps are reshaped to a square grid
      after discarding CLS token when possible.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def remove_hooks(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def _token_to_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token tensor [B, N, C] to [B, C, H, W].
        """
        if x.ndim != 3:
            raise ValueError("Expected token tensor with shape [B, N, C].")

        b, n, c = x.shape

        # Heuristic: remove CLS token if present and remaining tokens form a square.
        if n > 1:
            n_wo_cls = n - 1
            side = int(n_wo_cls ** 0.5)
            if side * side == n_wo_cls:
                x = x[:, 1:, :]
                n = n_wo_cls

        side = int(n ** 0.5)
        if side * side != n:
            raise ValueError(
                f"Cannot reshape token sequence of length {n} into square spatial map."
            )

        x = x.transpose(1, 2).reshape(b, c, side, side)
        return x

    def _compute_cam(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        """
        Compute Grad-CAM heatmap.
        """
        if activations.ndim == 3:
            activations = self._token_to_spatial(activations)
        if gradients.ndim == 3:
            gradients = self._token_to_spatial(gradients)

        if activations.ndim != 4 or gradients.ndim != 4:
            raise ValueError("Grad-CAM expects spatial activations/gradients with ndim=4.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        b = cam.shape[0]
        cam = cam.view(b, -1)
        cam_min = cam.min(dim=1, keepdim=True).values
        cam_max = cam.max(dim=1, keepdim=True).values
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam = cam.view(b, 1, *activations.shape[2:])
        return cam

    def generate(self, inputs: torch.Tensor, target_indices: torch.Tensor | None = None) -> GradCAMResult:
        """
        Generate saliency maps for a batch of inputs.
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        inputs = inputs.to(self.device)
        logits = self.model(inputs)
        predicted_indices = torch.argmax(logits, dim=1)

        if target_indices is None:
            target_indices = predicted_indices
        else:
            target_indices = target_indices.to(self.device)

        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_indices.view(-1, 1), 1.0)

        score = (logits * one_hot).sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        cam = self._compute_cam(self.activations, self.gradients)
        cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)

        return GradCAMResult(
            saliency=cam.detach().cpu(),
            logits=logits.detach().cpu(),
            predicted_indices=predicted_indices.detach().cpu(),
            target_indices=target_indices.detach().cpu(),
        )