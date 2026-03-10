"""
Core inference service logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.constants import MODEL_CONFIG_PATHS, TRAIN_CONFIG_DIR
from src.data.class_maps import build_class_map
from src.explainability.overlays import (
    cam_to_heatmap,
    overlay_heatmap_on_image,
    tensor_image_to_numpy,
)
from src.explainability.gradcam import GradCAM
from src.explainability.target_layers import resolve_target_layer
from src.models.factory import build_model
from src.utils.checkpoint import load_model_weights
from src.utils.config import load_yaml_config
from src.utils.device import get_device
from src.serving.demo_utils import build_demo_transform, load_image_from_bytes, pil_to_model_input


@dataclass
class LoadedInferenceBundle:
    model_name: str
    dataset_name: str
    checkpoint_path: str
    device: torch.device
    model: torch.nn.Module
    class_names: list[str]
    transform: Any


class InferenceService:
    """
    Stateful inference service for FastAPI or local demo use.
    """

    def __init__(self) -> None:
        self.bundle: LoadedInferenceBundle | None = None

    @property
    def is_loaded(self) -> bool:
        return self.bundle is not None

    def load_model(
        self,
        model_name: str,
        dataset_name: str,
        checkpoint_path: str | Path,
        model_config_path: str | Path | None = None,
        device_name: str = "auto",
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        device = get_device(device_name)

        if model_config_path is None:
            model_config_path = MODEL_CONFIG_PATHS[model_name]

        model_cfg = load_yaml_config(model_config_path)
        class_map = build_class_map(dataset_name)
        class_names = [class_map.index_to_class[i] for i in sorted(class_map.index_to_class)]

        model = build_model(
            model_name=model_name,
            model_cfg=model_cfg,
            num_classes=len(class_names),
        )
        model.to(device)
        load_model_weights(checkpoint_path, model, map_location=device, strict=True)
        model.eval()

        train_cfg = load_yaml_config(Path(TRAIN_CONFIG_DIR) / f"{dataset_name}.yaml")
        aug_name = train_cfg["augmentation"]["config_name"]
        aug_config_path = Path("configs/aug") / f"{aug_name}.yaml"
        transform = build_demo_transform(aug_config_path)

        self.bundle = LoadedInferenceBundle(
            model_name=model_name,
            dataset_name=dataset_name,
            checkpoint_path=str(checkpoint_path),
            device=device,
            model=model,
            class_names=class_names,
            transform=transform,
        )

    @torch.no_grad()
    def predict_from_bytes(self, image_bytes: bytes, topk: int = 3) -> dict[str, Any]:
        if self.bundle is None:
            raise RuntimeError("No model is loaded.")

        image = load_image_from_bytes(image_bytes)
        inputs = pil_to_model_input(image, self.bundle.transform).to(self.bundle.device)

        logits = self.bundle.model(inputs)
        probabilities = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=min(topk, probabilities.shape[1]), dim=1)

        predictions = []
        for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
            predictions.append(
                {
                    "class_index": int(idx),
                    "class_name": self.bundle.class_names[int(idx)],
                    "probability": float(prob),
                }
            )

        return {
            "predictions": predictions,
            "top1_class_index": predictions[0]["class_index"],
            "top1_class_name": predictions[0]["class_name"],
            "top1_probability": predictions[0]["probability"],
            "metadata": {
                "model_name": self.bundle.model_name,
                "dataset_name": self.bundle.dataset_name,
                "device": str(self.bundle.device),
            },
        }

    def explain_from_bytes(
        self,
        image_bytes: bytes,
        target_layer_path: str | None = None,
        alpha: float = 0.4,
    ) -> dict[str, Any]:
        """
        Generate a Grad-CAM overlay and return raw arrays.
        """
        if self.bundle is None:
            raise RuntimeError("No model is loaded.")

        image = load_image_from_bytes(image_bytes)
        inputs = pil_to_model_input(image, self.bundle.transform).to(self.bundle.device)

        target_layer = resolve_target_layer(
            self.bundle.model,
            model_name=self.bundle.model_name,
            layer_path=target_layer_path,
        )

        gradcam = GradCAM(
            model=self.bundle.model,
            target_layer=target_layer,
            device=self.bundle.device,
        )
        result = gradcam.generate(inputs)
        gradcam.remove_hooks()

        image_np = tensor_image_to_numpy(
            inputs[0].cpu(),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        cam_np = result.saliency[0, 0].numpy()
        heatmap_np = cam_to_heatmap(cam_np)
        overlay_np = overlay_heatmap_on_image(image_np, heatmap_np, alpha=alpha)

        pred_idx = int(result.predicted_indices[0].item())
        prob = float(torch.softmax(result.logits, dim=1)[0, pred_idx].item())

        return {
            "top1_class_index": pred_idx,
            "top1_class_name": self.bundle.class_names[pred_idx],
            "top1_probability": prob,
            "cam": cam_np,
            "overlay": overlay_np,
            "metadata": {
                "model_name": self.bundle.model_name,
                "dataset_name": self.bundle.dataset_name,
                "device": str(self.bundle.device),
            },
        }