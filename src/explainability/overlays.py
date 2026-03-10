"""
Utilities for creating saliency overlays.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def tensor_image_to_numpy(image_tensor, mean=None, std=None) -> np.ndarray:
    """
    Convert CHW tensor to HWC uint8 image.
    """
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    if mean is not None and std is not None:
        mean = np.asarray(mean).reshape(1, 1, 3)
        std = np.asarray(std).reshape(1, 1, 3)
        image = image * std + mean
    image = np.clip(image, 0.0, 1.0)
    return _to_uint8_image(image)


def cam_to_heatmap(cam: np.ndarray) -> np.ndarray:
    """
    Convert a normalized CAM map [H, W] into a simple red heatmap.
    """
    cam = np.clip(cam, 0.0, 1.0)
    heatmap = np.zeros((cam.shape[0], cam.shape[1], 3), dtype=np.float32)
    heatmap[..., 0] = cam
    heatmap[..., 1] = cam * 0.25
    heatmap[..., 2] = 0.0
    return _to_uint8_image(heatmap)


def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay heatmap on image.
    """
    image_f = image.astype(np.float32)
    heatmap_f = heatmap.astype(np.float32)
    overlay = (1.0 - alpha) * image_f + alpha * heatmap_f
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_numpy_image(image: np.ndarray, path: str) -> None:
    Image.fromarray(image).save(path)