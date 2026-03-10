"""
Utilities for loading and preprocessing images for the demo / API layer.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

from src.data.augmentations import build_train_val_test_transforms
from src.utils.config import load_yaml_config


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load a PIL RGB image from raw bytes."""
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def load_image_from_path(path: str | Path) -> Image.Image:
    """Load a PIL RGB image from a filesystem path."""
    return Image.open(path).convert("RGB")


def build_demo_transform(aug_config_path: str | Path):
    """
    Build inference transform from an augmentation config.

    Uses the deterministic val/test transform branch.
    """
    aug_cfg = load_yaml_config(aug_config_path)
    transforms = build_train_val_test_transforms(aug_cfg)
    return transforms["test"]


def pil_to_model_input(image: Image.Image, transform):
    """
    Convert PIL image into model-ready tensor batch.
    """
    tensor = transform(image)
    return tensor.unsqueeze(0)