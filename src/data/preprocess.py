"""
Image preprocessing helpers.
"""

from __future__ import annotations

from typing import Any

from PIL import Image
from torchvision import transforms


def build_base_preprocess(
    image_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Build a deterministic preprocessing transform."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_pil_image(image_path: str) -> Image.Image:
    """Load an image as RGB PIL image."""
    with Image.open(image_path) as img:
        return img.convert("RGB")


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Ensure image is RGB."""
    return img.convert("RGB")


def maybe_apply_transform(img: Image.Image, transform: Any = None):
    """Apply transform if provided."""
    return transform(img) if transform is not None else img