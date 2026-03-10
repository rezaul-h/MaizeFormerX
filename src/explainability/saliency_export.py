"""
Export utilities for saliency maps and overlays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.io import ensure_dir, write_json
from src.explainability.overlays import save_numpy_image


def save_saliency_array(path: str | Path, saliency: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path)
    np.save(path, saliency)


def save_overlay_image(path: str | Path, overlay: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path)
    save_numpy_image(overlay, str(path))


def save_explainability_metadata(path: str | Path, metadata: dict) -> None:
    path = Path(path)
    ensure_dir(path)
    write_json(path, metadata)