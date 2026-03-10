"""
Casebook helpers for collecting explainability examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, write_json


def build_case_entry(
    image_path: str,
    true_label: str,
    predicted_label: str,
    saliency_path: str | None = None,
    overlay_path: str | None = None,
    probability: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "image_path": image_path,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "probability": probability,
        "saliency_path": saliency_path,
        "overlay_path": overlay_path,
        "metadata": metadata or {},
    }


def save_casebook(path: str | Path, entries: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path)
    write_json(path, {"cases": entries})