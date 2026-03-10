"""
Class map utilities based on metadata JSON files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.constants import DATASET_CLASS_FILES, LABEL_MAPS_DIR
from src.utils.io import ensure_dir, read_json, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClassMap:
    dataset_name: str
    class_to_index: dict[str, int]
    index_to_class: dict[int, str]
    slug_to_class: dict[str, str]


def load_dataset_class_metadata(dataset_name: str) -> dict:
    """Load dataset class metadata JSON."""
    if dataset_name not in DATASET_CLASS_FILES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return read_json(DATASET_CLASS_FILES[dataset_name])


def build_class_map(dataset_name: str) -> ClassMap:
    """Build class lookup tables from metadata JSON."""
    metadata = load_dataset_class_metadata(dataset_name)
    classes = metadata["classes"]

    class_to_index = {item["class_name"]: int(item["class_index"]) for item in classes}
    index_to_class = {int(item["class_index"]): item["class_name"] for item in classes}
    slug_to_class = {item["slug"]: item["class_name"] for item in classes}

    return ClassMap(
        dataset_name=dataset_name,
        class_to_index=class_to_index,
        index_to_class=index_to_class,
        slug_to_class=slug_to_class,
    )


def save_runtime_class_map(
    dataset_name: str,
    output_dir: str | Path = LABEL_MAPS_DIR,
) -> Path:
    """Save a runtime-friendly class map JSON."""
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    class_map = build_class_map(dataset_name)
    payload = {
        "dataset_name": class_map.dataset_name,
        "class_to_index": class_map.class_to_index,
        "index_to_class": {str(k): v for k, v in class_map.index_to_class.items()},
        "slug_to_class": class_map.slug_to_class,
    }

    out_path = output_dir / f"{dataset_name}_label_map.json"
    write_json(out_path, payload)
    logger.info("Saved runtime class map to %s", out_path)
    return out_path