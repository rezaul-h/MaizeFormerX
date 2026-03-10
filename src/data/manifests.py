"""
Manifest construction utilities for maize datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.constants import (
    DATASET_RAW_DIRS,
    IMAGE_EXTENSIONS,
    MANIFESTS_DIR,
    SUPPORTED_DATASETS,
)
from src.utils.io import ensure_dir, list_files, write_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ManifestRecord:
    dataset: str
    image_path: str
    relative_path: str
    class_name: str
    class_index: int
    file_name: str
    stem: str
    extension: str


def discover_class_directories(dataset_root: Path) -> list[Path]:
    """Return immediate class directories under a dataset root."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    class_dirs = [p for p in dataset_root.iterdir() if p.is_dir()]
    return sorted(class_dirs)


def build_manifest_from_folder_tree(
    dataset_name: str,
    dataset_root: str | Path,
    class_to_index: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """
    Build a manifest from a folder-structured dataset.

    Expected layout
    ---------------
    dataset_root/
        class_a/
            img1.jpg
            img2.jpg
        class_b/
            img3.jpg
    """
    dataset_root = Path(dataset_root)
    class_dirs = discover_class_directories(dataset_root)

    if not class_dirs:
        raise ValueError(f"No class directories found under: {dataset_root}")

    if class_to_index is None:
        class_to_index = {p.name: idx for idx, p in enumerate(class_dirs)}

    records: list[dict[str, Any]] = []
    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name not in class_to_index:
            raise KeyError(f"Class {class_name!r} is missing from class_to_index.")

        image_paths = list_files(class_dir, suffixes=IMAGE_EXTENSIONS, recursive=True)
        for image_path in image_paths:
            rel_path = image_path.relative_to(dataset_root)
            row = ManifestRecord(
                dataset=dataset_name,
                image_path=str(image_path.resolve()),
                relative_path=str(rel_path.as_posix()),
                class_name=class_name,
                class_index=class_to_index[class_name],
                file_name=image_path.name,
                stem=image_path.stem,
                extension=image_path.suffix.lower(),
            )
            records.append(asdict(row))

    logger.info(
        "Built manifest for dataset=%s with %d samples and %d classes.",
        dataset_name,
        len(records),
        len(class_to_index),
    )
    return records


def save_manifest(
    dataset_name: str,
    records: list[dict[str, Any]],
    output_dir: str | Path = MANIFESTS_DIR,
) -> Path:
    """Save manifest records to CSV."""
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    output_path = output_dir / f"{dataset_name}_manifest.csv"
    write_csv(output_path, records)
    logger.info("Saved manifest to %s", output_path)
    return output_path


def build_and_save_manifest(
    dataset_name: str,
    dataset_root: str | Path | None = None,
    class_to_index: dict[str, int] | None = None,
    output_dir: str | Path = MANIFESTS_DIR,
) -> Path:
    """Build and save a manifest for one dataset."""
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_root = Path(dataset_root) if dataset_root is not None else DATASET_RAW_DIRS[dataset_name]
    records = build_manifest_from_folder_tree(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        class_to_index=class_to_index,
    )
    return save_manifest(dataset_name=dataset_name, records=records, output_dir=output_dir)