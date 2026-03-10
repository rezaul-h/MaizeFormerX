"""
Validation helpers for manifests, splits, and label maps.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.utils.io import path_exists


REQUIRED_MANIFEST_COLUMNS = {
    "dataset",
    "image_path",
    "relative_path",
    "class_name",
    "class_index",
    "file_name",
    "stem",
    "extension",
}


def validate_manifest_columns(records: list[dict]) -> None:
    """Ensure all required manifest columns are present."""
    if not records:
        raise ValueError("Manifest is empty.")

    keys = set(records[0].keys())
    missing = REQUIRED_MANIFEST_COLUMNS - keys
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")


def validate_manifest_paths(records: list[dict]) -> None:
    """Ensure all image paths exist."""
    missing_paths = [r["image_path"] for r in records if not path_exists(r["image_path"])]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} manifest paths do not exist. "
            f"First missing path: {missing_paths[0]}"
        )


def validate_class_index_consistency(records: list[dict]) -> None:
    """Ensure class_name always maps to exactly one class_index."""
    mapping: dict[str, set[int]] = {}
    for row in records:
        mapping.setdefault(row["class_name"], set()).add(int(row["class_index"]))

    bad = {k: v for k, v in mapping.items() if len(v) != 1}
    if bad:
        raise ValueError(f"Inconsistent class index mappings found: {bad}")


def validate_split_disjointness(
    train_records: list[dict],
    val_records: list[dict],
    test_records: list[dict],
) -> None:
    """Ensure split sets are mutually disjoint by image_path."""
    train_set = {r["image_path"] for r in train_records}
    val_set = {r["image_path"] for r in val_records}
    test_set = {r["image_path"] for r in test_records}

    if train_set & val_set:
        raise ValueError("Train and val splits overlap.")
    if train_set & test_set:
        raise ValueError("Train and test splits overlap.")
    if val_set & test_set:
        raise ValueError("Val and test splits overlap.")


def summarize_class_distribution(records: list[dict]) -> dict[str, int]:
    """Return class-name counts."""
    counter = Counter(r["class_name"] for r in records)
    return dict(sorted(counter.items()))


def validate_non_empty_split(records: list[dict], split_name: str) -> None:
    """Ensure a split is non-empty."""
    if not records:
        raise ValueError(f"{split_name} split is empty.")


def validate_dataset_root(path: str | Path) -> None:
    """Ensure dataset root exists and is a directory."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {path}")