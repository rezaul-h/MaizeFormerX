"""
Train/validation/test split generation utilities.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from src.constants import DEFAULT_SEED, DEFAULT_SPLIT_RATIOS, SPLIT_FILES_DIR
from src.utils.io import ensure_dir, read_csv, write_csv, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _validate_split_ratios(split_ratios: dict[str, float]) -> None:
    required = {"train", "val", "test"}
    if set(split_ratios.keys()) != required:
        raise ValueError(f"split_ratios must contain exactly {required}")
    total = sum(split_ratios.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def _extract_stratify_labels(records: list[dict[str, Any]]) -> list[Any]:
    return [row["class_name"] for row in records]


def build_stratified_splits(
    records: list[dict[str, Any]],
    split_ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[dict[str, Any]]]:
    """
    Build stratified train/val/test splits from manifest records.
    """
    if split_ratios is None:
        split_ratios = DEFAULT_SPLIT_RATIOS.copy()
    _validate_split_ratios(split_ratios)

    if not records:
        raise ValueError("Cannot split an empty manifest.")

    train_ratio = split_ratios["train"]
    val_ratio = split_ratios["val"]
    test_ratio = split_ratios["test"]

    labels = _extract_stratify_labels(records)

    train_records, temp_records = train_test_split(
        records,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels,
    )

    temp_labels = _extract_stratify_labels(temp_records)
    val_relative = val_ratio / (val_ratio + test_ratio)

    val_records, test_records = train_test_split(
        temp_records,
        test_size=(1.0 - val_relative),
        random_state=seed,
        stratify=temp_labels,
    )

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }


def attach_split_column(splits: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Add a split field to each record."""
    output: dict[str, list[dict[str, Any]]] = {}
    for split_name, rows in splits.items():
        output[split_name] = [{**row, "split": split_name} for row in rows]
    return output


def summarize_split_counts(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Create a summary of split sizes and per-class counts."""
    summary: dict[str, Any] = {"sizes": {}, "per_class": {}}

    for split_name, rows in splits.items():
        summary["sizes"][split_name] = len(rows)
        counter: dict[str, int] = defaultdict(int)
        for row in rows:
            counter[row["class_name"]] += 1
        summary["per_class"][split_name] = dict(sorted(counter.items()))
    return summary


def save_split_files(
    dataset_name: str,
    splits: dict[str, list[dict[str, Any]]],
    output_dir: str | Path = SPLIT_FILES_DIR,
) -> dict[str, Path]:
    """Save each split as a CSV and the overall summary as JSON."""
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    saved_paths: dict[str, Path] = {}
    split_rows = attach_split_column(splits)

    for split_name, rows in split_rows.items():
        out_path = output_dir / f"{dataset_name}_{split_name}.csv"
        write_csv(out_path, rows)
        saved_paths[split_name] = out_path

    summary = summarize_split_counts(splits)
    summary_path = output_dir / f"{dataset_name}_split_summary.json"
    write_json(summary_path, summary)
    saved_paths["summary"] = summary_path

    logger.info("Saved split files for dataset=%s into %s", dataset_name, output_dir)
    return saved_paths


def build_and_save_splits(
    manifest_csv: str | Path,
    dataset_name: str,
    split_ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
    output_dir: str | Path = SPLIT_FILES_DIR,
) -> dict[str, Path]:
    """Load a manifest CSV, build splits, and save artifacts."""
    records = read_csv(manifest_csv)
    splits = build_stratified_splits(records, split_ratios=split_ratios, seed=seed)
    return save_split_files(dataset_name=dataset_name, splits=splits, output_dir=output_dir)