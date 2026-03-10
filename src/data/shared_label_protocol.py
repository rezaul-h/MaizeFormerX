"""
Shared-label protocol for cross-dataset evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.constants import SHARED_LABEL_MAPS_JSON
from src.utils.io import read_json


@dataclass(frozen=True)
class SharedLabelSpec:
    source_dataset: str
    target_dataset: str
    shared_labels: list[str]
    source_to_shared: dict[str, str]
    target_to_shared: dict[str, str]


def load_shared_label_metadata() -> dict:
    """Load shared-label metadata JSON."""
    return read_json(SHARED_LABEL_MAPS_JSON)


def get_pair_key(source_dataset: str, target_dataset: str) -> str:
    """Return canonical pair key."""
    return f"{source_dataset}__{target_dataset}"


def get_shared_label_spec(source_dataset: str, target_dataset: str) -> SharedLabelSpec:
    """Load a pairwise shared-label specification."""
    payload = load_shared_label_metadata()
    pair_key = get_pair_key(source_dataset, target_dataset)

    if pair_key not in payload["pairwise_shared_labels"]:
        raise KeyError(f"Pair {pair_key!r} not found in shared label maps.")

    pair = payload["pairwise_shared_labels"][pair_key]
    return SharedLabelSpec(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        shared_labels=list(pair["shared_labels"]),
        source_to_shared=dict(pair["source_to_shared"]),
        target_to_shared=dict(pair["target_to_shared"]),
    )


def filter_records_to_shared_labels(
    records: list[dict],
    dataset_role: str,
    spec: SharedLabelSpec,
) -> list[dict]:
    """
    Filter records to shared labels and attach canonical_shared_label.

    Parameters
    ----------
    dataset_role:
        Either "source" or "target".
    """
    if dataset_role not in {"source", "target"}:
        raise ValueError("dataset_role must be either 'source' or 'target'.")

    mapping = spec.source_to_shared if dataset_role == "source" else spec.target_to_shared

    filtered = []
    for row in records:
        class_name = row["class_name"]
        if class_name in mapping:
            filtered.append({**row, "canonical_shared_label": mapping[class_name]})
    return filtered


def build_shared_label_index(shared_labels: Iterable[str]) -> dict[str, int]:
    """Create canonical shared-label index mapping."""
    shared_labels = list(shared_labels)
    return {label: idx for idx, label in enumerate(shared_labels)}