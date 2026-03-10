"""
Sampler utilities for imbalanced datasets.
"""

from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler


def compute_class_weights_from_records(records: list[dict]) -> dict[int, float]:
    """Compute inverse-frequency class weights from records."""
    labels = [int(r["class_index"]) for r in records]
    counter = Counter(labels)
    total = len(labels)
    num_classes = len(counter)

    weights = {
        cls_idx: total / (num_classes * count)
        for cls_idx, count in counter.items()
    }
    return weights


def build_weighted_sampler(records: list[dict]) -> WeightedRandomSampler:
    """Build a weighted random sampler from manifest records."""
    class_weights = compute_class_weights_from_records(records)
    sample_weights = [class_weights[int(r["class_index"])] for r in records]

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )