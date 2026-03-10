"""
Aggregation helpers for multi-seed / multi-run metrics.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def aggregate_metrics(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """
    Aggregate scalar metrics across runs.
    """
    bucket: dict[str, list[float]] = defaultdict(list)

    for metrics in metric_dicts:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                bucket[key].append(float(value))

    aggregated: dict[str, dict[str, float]] = {}
    for key, values in bucket.items():
        arr = np.asarray(values, dtype=float)
        aggregated[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return aggregated