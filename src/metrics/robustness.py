"""
Robustness and variability summaries across runs.
"""

from __future__ import annotations

import numpy as np


def summarize_metric_distribution(values: list[float]) -> dict[str, float]:
    """Summarize a metric over repeated runs."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def relative_drop(reference: float, perturbed: float) -> float:
    """Compute relative drop from reference to perturbed."""
    if reference == 0:
        return 0.0
    return float((reference - perturbed) / reference)