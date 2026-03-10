"""
Per-class reporting utilities.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report


def compute_per_class_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """
    Return sklearn classification report as dict.
    """
    labels = list(range(len(class_names))) if class_names is not None else None
    return classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )