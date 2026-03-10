"""
Confusion matrix utilities.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    labels: list[int] | None = None,
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(targets, predictions, labels=labels)


def normalize_confusion_matrix(cm: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Normalize confusion matrix by row or column.
    """
    denom = cm.sum(axis=axis, keepdims=True)
    denom = np.where(denom == 0, 1, denom)
    return cm / denom