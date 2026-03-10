"""
Precision-recall curve utilities.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize


def compute_per_class_pr_curves(
    targets: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Compute per-class precision-recall curves.
    """
    y_true_bin = label_binarize(targets, classes=list(range(num_classes)))
    if y_true_bin.shape[1] == 1 and num_classes == 2:
        y_true_bin = np.concatenate([1 - y_true_bin, y_true_bin], axis=1)

    curves: dict[int, dict[str, np.ndarray]] = {}
    for class_idx in range(num_classes):
        precision, recall, thresholds = precision_recall_curve(
            y_true_bin[:, class_idx],
            probabilities[:, class_idx],
        )
        curves[class_idx] = {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
        }
    return curves