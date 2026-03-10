"""
Classification metrics.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
    num_classes: int | None = None,
) -> dict[str, float]:
    """
    Compute core classification metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "micro_f1": float(f1_score(targets, predictions, average="micro")),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "mcc": float(matthews_corrcoef(targets, predictions)),
    }

    if probabilities is not None and num_classes is not None and len(targets) > 0:
        y_true_bin = label_binarize(targets, classes=list(range(num_classes)))
        if y_true_bin.shape[1] == 1 and num_classes == 2:
            y_true_bin = np.concatenate([1 - y_true_bin, y_true_bin], axis=1)

        try:
            metrics["pr_auc"] = float(
                average_precision_score(y_true_bin, probabilities, average="macro")
            )
        except ValueError:
            metrics["pr_auc"] = float("nan")
    else:
        metrics["pr_auc"] = float("nan")

    return metrics