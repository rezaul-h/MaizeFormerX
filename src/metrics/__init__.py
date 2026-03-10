"""
Metric exports.
"""

from src.metrics.aggregation import aggregate_metrics
from src.metrics.classification import compute_classification_metrics
from src.metrics.confusion_matrix import compute_confusion_matrix, normalize_confusion_matrix
from src.metrics.per_class import compute_per_class_metrics
from src.metrics.pr_curves import compute_per_class_pr_curves
from src.metrics.robustness import relative_drop, summarize_metric_distribution

__all__ = [
    "aggregate_metrics",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "compute_per_class_metrics",
    "compute_per_class_pr_curves",
    "normalize_confusion_matrix",
    "relative_drop",
    "summarize_metric_distribution",
]