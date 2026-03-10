"""
Statistical report builders.
"""

from __future__ import annotations

from typing import Any

from src.stats.confidence_intervals import mean_confidence_interval
from src.stats.effect_sizes import cliffs_delta, cohens_d
from src.stats.wilcoxon import wilcoxon_signed_rank


def build_pairwise_stat_report(
    reference_values: list[float],
    comparison_values: list[float],
    confidence: float = 0.95,
) -> dict[str, Any]:
    """
    Build a compact statistical report for two result vectors.
    """
    return {
        "reference_ci": mean_confidence_interval(reference_values, confidence=confidence),
        "comparison_ci": mean_confidence_interval(comparison_values, confidence=confidence),
        "wilcoxon": wilcoxon_signed_rank(reference_values, comparison_values),
        "cliffs_delta": cliffs_delta(reference_values, comparison_values),
        "cohens_d": cohens_d(reference_values, comparison_values),
    }