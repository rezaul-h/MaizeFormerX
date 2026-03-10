"""
Confidence interval utilities.
"""

from __future__ import annotations

import math


def mean_confidence_interval(values: list[float], confidence: float = 0.95) -> dict[str, float]:
    """
    Approximate normal-based confidence interval for the mean.
    """
    if len(values) == 0:
        raise ValueError("values must be non-empty.")

    n = len(values)
    mean = sum(values) / n

    if n == 1:
        return {
            "mean": float(mean),
            "lower": float(mean),
            "upper": float(mean),
            "half_width": 0.0,
            "confidence": float(confidence),
        }

    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    se = std / math.sqrt(n)

    # Approximate z critical values
    z = 1.96 if abs(confidence - 0.95) < 1e-8 else 1.96
    half_width = z * se

    return {
        "mean": float(mean),
        "lower": float(mean - half_width),
        "upper": float(mean + half_width),
        "half_width": float(half_width),
        "confidence": float(confidence),
    }