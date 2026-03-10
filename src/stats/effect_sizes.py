"""
Effect size utilities.
"""

from __future__ import annotations

import math


def cliffs_delta(x: list[float], y: list[float]) -> dict[str, float | str]:
    """
    Compute Cliff's delta for two paired or unpaired samples.
    """
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        raise ValueError("Samples must be non-empty.")

    more = 0
    less = 0
    for a in x:
        for b in y:
            if a > b:
                more += 1
            elif a < b:
                less += 1

    delta = (more - less) / (n_x * n_y)
    magnitude = interpret_cliffs_delta(delta)
    return {"cliffs_delta": float(delta), "magnitude": magnitude}


def interpret_cliffs_delta(delta: float) -> str:
    """
    Standard magnitude interpretation for Cliff's delta.
    """
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def cohens_d(x: list[float], y: list[float]) -> dict[str, float | str]:
    """
    Compute Cohen's d for two samples.
    """
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Each sample must contain at least two values.")

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    var_x = sum((v - mean_x) ** 2 for v in x) / (len(x) - 1)
    var_y = sum((v - mean_y) ** 2 for v in y) / (len(y) - 1)

    pooled_std = math.sqrt(((len(x) - 1) * var_x + (len(y) - 1) * var_y) / (len(x) + len(y) - 2))
    if pooled_std == 0:
        d = 0.0
    else:
        d = (mean_x - mean_y) / pooled_std

    return {"cohens_d": float(d), "magnitude": interpret_cohens_d(d)}


def interpret_cohens_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"