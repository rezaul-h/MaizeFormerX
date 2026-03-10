"""
Wilcoxon signed-rank test utilities.
"""

from __future__ import annotations

import math


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> dict[str, float]:
    """
    Lightweight paired Wilcoxon signed-rank test.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    diffs = [a - b for a, b in zip(x, y) if (a - b) != 0]
    n = len(diffs)

    if n == 0:
        return {"statistic": 0.0, "pvalue": 1.0, "n": 0}

    abs_diffs = [abs(d) for d in diffs]
    order = sorted(range(n), key=lambda i: abs_diffs[i])

    ranks = [0.0] * n
    rank = 1
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[order[j + 1]] == abs_diffs[order[i]]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        rank += j - i + 1
        i = j + 1

    w_pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    statistic = min(w_pos, w_neg)

    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    z = (statistic - mean) / math.sqrt(variance + 1e-12)
    pvalue = 2.0 * min(_normal_cdf(z), 1.0 - _normal_cdf(z))

    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "n": int(n),
        "w_pos": float(w_pos),
        "w_neg": float(w_neg),
        "z": float(z),
    }