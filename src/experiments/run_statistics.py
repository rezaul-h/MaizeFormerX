"""
Statistical significance experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import math

from src.constants import EXPERIMENT_CONFIG_PATHS, OUTPUT_REPORTS_DIR
from src.utils.config import load_yaml_config
from src.utils.io import read_json, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> dict[str, float]:
    """
    Lightweight Wilcoxon signed-rank implementation without tie correction.
    Suitable for small paired seed-wise comparisons.
    """
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
    stat = min(w_pos, w_neg)

    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (stat - mean) / math.sqrt(var + 1e-12)
    pvalue = 2.0 * min(_normal_cdf(z), 1.0 - _normal_cdf(z))

    return {"statistic": float(stat), "pvalue": float(pvalue), "n": n}


def run_statistics_experiment(
    experiment_config_path: str | Path | None = None,
    in_domain_results_path: str | Path = "outputs/metrics/in_domain_results.json",
) -> dict[str, Any]:
    exp_cfg = load_yaml_config(experiment_config_path or EXPERIMENT_CONFIG_PATHS["statistics"])
    results = read_json(in_domain_results_path)

    reference_model = exp_cfg["reference_model"]
    metrics_to_compare = exp_cfg["metrics"]

    report: dict[str, Any] = {}

    for dataset_name in exp_cfg["datasets"]:
        report[dataset_name] = {}

        for comparison_model in exp_cfg["comparison_models"]:
            report[dataset_name][comparison_model] = {}

            aug_names = exp_cfg["augmentation_configs"][dataset_name]
            for aug_name in aug_names:
                ref_runs = results[dataset_name][reference_model][aug_name]["runs"]
                cmp_runs = results[dataset_name][comparison_model][aug_name]["runs"]

                metric_report = {}
                for metric_name in metrics_to_compare:
                    ref_vals = [float(run["test"]["metrics"][metric_name]) for run in ref_runs]
                    cmp_vals = [float(run["test"]["metrics"][metric_name]) for run in cmp_runs]
                    metric_report[metric_name] = wilcoxon_signed_rank(ref_vals, cmp_vals)

                report[dataset_name][comparison_model][aug_name] = metric_report

    output_path = OUTPUT_REPORTS_DIR / "statistics_report.json"
    write_json(output_path, report)
    logger.info("Saved statistics report to %s", output_path)
    return report