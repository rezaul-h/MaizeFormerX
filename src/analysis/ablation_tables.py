"""
Ablation table builders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, read_json, write_csv


DEFAULT_METRICS = ["accuracy", "micro_f1", "macro_f1", "pr_auc", "mcc"]


def build_ablation_rows(
    results: dict[str, Any],
    metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    metrics = metrics or DEFAULT_METRICS
    rows: list[dict[str, Any]] = []

    for dataset_name, dataset_block in results.items():
        for ablation_name, ablation_block in dataset_block.items():
            row = {
                "dataset": dataset_name,
                "ablation": ablation_name,
            }
            aggregated = ablation_block.get("aggregated_test_metrics", {})
            for metric in metrics:
                stat = aggregated.get(metric, {})
                row[f"{metric}_mean"] = stat.get("mean")
                row[f"{metric}_std"] = stat.get("std")
            rows.append(row)

    return rows


def export_ablation_table(
    results_json_path: str | Path,
    output_csv_path: str | Path,
    metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    results = read_json(results_json_path)
    rows = build_ablation_rows(results, metrics=metrics)
    output_csv_path = Path(output_csv_path)
    ensure_dir(output_csv_path)
    write_csv(output_csv_path, rows)
    return rows