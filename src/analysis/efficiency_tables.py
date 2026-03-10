"""
Efficiency table builders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, read_json, write_csv


def build_efficiency_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for model_name, block in results.items():
        row = {
            "model": model_name,
            "trainable_params": block.get("trainable_params"),
            "flops": block.get("flops"),
            "cpu_latency_ms": block.get("cpu_latency_ms"),
            "gpu_latency_ms": block.get("gpu_latency_ms"),
            "peak_memory_mb": block.get("peak_memory_mb"),
            "device": block.get("device"),
        }
        rows.append(row)

    return rows


def export_efficiency_table(
    results_json_path: str | Path,
    output_csv_path: str | Path,
) -> list[dict[str, Any]]:
    results = read_json(results_json_path)
    rows = build_efficiency_rows(results)
    output_csv_path = Path(output_csv_path)
    ensure_dir(output_csv_path)
    write_csv(output_csv_path, rows)
    return rows