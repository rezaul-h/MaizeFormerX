"""
Augmentation sweep experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.constants import EXPERIMENT_CONFIG_PATHS, OUTPUT_METRICS_DIR
from src.experiments.run_in_domain import _prepare_single_run
from src.metrics.aggregation import aggregate_metrics
from src.utils.config import load_yaml_config
from src.utils.io import write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_augmentation_sweep_experiment(
    experiment_config_path: str | Path | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    exp_cfg = load_yaml_config(experiment_config_path or EXPERIMENT_CONFIG_PATHS["augmentation_sweep"])
    results: dict[str, Any] = {}

    for dataset_name in exp_cfg["datasets"]:
        results[dataset_name] = {}
        for model_name in exp_cfg["models"]:
            results[dataset_name][model_name] = {}

            for aug_name in exp_cfg["sweep"][dataset_name]:
                logger.info(
                    "Running augmentation sweep: dataset=%s, model=%s, aug=%s",
                    dataset_name,
                    model_name,
                    aug_name,
                )

                run_rows = []
                metric_rows = []
                for seed in exp_cfg["execution"]["seeds"]:
                    single = _prepare_single_run(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        aug_config_name=aug_name,
                        seed=seed,
                        device_name=device_name,
                    )
                    run_rows.append(single)
                    metric_rows.append(single["test"]["metrics"])

                results[dataset_name][model_name][aug_name] = {
                    "runs": run_rows,
                    "aggregated_test_metrics": aggregate_metrics(metric_rows),
                }

    output_path = OUTPUT_METRICS_DIR / "augmentation_sweep_results.json"
    write_json(output_path, results)
    logger.info("Saved augmentation sweep results to %s", output_path)
    return results