"""
Experiment orchestration.
"""

from __future__ import annotations

from typing import Any

from src.experiments.run_ablation import run_ablation_experiment
from src.experiments.run_augmentation_sweep import run_augmentation_sweep_experiment
from src.experiments.run_cross_domain import run_cross_domain_experiment
from src.experiments.run_efficiency import run_efficiency_experiment
from src.experiments.run_explainability import run_explainability_experiment
from src.experiments.run_in_domain import run_in_domain_experiment
from src.experiments.run_statistics import run_statistics_experiment


def run_experiment(experiment_name: str, **kwargs) -> dict[str, Any]:
    name = experiment_name.lower()

    if name == "in_domain":
        return run_in_domain_experiment(**kwargs)
    if name == "augmentation_sweep":
        return run_augmentation_sweep_experiment(**kwargs)
    if name == "cross_domain":
        return run_cross_domain_experiment(**kwargs)
    if name == "ablation":
        return run_ablation_experiment(**kwargs)
    if name == "explainability":
        return run_explainability_experiment(**kwargs)
    if name == "efficiency":
        return run_efficiency_experiment(**kwargs)
    if name == "statistics":
        return run_statistics_experiment(**kwargs)

    raise ValueError(f"Unsupported experiment: {experiment_name}")