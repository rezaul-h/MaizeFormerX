"""
Efficiency profiling experiment.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from src.constants import EXPERIMENT_CONFIG_PATHS, MODEL_CONFIG_PATHS, OUTPUT_PROFILES_DIR
from src.models.factory import build_model
from src.utils.config import load_yaml_config
from src.utils.device import get_device
from src.utils.io import write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _measure_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup_iterations: int = 20,
    benchmark_iterations: int = 100,
) -> float:
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(benchmark_iterations):
            _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    return (end - start) * 1000.0 / benchmark_iterations


def run_efficiency_experiment(
    experiment_config_path: str | Path | None = None,
    device_name: str = "auto",
    num_classes: int = 3,
) -> dict[str, Any]:
    exp_cfg = load_yaml_config(experiment_config_path or EXPERIMENT_CONFIG_PATHS["efficiency"])
    device = get_device(device_name)

    results: dict[str, Any] = {}
    input_size = exp_cfg["profiling"]["input_size"]
    warmup_iterations = exp_cfg["profiling"]["warmup_iterations"]
    benchmark_iterations = exp_cfg["profiling"]["benchmark_iterations"]

    for model_name in exp_cfg["models"]:
        logger.info("Profiling model=%s on device=%s", model_name, device)
        model_cfg = load_yaml_config(MODEL_CONFIG_PATHS[model_name])
        model = build_model(model_name=model_name, model_cfg=model_cfg, num_classes=num_classes)
        model.to(device)

        input_tensor = torch.randn(*input_size, device=device)
        latency_ms = _measure_latency(
            model=model,
            input_tensor=input_tensor,
            device=device,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
        )

        results[model_name] = {
            "trainable_params": _count_trainable_params(model),
            "latency_ms": latency_ms,
            "device": str(device),
            "input_size": input_size,
        }

    output_path = OUTPUT_PROFILES_DIR / "efficiency_results.json"
    write_json(output_path, results)
    logger.info("Saved efficiency results to %s", output_path)
    return results