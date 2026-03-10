"""
CPU latency profiling.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn


def measure_cpu_latency(
    model: nn.Module,
    input_size: tuple[int, ...] = (1, 3, 224, 224),
    warmup_iterations: int = 20,
    benchmark_iterations: int = 100,
    num_threads: int = 1,
) -> dict[str, float]:
    """
    Measure average CPU inference latency in milliseconds.
    """
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)

    model = model.to("cpu").eval()
    dummy = torch.randn(*input_size, device="cpu")

    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy)

        start = time.perf_counter()
        for _ in range(benchmark_iterations):
            _ = model(dummy)
        end = time.perf_counter()

    torch.set_num_threads(old_threads)

    avg_ms = (end - start) * 1000.0 / benchmark_iterations
    return {
        "cpu_latency_ms": float(avg_ms),
        "warmup_iterations": int(warmup_iterations),
        "benchmark_iterations": int(benchmark_iterations),
        "num_threads": int(num_threads),
    }