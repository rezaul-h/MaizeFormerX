"""
GPU latency profiling.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn


def measure_gpu_latency(
    model: nn.Module,
    input_size: tuple[int, ...] = (1, 3, 224, 224),
    warmup_iterations: int = 20,
    benchmark_iterations: int = 100,
    use_fp16: bool = False,
) -> dict[str, float | None]:
    """
    Measure average GPU inference latency in milliseconds.
    """
    if not torch.cuda.is_available():
        return {
            "gpu_latency_ms": None,
            "warmup_iterations": int(warmup_iterations),
            "benchmark_iterations": int(benchmark_iterations),
            "use_fp16": bool(use_fp16),
        }

    device = torch.device("cuda")
    model = model.to(device).eval()
    dummy = torch.randn(*input_size, device=device)

    with torch.no_grad():
        for _ in range(warmup_iterations):
            with torch.autocast(device_type="cuda", enabled=use_fp16):
                _ = model(dummy)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(benchmark_iterations):
            with torch.autocast(device_type="cuda", enabled=use_fp16):
                _ = model(dummy)
        torch.cuda.synchronize()
        end = time.perf_counter()

    avg_ms = (end - start) * 1000.0 / benchmark_iterations
    return {
        "gpu_latency_ms": float(avg_ms),
        "warmup_iterations": int(warmup_iterations),
        "benchmark_iterations": int(benchmark_iterations),
        "use_fp16": bool(use_fp16),
    }