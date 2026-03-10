"""
Memory profiling utilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def measure_peak_gpu_memory(
    model: nn.Module,
    input_size: tuple[int, ...] = (1, 3, 224, 224),
    use_fp16: bool = False,
) -> dict[str, float | None]:
    """
    Measure peak CUDA memory usage during a forward pass in MB.
    """
    if not torch.cuda.is_available():
        return {"peak_memory_mb": None, "use_fp16": bool(use_fp16)}

    device = torch.device("cuda")
    model = model.to(device).eval()
    dummy = torch.randn(*input_size, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=use_fp16):
            _ = model(dummy)
        torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb = peak_bytes / (1024.0 ** 2)

    return {"peak_memory_mb": float(peak_mb), "use_fp16": bool(use_fp16)}