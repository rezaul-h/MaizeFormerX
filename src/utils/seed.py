"""
Reproducibility helpers.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Random seed.
    deterministic:
        Whether to enable deterministic PyTorch behavior where possible.
    cudnn_benchmark:
        Whether to enable cuDNN benchmark mode.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark

    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        # Some environments or operators may not fully support this.
        pass