"""
Distributed training helpers.

This module is intentionally lightweight at this stage and can later be extended
for full DDP workflows.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed_available() -> bool:
    """Return True if torch.distributed is available."""
    return dist.is_available()


def is_distributed_initialized() -> bool:
    """Return True if torch.distributed is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return process rank, defaulting to 0 in non-distributed mode."""
    if is_distributed_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Return world size, defaulting to 1 in non-distributed mode."""
    if is_distributed_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Return True for rank 0 or non-distributed mode."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes if distributed is initialized."""
    if is_distributed_initialized():
        dist.barrier()


def setup_distributed(backend: str = "nccl") -> bool:
    """
    Initialize the distributed process group if environment variables are set.

    Returns
    -------
    bool
        True if initialization succeeded, otherwise False.
    """
    required_envs = {"RANK", "WORLD_SIZE", "LOCAL_RANK"}
    if not required_envs.issubset(os.environ.keys()):
        return False

    if is_distributed_initialized():
        return True

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend)
    return True


def cleanup_distributed() -> None:
    """Destroy the process group if initialized."""
    if is_distributed_initialized():
        dist.destroy_process_group()