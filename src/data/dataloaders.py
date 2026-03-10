"""
Dataloader construction helpers.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from src.data.datasets import ManifestImageDataset, SharedLabelDataset
from src.data.samplers import build_weighted_sampler


def build_dataloader(
    dataset,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    shuffle: bool = False,
    sampler=None,
    drop_last: bool = False,
) -> DataLoader:
    """Create a PyTorch DataLoader."""
    use_persistent = persistent_workers and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        drop_last=drop_last,
    )


def build_manifest_dataloader(
    records,
    transform,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    shuffle: bool = False,
    weighted_sampling: bool = False,
    drop_last: bool = False,
    return_metadata: bool = False,
):
    """Build a dataloader from manifest-backed records."""
    dataset = ManifestImageDataset(records=records, transform=transform, return_metadata=return_metadata)
    sampler = build_weighted_sampler(dataset.records) if weighted_sampling else None

    return build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
    )


def build_shared_label_dataloader(
    records,
    shared_label_to_index: dict[str, int],
    transform,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    shuffle: bool = False,
    drop_last: bool = False,
    return_metadata: bool = False,
):
    """Build a shared-label dataloader."""
    dataset = SharedLabelDataset(
        records=records,
        shared_label_to_index=shared_label_to_index,
        transform=transform,
        return_metadata=return_metadata,
    )
    return build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        shuffle=shuffle,
        sampler=None,
        drop_last=drop_last,
    )