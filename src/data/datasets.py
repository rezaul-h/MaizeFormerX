"""
Dataset definitions for manifest-driven training and evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.preprocess import load_pil_image
from src.utils.io import read_csv


class ManifestImageDataset(Dataset):
    """
    Generic image dataset backed by a manifest CSV or list of row dictionaries.
    """

    def __init__(
        self,
        records: str | Path | list[dict[str, Any]],
        transform=None,
        return_metadata: bool = False,
    ) -> None:
        if isinstance(records, (str, Path)):
            self.records = read_csv(records)
        else:
            self.records = records

        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        image = load_pil_image(row["image_path"])
        if self.transform is not None:
            image = self.transform(image)

        label = int(row["class_index"])
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_metadata:
            meta = {
                "dataset": row.get("dataset"),
                "class_name": row.get("class_name"),
                "image_path": row.get("image_path"),
                "relative_path": row.get("relative_path"),
                "file_name": row.get("file_name"),
                "split": row.get("split"),
                "index": index,
            }
            return image, label_tensor, meta

        return image, label_tensor


class SharedLabelDataset(Dataset):
    """
    Dataset that uses canonical_shared_label instead of original class_name.

    Useful for cross-dataset shared-label evaluation.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        shared_label_to_index: dict[str, int],
        transform=None,
        return_metadata: bool = False,
    ) -> None:
        self.records = records
        self.shared_label_to_index = shared_label_to_index
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        image = load_pil_image(row["image_path"])
        if self.transform is not None:
            image = self.transform(image)

        canonical_label = row["canonical_shared_label"]
        label = int(self.shared_label_to_index[canonical_label])
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_metadata:
            meta = {
                "dataset": row.get("dataset"),
                "original_class_name": row.get("class_name"),
                "canonical_shared_label": canonical_label,
                "image_path": row.get("image_path"),
                "relative_path": row.get("relative_path"),
                "index": index,
            }
            return image, label_tensor, meta

        return image, label_tensor