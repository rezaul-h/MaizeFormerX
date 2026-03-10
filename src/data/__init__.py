"""
Data-layer exports for MaizeFormerX.
"""

from src.data.augmentations import build_train_val_test_transforms, build_transform_from_section
from src.data.class_maps import ClassMap, build_class_map, load_dataset_class_metadata, save_runtime_class_map
from src.data.dataloaders import (
    build_dataloader,
    build_manifest_dataloader,
    build_shared_label_dataloader,
)
from src.data.datasets import ManifestImageDataset, SharedLabelDataset
from src.data.manifests import build_and_save_manifest, build_manifest_from_folder_tree, save_manifest
from src.data.preprocess import build_base_preprocess, ensure_rgb, load_pil_image, maybe_apply_transform
from src.data.samplers import build_weighted_sampler, compute_class_weights_from_records
from src.data.shared_label_protocol import (
    SharedLabelSpec,
    build_shared_label_index,
    filter_records_to_shared_labels,
    get_shared_label_spec,
)
from src.data.split_builder import (
    attach_split_column,
    build_and_save_splits,
    build_stratified_splits,
    save_split_files,
    summarize_split_counts,
)
from src.data.validators import (
    summarize_class_distribution,
    validate_class_index_consistency,
    validate_dataset_root,
    validate_manifest_columns,
    validate_manifest_paths,
    validate_non_empty_split,
    validate_split_disjointness,
)

__all__ = [
    "ClassMap",
    "ManifestImageDataset",
    "SharedLabelDataset",
    "SharedLabelSpec",
    "attach_split_column",
    "build_and_save_manifest",
    "build_and_save_splits",
    "build_base_preprocess",
    "build_class_map",
    "build_dataloader",
    "build_manifest_dataloader",
    "build_manifest_from_folder_tree",
    "build_shared_label_index",
    "build_stratified_splits",
    "build_train_val_test_transforms",
    "build_transform_from_section",
    "build_weighted_sampler",
    "build_shared_label_dataloader",
    "compute_class_weights_from_records",
    "ensure_rgb",
    "filter_records_to_shared_labels",
    "get_shared_label_spec",
    "load_dataset_class_metadata",
    "load_pil_image",
    "maybe_apply_transform",
    "save_manifest",
    "save_runtime_class_map",
    "save_split_files",
    "summarize_class_distribution",
    "summarize_split_counts",
    "validate_class_index_consistency",
    "validate_dataset_root",
    "validate_manifest_columns",
    "validate_manifest_paths",
    "validate_non_empty_split",
    "validate_split_disjointness",
]