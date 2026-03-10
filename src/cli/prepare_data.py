"""
CLI for dataset preparation:
- build manifests
- save runtime class maps
- build split files
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.constants import DEFAULT_SEED, SUPPORTED_DATASETS
from src.data.class_maps import build_class_map, save_runtime_class_map
from src.data.manifests import build_and_save_manifest
from src.data.split_builder import build_and_save_splits
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare data artifacts for MaizeFormerX.")
    parser.add_argument("--dataset", type=str, choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(args=None) -> int:
    if args is None or not hasattr(args, "dataset"):
        parser = build_parser()
        args = parser.parse_args()

    dataset_name = args.dataset
    dataset_root = args.dataset_root
    seed = args.seed

    logger.info("Preparing dataset=%s", dataset_name)

    class_map = build_class_map(dataset_name)
    manifest_path = build_and_save_manifest(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        class_to_index=class_map.class_to_index,
    )
    save_runtime_class_map(dataset_name)
    build_and_save_splits(
        manifest_csv=manifest_path,
        dataset_name=dataset_name,
        seed=seed,
    )

    logger.info("Finished preparing dataset=%s", dataset_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())