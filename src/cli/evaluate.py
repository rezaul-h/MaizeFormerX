"""
CLI for evaluating a trained model checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.constants import MODEL_CONFIG_PATHS, TRAIN_CONFIG_DIR
from src.data.augmentations import build_train_val_test_transforms
from src.data.class_maps import build_class_map
from src.data.dataloaders import build_manifest_dataloader
from src.engine.evaluator import Evaluator
from src.losses.classification import build_classification_loss
from src.models.factory import build_model
from src.utils.checkpoint import load_model_weights
from src.utils.config import load_yaml_config
from src.utils.device import get_device
from src.utils.io import read_csv, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--aug-config", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=str, default="outputs/reports/eval_results.json")
    return parser


def main(args=None) -> int:
    if args is None or not hasattr(args, "dataset"):
        parser = build_parser()
        args = parser.parse_args()

    device = get_device(args.device)
    train_cfg = load_yaml_config(Path(TRAIN_CONFIG_DIR) / f"{args.dataset}.yaml")
    model_cfg = load_yaml_config(MODEL_CONFIG_PATHS[args.model])
    class_map = build_class_map(args.dataset)

    aug_name = args.aug_config or train_cfg["augmentation"]["config_name"]
    aug_cfg = load_yaml_config(Path("configs/aug") / f"{aug_name}.yaml")
    transforms = build_train_val_test_transforms(aug_cfg)

    split_records = read_csv(Path("data/interim/split_files") / f"{args.dataset}_{args.split}.csv")
    loader_cfg = train_cfg["loader"]
    dataloader = build_manifest_dataloader(
        split_records,
        transforms["test"],
        batch_size=loader_cfg["batch_size"],
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        persistent_workers=loader_cfg["persistent_workers"],
        shuffle=False,
        weighted_sampling=False,
        drop_last=False,
    )

    model = build_model(args.model, model_cfg, num_classes=len(class_map.class_to_index))
    model.to(device)
    load_model_weights(args.checkpoint, model, map_location=device, strict=True)

    criterion = build_classification_loss(train_cfg["training"]["criterion"], device=device)
    evaluator = Evaluator(
        model=model,
        criterion=criterion,
        device=device,
        num_classes=len(class_map.class_to_index),
        class_names=list(class_map.class_to_index.keys()),
        use_amp=train_cfg["training"].get("precision", {}).get("amp", True),
    )
    result = evaluator.evaluate(dataloader)
    write_json(args.output_json, result)

    logger.info("Saved evaluation results to %s", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())