"""
CLI for single training runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.constants import MODEL_CONFIG_PATHS, TRAIN_CONFIG_DIR
from src.data.augmentations import build_train_val_test_transforms
from src.data.class_maps import build_class_map
from src.data.dataloaders import build_manifest_dataloader
from src.engine.trainer import Trainer
from src.losses.classification import build_classification_loss
from src.models.factory import build_model
from src.optim.optimizer_builder import build_optimizer
from src.optim.scheduler_builder import build_scheduler
from src.utils.config import load_yaml_config
from src.utils.device import get_device
from src.utils.io import read_csv
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a MaizeFormerX or baseline model.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--aug-config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints/manual_train")
    return parser


def main(args=None) -> int:
    if args is None or not hasattr(args, "dataset"):
        parser = build_parser()
        args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)

    train_cfg = load_yaml_config(Path(TRAIN_CONFIG_DIR) / f"{args.dataset}.yaml")
    model_cfg = load_yaml_config(MODEL_CONFIG_PATHS[args.model])
    class_map = build_class_map(args.dataset)

    aug_name = args.aug_config or train_cfg["augmentation"]["config_name"]
    aug_cfg = load_yaml_config(Path("configs/aug") / f"{aug_name}.yaml")
    transforms = build_train_val_test_transforms(aug_cfg)

    split_dir = Path("data/interim/split_files")
    train_records = read_csv(split_dir / f"{args.dataset}_train.csv")
    val_records = read_csv(split_dir / f"{args.dataset}_val.csv")

    loader_cfg = train_cfg["loader"]
    training_cfg = train_cfg["training"]

    train_loader = build_manifest_dataloader(
        train_records,
        transforms["train"],
        batch_size=loader_cfg["batch_size"],
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        persistent_workers=loader_cfg["persistent_workers"],
        shuffle=loader_cfg["shuffle_train"],
        weighted_sampling=False,
        drop_last=loader_cfg["drop_last_train"],
    )
    val_loader = build_manifest_dataloader(
        val_records,
        transforms["val"],
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

    criterion = build_classification_loss(training_cfg["criterion"], device=device)
    optimizer = build_optimizer(model, training_cfg["optimizer"])
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_cfg=training_cfg["scheduler"],
        steps_per_epoch=len(train_loader),
        max_epochs=training_cfg["epochs"],
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=len(class_map.class_to_index),
        class_names=list(class_map.class_to_index.keys()),
        training_cfg=training_cfg,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer.fit(train_loader, val_loader=val_loader, max_epochs=training_cfg["epochs"])
    logger.info("Training completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())