"""
Cross-domain shared-label evaluation experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.constants import (
    EXPERIMENT_CONFIG_PATHS,
    MODEL_CONFIG_PATHS,
    OUTPUT_CHECKPOINTS_DIR,
    OUTPUT_METRICS_DIR,
    TRAIN_CONFIG_DIR,
)
from src.data.augmentations import build_train_val_test_transforms
from src.data.class_maps import build_class_map
from src.data.dataloaders import build_manifest_dataloader, build_shared_label_dataloader
from src.data.shared_label_protocol import (
    build_shared_label_index,
    filter_records_to_shared_labels,
    get_shared_label_spec,
)
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.losses.classification import build_classification_loss
from src.metrics.aggregation import aggregate_metrics
from src.models.factory import build_model
from src.optim.optimizer_builder import build_optimizer
from src.optim.scheduler_builder import build_scheduler
from src.utils.config import load_yaml_config
from src.utils.device import get_device
from src.utils.io import read_csv, write_json
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

logger = get_logger(__name__)


def _load_split_records(dataset_name: str, split_name: str) -> list[dict]:
    return read_csv(Path("data/interim/split_files") / f"{dataset_name}_{split_name}.csv")


def _train_source_model(
    source_dataset: str,
    model_name: str,
    aug_name: str,
    seed: int,
    device_name: str,
):
    train_cfg = load_yaml_config(TRAIN_CONFIG_DIR / f"{source_dataset}.yaml")
    model_cfg = load_yaml_config(MODEL_CONFIG_PATHS[model_name])
    aug_cfg = load_yaml_config(Path("configs/aug") / f"{aug_name}.yaml")

    class_map = build_class_map(source_dataset)
    transforms = build_train_val_test_transforms(aug_cfg)
    train_records = _load_split_records(source_dataset, "train")
    val_records = _load_split_records(source_dataset, "val")

    loader_cfg = train_cfg["loader"]
    training_cfg = train_cfg["training"]

    train_loader = build_manifest_dataloader(
        records=train_records,
        transform=transforms["train"],
        batch_size=loader_cfg["batch_size"],
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        persistent_workers=loader_cfg["persistent_workers"],
        shuffle=loader_cfg["shuffle_train"],
        weighted_sampling=False,
        drop_last=loader_cfg["drop_last_train"],
    )
    val_loader = build_manifest_dataloader(
        records=val_records,
        transform=transforms["val"],
        batch_size=loader_cfg["batch_size"],
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        persistent_workers=loader_cfg["persistent_workers"],
        shuffle=False,
        weighted_sampling=False,
        drop_last=False,
    )

    device = get_device(device_name)
    seed_everything(seed)

    model = build_model(model_name=model_name, model_cfg=model_cfg, num_classes=len(class_map.class_to_index))
    model.to(device)

    criterion = build_classification_loss(training_cfg["criterion"], device=device)
    optimizer = build_optimizer(model, training_cfg["optimizer"])
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_cfg=training_cfg["scheduler"],
        steps_per_epoch=len(train_loader),
        max_epochs=training_cfg["epochs"],
    )

    checkpoint_dir = OUTPUT_CHECKPOINTS_DIR / "cross_domain" / source_dataset / model_name / f"seed_{seed}"
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=len(class_map.class_to_index),
        class_names=list(class_map.class_to_index.keys()),
        training_cfg=train_cfg["training"],
        checkpoint_dir=checkpoint_dir,
    )
    trainer.fit(train_loader, val_loader=val_loader, max_epochs=training_cfg["epochs"])

    return trainer.model, device, training_cfg


def run_cross_domain_experiment(
    experiment_config_path: str | Path | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    exp_cfg = load_yaml_config(experiment_config_path or EXPERIMENT_CONFIG_PATHS["cross_domain"])
    results: dict[str, Any] = {}

    for pair in exp_cfg["source_target_pairs"]:
        source_dataset = pair["source"]
        target_dataset = pair["target"]
        pair_key = f"{source_dataset}__{target_dataset}"
        results[pair_key] = {}

        spec = get_shared_label_spec(source_dataset, target_dataset)
        shared_label_to_index = build_shared_label_index(spec.shared_labels)
        if len(shared_label_to_index) == 0:
            logger.warning("Skipping pair with no shared labels: %s", pair_key)
            continue

        target_records = _load_split_records(target_dataset, "test")
        target_records = filter_records_to_shared_labels(target_records, dataset_role="target", spec=spec)

        for model_name in exp_cfg["models"]:
            logger.info("Running cross-domain pair=%s model=%s", pair_key, model_name)
            run_rows = []
            metric_rows = []

            for seed in exp_cfg["execution"]["seeds"]:
                source_aug = exp_cfg["augmentation_for_source_training"][source_dataset]
                model, device, training_cfg = _train_source_model(
                    source_dataset=source_dataset,
                    model_name=model_name,
                    aug_name=source_aug,
                    seed=seed,
                    device_name=device_name,
                )

                aug_cfg = load_yaml_config(Path("configs/aug") / f"{source_aug}.yaml")
                transforms = build_train_val_test_transforms(aug_cfg)
                test_loader = build_shared_label_dataloader(
                    records=target_records,
                    shared_label_to_index=shared_label_to_index,
                    transform=transforms["test"],
                    batch_size=32,
                    num_workers=8,
                    pin_memory=True,
                    persistent_workers=True,
                    shuffle=False,
                    drop_last=False,
                )

                criterion = build_classification_loss({"name": "cross_entropy"}, device=device)
                evaluator = Evaluator(
                    model=model,
                    criterion=criterion,
                    device=device,
                    num_classes=len(shared_label_to_index),
                    class_names=list(shared_label_to_index.keys()),
                    use_amp=training_cfg.get("precision", {}).get("amp", True),
                )
                out = evaluator.evaluate(test_loader)

                row = {
                    "seed": seed,
                    "metrics": out["metrics"],
                    "shared_labels": list(shared_label_to_index.keys()),
                }
                run_rows.append(row)
                metric_rows.append(out["metrics"])

            results[pair_key][model_name] = {
                "runs": run_rows,
                "aggregated_test_metrics": aggregate_metrics(metric_rows),
            }

    output_path = OUTPUT_METRICS_DIR / "cross_domain_results.json"
    write_json(output_path, results)
    logger.info("Saved cross-domain results to %s", output_path)
    return results