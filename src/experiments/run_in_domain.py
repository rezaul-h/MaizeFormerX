"""
In-domain benchmark experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.constants import (
    DEFAULT_MULTI_SEEDS,
    EXPERIMENT_CONFIG_PATHS,
    MODEL_CONFIG_PATHS,
    OUTPUT_CHECKPOINTS_DIR,
    OUTPUT_METRICS_DIR,
    TRAIN_CONFIG_DIR,
)
from src.data.augmentations import build_train_val_test_transforms
from src.data.class_maps import build_class_map
from src.data.dataloaders import build_manifest_dataloader
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
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


def _load_split_records(dataset_name: str) -> tuple[list[dict], list[dict], list[dict]]:
    split_dir = Path("data/interim/split_files")
    train_records = read_csv(split_dir / f"{dataset_name}_train.csv")
    val_records = read_csv(split_dir / f"{dataset_name}_val.csv")
    test_records = read_csv(split_dir / f"{dataset_name}_test.csv")
    return train_records, val_records, test_records


def _prepare_single_run(
    dataset_name: str,
    model_name: str,
    aug_config_name: str,
    seed: int,
    device_name: str = "auto",
):
    train_cfg = load_yaml_config(TRAIN_CONFIG_DIR / f"{dataset_name}.yaml")
    model_cfg = load_yaml_config(MODEL_CONFIG_PATHS[model_name])
    aug_cfg = load_yaml_config(Path("configs/aug") / f"{aug_config_name}.yaml")

    class_map = build_class_map(dataset_name)
    transforms = build_train_val_test_transforms(aug_cfg)
    train_records, val_records, test_records = _load_split_records(dataset_name)

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
        return_metadata=False,
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
        return_metadata=False,
    )
    test_loader = build_manifest_dataloader(
        records=test_records,
        transform=transforms["test"],
        batch_size=loader_cfg["batch_size"],
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        persistent_workers=loader_cfg["persistent_workers"],
        shuffle=False,
        weighted_sampling=False,
        drop_last=False,
        return_metadata=False,
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

    checkpoint_dir = OUTPUT_CHECKPOINTS_DIR / "in_domain" / dataset_name / model_name / aug_config_name / f"seed_{seed}"
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

    train_out = trainer.fit(train_loader, val_loader=val_loader, max_epochs=training_cfg["epochs"])

    evaluator = Evaluator(
        model=trainer.model,
        criterion=criterion,
        device=device,
        num_classes=len(class_map.class_to_index),
        class_names=list(class_map.class_to_index.keys()),
        use_amp=training_cfg.get("precision", {}).get("amp", True),
    )
    test_out = evaluator.evaluate(test_loader)

    return {
        "seed": seed,
        "train": train_out,
        "test": test_out,
    }


def run_in_domain_experiment(
    experiment_config_path: str | Path | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    """
    Run the full in-domain benchmark.
    """
    exp_cfg = load_yaml_config(experiment_config_path or EXPERIMENT_CONFIG_PATHS["in_domain"])

    all_results: dict[str, Any] = {}

    for dataset_name in exp_cfg["datasets"]:
        all_results[dataset_name] = {}

        for model_name in exp_cfg["models"]:
            all_results[dataset_name][model_name] = {}

            for aug_name in exp_cfg["augmentation_configs"][dataset_name]:
                logger.info(
                    "Running in-domain experiment: dataset=%s, model=%s, aug=%s",
                    dataset_name,
                    model_name,
                    aug_name,
                )

                run_results = []
                metric_rows = []

                for seed in exp_cfg["execution"].get("seeds", list(DEFAULT_MULTI_SEEDS)):
                    single = _prepare_single_run(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        aug_config_name=aug_name,
                        seed=seed,
                        device_name=device_name,
                    )
                    run_results.append(single)
                    metric_rows.append(single["test"]["metrics"])

                aggregated = aggregate_metrics(metric_rows)
                all_results[dataset_name][model_name][aug_name] = {
                    "runs": run_results,
                    "aggregated_test_metrics": aggregated,
                }

    output_path = OUTPUT_METRICS_DIR / "in_domain_results.json"
    write_json(output_path, all_results)
    logger.info("Saved in-domain experiment results to %s", output_path)
    return all_results