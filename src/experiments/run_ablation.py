"""
Ablation study experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.constants import EXPERIMENT_CONFIG_PATHS, MODEL_CONFIG_PATHS, OUTPUT_CHECKPOINTS_DIR, OUTPUT_METRICS_DIR
from src.data.augmentations import build_train_val_test_transforms
from src.data.class_maps import build_class_map
from src.data.dataloaders import build_manifest_dataloader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.losses.classification import build_classification_loss
from src.metrics.aggregation import aggregate_metrics
from src.models.factory import build_ablation_model
from src.optim.optimizer_builder import build_optimizer
from src.optim.scheduler_builder import build_scheduler
from src.utils.config import load_yaml_config
from src.utils.device import get_device
from src.utils.io import read_csv, write_json
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

logger = get_logger(__name__)


def run_ablation_experiment(
    experiment_config_path: str | Path | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    exp_cfg = load_yaml_config(experiment_config_path or EXPERIMENT_CONFIG_PATHS["ablation"])
    base_model_cfg = load_yaml_config(MODEL_CONFIG_PATHS["maizeformerx"])

    results: dict[str, Any] = {}

    for dataset_name in exp_cfg["dataset_priority"]:
        class_map = build_class_map(dataset_name)
        train_cfg = load_yaml_config(Path("configs/train") / f"{dataset_name}.yaml")
        loader_cfg = train_cfg["loader"]
        training_cfg = train_cfg["training"]

        aug_name = exp_cfg["augmentation_for_training"][dataset_name]
        aug_cfg = load_yaml_config(Path("configs/aug") / f"{aug_name}.yaml")
        transforms = build_train_val_test_transforms(aug_cfg)

        train_records = read_csv(Path("data/interim/split_files") / f"{dataset_name}_train.csv")
        val_records = read_csv(Path("data/interim/split_files") / f"{dataset_name}_val.csv")
        test_records = read_csv(Path("data/interim/split_files") / f"{dataset_name}_test.csv")

        train_loader = build_manifest_dataloader(
            train_records, transforms["train"],
            batch_size=loader_cfg["batch_size"], num_workers=loader_cfg["num_workers"],
            pin_memory=loader_cfg["pin_memory"], persistent_workers=loader_cfg["persistent_workers"],
            shuffle=loader_cfg["shuffle_train"], weighted_sampling=False,
            drop_last=loader_cfg["drop_last_train"],
        )
        val_loader = build_manifest_dataloader(
            val_records, transforms["val"],
            batch_size=loader_cfg["batch_size"], num_workers=loader_cfg["num_workers"],
            pin_memory=loader_cfg["pin_memory"], persistent_workers=loader_cfg["persistent_workers"],
            shuffle=False, weighted_sampling=False, drop_last=False,
        )
        test_loader = build_manifest_dataloader(
            test_records, transforms["test"],
            batch_size=loader_cfg["batch_size"], num_workers=loader_cfg["num_workers"],
            pin_memory=loader_cfg["pin_memory"], persistent_workers=loader_cfg["persistent_workers"],
            shuffle=False, weighted_sampling=False, drop_last=False,
        )

        results[dataset_name] = {}

        for ablation_spec in exp_cfg["ablations"]:
            ablation_name = ablation_spec["name"]
            logger.info("Running ablation: dataset=%s, ablation=%s", dataset_name, ablation_name)

            run_rows = []
            metric_rows = []

            for seed in exp_cfg["execution"]["seeds"]:
                seed_everything(seed)
                device = get_device(device_name)

                model = build_ablation_model(ablation_name, base_model_cfg, len(class_map.class_to_index))
                model.to(device)

                criterion = build_classification_loss(training_cfg["criterion"], device=device)
                optimizer = build_optimizer(model, training_cfg["optimizer"])
                scheduler = build_scheduler(
                    optimizer=optimizer,
                    scheduler_cfg=training_cfg["scheduler"],
                    steps_per_epoch=len(train_loader),
                    max_epochs=training_cfg["epochs"],
                )

                checkpoint_dir = OUTPUT_CHECKPOINTS_DIR / "ablation" / dataset_name / ablation_name / f"seed_{seed}"
                trainer = Trainer(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    num_classes=len(class_map.class_to_index),
                    class_names=list(class_map.class_to_index.keys()),
                    training_cfg=training_cfg,
                    checkpoint_dir=checkpoint_dir,
                )
                trainer.fit(train_loader, val_loader=val_loader, max_epochs=training_cfg["epochs"])

                evaluator = Evaluator(
                    model=trainer.model,
                    criterion=criterion,
                    device=device,
                    num_classes=len(class_map.class_to_index),
                    class_names=list(class_map.class_to_index.keys()),
                    use_amp=training_cfg.get("precision", {}).get("amp", True),
                )
                test_out = evaluator.evaluate(test_loader)

                row = {"seed": seed, "metrics": test_out["metrics"]}
                run_rows.append(row)
                metric_rows.append(test_out["metrics"])

            results[dataset_name][ablation_name] = {
                "runs": run_rows,
                "aggregated_test_metrics": aggregate_metrics(metric_rows),
            }

    output_path = OUTPUT_METRICS_DIR / "ablation_results.json"
    write_json(output_path, results)
    logger.info("Saved ablation results to %s", output_path)
    return results