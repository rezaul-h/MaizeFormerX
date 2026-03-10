"""
Training loop for classification models.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.engine.early_stopping import EarlyStopping
from src.engine.evaluator import Evaluator
from src.engine.hooks import HookManager
from src.metrics.aggregation import aggregate_metrics
from src.optim.ema import ModelEMA
from src.utils.checkpoint import save_checkpoint
from src.utils.device import move_to_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Generic trainer for classification tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        num_classes: int,
        class_names: list[str] | None = None,
        training_cfg: dict[str, Any] | None = None,
        checkpoint_dir: str | Path | None = None,
        hooks: list | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        self.training_cfg = training_cfg or {}
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

        precision_cfg = self.training_cfg.get("precision", {})
        self.use_amp = bool(precision_cfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        regularization_cfg = self.training_cfg.get("regularization", {})
        self.grad_clip_norm = regularization_cfg.get("grad_clip_norm", None)

        ema_cfg = regularization_cfg.get("ema", {})
        self.ema = ModelEMA(model, decay=float(ema_cfg.get("decay", 0.999))).to(device) if ema_cfg.get("enabled", False) else None

        early_cfg = self.training_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=int(early_cfg.get("patience", 4)),
            min_delta=float(early_cfg.get("min_delta", 0.0)),
            mode=early_cfg.get("mode", "max"),
        ) if early_cfg.get("enabled", True) else None

        self.monitor_metric = self.training_cfg.get("checkpointing", {}).get("metric", "val_accuracy")
        self.monitor_mode = self.training_cfg.get("checkpointing", {}).get("mode", "max")

        self.hooks = HookManager(hooks)

        self.best_metric: float | None = None
        self.best_state_dict = None
        self.history: list[dict[str, Any]] = []

    def _is_better(self, new_value: float, old_value: float | None) -> bool:
        if old_value is None:
            return True
        if self.monitor_mode == "max":
            return new_value > old_value
        return new_value < old_value

    def _train_one_epoch(self, dataloader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for step, batch in enumerate(dataloader):
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            inputs = move_to_device(inputs, self.device)
            targets = move_to_device(targets, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()

            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.grad_clip_norm))

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step()

            if self.ema is not None:
                self.ema.update(self.model)

            batch_size = inputs.size(0)
            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            self.hooks.on_batch_end(
                self,
                step=step,
                logs={
                    "epoch": epoch,
                    "batch_loss": float(loss.item()),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                },
            )

        epoch_loss = running_loss / max(1, total_samples)
        return {"train_loss": epoch_loss}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_epochs: int = 30,
    ) -> dict[str, Any]:
        self.model.to(self.device)
        self.hooks.on_train_start(self)

        for epoch in range(1, max_epochs + 1):
            self.hooks.on_epoch_start(self, epoch)

            train_logs = self._train_one_epoch(train_loader, epoch)
            epoch_logs = {"epoch": epoch, **train_logs}

            if val_loader is not None:
                eval_model = self.ema.ema_model if self.ema is not None else self.model
                evaluator = Evaluator(
                    model=eval_model,
                    criterion=self.criterion,
                    device=self.device,
                    num_classes=self.num_classes,
                    class_names=self.class_names,
                    use_amp=self.use_amp,
                )
                val_result = evaluator.evaluate(val_loader)

                for key, value in val_result["metrics"].items():
                    epoch_logs[f"val_{key}"] = value

                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric_name = self.monitor_metric.replace("val_", "")
                    self.scheduler.step(val_result["metrics"].get(metric_name, val_result["metrics"]["accuracy"]))

                monitor_value = epoch_logs.get(self.monitor_metric)
                if monitor_value is None:
                    fallback_metric = self.monitor_metric.replace("val_", "")
                    monitor_value = val_result["metrics"].get(fallback_metric)

                if self._is_better(float(monitor_value), self.best_metric):
                    self.best_metric = float(monitor_value)
                    state_source = self.ema.ema_model if self.ema is not None else self.model
                    self.best_state_dict = deepcopy(state_source.state_dict())

                    if self.checkpoint_dir is not None:
                        save_checkpoint(
                            path=self.checkpoint_dir / "best.pt",
                            model=state_source,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            epoch=epoch,
                            metrics=epoch_logs,
                            config=self.training_cfg,
                            extra={"best_metric": self.best_metric},
                        )

                if self.early_stopping is not None:
                    should_stop = self.early_stopping.step(float(monitor_value))
                    if should_stop:
                        self.history.append(epoch_logs)
                        self.hooks.on_epoch_end(self, epoch, epoch_logs)
                        logger.info("Early stopping triggered at epoch %d", epoch)
                        break

            if self.checkpoint_dir is not None:
                save_checkpoint(
                    path=self.checkpoint_dir / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    metrics=epoch_logs,
                    config=self.training_cfg,
                )

            self.history.append(epoch_logs)
            self.hooks.on_epoch_end(self, epoch, epoch_logs)

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        self.hooks.on_train_end(self)
        return {
            "history": self.history,
            "best_metric": self.best_metric,
            "aggregated_history": aggregate_metrics(
                [{k: v for k, v in row.items() if isinstance(v, (float, int))} for row in self.history]
            ) if self.history else {},
        }