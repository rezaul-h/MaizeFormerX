"""
Training hooks / callbacks.
"""

from __future__ import annotations

from typing import Any


class Hook:
    def on_train_start(self, trainer) -> None:
        pass

    def on_train_end(self, trainer) -> None:
        pass

    def on_epoch_start(self, trainer, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer, epoch: int, logs: dict[str, Any]) -> None:
        pass

    def on_batch_end(self, trainer, step: int, logs: dict[str, Any]) -> None:
        pass


class HookManager:
    def __init__(self, hooks: list[Hook] | None = None) -> None:
        self.hooks = hooks or []

    def on_train_start(self, trainer) -> None:
        for hook in self.hooks:
            hook.on_train_start(trainer)

    def on_train_end(self, trainer) -> None:
        for hook in self.hooks:
            hook.on_train_end(trainer)

    def on_epoch_start(self, trainer, epoch: int) -> None:
        for hook in self.hooks:
            hook.on_epoch_start(trainer, epoch)

    def on_epoch_end(self, trainer, epoch: int, logs: dict[str, Any]) -> None:
        for hook in self.hooks:
            hook.on_epoch_end(trainer, epoch, logs)

    def on_batch_end(self, trainer, step: int, logs: dict[str, Any]) -> None:
        for hook in self.hooks:
            hook.on_batch_end(trainer, step, logs)