"""
Early stopping utility.
"""

from __future__ import annotations


class EarlyStopping:
    """
    Track validation metric and stop when improvement stalls.
    """

    def __init__(
        self,
        patience: int = 4,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode

        self.best_score: float | None = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True

        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float) -> bool:
        if self._is_improvement(score):
            self.best_score = float(score)
            self.num_bad_epochs = 0
            self.should_stop = False
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True

        return self.should_stop