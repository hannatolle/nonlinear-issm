"""
Early stopping for iSSM training.
"""

from typing import Dict
import torch.nn as nn


class EarlyStopper:
    """Stop training when a monitored metric (lower is better) stops improving.

    Patience is counted in evaluation intervals (calls to step()), not epochs:
    with log_every=10 and patience=20, training stops after 200 epochs without
    improvement.
    """

    def __init__(self, patience: int, metric: str, min_delta: float = 0.0):
        """
        Args:
            patience:  Consecutive evaluations without improvement before stopping.
            metric:    Key in the metrics dict to monitor.
            min_delta: Minimum decrease that counts as an improvement.
        """
        self.patience = patience
        self.metric = metric
        self.min_delta = min_delta

        self.best_value = float("inf")
        self.best_epoch: int = 0
        self.counter = 0
        self._best_state: Dict[str, dict] = {}

    def step(self, metrics: dict, models: Dict[str, nn.Module]) -> bool:
        """Record metrics, checkpoint models on improvement; True once patience is exhausted."""
        value = metrics[self.metric]
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.best_epoch = metrics["epoch"]
            self.counter = 0
            self._best_state = {
                name: {k: v.clone() for k, v in m.state_dict().items()}
                for name, m in models.items()
            }
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, models: Dict[str, nn.Module]) -> None:
        """Load the best saved weights back into models."""
        for name, m in models.items():
            if name in self._best_state:
                m.load_state_dict(self._best_state[name])
