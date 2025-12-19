from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .base import Bandit


@dataclass
class RandomBandit(Bandit):
    def select(self, context: np.ndarray, rng: np.random.Generator) -> int:
        return int(rng.integers(low=0, high=self.n_actions))

    def update(self, action: int, reward: float, context: np.ndarray) -> None:
        return
