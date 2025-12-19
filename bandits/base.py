from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Bandit:
    name: str
    n_actions: int
    context_dim: int

    def select(self, context: np.ndarray, rng: np.random.Generator) -> int:
        raise NotImplementedError

    def update(self, action: int, reward: float, context: np.ndarray) -> None:
        raise NotImplementedError
