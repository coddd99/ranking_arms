from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .base import Bandit


@dataclass
class LinUCB(Bandit):
    alpha: float = 0.5
    gamma: float = 1.0  # <1이면 드리프트 대응: 과거를 할인

    def __post_init__(self):
        d = int(self.context_dim)
        a = int(self.n_actions)
        self.A_inv = np.stack(
            [np.eye(d, dtype="float32") for _ in range(a)], axis=0
        )  # (A,d,d)
        self.b = np.zeros((a, d), dtype="float32")

    def _theta(self) -> np.ndarray:
        return np.einsum("aij,aj->ai", self.A_inv, self.b).astype("float32")

    def select(self, context: np.ndarray, rng: np.random.Generator) -> int:
        x = context.astype(np.float64)
        theta = self._theta()  # (A,d)
        mean = theta @ x  # (A,)
        tmp = self.A_inv @ x  # (A,d)
        var = (tmp * x).sum(axis=1)  # (A,)
        bonus = float(self.alpha) * np.sqrt(np.clip(var, 1e-12, None))
        p = mean + bonus
        return int(np.argmax(p))

    def update(self, action: int, reward: float, context: np.ndarray) -> None:
        a = int(action)
        r = float(reward)
        x = context.astype(np.float64)
        g = float(self.gamma)

        # 할인 적용: b <- gamma*b (과거 영향 감소)
        if g < 1.0:
            self.b[a] *= g
            Ainv = self.A_inv[a] / g
        else:
            Ainv = self.A_inv[a]

        Ax = Ainv @ x
        denom = 1.0 + float(x.T @ Ax)
        self.A_inv[a] = Ainv - np.outer(Ax, Ax) / denom
        self.b[a] += r * x
