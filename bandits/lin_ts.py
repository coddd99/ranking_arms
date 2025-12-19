from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .base import Bandit


@dataclass
class LinTS(Bandit):
    lam: float = 1.0  # prior precision
    sigma2: float = 1.0  # noise variance proxy
    clip_var: float = 1e-8

    def __post_init__(self):
        d = int(self.context_dim)
        a = int(self.n_actions)
        A0 = (1.0 / float(self.lam)) * np.eye(d, dtype="float32")  # covariance ~ A_inv
        self.A_inv = np.stack([A0.copy() for _ in range(a)], axis=0)
        self.b = np.zeros((a, d), dtype="float32")

    def _theta_hat(self) -> np.ndarray:
        return np.einsum("aij,aj->ai", self.A_inv, self.b).astype("float32")

    def select(self, context: np.ndarray, rng: np.random.Generator) -> int:
        x = context.astype("float32")
        theta_hat = self._theta_hat()  # (A,d)

        # 각 action에 대해 theta ~ N(theta_hat[a], sigma2 * A_inv[a]) 샘플
       
        scores = np.empty(self.n_actions, dtype="float32")
        for a in range(self.n_actions):
            Ainv = self.A_inv[a]
            Ainv_sym = 0.5 * (Ainv + Ainv.T) # 수치 안정화
            Ainv_sym = Ainv_sym + self.clip_var * np.eye(
                Ainv_sym.shape[0], dtype=Ainv_sym.dtype
            )

            L = np.linalg.cholesky(Ainv_sym)  # Ainv = L L^T
            z = rng.standard_normal(size=(self.context_dim,)).astype("float32")
            theta_s = theta_hat[a] + np.sqrt(float(self.sigma2)) * (L @ z)
            scores[a] = float(theta_s @ x)

        return int(np.argmax(scores))

    def update(self, action: int, reward: float, context: np.ndarray) -> None:
        a = int(action)
        r = float(reward)
        x = context.astype("float32")

        Ainv = self.A_inv[a]
        Ax = Ainv @ x
        denom = 1.0 + float(x.T @ Ax)
        self.A_inv[a] = Ainv - np.outer(Ax, Ax) / denom
        self.b[a] += r * x
