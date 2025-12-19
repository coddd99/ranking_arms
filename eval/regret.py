from __future__ import annotations
import numpy as np


def best_fixed_action(reward_matrix: np.ndarray) -> int:
    """reward_matrix[t, a]"""
    sums = reward_matrix.sum(axis=0)
    return int(np.argmax(sums))


def cumulative(x: np.ndarray) -> np.ndarray:
    return np.cumsum(x).astype("float32")
