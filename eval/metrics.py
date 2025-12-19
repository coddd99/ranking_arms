from __future__ import annotations
import numpy as np


def dcg_reward_from_rank(pos0: int, k: int) -> float:
    if pos0 < 0 or pos0 >= int(k):
        return 0.0
    return float(1.0 / np.log2(pos0 + 2.0))


def hit_at_k(pos0: int, k: int) -> float:
    return 1.0 if (0 <= pos0 < int(k)) else 0.0
