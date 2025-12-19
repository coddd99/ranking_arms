from __future__ import annotations

import numpy as np
import pandas as pd


def build_user_profiles(
    interactions: pd.DataFrame, item_genre: np.ndarray, num_users: int
) -> np.ndarray:
    """
    user_profile[u] = 장르 분포(합=1)
    interactions: (user_idx, item_idx) 포함
    """
    num_genres = item_genre.shape[1]
    prof = np.zeros((num_users, num_genres), dtype="float32")
    for u, it in zip(
        interactions["user_idx"].to_numpy(), interactions["item_idx"].to_numpy()
    ):
        prof[int(u)] += item_genre[int(it)]
    row_sum = prof.sum(axis=1, keepdims=True) + 1e-6
    return prof / row_sum
