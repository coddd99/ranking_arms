from __future__ import annotations

import numpy as np
import pandas as pd


def build_item_trend_features(
    interactions: pd.DataFrame, num_items: int, n_time_bins: int = 4
) -> np.ndarray:
    """
    item_trend[item] = [log(total), log(recent), log(past), recent_ratio] (4차원)
    time_norm in [0,1] 필요
    """
    bins = np.linspace(0.0, 1.0, n_time_bins + 1)
    counts = np.zeros((num_items, n_time_bins), dtype="float32")

    item_idx = interactions["item_idx"].to_numpy()
    tnorm = interactions["time_norm"].to_numpy()

    b = np.digitize(tnorm, bins[:-1], right=False) - 1
    b = np.clip(b, 0, n_time_bins - 1)

    for it, bi in zip(item_idx, b):
        counts[int(it), int(bi)] += 1.0

    total = counts.sum(axis=1) + 1e-6
    recent = counts[:, -1] + 1e-6
    past = counts[:, 0] + 1e-6

    total_log = np.log1p(total)
    recent_log = np.log1p(recent)
    past_log = np.log1p(past)
    recent_ratio = recent / total

    return np.stack([total_log, recent_log, past_log, recent_ratio], axis=1).astype(
        "float32"
    )
