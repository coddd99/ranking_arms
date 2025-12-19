from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .interactions import Interactions


def load_movielens10m_events(
    data_dir: str,
    min_rating: float,
    min_item_interactions: int,
) -> Tuple[Interactions, Dict[int, int], Dict[int, int], pd.DataFrame]:
    ratings_path = os.path.join(data_dir, "ratings.dat")
    movies_path = os.path.join(data_dir, "movies.dat")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings.dat not found: {ratings_path}")
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"movies.dat not found: {movies_path}")

    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_raw", "item_raw", "rating", "timestamp"],
    )

    df = df[df["rating"] >= float(min_rating)].copy()

    vc_raw = df["item_raw"].value_counts()
    keep_raw = vc_raw[vc_raw >= int(min_item_interactions)].index
    df = df[df["item_raw"].isin(keep_raw)].copy()

    df["user_idx"], user_uniques = pd.factorize(df["user_raw"])
    df["item_idx"], item_uniques = pd.factorize(df["item_raw"])

    ts = df["timestamp"].astype("int64").to_numpy()
    ts_min, ts_max = int(ts.min()), int(ts.max())
    if ts_max == ts_min:
        df["time_norm"] = np.zeros_like(ts, dtype="float32")
    else:
        df["time_norm"] = ((ts - ts_min) / (ts_max - ts_min)).astype("float32")

    df["label"] = 1.0
    df = df.sort_values("timestamp").reset_index(drop=True)

    # raw -> idx 맵
    user_id_map = {int(raw): int(i) for i, raw in enumerate(user_uniques)}
    item_id_map = {int(raw): int(i) for i, raw in enumerate(item_uniques)}

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["item_raw", "title", "genres"],
    )

    movies = movies[movies["item_raw"].isin(item_uniques)].copy()
    movies["item_idx"] = movies["item_raw"].map(item_id_map).astype("int64")

    num_users = len(user_uniques)
    num_items = len(item_uniques)

    interactions = Interactions(
        df=df[["user_idx", "item_idx", "timestamp", "time_norm", "label"]].copy(),
        num_users=num_users,
        num_items=num_items,
    )
    return interactions, user_id_map, item_id_map, movies
