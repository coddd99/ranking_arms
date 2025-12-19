from __future__ import annotations

from typing import List, Dict
import numpy as np
import pandas as pd

# MovieLens 장르 문자열을 위한 기본 리스트 (데이터셋 바뀌면 교체/확장 가능)
DEFAULT_GENRES: List[str] = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def build_item_genre_matrix(
    movies_df: pd.DataFrame, num_items: int, genres: List[str] = DEFAULT_GENRES
) -> np.ndarray:
    """
    item_genre[item, g] = 1/0
    movies_df는 최소 (item_idx, genres) 컬럼이 있어야 함.
    """
    g2i: Dict[str, int] = {g: i for i, g in enumerate(genres)}
    mat = np.zeros((num_items, len(genres)), dtype="float32")

    for _, row in movies_df.iterrows():
        item = int(row["item_idx"])
        gs = str(row["genres"]).split("|")
        for g in gs:
            if g in g2i:
                mat[item, g2i[g]] = 1.0
    return mat
