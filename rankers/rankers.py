from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional


# 0으로 나누기 방지 및 안정적인 z-score 계산
def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if x.size <= 1:  # 원소가 하나면 정규화 의미 없음
        return np.zeros_like(x)
    mu = x.mean()
    sd = x.std()
    return (x - mu) / (sd + eps)


class BaseRanker:
    def __init__(self, name: str):
        self.name = name

    # 모든 자식들이 공통적으로 가져야 할 시그니처 정의
    def score(
        self,
        store,
        user_id: int,
        item_ids: np.ndarray,
        time_norm: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError

    def rank(
        self,
        store,
        user_id: int,
        candidate_item_ids: np.ndarray,
        k: int,
        time_norm: float,
        rng: np.random.Generator,
    ):
        item_ids = np.asarray(candidate_item_ids, dtype=np.int64)
        # 시그니처에 맞춰 인자 전달
        scores = self.score(store, user_id, item_ids, time_norm, rng)

        k = min(int(k), item_ids.size)
        idx_part = np.argpartition(-scores, kth=k - 1)[:k]
        top_ids = item_ids[idx_part]
        top_scores = scores[idx_part]

        order = np.lexsort((top_ids, -top_scores))
        return top_ids[order], top_scores[order]


class RandomRanker(BaseRanker):
    def score(self, store, user_id, item_ids, time_norm, rng):
        # 주입받은 rng를 사용하여 재현성 확보
        return rng.standard_normal(size=len(item_ids)).astype("float32")


class TrendRanker(BaseRanker):
    def __init__(self, name: str, w: np.ndarray, normalize: bool = True):
        super().__init__(name)
        self.w = np.asarray(w, dtype=np.float32)
        self.normalize = normalize

    def score(self, store, user_id, item_ids, time_norm, rng):
        X = store.item_trend_features(item_ids)  # (N,4)
        s = X @ self.w
        if self.normalize:
            s = _zscore(s)
        return s


class PreferenceRanker(BaseRanker):
    def __init__(self, name: str = "Preference_genre_dot", normalize: bool = True):
        super().__init__(name)
        self.normalize = normalize

    def score(self, store, user_id, item_ids, time_norm, rng):
        p = store.user_profile[int(user_id)]
        M = store.item_genre[item_ids]
        s = M @ p
        if self.normalize:  # 누락된 정규화 로직 추가
            s = _zscore(s)
        return s


@dataclass
class HybridRanker(BaseRanker):
    name: str
    w_trend: np.ndarray
    lambda_pref: float = 0.7

    # dataclass가 생성하는 __init__ 대신 부모의 name을 받기 위해 post_init 활용
    def __post_init__(self):
        super().__init__(self.name)

    def score(self, store, user_idx, item_ids, time_norm, rng):
        # 1. Trend Score
        feats = store.item_trend_features(item_ids)
        s_trend = feats @ self.w_trend
        s_trend = _zscore(s_trend)

        # 2. Preference Score
        p = store.user_profile[int(user_idx)]
        M = store.item_genre[item_ids]
        s_pref = M @ p
        s_pref = _zscore(s_pref)

        # 3. Hybrid 결합
        lam = self.lambda_pref
        return (1.0 - lam) * s_trend + lam * s_pref
