from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set
import numpy as np
import pandas as pd
import warnings

from ..features.store import FeatureStore
from ..rankers.rankers import BaseRanker as Ranker


@dataclass
class StepBatch:
    t: int
    user_idx: int
    pos_item: int
    time_norm: float
    cand_items: np.ndarray  # (C,)


class OfflineRankEnv:
    """
    오프라인 로그(positive 이벤트)로 bandit-over-rankers를 시뮬레이션하는 환경

    - Candidate는 초기화 시점에 전부 고정(precompute)
    - Oracle/Best-fixed 계산과 Bandit 실행이 같은 후보군을 공유하여 평가가 일관되게 함
    - negative는 (user, 시간 흐름) 기준으로 아직 보지 않은 아이템에서 샘플링
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        store: FeatureStore,
        rankers: List[Ranker],
        n_steps: int,
        n_candidates: int,
        seed: int,
    ):
        self.df = interactions.reset_index(drop=True)
        self.store = store
        self.rankers = list(rankers)

        if n_steps > len(self.df):
            warnings.warn(
                f"n_steps ({n_steps})> data length ({len(self.df)}). "
                "n-steps를 data length로 설정합니다.",
                UserWarning,
            )

        self.n_steps = min(int(n_steps), len(self.df))
        self.n_candidates = n_candidates
        self.rng = np.random.default_rng(int(seed))

        self.num_users = int(self.df["user_idx"].max()) + 1
        self.num_items = int(store.item_genre.shape[0])
        self.n_actions = len(self.rankers)

        self.steps: List[StepBatch] = self._precompute_steps()

    def context(self, user_idx: int, time_norm: float) -> np.ndarray:
        return self.store.context_vector(user_idx, time_norm)

    def _precompute_steps(self) -> List[StepBatch]:
        C = self.n_candidates
        steps: List[StepBatch] = []
        user_seen: List[Set[int]] = [set() for _ in range(self.num_users)]

        for t in range(self.n_steps):
            row = self.df.iloc[int(t)]
            u = int(row["user_idx"])
            pos = int(row["item_idx"])
            tn = float(row["time_norm"])

            cand = np.empty(C, dtype="int64")
            cand[0] = pos

            seen = user_seen[u]
            # negative는 "그 시점까지" user가 보지 않은 아이템에서 샘플
            j = 1
            while j < C:
                it = int(self.rng.integers(low=0, high=self.num_items))
                if it == pos:
                    continue
                if it in seen:
                    continue
                cand[j] = it
                j += 1

            steps.append(
                StepBatch(t=t, user_idx=u, pos_item=pos, time_norm=tn, cand_items=cand)
            )
            # time 흐름에 따라 seen 갱신
            seen.add(pos)

        return steps

    def step(self, t: int) -> StepBatch:
        return self.steps[int(t)]
