from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
from collections import deque
import numpy as np


@dataclass
class FeatureStore:
    """
    user-profile:과거 상호작용(<=t-1)까지 누적해 갱신함
    item-trend: 최근 W-step 윈도우 기준으로 'recent'를 계산
    아이템 장르, 유저수, 아이템수, 윈도우사이즈, 페르소나여부, 유저별 장르 카운팅,
    유저별 프로필, 아이템 전체 클릭횟수, 최근 클릭 횟수
    """

    # user_profile: np.ndarray        # (U, G)
    item_genre: np.ndarray  # (I, G)
    num_users: int
    num_items: int
    trend_window_steps: int = 20_000
    persona: Optional[np.ndarray] = None  # (U, P) optional

    def __post_init__(self):
        U, I = int(self.num_users), int(self.num_items)
        G = int(self.item_genre.shape[1])

        self._user_genre_cnt = np.zeros((U, G), dtype="float32")
        self.user_profile = np.zeros((U, G), dtype="float32")

        self._item_total = np.zeros(I, dtype="float32")
        self._item_recent_steps: List[deque] = [deque() for _ in range(I)]

        self.current_step: int = 0

    @property
    def context_dim(self) -> int:
        d = int(self.user_profile.shape[1]) + 3
        if self.persona is not None:
            d += int(self.persona.shape[1])
        return d

    def set_step(self, t: int) -> None:
        self.current_step = int(t)

    def _update_user_profile(self, u: int) -> None:
        cnt = self._user_genre_cnt[u]
        s = float(cnt.sum()) + 1e-6
        self.user_profile[u] = (cnt / s).astype("float32")

    def update_from_event(self, user_idx: int, pos_item: int) -> None:
        """
        step t가 끝난 뒤 호출.
        이 이벤트는 "로그에 실제로 있었던 positive" 이므로
        미래 누수 없이 t+1 이후 컨텍스트에 반영된다.
        """
        u = user_idx
        it = pos_item
        t = self.current_step
        W = self.trend_window_steps

        # 1) 유저 프로필 갱신 (누적)
        self._user_genre_cnt[u] += self.item_genre[it]
        self._update_user_profile(u)

        # 2) 아이템 트렌드 갱신
        self._item_total[it] += 1.0
        dq = self._item_recent_steps[it]
        dq.append(t)

        # recent window 밖 제거 (해당 아이템 dq만 정리하면 충분)
        cutoff = t - W
        while dq and dq[0] <= cutoff:
            dq.popleft()

    def reset(self) -> None:
        self._user_genre_cnt.fill(0.0)
        self.user_profile.fill(0.0)
        self._item_total.fill(0.0)
        for dq in self._item_recent_steps:
            dq.clear()
        self.current_step = 0

    def item_trend_features(self, items: np.ndarray) -> np.ndarray:
        """
        items: (C,)
        반환: (C,4) = [log(total), log(recent), log(past), recent_ratio]
        - recent: 최근 W-step 내 등장 횟수
        - past: total - recent
        """
        t = self.current_step
        W = self.trend_window_steps
        eps = 1e-6

        items = items.astype("int64", copy=False)
        total = self._item_total[items]

        recent = np.zeros_like(total, dtype="float32")
        cutoff = t - W

        # 후보가 50개라 여기 loop는 매우 싸다
        for j, it in enumerate(items):
            dq = self._item_recent_steps[int(it)]
            while dq and dq[0] <= cutoff:
                dq.popleft()
            recent[j] = float(len(dq))  # 최근 W동안 it 아이템이 클릭된 횟수

        past = np.clip(total - recent, 0.0, None)
        recent_ratio = recent / (total + eps)

        return np.stack(
            [np.log1p(total), np.log1p(recent), np.log1p(past), recent_ratio], axis=1
        ).astype("float32")

    def context_vector(self, user_idx: int, time_norm: float) -> np.ndarray:
        """
        context = [user_profile(u), (persona(u) if any), time features]
        time features: [t, sin(2πt), cos(2πt)]
        """
        u = user_idx
        tn = time_norm
        time_feat = np.array(
            [tn, np.sin(2.0 * np.pi * tn), np.cos(2.0 * np.pi * tn)], dtype="float32"
        )

        parts = [self.user_profile[u].astype("float32")]
        if self.persona is not None:
            parts.append(self.persona[u].astype("float32"))
        parts.append(time_feat)
        return np.concatenate(parts).astype("float32")
