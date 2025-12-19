from __future__ import annotations
import os, re, json
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from .env.offline_env import OfflineRankEnv, StepBatch
from .bandits.base import Bandit
from .eval.metrics import dcg_reward_from_rank, hit_at_k
from .eval.regret import best_fixed_action


@dataclass
class RunResult:
    chosen_action: np.ndarray  # (T,)
    reward: np.ndarray  # (T,)
    hit1: np.ndarray  # (T,)
    hitk: np.ndarray  # (T,)
    oracle_reward: np.ndarray  # (T,)
    best_fixed_reward: np.ndarray  # (T,)
    gap_to_oracle: np.ndarray  # (T,)
    regret_fixed: np.ndarray  # (T,)
    reward_matrix: np.ndarray  # (T, A)
    best_fixed_action: int


def warm_start_store(env: OfflineRankEnv, df_warm: pd.DataFrame) -> None:
    env.store.reset()
    if df_warm is None or len(df_warm) == 0:
        env.store.set_step(0)
        return

    base = -len(df_warm)
    for i in range(len(df_warm)):
        row = df_warm.iloc[i]
        env.store.set_step(base + i)
        env.store.update_from_event(int(row["user_idx"]), int(row["item_idx"]))

    env.store.set_step(0)


def rank_and_reward(
    env: OfflineRankEnv, step: StepBatch, action: int, k: int
) -> Tuple[float, int]:
    ranker = env.rankers[int(action)]
    # 점수는 유저+아이템+시간을 모두 사용할 수 있음 (ranker 구현에 따라 다름)
    scores = ranker.score(
        env.store,
        step.user_idx,
        step.cand_items,
        step.time_norm,
        rng=np.random.default_rng(0),
    )
    order = np.argsort(-scores)
    ranked = step.cand_items[order]
    pos0 = int(np.where(ranked == int(step.pos_item))[0][0])
    r = dcg_reward_from_rank(pos0, k=int(k))
    return float(r), pos0


def compute_reward_matrix(
    env: OfflineRankEnv, k: int, df_warm: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    warm_start_store(env, df_warm)

    T, A = env.n_steps, env.n_actions
    reward_mat = np.zeros((T, A), dtype="float32")
    oracle = np.zeros(T, dtype="float32")

    for t in tqdm(range(T), desc="compute_reward_matrix", unit="step"):
        step = env.step(t)

        # t시점 이전의 피처만 반영
        env.store.set_step(t)

        for a in range(A):
            r, _ = rank_and_reward(env, step, a, k=k)
            reward_mat[t, a] = r

        oracle[t] = float(reward_mat[t].max())

        env.store.update_from_event(step.user_idx, step.pos_item)

    a_star = best_fixed_action(reward_mat)
    best_fixed = reward_mat[:, a_star].copy()
    return reward_mat, oracle, best_fixed, a_star


def run_bandit(
    env: OfflineRankEnv,
    bandit: Bandit,
    reward_mat: np.ndarray,
    oracle: np.ndarray,
    best_fixed: np.ndarray,
    best_fixed_action_idx: int,
    k: int,
    seed: int,
    df_warm: pd.DataFrame,
    use_user_profile: bool = True
) -> RunResult:
    warm_start_store(env, df_warm)
    T, A = env.n_steps, env.n_actions
    rng = np.random.default_rng(int(seed) + 999)

    chosen = np.zeros(T, dtype="int64")
    rew = np.zeros(T, dtype="float32")

    eval_ks = [1, int(k)]
    hits_track = {f"hit@{vk}": np.zeros(T) for vk in eval_ks}

    for t in tqdm(range(T), desc=f"run_bandit:{bandit.name}", unit="step"):
        step = env.step(t)

        env.store.set_step(t)
        ctx = env.context(step.user_idx, step.time_norm)

        if not use_user_profile:
            G = int(env.store.user_profile.shape[1])   # user_profile dim
            ctx = ctx.astype("float32", copy=True)
            ctx[:G] = 0.0

        a_t = int(bandit.select(ctx, rng=rng))
        r_t = float(reward_mat[t, a_t])
        bandit.update(a_t, r_t, ctx)

        chosen[t] = a_t
        rew[t] = r_t

        # hit 계산 (선택한 action으로 실제 랭킹을 만들고 pos 위치 확인)
        _, pos0 = rank_and_reward(env, step, a_t, k=k)
        for vk in eval_ks:
            hits_track[f"hit@{vk}"][t] = hit_at_k(pos0, vk)

        env.store.update_from_event(step.user_idx, step.pos_item)

    gap_to_oracle = (oracle - rew).astype("float32")
    regret_fixed = (best_fixed - rew).astype("float32")

    step_best_action = np.argmax(reward_mat, axis=1)  # (T,)
    best_fixed_is_stepbest_rate = float(
        np.mean(step_best_action == int(best_fixed_action_idx))
    )

    # 밴딧이 한 arm으로 수렴했는지: window별 top-share 평균
    window = 5000
    n_win = T // window + 1
    counts = np.zeros((n_win, A), dtype=np.int32)
    for i in range(T):
        counts[i // window, chosen[i]] += 1
    share = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)
    top_share_mean_per_window = float(np.mean(np.max(share, axis=1)))

    top_arm_by_window = np.argmax(counts, axis=1)  # (n_win,)
    top_arm_switch_count = int(np.sum(top_arm_by_window[1:] != top_arm_by_window[:-1]))
    
    out_root = Path(os.environ.get("RANKING_ARMS_OUTDIR"))
    if not out_root:
        raise RuntimeError("RANKING_ARMS_OUTDIR is not set. Run via cli.py to set output directory.")


    out_dir = out_root / "run_bandit_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    up_suffix = "UP1" if use_user_profile else "UP0"
    path = out_dir / f"{bandit.name}_{up_suffix}_seed{int(seed)}.json"
    
    
    up_suffix = "UP1" if use_user_profile else "UP0"
    path = os.path.join(out_dir, f"{bandit.name}_{up_suffix}_seed{int(seed)}.json")

    ### ===== tail(마지막 20%) 평균 성능 측정 =====
    tail_frac = 0.2
    tail_start = int((1.0 - tail_frac) * T)

    tail_mean_reward = float(rew[tail_start:].mean())
    tail_mean_hit1 = float(hits_track["hit@1"][tail_start:].mean())
    tail_mean_hitk = float(hits_track[f"hit@{k}"][tail_start:].mean())

    full_mean_reward = float(rew.mean())
    full_mean_hit1 = float(hits_track["hit@1"].mean())
    full_mean_hitk = float(hits_track[f"hit@{k}"].mean())

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "bandit": bandit.name,
                "seed": int(seed),
                "T": int(T),
                "A": int(A),
                "best_fixed_action_idx": int(best_fixed_action_idx),
                "best_fixed_is_stepbest_rate": best_fixed_is_stepbest_rate,
                "top_share_mean_per_window": top_share_mean_per_window,
                "tail_frac": float(tail_frac),
                "tail_mean_reward": tail_mean_reward,
                "tail_mean_hit1": tail_mean_hit1,
                "tail_mean_hitk": tail_mean_hitk,
                "full_mean_reward": full_mean_reward,
                "full_mean_hit1": full_mean_hit1,
                "full_mean_hitk": full_mean_hitk,
                "top_arm_switch_count": top_arm_switch_count,
                "window": int(window),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return RunResult(
        chosen_action=chosen,
        reward=rew,
        hit1=hits_track["hit@1"],
        hitk=hits_track[f"hit@{k}"],
        oracle_reward=oracle,
        best_fixed_reward=best_fixed,
        gap_to_oracle=gap_to_oracle,
        regret_fixed=regret_fixed,
        reward_matrix=reward_mat,
        best_fixed_action=int(best_fixed_action_idx),
    )
