from __future__ import annotations

import argparse
import json
import os
import yaml
from typing import List

from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from .config import SimConfig
from .data.movielens10m import load_movielens10m_events
from .features.genres import build_item_genre_matrix
from .features.user_profile import build_user_profiles
from .features.trend import build_item_trend_features
from .features.store import FeatureStore

from .rankers import RandomRanker, TrendRanker, PreferenceRanker, HybridRanker

from .bandits.linucb import LinUCB
from .bandits.random_bandit import RandomBandit
from .bandits.lin_ts import LinTS

from .env.offline_env import OfflineRankEnv
from .simulate import compute_reward_matrix, run_bandit
from .eval.regret import cumulative


def parse_args() -> SimConfig:
    p = argparse.ArgumentParser()

    p.add_argument("--n-steps", type=int, default=100_000)
    p.add_argument("--n-candidates", type=int, default=50)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--loader", type=str, default="movielens10m")
    p.add_argument("--data-dir", type=str, default="./data/ml-10M")
    p.add_argument("--min-rating", type=float, default=3.0)
    p.add_argument("--min-item-interactions", type=int, default=20)
    p.add_argument("--trend-window-size", type=int, default=20000)
    
    p.add_argument("--user-profile", type=str, default="True")
    p.add_argument("--persona-npy", type=str, default=None)
    p.add_argument("--start-index", type=int, default=-1)  # >=0이면 고정 시작점
    p.add_argument("--warmup-steps", type=int, default=20000)  # store warm-start 용
    p.add_argument("--n-seeds", type=int, default=5)  # 서로 다른 시드로 반복하는 횟수

    def _to_bool(s: str) -> bool:
        return str(s).strip().lower() in ("1", "true", "t", "yes", "y")

    args = p.parse_args()
    return SimConfig(
        loader=args.loader,
        data_dir=args.data_dir,
        n_steps=args.n_steps,
        n_candidates=args.n_candidates,
        k=args.k,
        seed=args.seed,
        min_rating=args.min_rating,
        min_item_interactions=args.min_item_interactions,
        trend_window_size=args.trend_window_size,
        persona_npy=args.persona_npy,
        start_index=args.start_index,
        warmup_steps=args.warmup_steps,
        n_seeds=args.n_seeds,
        user_profile=_to_bool(args.user_profile),
    )

def make_run_outdir() -> Path:
    base_dir = Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = base_dir / ".outputs" / ts
    out_root.mkdir(parents=True, exist_ok=True)

    os.environ["RANKING_ARMS_OUTDIR"] = str(out_root)
    return out_root

def save_cumulative_plot(stats, n_steps: int, out_path: str, title_prefix: str = ""):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    algo_names = sorted(stats.keys())
    if "Oracle" in algo_names:
        algo_names = [n for n in algo_names if n != "Oracle"] + ["Oracle"]

    STYLES = {
        "RandomBandit": dict(linestyle="--", linewidth=2.8),
        "Oracle": dict(linestyle=":", linewidth=3.2),
    }

    def _style_for(name: str):
        if name in STYLES:
            return STYLES[name]
        if name.startswith("LinUCB"):
            return dict(
                linestyle="--", linewidth=1.6, marker="o", markevery=5000, markersize=3
            )
        if name.startswith("LinTS"):
            return dict(
                linestyle="-", linewidth=1.6, marker="^", markevery=5000, markersize=3
            )
        return dict(linestyle="-", linewidth=1.2)

    # 1) 누적 reward
    ax = axes[0]
    for name in algo_names:
        data = stats[name]
        mean = data["cum_reward_mean"]
        std = data["cum_reward_std"]
        ax.plot(mean, label=name, **_style_for(name))
        ax.fill_between(
            range(n_steps),
            mean - 1.0 * std,
            mean + 1.0 * std,
            alpha=0.2,
        )
    ax.set_title(f"{title_prefix} Cumulative Rewards (Mean ± Std)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Reward")
    ax.set_xlim([0, n_steps - 1])
    ax.legend()

    # 2) 누적 regret (regret_fixed)
    ax = axes[1]
    for name in algo_names:
        data = stats[name]
        mean = data["cum_regret_mean"]
        std = data["cum_regret_std"]
        ax.plot(mean, label=name, **_style_for(name))
        ax.fill_between(
            range(n_steps),
            mean - std,
            mean + std,
            alpha=0.2,
        )
    ax.set_title(f"{title_prefix} Cumulative Regrets (Mean ± Std)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Regret")
    ax.set_xlim([0, n_steps - 1])
    ax.legend()

    # 3) 누적 regert (oracle)
    ax = axes[2]
    for name in algo_names:
        mean = stats[name]["cum_gap_mean"]
        std = stats[name]["cum_gap_std"]
        ax.plot(mean, label=name, **_style_for(name))
        ax.fill_between(range(n_steps), mean - std, mean + std, alpha=0.2)
    ax.set_title(f"{title_prefix} Cumulative Regrets [Oracle] (Mean ± Std)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gap_to_oracle")
    ax.set_xlim([0, n_steps - 1])
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=800, bbox_inches="tight")
    plt.close(fig)


def build_rankers() -> List:
    r1 = TrendRanker(
        name="Trend_recent", w=np.array([0.5, 1.5, 0.0, 1.0], dtype="float32")
    )
    r2 = TrendRanker(
        name="Trend_long", w=np.array([1.2, 0.5, 0.3, 0.2], dtype="float32")
    )
    r3 = TrendRanker(
        name="Trend_old", w=np.array([0.8, 0.1, 1.0, -0.5], dtype="float32")
    )
    r4 = TrendRanker(
        name="Trend_spike", w=np.array([0.3, 1.5, -0.5, 1.2], dtype="float32")
    )

    # 개인화 ranker (유저-아이템 매치)
    pref = PreferenceRanker(name="Preference_genre_dot")

    # 결합 ranker (개인화+트렌드)
    h1 = HybridRanker(name="Hybrid_pref80", w_trend=r1.w, lambda_pref=0.8)
    h2 = HybridRanker(name="Hybrid_pref50", w_trend=r1.w, lambda_pref=0.5)

    return [r1, r2, r3, r4, pref, h1, h2]


def main():
    cfg = parse_args()
    print(cfg)
    out_root = make_run_outdir()
    print(f"[OUT] {out_root}") 

    if cfg.loader != "movielens10m":
        raise ValueError(
            "현재 예시는 movielens10m loader만 포함합니다. (코어는 dataset-agnostic)"
        )

    interactions, _, _, movies = load_movielens10m_events(
        data_dir=cfg.data_dir,
        min_rating=cfg.min_rating,
        min_item_interactions=cfg.min_item_interactions,
    )

    df_all = interactions.df.sort_values("timestamp").reset_index(drop=True)

    N = len(df_all)
    T = min(int(cfg.n_steps), N)

    if cfg.start_index == -1 or N == T:
        start = 0
    else:
        start = cfg.start_index

    df = df_all.iloc[start : start + T].reset_index(drop=True)

    W = int(cfg.warmup_steps)
    warm_start = max(0, start - W)
    df_warm = df_all.iloc[warm_start:start].reset_index(drop=True)

    item_genre = build_item_genre_matrix(movies, num_items=interactions.num_items)

    persona = None
    if cfg.persona_npy:
        persona = np.load(cfg.persona_npy).astype("float32")

    store = FeatureStore(
        item_genre=item_genre,
        num_users=interactions.num_users,
        num_items=interactions.num_items,
        trend_window_steps=cfg.trend_window_size,
        persona=persona,
    )
    rankers = build_rankers()
    env = OfflineRankEnv(
        interactions=df,
        store=store,
        rankers=rankers,
        n_steps=cfg.n_steps,
        n_candidates=cfg.n_candidates,
        seed=cfg.seed,
    )

    reward_mat, oracle, best_fixed, a_star = compute_reward_matrix(
        env, k=cfg.k, df_warm=df_warm
    )
    
    print("Checking variance across actions in reward_mat")
    action_var = reward_mat.var(axis=1)  # 각 step에서 action 방향으로 분산
    print("Max var across actions:", action_var.max())
    print("Number of steps with non-zero variance:", np.sum(action_var > 0))
    
    hp_path = Path(__file__).resolve().parent / "hyperparams" / "exp.yaml"
    with open(hp_path, "r", encoding="utf-8") as f:
        exp = yaml.safe_load(f)

    alpha_gamma = [tuple(x) for x in exp["bandits"]["linucb"]["alpha_gamma"]]
    sigma2s = list(exp["bandits"]["lints"]["sigma2s"])

    all_rows = []
    curves = {}

    def _push(
        name: str,
        reward_step: np.ndarray,
        regret_step: np.ndarray,
        gap_step: np.ndarray,
    ):
        if name not in curves:
            curves[name] = {"cum_reward": [], "cum_regret": [], "cum_gap": []}
        curves[name]["cum_reward"].append(cumulative(reward_step))
        curves[name]["cum_regret"].append(cumulative(regret_step))
        curves[name]["cum_gap"].append(cumulative(gap_step))

    for s in range(int(cfg.seed), int(cfg.seed) + int(cfg.n_seeds)):
        seed_results = {"seed": s}

        rnd_bandit = RandomBandit(
            name="RandomBandit", n_actions=env.n_actions, context_dim=store.context_dim
        )
        res_rnd = run_bandit(
            env,
            rnd_bandit,
            reward_mat,
            oracle,
            best_fixed,
            a_star,
            k=cfg.k,
            seed=s + 1000,
            df_warm=df_warm,
            use_user_profile=cfg.user_profile,
        )
        _push(
            rnd_bandit.name, res_rnd.reward, res_rnd.regret_fixed, res_rnd.gap_to_oracle
        )

        seed_results["RandomBandit"] = {
            "mean_reward": float(res_rnd.reward.mean()),
            "mean_hit1": float(res_rnd.hit1.mean()),
            f"mean_hit@{cfg.k}": float(res_rnd.hitk.mean()),
            "cum_gap_to_oracle": float(cumulative(res_rnd.gap_to_oracle)[-1]),
            "cum_regret_fixed": float(cumulative(res_rnd.regret_fixed)[-1]),
        }

        for para in alpha_gamma:
            bandit = LinUCB(
                name=f"LinUCB(alpha={para[0]},gamma={para[1]})",
                n_actions=env.n_actions,
                context_dim=store.context_dim,
                alpha=para[0],
                gamma=para[1],
            )
            agent_name = bandit.name

            res = run_bandit(
                env,
                bandit,
                reward_mat,
                oracle,
                best_fixed,
                a_star,
                k=cfg.k,
                seed=s,
                df_warm=df_warm,
                use_user_profile=cfg.user_profile,
            )

            _push(agent_name, res.reward, res.regret_fixed, res.gap_to_oracle)

            seed_results[agent_name] = {
                "mean_reward": float(res.reward.mean()),
                "mean_hit1": float(res.hit1.mean()),
                f"mean_hit@{cfg.k}": float(res.hitk.mean()),
                "cum_gap_to_oracle": float(cumulative(res.gap_to_oracle)[-1]),
                "cum_regret_fixed": float(cumulative(res.regret_fixed)[-1]),
            }

        for s2 in sigma2s:
            ts = LinTS(
                name=f"LinTS(lam=1.0,sigma2={s2})",
                n_actions=env.n_actions,
                context_dim=store.context_dim,
                lam=1.0,
                sigma2=s2,
            )
            res_ts = run_bandit(
                env,
                ts,
                reward_mat,
                oracle,
                best_fixed,
                a_star,
                k=cfg.k,
                seed=s,
                df_warm=df_warm,
                use_user_profile=cfg.user_profile,
            )

            _push(ts.name, res_ts.reward, res_ts.regret_fixed, res_ts.gap_to_oracle)

            seed_results[ts.name] = {
                "mean_reward": float(res_ts.reward.mean()),
                "mean_hit1": float(res_ts.hit1.mean()),
                f"mean_hit@{cfg.k}": float(res_ts.hitk.mean()),
                "cum_gap_to_oracle": float(cumulative(res_ts.gap_to_oracle)[-1]),
                "cum_regret_fixed": float(cumulative(res_ts.regret_fixed)[-1]),
            }

        all_rows.append(seed_results)

        with open(
            os.path.join(out_root, f"summary_seed{s}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(seed_results, f, ensure_ascii=False, indent=2)

    
    def _agg(algo_name: str, key: str):
        vals = [row[algo_name][key] for row in all_rows]
        arr = np.array(vals, dtype="float64")
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    all_agent_names = [k for k in all_rows[0].keys() if k != "seed"]

    agg_summary = {
        "config": cfg.__dict__,
        "window": {
            "start": int(start),
            "n_steps": int(T),
            "warmup_used": int(len(df_warm)),
        },
        "best_fixed_ranker_name": env.rankers[int(a_star)].name,
        "aggregate": {},
    }

    for name in all_agent_names:
        agg_summary["aggregate"][name] = {
            "mean_reward_mean": _agg(name, "mean_reward")[0],
            "mean_reward_std": _agg(name, "mean_reward")[1],
            "cum_regret_fixed_mean": _agg(name, "cum_regret_fixed")[0],
            "cum_regret_fixed_std": _agg(name, "cum_regret_fixed")[1],
        }

    with open(os.path.join(out_root, "summary_agg.json"), "w", encoding="utf-8") as f:
        json.dump(agg_summary, f, ensure_ascii=False, indent=2)

    def _mean_std(list_of_arrays: List[np.ndarray]):
        arr = np.stack(list_of_arrays, axis=0)  # [n_seeds, T]
        mean = arr.mean(axis=0)
        if arr.shape[0] > 1:
            std = arr.std(axis=0, ddof=1)
        else:
            std = np.zeros_like(mean)
        return mean, std

    stats_plot = {}
    n_steps_plot = None
    for name, d in curves.items():
        cr_mean, cr_std = _mean_std(d["cum_reward"])
        cg_mean, cg_std = _mean_std(d["cum_regret"])
        gap_mean, gap_std = _mean_std(d["cum_gap"])
        stats_plot[name] = {
            "cum_reward_mean": cr_mean,
            "cum_reward_std": cr_std,
            "cum_regret_mean": cg_mean,
            "cum_regret_std": cg_std,
            "cum_gap_mean": gap_mean,
            "cum_gap_std": gap_std,
        }
        if n_steps_plot is None:
            n_steps_plot = len(cr_mean)

    out_png = os.path.join(out_root, f"cumulative_alpha_comparison.png")
    save_cumulative_plot(
        stats_plot,
        n_steps=n_steps_plot,
        out_path=out_png,
        title_prefix=f"{cfg.loader} (k={cfg.k})",
    )


if __name__ == "__main__":
    main()
