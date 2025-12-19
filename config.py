from dataclasses import dataclass
from typing import Optional


@dataclass
class SimConfig:
    n_steps: int = 100_000
    n_candidates: int = 50
    k: int = 10
    seed: int = 42

    loader: str = "movielens10m"
    data_dir: str = "./data/ml-10M"
    min_rating: float = 3.0
    min_item_interactions: int = 20
    trend_window_size:int = 20_000

    user_profile: str = "True"
    persona_npy: Optional[str] = None  # (num_users, d_p) numpy 파일 경로
    start_index: int = -1
    warmup_steps: int = 20_000
    n_seeds: int = 1
    
