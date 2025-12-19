from dataclasses import dataclass
import pandas as pd


@dataclass
class Interactions:
    df: pd.DataFrame
    num_users: int
    num_items: int

    @property
    def n(self) -> int:
        return len(self.df)
