import torch

from typing import Callable
from enum import Enum


class ForwardMode(Enum):
    SQL_ON = "SQL_ON"
    SQL_OFF_GT = "SQL_OFF_GT"
   
    INFER = "INFER"

def get_reward_shaping_func(
    old_min: float,
    old_max: float,
    new_min: float,
    new_max: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    def _shaping_func(reward: torch.Tensor) -> torch.Tensor:
        percentile = (reward - old_min) / (old_max - old_min)
        return percentile * (new_max - new_min) + new_min

    return _shaping_func

