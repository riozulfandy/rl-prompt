from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig

from rlprompt.modules import SQLModule, PPOModule
from rlprompt.models import BaseModel
from rlprompt.rewards import BaseReward

def make_sql_module(model: BaseModel,
                    reward: BaseReward,
                    config: "DictConfig",
                    target_model: Optional[BaseModel] = None) -> SQLModule:
    return SQLModule(model, target_model, reward, 
                     config.sql_loss_impl, config.training_mode, 
                     config.mix_strategy, config.target_update_method, 
                     config.target_update_steps, config.target_learning_rate, 
                     config.reward_shaping, config.reward_shaping_old_min, 
                     config.reward_shaping_old_max, 
                     config.reward_shaping_new_min, 
                     config.reward_shaping_new_max, 
                     config.top_k, config.top_p, config.num_beams)

def make_ppo_module(model: BaseModel,
                   reward: BaseReward,
                   config: "DictConfig",
                   value_model: Optional[BaseModel] = None) -> PPOModule:
    return PPOModule(model, value_model, reward,
                    config.clip_ratio, config.value_coef,
                    config.entropy_coef, config.gamma,
                    config.lam, config.n_epochs,
                    config.training_mode, config.mix_strategy,
                    config.reward_shaping, config.reward_shaping_old_min,
                    config.reward_shaping_old_max, config.reward_shaping_new_min,
                    config.reward_shaping_new_max, config.top_k,
                    config.top_p, config.num_beams)

@dataclass
class SQLModuleConfig:
    sql_loss_impl: str = "v2_v2r_v3_v3r"
    training_mode: str = "sql-onpolicy"
    mix_strategy: Optional[str] = None
    # Target model setting
    target_update_method: str = "polyak"
    target_update_steps: Optional[int] = None
    target_learning_rate: float = 0.001
    # Reward shaping linearly transforms reward range of [old_min, old_max] to [new_min, new_max]
    reward_shaping: bool = True
    reward_shaping_old_min: float = 0
    reward_shaping_old_max: float = 100
    reward_shaping_new_min: float = -10
    reward_shaping_new_max: float = 10
    # Prompt generation setting
    top_k: Optional[int] = None
    top_p: float = 1.0
    num_beams: int = 1

@dataclass
class PPOModuleConfig:
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    n_epochs: int = 4
    training_mode: str = "ppo-onpolicy"
    mix_strategy: Optional[str] = None
    # Reward shaping linearly transforms reward range
    reward_shaping: bool = True
    reward_shaping_old_min: float = 0
    reward_shaping_old_max: float = 100
    reward_shaping_new_min: float = -10
    reward_shaping_new_max: float = 10
    # Prompt generation setting
    top_k: Optional[int] = None
    top_p: float = 1.0
    num_beams: int = 1