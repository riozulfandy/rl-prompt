import torch
import copy
from typing import Optional, List, Dict, Any, Union, Tuple

from rlprompt.models import BaseModel
from rlprompt.modules import BaseModule
from rlprompt.rewards import BaseReward
from rlprompt.modules.module_utils import ForwardMode, get_reward_shaping_func
from rlprompt.losses.ppo_losses import ppo_loss, compute_gae
from rlprompt.utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOModule(BaseModule):
    def __init__(
        self,
        model: BaseModel,
        value_model: Optional[BaseModel],
        reward: Optional[BaseReward],
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gamma: float = 0.99,
        lam: float = 0.95,
        training_mode: str = "ppo-onpolicy",
        mix_strategy: Optional[str] = None,
        reward_shaping: bool = True,
        reward_shaping_old_min: float = 0,
        reward_shaping_old_max: float = 100,
        reward_shaping_new_min: float = -10,
        reward_shaping_new_max: float = 10,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        num_beams: int = 1,
    ):
        super().__init__()
        
        self._model = model
        
        # If a separate value model is not provided, duplicate the model
        if value_model is None:
            self._value_model = copy.deepcopy(self._model)
        else:
            self._value_model = value_model
            
        self._reward = reward
        
        # PPO hyperparameters
        self._clip_ratio = clip_ratio
        self._value_coef = value_coef
        self._entropy_coef = entropy_coef
        self._gamma = gamma
        self._lam = lam
        
        # Training settings
        self._training_mode = training_mode
        self._mix_strategy = mix_strategy
        self._forward_modes = _get_forward_modes(training_mode, mix_strategy)
        
        # Generation settings
        self._top_k = top_k
        self._top_p = top_p
        self._num_beams = num_beams
        
        # Store the old policy logits for PPO update
        self._old_logits = None
        
        # Reward shaping
        if reward_shaping is True:
            self._reward_shaping_func = get_reward_shaping_func(
                old_min=reward_shaping_old_min,
                old_max=reward_shaping_old_max,
                new_min=reward_shaping_new_min,
                new_max=reward_shaping_new_max)
        else:
            self._reward_shaping_func = lambda _r: _r
    
    def forward(self, batch: Dict[str, Any]) -> Tuple[Union[torch.Tensor, Dict],
                                                      Dict[str, Any]]:
        loss_list = []
        loss_log_list = []
        
        for mode in self._forward_modes:
            _loss, _loss_log = self._forward(mode=mode, batch=batch)
            loss_list.append(_loss)
            loss_log_list.append(_loss_log)
        
        loss = torch.mean(torch.stack(loss_list))
        loss_log = utils.unionize_dicts(loss_log_list)
        
        return loss, loss_log
    
    def _forward(
        self,
        mode: ForwardMode,
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict]:
        if mode != ForwardMode.PPO_ON and mode != ForwardMode.INFER:
            raise NotImplementedError('Only on-policy sampling and greedy '
                                      'inference is supported for PPO')
        
        if mode == ForwardMode.PPO_ON:
            # Get current policy outputs
            (logits, values, output_tokens, output_ids, sequence_lengths) = \
                self._decode_sampling(batch=batch)
            
            # Compute rewards
            raw_rewards, rewards_log = \
                self.compute_rewards(batch=batch, 
                                     output_tokens=output_tokens,
                                     mode="train")
            shaped_rewards = self._reward_shaping_func(raw_rewards)
            
            # Compute advantages and returns using GAE
            advantages, returns = compute_gae(
                rewards=shaped_rewards,
                values=values,
                sequence_length=sequence_lengths,
                gamma=self._gamma,
                lam=self._lam
            )
            
            # If this is the first forward pass, store the logits as old_logits
            if self._old_logits is None:
                self._old_logits = logits.detach()
            
            # Compute PPO loss
            ppo_loss_val, ppo_loss_log = ppo_loss(
                logits=logits,
                old_logits=self._old_logits,
                values=values,
                actions=output_ids,
                rewards=shaped_rewards,
                returns=returns,
                advantages=advantages,
                sequence_length=sequence_lengths,
                clip_ratio=self._clip_ratio,
                value_coef=self._value_coef,
                entropy_coef=self._entropy_coef
            )
            
            # Update old_logits for next iteration
            self._old_logits = logits.detach()
            
            # Add prefixes to logs
            utils.add_prefix_to_dict_keys_inplace(
                rewards_log, prefix=f"{mode.value}/rewards/")
            utils.add_prefix_to_dict_keys_inplace(
                ppo_loss_log, prefix=f"{mode.value}/")
            
            ppo_loss_log = utils.unionize_dicts([
                rewards_log,
                ppo_loss_log,
                {
                    f"{mode.value}/rewards/raw": raw_rewards.mean(),
                    f"{mode.value}/rewards/shaped": shaped_rewards.mean(),
                },
            ])
            
            return ppo_loss_val, ppo_loss_log
        
        elif mode == ForwardMode.INFER:
            # For inference mode, just return the model's outputs
            outputs = self.infer(batch=batch)
            return outputs, {}
    
    def _decode_sampling(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]],
               torch.LongTensor, torch.LongTensor]:
        # Generate samples from the policy
        outputs = self._model.generate(**batch,
                                      do_sample=True,
                                      top_k=self._top_k,
                                      top_p=self._top_p,
                                      num_beams=self._num_beams)
        
        # Get policy logits
        logits = outputs['sample_logits'].contiguous()
        
        # Get values from the value model
        batch_ = {k: v for k, v in batch.items()}
        batch_.update(outputs)
        
        value_outputs = self._value_model.teacher_forcing(**batch_)
        values = value_outputs.get('value_estimates', torch.zeros_like(logits[:, :, 0])).contiguous()
        
        return (logits,
                values,
                outputs['sample_tokens'],
                outputs['sample_ids'].contiguous(),
                outputs['sample_lengths'].contiguous())
    
    def compute_rewards(
        self,
        batch: Dict[str, Any],
        output_tokens: List[List[str]],
        to_tensor: bool = True,
        mode: str = "infer"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        rewards_tensor, rewards_log = self._reward(
            **batch,
            output_tokens=output_tokens,
            to_tensor=to_tensor,
            mode=mode)
        
        rewards_tensor = rewards_tensor.to(device)            
        return rewards_tensor, rewards_log
    
    def infer(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, Union[torch.Tensor, torch.LongTensor, List[List[str]]]]:
        return self._model.generate(**batch,
                                    do_sample=False,
                                    top_k=self._top_k,
                                    top_p=self._top_p,
                                    num_beams=self._num_beams,
                                    infer=True)


def _get_forward_modes(
    training_mode: str,
    mix_strategy: Optional[str]
) -> List[ForwardMode]:
    if training_mode == "ppo-mixed":
        candidate_modes = [
            ForwardMode.SQL_OFF_GT,  # Reuse existing off-policy mode 
            ForwardMode.PPO_ON]
        
        if mix_strategy == "alternate":
            # This is simplified; would need global step tracking in practice
            step = 0  
            modes = [candidate_modes[step % len(candidate_modes)]]
        elif mix_strategy == "mix":
            modes = candidate_modes
    else:
        training_mode_map = {"ppo-onpolicy": ForwardMode.PPO_ON}
        
        modes = [training_mode_map[training_mode]]
    
    return modes