import torch
import copy
from typing import Optional, List, Dict, Any, Tuple, Union

from rlprompt.models import BaseModel
from rlprompt.modules import BaseModule
from rlprompt.rewards import BaseReward
from rlprompt.modules.module_utils import get_reward_shaping_func
from rlprompt.losses.q_losses import q_learning_loss_with_sparse_rewards
from rlprompt.utils import utils
from rlprompt.losses import loss_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QModule(BaseModule):
    def __init__(
        self,
        model: BaseModel,
        target_model: Optional[BaseModel],
        reward_computer: BaseReward,
        # Q learning specific hyperparameters
        gamma: float,
        # target network update param
        target_update_method: str, 
        target_update_steps: Optional[int],
        target_learning_rate: float,
        # reward shaping params
        reward_shaping: bool,
        reward_shaping_old_min: float,
        reward_shaping_old_max: float,
        reward_shaping_new_min: float,
        reward_shaping_new_max: float,
        # generation params for policy
        top_k: Optional[int],
        top_p: float,
        num_beams: int,
    ):
        super().__init__()
        assert target_update_method in ["copy", "polyak"]
        assert not (top_k is not None and top_p < 1.0), \
               "Only one of top_k or top_p should be selected"
        
        self._model = model.to(device)
        if target_model is None:
            self._target_model = copy.deepcopy(self._model)
        else:
            self._target_model = target_model.to(device)
        for param in self._target_model.parameters():
            param.requires_grad = False
        
        self._reward_computer = reward_computer
        self._gamma = gamma

        self._target_update_method = target_update_method
        self._target_update_steps = target_update_steps
        self._target_learning_rate = target_learning_rate

        self._top_k = top_k
        self._top_p = top_p
        self._num_beams = num_beams

        if reward_shaping:
            self._reward_shaping_func = get_reward_shaping_func(
                old_min=reward_shaping_old_min,
                old_max=reward_shaping_old_max,
                new_min=reward_shaping_new_min,
                new_max=reward_shaping_new_max,
            )
        else:
            self._reward_shaping_func = lambda _r: _r
        
        self._current_step = 0
    
    def _sync_target_model(self) -> None:
        if self._target_update_method == "copy":
            self._target_model.load_state_dict(self._model.state_dict())
        elif self._target_update_method == "polyak":
            for param_, param in zip(self._target_model.parameters(), self._model.parameters()):
                param_.data.copy_((1 - self._target_learning_rate) * param_.data + self._target_learning_rate * param.data)
    
    def _pre_steps(self, step):
        self._current_step = step
        if self._target_update_method == "polyak":
            self._sync_target_model()
        elif self._target_update_method == "copy" \
            and self._current_step > 0 and self._current_step % self._target_update_steps == 0:
            self._sync_target_model()

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self._model.train()
        self._target_model.eval()

        online_model_outputs = self._model.generate(
            **batch,
            do_sample=True,
            top_k=self._top_k,
            top_p=self._top_p,
            num_beams=self._num_beams,
        )

        online_logits = online_model_outputs["sample_logits"].contiguous().to(device)
        output_ids = online_model_outputs['sample_ids'].contiguous().to(device)
        output_tokens = online_model_outputs['sample_tokens']
        sequence_lengths = online_model_outputs['sample_lengths'].contiguous().to(device)

        target_model_input_batch = {**batch, "sample_ids": output_ids}
        target_model_outputs = self._target_model.teacher_forcing(**target_model_input_batch)
        target_logits = target_model_outputs['sample_logits'].contiguous().to(device)

        raw_rewards, rewards_log = self.compute_rewards(
            batch=batch,
            output_tokens=output_tokens,
            to_tensor=True,
            mode="train"
        )
        raw_rewards = raw_rewards.to(device)
        shaped_rewards = self._reward_shaping_func(raw_rewards)

        q_loss, q_loss_log_terms = q_learning_loss_with_sparse_rewards(
            logits=online_logits,
            target_logits=target_logits,
            actions=output_ids,
            rewards=shaped_rewards,
            sequence_lengths=sequence_lengths,
            gamma=self._gamma,
            device=device
        )

        final_loss = loss_utils.mask_and_reduce(
            sequence=q_loss,
            sequence_length=sequence_lengths,
            average_across_batch=True,
            sum_over_timesteps=True,
        )

        # Logging
        loss_log = {
            "q_loss/total_loss": final_loss.item(),
            "q_loss/sequence_length_mean": sequence_lengths.float().mean().item(),
        }
        for key, value in q_loss_log_terms.items():
            loss_log[f"q_loss/{key}"] = value.item() if torch.is_tensor(value) else value
            
        for key, value in rewards_log.items():
             loss_log[f"rewards/{key}"] = value.item() if torch.is_tensor(value) else value
        loss_log["rewards/raw_mean"] = raw_rewards.mean().item()
        loss_log["rewards/shaped_mean"] = shaped_rewards.mean().item()
        
        masked_mean_raw_loss_per_step = loss_utils.mask_and_reduce(
            sequence=q_loss,
            sequence_length=sequence_lengths,
            average_across_batch=True,
            average_across_timesteps=True, # Average over active timesteps
            sum_over_timesteps=False
        )
        loss_log["q_loss/mean_loss_per_step"] = masked_mean_raw_loss_per_step.item()

        return final_loss, loss_log

    def compute_rewards(
        self,
        batch: Dict[str, Any],
        output_tokens: List[List[str]],
        to_tensor: bool = True,
        mode: str = "infer"
    ) -> Tuple[Union[torch.Tensor, List[float]], Dict[str, Any]]:
        rewards_tensor, rewards_log = self._reward_computer(
            **batch,
            output_tokens=output_tokens,
            to_tensor=to_tensor, 
            mode=mode
        )
    
        if to_tensor and torch.is_tensor(rewards_tensor):
            rewards_tensor = rewards_tensor.to(device)
            
        return rewards_tensor, rewards_log

    def infer(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, Union[torch.Tensor, torch.LongTensor, List[List[str]]]]:
        """
        Greedy inference using the online model.
        """
        self._model.eval()
        return self._model.generate(**batch,
                                    do_sample=False,
                                    top_k=self._top_k,
                                    top_p=self._top_p,
                                    num_beams=self._num_beams,
                                    infer=True)
