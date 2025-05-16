import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List

from rlprompt.losses import loss_utils
from rlprompt.utils import utils

def ppo_policy_loss(
    logits: torch.Tensor,
    old_logits: torch.Tensor,
    actions: torch.LongTensor,
    advantages: torch.Tensor,
    sequence_length: torch.LongTensor,
    clip_ratio: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """PPO policy loss function with clipped objective
    
    Arguments:
        logits: [batch_size, sequence_length, vocab_size] - Current policy logits
        old_logits: [batch_size, sequence_length, vocab_size] - Old policy logits
        actions: [batch_size, sequence_length] - Actions that were taken
        advantages: [batch_size] - Advantage estimates
        sequence_length: [batch_size] - Length of each sequence
        clip_ratio: float - Clip parameter for PPO
    """
    # Get log probabilities of actions under current and old policies
    log_probs = F.log_softmax(logits, dim=-1)
    old_log_probs = F.log_softmax(old_logits, dim=-1)
    
    # Gather the log probs for the actions that were actually taken
    log_probs_taken = loss_utils.gather_2d_on_last_dim(
        tensor=log_probs,
        index=actions,
        shape=actions.shape)
    old_log_probs_taken = loss_utils.gather_2d_on_last_dim(
        tensor=old_log_probs,
        index=actions,
        shape=actions.shape)
    
    # Calculate the ratio of new/old policy probabilities
    ratio = torch.exp(log_probs_taken - old_log_probs_taken.detach())
    
    # Expand advantages to match the shape of ratio
    advantages_expanded = advantages.view(-1, 1).expand_as(ratio)
    
    # Calculate the surrogate loss with clipping
    surr1 = ratio * advantages_expanded
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_expanded
    
    # PPO objective is negative because we want to maximize it
    raw_losses = -torch.min(surr1, surr2)
    
    # Logging quantities
    quantities_to_log = {
        "ratio": ratio,
        "surr1": surr1,
        "surr2": surr2,
        "advantages": advantages_expanded,
    }
    
    return raw_losses, quantities_to_log

def ppo_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    sequence_length: torch.LongTensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Value function loss for PPO
    
    Arguments:
        values: [batch_size, sequence_length] - Value estimates
        returns: [batch_size] - Returns
        sequence_length: [batch_size] - Length of each sequence
    """
    # Expand returns to match the shape of values
    returns_expanded = returns.view(-1, 1).expand_as(values)
    
    # Calculate MSE loss between values and returns
    raw_losses = F.mse_loss(values, returns_expanded, reduction="none")
    
    quantities_to_log = {
        "values": values,
        "returns": returns_expanded,
    }
    
    return raw_losses, quantities_to_log

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    sequence_length: torch.LongTensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) and returns
    
    Arguments:
        rewards: [batch_size] - Rewards
        values: [batch_size, sequence_length] - Value estimates
        sequence_length: [batch_size] - Length of each sequence
        gamma: float - Discount factor
        lam: float - GAE lambda parameter
    
    Returns:
        advantages: [batch_size] - Advantage estimates
        returns: [batch_size] - Returns
    """
    batch_size, max_seq_len = values.shape
    
    # Initialize advantages and returns
    advantages = torch.zeros_like(values)
    returns = torch.zeros_like(values)
    
    # For each sequence in the batch
    for i in range(batch_size):
        seq_len = sequence_length[i]
        reward = rewards[i]
        
        # Initialize the GAE advantage
        gae = 0
        
        # The return at the end of the sequence is the reward
        returns[i, seq_len-1] = reward
        
        # Work backwards through the sequence
        for t in reversed(range(seq_len)):
            # If we're at the end of the sequence, the next value is 0
            next_val = values[i, t+1] if t < seq_len-1 else 0
            
            # Delta is the TD error
            delta = reward - values[i, t] + gamma * next_val
            
            # Update the GAE advantage
            gae = delta + gamma * lam * gae
            
            # Store the advantage
            advantages[i, t] = gae
            
            # Compute the return at this time step
            returns[i, t] = advantages[i, t] + values[i, t]
    
    return advantages, returns

def ppo_loss(
    logits: torch.Tensor,
    old_logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.LongTensor,
    rewards: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    sequence_length: torch.LongTensor,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Combined PPO loss function
    
    Arguments:
        logits: [batch_size, sequence_length, vocab_size] - Current policy logits
        old_logits: [batch_size, sequence_length, vocab_size] - Old policy logits
        values: [batch_size, sequence_length] - Value estimates
        actions: [batch_size, sequence_length] - Actions that were taken
        rewards: [batch_size] - Rewards
        returns: [batch_size] - Returns
        advantages: [batch_size] - Advantage estimates
        sequence_length: [batch_size] - Length of each sequence
        clip_ratio: float - Clip parameter for PPO
        value_coef: float - Value loss coefficient
        entropy_coef: float - Entropy coefficient
    """
    # Policy loss
    policy_raw_losses, policy_quantities_to_log = ppo_policy_loss(
        logits=logits,
        old_logits=old_logits,
        actions=actions,
        advantages=advantages,
        sequence_length=sequence_length,
        clip_ratio=clip_ratio
    )
    
    # Value loss
    value_raw_losses, value_quantities_to_log = ppo_value_loss(
        values=values,
        returns=returns,
        sequence_length=sequence_length
    )
    
    # Entropy loss (to encourage exploration)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    entropy = -(log_probs * probs).sum(dim=-1)
    entropy_loss = -entropy  # negative because we want to maximize entropy
    
    # Combined loss
    raw_losses = policy_raw_losses + value_coef * value_raw_losses + entropy_coef * entropy_loss
    
    # Apply masking and reduce
    loss = loss_utils.mask_and_reduce(
        sequence=raw_losses,
        sequence_length=sequence_length)
    
    # Prepare logging info
    utils.add_prefix_to_dict_keys_inplace(
        policy_quantities_to_log, prefix="policy/")
    utils.add_prefix_to_dict_keys_inplace(
        value_quantities_to_log, prefix="value/")
    
    quantities_to_log = utils.unionize_dicts([
        policy_quantities_to_log,
        value_quantities_to_log,
        {"entropy": entropy}
    ])
    
    loss_log = {
        "loss": loss,
        "sequence_length": sequence_length.float().mean(),
        "loss-normalized": loss_utils.mask_and_reduce(
            sequence=raw_losses,
            sequence_length=sequence_length,
            average_across_timesteps=True,
            sum_over_timesteps=False),
        "rewards": rewards.mean(),
        "returns": returns.mean(),
        "advantages": advantages.mean(),
    }
    
    for key, value in quantities_to_log.items():
        masked_mean, masked_min, masked_max = \
            loss_utils.get_masked_mean_min_max(value,
                                              lengths=sequence_length)
        loss_log[f"{key}/min"] = masked_min
        loss_log[f"{key}/max"] = masked_max
        loss_log[f"{key}/mean"] = masked_mean
    
    return loss, loss_log