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
    """Policy loss for PPO with robust tensor shape handling
    
    Arguments:
        logits: [batch_size, sequence_length, vocab_size] - Policy logits
        old_logits: [batch_size, sequence_length, vocab_size] - Old policy logits
        actions: [batch_size, sequence_length] - Taken actions
        advantages: Tensor of advantages - Can be [batch_size, sequence_length], [batch_size], or [batch_size*sequence_length]
        sequence_length: [batch_size] - Length of each sequence
        clip_ratio: PPO clipping parameter
    """
    batch_size, seq_len = logits.shape[:2]
    
    # Get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    old_log_probs = F.log_softmax(old_logits, dim=-1)
    
    # Gather log probs of actions
    log_probs_taken = loss_utils.gather_2d_on_last_dim(
        tensor=log_probs,
        index=actions,
        shape=actions.shape)
    old_log_probs_taken = loss_utils.gather_2d_on_last_dim(
        tensor=old_log_probs,
        index=actions,
        shape=actions.shape)
    
    # Calculate probability ratio
    ratio = torch.exp(log_probs_taken - old_log_probs_taken.detach())
    
    # Robustly reshape advantages to match ratio's shape
    if advantages.shape == ratio.shape:
        # Already the correct shape, no change needed
        advantages_expanded = advantages
    elif advantages.dim() == 1:
        if advantages.shape[0] == batch_size:
            # [batch_size] → [batch_size, seq_len]
            advantages_expanded = advantages.unsqueeze(1).expand_as(ratio)
        elif advantages.shape[0] == batch_size * seq_len:
            # [batch_size*seq_len] → [batch_size, seq_len]
            advantages_expanded = advantages.view(batch_size, seq_len)
        else:
            raise ValueError(f"Cannot reshape advantages of shape {advantages.shape} to match ratio shape {ratio.shape}")
    elif advantages.dim() == 2 and advantages.shape[0] == batch_size:
        if advantages.shape[1] == 1:
            # [batch_size, 1] → [batch_size, seq_len]
            advantages_expanded = advantages.expand_as(ratio)
        elif advantages.shape[1] == seq_len:
            # [batch_size, seq_len] - already correct
            advantages_expanded = advantages
        else:
            raise ValueError(f"Cannot reshape advantages of shape {advantages.shape} to match ratio shape {ratio.shape}")
    else:
        raise ValueError(f"Unsupported advantages shape: {advantages.shape}, expected compatible with {ratio.shape}")
    
    # Compute surrogate objectives
    surr1 = ratio * advantages_expanded
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_expanded
    raw_losses = -torch.min(surr1, surr2)
    
    # Quantities to log
    quantities_to_log = {
        "ratio": ratio,
        "ratio_mean": ratio.mean(),
        "ratio_min": ratio.min(),
        "ratio_max": ratio.max(),
        "advantages": advantages_expanded,
        "advantages_mean": advantages_expanded.mean(),
        "entropy": -(log_probs * torch.exp(log_probs)).sum(-1),
        "surr1": surr1,
        "surr2": surr2,
    }
    
    return raw_losses, quantities_to_log

def ppo_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    sequence_length: torch.LongTensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Value function loss for PPO with robust tensor shape handling
    
    Arguments:
        values: [batch_size, sequence_length] - Value estimates
        returns: Tensor of returns - Can be [batch_size, sequence_length], [batch_size], or [batch_size*sequence_length]
        sequence_length: [batch_size] - Length of each sequence
    """
    batch_size, seq_len = values.shape
    
    # Robustly reshape returns to match values' shape
    if returns.shape == values.shape:
        # Already the correct shape, no change needed
        returns_expanded = returns
    elif returns.dim() == 1:
        if returns.shape[0] == batch_size:
            # [batch_size] → [batch_size, seq_len]
            returns_expanded = returns.unsqueeze(1).expand_as(values)
        elif returns.shape[0] == batch_size * seq_len:
            # [batch_size*seq_len] → [batch_size, seq_len]
            returns_expanded = returns.view(batch_size, seq_len)
        else:
            raise ValueError(f"Cannot reshape returns of shape {returns.shape} to match values shape {values.shape}")
    elif returns.dim() == 2 and returns.shape[0] == batch_size:
        if returns.shape[1] == 1:
            # [batch_size, 1] → [batch_size, seq_len]
            returns_expanded = returns.expand_as(values)
        elif returns.shape[1] == seq_len:
            # [batch_size, seq_len] - already correct
            returns_expanded = returns
        else:
            raise ValueError(f"Cannot reshape returns of shape {returns.shape} to match values shape {values.shape}")
    else:
        raise ValueError(f"Unsupported returns shape: {returns.shape}, expected compatible with {values.shape}")
    
    # Calculate MSE loss
    raw_losses = F.mse_loss(values, returns_expanded, reduction="none")
    
    quantities_to_log = {
        "values": values,
        "returns": returns_expanded,
        "value_loss_mean": raw_losses.mean(),
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