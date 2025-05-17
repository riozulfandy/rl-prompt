import torch
import torch.functional as F
from typing import Dict, Tuple, Any

from rlprompt.losses import loss_utils

def q_learning_loss_with_sparse_rewards(
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    actions: torch.LongTensor,
    rewards: torch.Tensor,
    sequence_lengths: torch.LongTensor,
    gamma: float = 0.99,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Standard Q-Learning Loss Function with Sparse Rewards.

    Args:
        logits: Q-values from the online model (batch_size, sequence_length, vocab_size).
                These are Q(s_t, a) for all 'a' at each step 't'.
        target_logits: Q-values from the target model (batch_size, sequence_length, vocab_size).
                       These are Q_target(s_t, a) for all 'a' at each step 't'.
        actions: Chosen actions (batch_size, sequence_length). These are a_t.
        rewards: Final rewards for each sequence in the batch (batch_size).
        sequence_length: Length of each sequence in the batch (batch_size).
        gamma: Discount factor.
        device: The torch device.

    Returns:
        A tuple containing:
            - raw_losses: The Q-learning loss for each element (batch_size, sequence_length).
            - quantities_to_log: A dictionary of values to log.
    """
    q_values_s_t_a_t = loss_utils.gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=logits.shape,
    )

    max_q_target_s_t_a = target_logits.max(dim=-1).values.detach()
    y_t = torch.zeros_like(q_values_s_t_a_t, device=device)

    if q_values_s_t_a_t.shape[1] > 1:
        y_t[:, :-1] = gamma * max_q_target_s_t_a[:, 1:]
    
    batch_indices = torch.arange(rewards.shape[0], device=device)
    terminal_step_indices = sequence_lengths - 1
    y_t[batch_indices, terminal_step_indices] = rewards

    td_error = y_t - q_values_s_t_a_t
    raw_losses = F.mse_loss(q_values_s_t_a_t, y_t, reduction="none")

    quantities_to_log = {
        "q_values_selected_actions": q_values_s_t_a_t.mean(),
        "target_y_t": y_t.mean(),
        "td_error_abs": td_error.abs().mean(),
        "max_q_target_mean": max_q_target_s_t_a.mean()
    }

    if logits.shape[1] > 0:
        quantities_to_log["online_q_s0_mean"] = logits[:, 0, :].mean()
        quantities_to_log["target_q_s0_mean"] = target_logits[:, 0, :].mean()
        if logits.shape[1] > 1:
            quantities_to_log["target_max_q_s1_mean"] = max_q_target_s_t_a[:, 1:].mean()

    return raw_losses, quantities_to_log