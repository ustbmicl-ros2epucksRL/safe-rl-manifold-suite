"""
MAPPO trainer with CTDE (Centralized Training, Decentralized Execution).
Parameter-shared actor + shared critic.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from formation_nav.algo.networks import Actor, Critic, CostCritic
from formation_nav.algo.buffer import RolloutBuffer
from formation_nav.config import AlgoConfig


class MAPPO:
    """
    Multi-Agent PPO with parameter sharing.

    - Shared Actor: all agents use the same policy network
    - Shared Critic: centralized value function on global state
    - Stores raw policy outputs (alphas) in the buffer, not safe actions,
      to keep log_prob consistent.
    """

    def __init__(self, obs_dim: int, share_obs_dim: int, act_dim: int,
                 cfg: AlgoConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.act_dim = act_dim

        # Shared networks
        self.actor = Actor(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.critic = Critic(share_obs_dim, cfg.hidden_sizes).to(device)
        self.cost_critic = CostCritic(share_obs_dim, cfg.hidden_sizes).to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr)
        self.cost_critic_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(), lr=cfg.critic_lr)

    @torch.no_grad()
    def get_actions(self, obs: np.ndarray, deterministic: bool = False
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample actions for all agents using shared actor.

        Args:
            obs: (num_agents, obs_dim) per-agent observations

        Returns:
            actions: (num_agents, act_dim) raw policy outputs (alphas)
            log_probs: (num_agents, 1)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        actions, log_probs = self.actor.get_actions(obs_t, deterministic)
        # Clip to [-1, 1] for ATACOM input
        actions = torch.tanh(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    @torch.no_grad()
    def get_values(self, share_obs: np.ndarray) -> np.ndarray:
        """
        Compute value estimates using shared critic.

        Args:
            share_obs: (num_agents, share_obs_dim)

        Returns:
            values: (num_agents, 1)
        """
        share_obs_t = torch.as_tensor(share_obs, dtype=torch.float32).to(self.device)
        values = self.critic(share_obs_t)
        return values.cpu().numpy()

    def update(self, buffer: RolloutBuffer) -> dict:
        """
        Run PPO update for ppo_epochs.

        Returns:
            info dict with loss values.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            for batch in buffer.feed_forward_generator(self.cfg.num_mini_batch):
                # Actor update
                new_log_probs, entropy = self.actor.evaluate_actions(
                    batch.obs, batch.actions)

                ratio = torch.exp(new_log_probs - batch.log_probs)
                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.cfg.clip_param,
                    1.0 + self.cfg.clip_param,
                ) * batch.advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                actor_loss = policy_loss - self.cfg.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_optimizer.step()

                # Critic update
                values = self.critic(batch.share_obs)
                value_pred_clipped = batch.values + (values - batch.values).clamp(
                    -self.cfg.clip_param, self.cfg.clip_param)
                value_loss1 = (values - batch.returns) ** 2
                value_loss2 = (value_pred_clipped - batch.returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                self.critic_optimizer.zero_grad()
                (value_loss * self.cfg.value_loss_coef).backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "cost_critic": self.cost_critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "cost_critic" in ckpt:
            self.cost_critic.load_state_dict(ckpt["cost_critic"])
