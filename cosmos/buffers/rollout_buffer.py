"""
Rollout Buffer for On-Policy MARL Algorithms.

Provides on-policy storage for algorithms like MAPPO, IPPO.
Compatible with formation_nav buffer interface.
"""

from typing import Dict, Any, Optional, Generator, NamedTuple
import numpy as np
import torch


class RolloutData(NamedTuple):
    """Named tuple for mini-batch data."""
    obs: torch.Tensor
    share_obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    masks: torch.Tensor


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms (PPO family).

    Stores rollout data and computes advantages using GAE.
    Data layout: (episode_length, num_agents, *)

    Compatible with formation_nav.algo.buffer.RolloutBuffer interface.
    """

    def __init__(
        self,
        episode_length: int,
        num_agents: int,
        obs_dim: int,
        share_obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu"
    ):
        """
        Args:
            episode_length: Maximum steps per episode (rollout_length).
            num_agents: Number of agents.
            obs_dim: Observation dimension per agent.
            share_obs_dim: Shared observation dimension.
            act_dim: Action dimension per agent.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
            device: Torch device.
        """
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.share_obs_dim = share_obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Buffers: (T+1, N, dim) for obs/values, (T, N, dim) for rest
        self.obs = torch.zeros(episode_length + 1, num_agents, obs_dim)
        self.share_obs = torch.zeros(episode_length + 1, num_agents, share_obs_dim)
        self.actions = torch.zeros(episode_length, num_agents, act_dim)
        self.log_probs = torch.zeros(episode_length, num_agents, 1)
        self.values = torch.zeros(episode_length + 1, num_agents, 1)
        self.rewards = torch.zeros(episode_length, num_agents, 1)
        self.costs = torch.zeros(episode_length, num_agents, 1)
        self.masks = torch.ones(episode_length + 1, num_agents, 1)

        # Computed quantities
        self.returns = torch.zeros(episode_length, num_agents, 1)
        self.advantages = torch.zeros(episode_length, num_agents, 1)

        # For constrained optimization
        self.cost_values = torch.zeros(episode_length + 1, num_agents, 1)
        self.cost_returns = torch.zeros(episode_length, num_agents, 1)
        self.cost_advantages = torch.zeros(episode_length, num_agents, 1)

        self.step = 0

    def reset(self):
        """Reset buffer for new rollout (alternative to after_update)."""
        self.step = 0

    def set_first_obs(self, obs: np.ndarray, share_obs: np.ndarray):
        """Set the initial observation."""
        self.obs[0] = torch.as_tensor(obs, dtype=torch.float32)
        self.share_obs[0] = torch.as_tensor(share_obs, dtype=torch.float32)

    def insert(
        self,
        obs: np.ndarray,
        share_obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        costs: np.ndarray,
        masks: np.ndarray
    ):
        """
        Insert one timestep of data.

        All inputs are numpy arrays or tensors with shape (num_agents, dim).

        Args:
            obs: Next observations (num_agents, obs_dim)
            share_obs: Next shared observations (num_agents, share_obs_dim)
            actions: Actions taken (num_agents, act_dim)
            log_probs: Log probabilities (num_agents, 1)
            values: Value estimates (num_agents, 1)
            rewards: Rewards received (num_agents, 1)
            costs: Costs received (num_agents, 1)
            masks: Episode masks (num_agents, 1)
        """
        t = self.step

        self.obs[t + 1] = torch.as_tensor(obs, dtype=torch.float32)
        self.share_obs[t + 1] = torch.as_tensor(share_obs, dtype=torch.float32)
        self.actions[t] = torch.as_tensor(actions, dtype=torch.float32)
        self.log_probs[t] = torch.as_tensor(log_probs, dtype=torch.float32)
        self.values[t] = torch.as_tensor(values, dtype=torch.float32)
        self.rewards[t] = torch.as_tensor(rewards, dtype=torch.float32)
        self.costs[t] = torch.as_tensor(costs, dtype=torch.float32)
        self.masks[t + 1] = torch.as_tensor(masks, dtype=torch.float32)

        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        last_cost_values: Optional[np.ndarray] = None
    ):
        """
        Compute GAE advantages and returns.

        Args:
            last_values: Value estimates for the last state (num_agents, 1).
            last_cost_values: Cost value estimates for final state (optional).
        """
        last_values = torch.as_tensor(last_values, dtype=torch.float32)
        self.values[self.step] = last_values

        # Compute GAE for rewards
        gae = torch.zeros(self.num_agents, 1)
        for t in reversed(range(self.step)):
            delta = (
                self.rewards[t]
                + self.gamma * self.values[t + 1] * self.masks[t + 1]
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * self.masks[t + 1] * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

        # Compute GAE for costs (if using constrained optimization)
        if last_cost_values is not None:
            last_cost_values = torch.as_tensor(last_cost_values, dtype=torch.float32)
            self.cost_values[self.step] = last_cost_values

            cost_gae = torch.zeros(self.num_agents, 1)
            for t in reversed(range(self.step)):
                cost_delta = (
                    self.costs[t]
                    + self.gamma * self.cost_values[t + 1] * self.masks[t + 1]
                    - self.cost_values[t]
                )
                cost_gae = cost_delta + self.gamma * self.gae_lambda * self.masks[t + 1] * cost_gae
                self.cost_advantages[t] = cost_gae
                self.cost_returns[t] = cost_gae + self.cost_values[t]

    def feed_forward_generator(self, num_mini_batch: int) -> Generator[RolloutData, None, None]:
        """
        Yield mini-batches of flattened (T*N, dim) data.

        Args:
            num_mini_batch: Number of mini-batches to split data into.

        Yields:
            RolloutData named tuples.
        """
        T = self.step
        N = self.num_agents
        batch_size = T * N
        mini_batch_size = batch_size // num_mini_batch

        # Flatten: (T, N, dim) -> (T*N, dim)
        obs = self.obs[:T].reshape(batch_size, -1)
        share_obs = self.share_obs[:T].reshape(batch_size, -1)
        actions = self.actions[:T].reshape(batch_size, -1)
        log_probs = self.log_probs[:T].reshape(batch_size, -1)
        values = self.values[:T].reshape(batch_size, -1)
        returns = self.returns[:T].reshape(batch_size, -1)
        masks = self.masks[:T].reshape(batch_size, -1)

        # Normalize advantages across all agents jointly
        advantages = self.advantages[:T].reshape(batch_size, -1)
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Random shuffle
        indices = torch.randperm(batch_size)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            if end > batch_size:
                break
            idx = indices[start:end]
            yield RolloutData(
                obs=obs[idx].to(self.device),
                share_obs=share_obs[idx].to(self.device),
                actions=actions[idx].to(self.device),
                log_probs=log_probs[idx].to(self.device),
                values=values[idx].to(self.device),
                returns=returns[idx].to(self.device),
                advantages=advantages[idx].to(self.device),
                masks=masks[idx].to(self.device),
            )

    def after_update(self):
        """Reset buffer for next rollout, carrying over last observation."""
        self.obs[0].copy_(self.obs[self.step])
        self.share_obs[0].copy_(self.share_obs[self.step])
        self.masks[0].copy_(self.masks[self.step])
        self.step = 0

    def get_data(self) -> Dict[str, torch.Tensor]:
        """Get all data as a dict of tensors (alternative interface)."""
        T = self.step
        return {
            "obs": self.obs[:T].to(self.device),
            "share_obs": self.share_obs[:T].to(self.device),
            "actions": self.actions[:T].to(self.device),
            "rewards": self.rewards[:T].to(self.device),
            "costs": self.costs[:T].to(self.device),
            "values": self.values[:T].to(self.device),
            "log_probs": self.log_probs[:T].to(self.device),
            "advantages": self.advantages[:T].to(self.device),
            "returns": self.returns[:T].to(self.device),
            "masks": self.masks[:T].to(self.device),
            "cost_advantages": self.cost_advantages[:T].to(self.device),
            "cost_returns": self.cost_returns[:T].to(self.device),
        }
