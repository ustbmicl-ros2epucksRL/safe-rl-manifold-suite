"""
RolloutBuffer with GAE computation and mini-batch sampling.
"""

import torch
import numpy as np
from typing import Generator, NamedTuple


class RolloutData(NamedTuple):
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
    Stores rollout data for all agents. Supports GAE and mini-batch sampling.

    Data layout: (episode_length, num_agents, *)
    """

    def __init__(self, episode_length: int, num_agents: int,
                 obs_dim: int, share_obs_dim: int, act_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 device: str = "cpu"):
        self.episode_length = episode_length
        self.num_agents = num_agents
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

        self.returns = torch.zeros(episode_length, num_agents, 1)
        self.advantages = torch.zeros(episode_length, num_agents, 1)

        self.step = 0

    def insert(self, obs, share_obs, actions, log_probs, values, rewards, costs, masks):
        """
        Insert one timestep of data.

        All inputs are numpy arrays or tensors with shape (num_agents, dim).
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

    def set_first_obs(self, obs, share_obs):
        """Set the initial observation."""
        self.obs[0] = torch.as_tensor(obs, dtype=torch.float32)
        self.share_obs[0] = torch.as_tensor(share_obs, dtype=torch.float32)

    def compute_returns_and_advantages(self, last_values):
        """
        Compute GAE advantages and returns.

        Args:
            last_values: (num_agents, 1) value estimates for the last state.
        """
        last_values = torch.as_tensor(last_values, dtype=torch.float32)
        self.values[self.step] = last_values

        gae = torch.zeros(self.num_agents, 1)
        for t in reversed(range(self.step)):
            delta = (self.rewards[t]
                     + self.gamma * self.values[t + 1] * self.masks[t + 1]
                     - self.values[t])
            gae = delta + self.gamma * self.gae_lambda * self.masks[t + 1] * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def feed_forward_generator(self, num_mini_batch: int) -> Generator:
        """
        Yield mini-batches of flattened (T*N, dim) data.
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
        """Reset buffer for next rollout."""
        self.obs[0].copy_(self.obs[self.step])
        self.share_obs[0].copy_(self.share_obs[self.step])
        self.masks[0].copy_(self.masks[self.step])
        self.step = 0
