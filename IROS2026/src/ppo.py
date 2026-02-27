"""
Proximal Policy Optimization (PPO) Implementation

Simple, clean implementation for single-agent safe RL experiments.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    hidden_dim: int = 256
    n_layers: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64


class ActorCritic(nn.Module if TORCH_AVAILABLE else object):
    """Actor-Critic network for PPO."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ActorCritic")

        super().__init__()

        # Shared feature extractor
        layers = []
        in_dim = obs_dim

        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        # Actor head (mean and log_std)
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        # Smaller initialization for output layers
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            action_mean: [batch, act_dim]
            action_log_std: [act_dim]
            value: [batch]
        """
        features = self.features(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features).squeeze(-1)

        return action_mean, self.actor_log_std, value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Returns:
            action: [batch, act_dim]
            log_prob: [batch]
            value: [batch]
        """
        action_mean, action_log_std, value = self.forward(obs)
        action_std = action_log_std.exp()

        if deterministic:
            action = action_mean
            log_prob = torch.zeros(obs.shape[0])
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob, value

    def evaluate_action(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and value for given action.

        Returns:
            log_prob: [batch]
            value: [batch]
            entropy: [batch]
        """
        action_mean, action_log_std, value = self.forward(obs)
        action_std = action_log_std.exp()

        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return log_prob, value, entropy


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self, size: int, obs_dim: int, act_dim: int):
        self.size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.observations = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)

        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)

        self.ptr = 0
        self.path_start_idx = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Add transition to buffer."""
        assert self.ptr < self.size

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)

        self.ptr += 1

    def finish_path(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute GAE and returns for completed episode."""
        path_slice = slice(self.path_start_idx, self.ptr)

        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        dones = np.append(self.dones[path_slice], 0)

        # GAE computation
        deltas = rewards[:-1] + gamma * values[1:] * (1 - dones[:-1]) - values[:-1]

        advantages = np.zeros_like(deltas)
        gae = 0

        for t in reversed(range(len(deltas))):
            gae = deltas[t] + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        self.advantages[path_slice] = advantages
        self.returns[path_slice] = advantages + values[:-1]

        self.path_start_idx = self.ptr

    def get(self) -> Dict[str, np.ndarray]:
        """Get all data from buffer."""
        assert self.ptr == self.size

        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

        return {
            'observations': self.observations,
            'actions': self.actions,
            'log_probs': self.log_probs,
            'advantages': self.advantages,
            'returns': self.returns,
        }

    def reset(self):
        """Reset buffer."""
        self.ptr = 0
        self.path_start_idx = 0


class PPO:
    """Proximal Policy Optimization algorithm."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: PPOConfig = None,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PPO")

        self.config = config or PPOConfig()
        self.device = torch.device(device)

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Actor-Critic network
        self.ac = ActorCritic(
            obs_dim,
            act_dim,
            self.config.hidden_dim,
            self.config.n_layers,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.ac.parameters(),
            lr=self.config.learning_rate,
        )

    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Get action for given observation.

        Returns:
            action: [act_dim]
            log_prob: scalar
            value: scalar
        """
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.ac.get_action(obs, deterministic)

        return (
            action.cpu().numpy()[0],
            log_prob.cpu().item(),
            value.cpu().item(),
        )

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Update policy using PPO.

        Returns:
            Dictionary of training metrics
        """
        data = buffer.get()

        # Convert to tensors
        obs = torch.FloatTensor(data['observations']).to(self.device)
        actions = torch.FloatTensor(data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        advantages = torch.FloatTensor(data['advantages']).to(self.device)
        returns = torch.FloatTensor(data['returns']).to(self.device)

        n_samples = len(obs)
        batch_size = self.config.batch_size

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.config.n_epochs):
            # Random permutation
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                # Get batch
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Evaluate actions
                log_probs, values, entropy = self.ac.evaluate_action(
                    batch_obs, batch_actions
                )

                # Policy loss (clipped)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }

    def save(self, path: str):
        """Save model."""
        torch.save({
            'ac_state_dict': self.ac.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(checkpoint['ac_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
