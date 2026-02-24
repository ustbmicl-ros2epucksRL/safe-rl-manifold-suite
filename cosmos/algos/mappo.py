"""
Multi-Agent PPO (MAPPO) Algorithm

MAPPO with parameter sharing for cooperative multi-agent tasks.
Uses CTDE: centralized critic, decentralized actors.

Reference:
    Yu et al., "The Surprising Effectiveness of PPO in Cooperative
    Multi-Agent Games", NeurIPS 2022
"""

from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from cosmos.registry import ALGO_REGISTRY
from cosmos.algos.base import BaseMARLAlgo, AlgoConfig, OnPolicyAlgo


@dataclass
class MAPPOConfig:
    """MAPPO configuration."""
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    ppo_epochs: int = 10
    num_mini_batch: int = 4
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_shared_actor: bool = True


def orthogonal_init(layer, gain=1.0):
    """Orthogonal weight initialization."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    """Gaussian policy network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int] = [128, 128]):
        super().__init__()

        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        self.base = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_size, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Initialize
        self.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2)))
        orthogonal_init(self.mean, gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and std."""
        features = self.base(obs)
        mean = self.mean(features)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action and compute log probability."""
        mean, std = self.forward(obs)

        if deterministic:
            action = mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate log probability and entropy of actions."""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class Critic(nn.Module):
    """Value function network."""

    def __init__(self, obs_dim: int, hidden_sizes: List[int] = [128, 128]):
        super().__init__()

        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)

        self.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2)))
        orthogonal_init(self.net[-1], gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate value."""
        return self.net(obs).squeeze(-1)


@ALGO_REGISTRY.register("mappo", aliases=["ppo", "ippo"])
class MAPPO(OnPolicyAlgo):
    """
    Multi-Agent PPO with parameter sharing.

    Features:
    - Shared actor network across all agents
    - Centralized critic with shared observations
    - PPO clipping for stable updates
    - GAE for advantage estimation

    Config options:
        actor_lr: Actor learning rate (default: 3e-4)
        critic_lr: Critic learning rate (default: 3e-4)
        clip_param: PPO clipping parameter (default: 0.2)
        ppo_epochs: Number of PPO update epochs (default: 10)
        num_mini_batch: Number of mini-batches (default: 4)
        entropy_coef: Entropy bonus coefficient (default: 0.01)
        gamma: Discount factor (default: 0.99)
        gae_lambda: GAE lambda (default: 0.95)
    """

    def __init__(
        self,
        obs_dim: int,
        share_obs_dim: int,
        act_dim: int,
        num_agents: int = 1,
        cfg: Optional[Union[AlgoConfig, Dict[str, Any], MAPPOConfig]] = None,
        device: str = "cpu"
    ):
        super().__init__(obs_dim, share_obs_dim, act_dim, num_agents, cfg, device)

        # Parse config
        if cfg is None:
            self._cfg = MAPPOConfig()
        elif isinstance(cfg, dict):
            self._cfg = MAPPOConfig(**{k: v for k, v in cfg.items() if hasattr(MAPPOConfig, k)})
        elif isinstance(cfg, MAPPOConfig):
            self._cfg = cfg
        else:
            # Try to extract values from generic config
            self._cfg = MAPPOConfig()
            for attr in ['actor_lr', 'critic_lr', 'gamma', 'gae_lambda', 'clip_param',
                        'ppo_epochs', 'num_mini_batch', 'entropy_coef', 'hidden_sizes']:
                if hasattr(cfg, attr):
                    setattr(self._cfg, attr, getattr(cfg, attr))

        self.device = device
        hidden_sizes = list(self._cfg.hidden_sizes)

        # Networks
        self.actor = Actor(obs_dim, act_dim, hidden_sizes).to(device)
        self.critic = Critic(share_obs_dim, hidden_sizes).to(device)
        self.cost_critic = Critic(share_obs_dim, hidden_sizes).to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self._cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self._cfg.critic_lr)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self._cfg.critic_lr)

    def get_actions(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample actions from policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            actions, log_probs = self.actor.get_action(obs_tensor, deterministic)
            return actions.cpu().numpy(), log_probs.cpu().numpy()

    def get_values(self, share_obs: np.ndarray) -> np.ndarray:
        """Estimate state values."""
        with torch.no_grad():
            share_obs_tensor = torch.FloatTensor(share_obs).to(self.device)
            values = self.critic(share_obs_tensor)
            return values.cpu().numpy()

    def get_cost_values(self, share_obs: np.ndarray) -> np.ndarray:
        """Estimate cost values."""
        with torch.no_grad():
            share_obs_tensor = torch.FloatTensor(share_obs).to(self.device)
            cost_values = self.cost_critic(share_obs_tensor)
            return cost_values.cpu().numpy()

    def update(self, buffer: Any) -> Dict[str, float]:
        """Update policy from buffer."""
        # Get data from buffer - obs has T+1 steps, actions has T steps
        # Use only the first T steps of obs to match actions
        T = buffer.step  # Number of actual steps taken
        obs = buffer.obs[:T].to(self.device)  # (T, N, obs_dim)
        share_obs = buffer.share_obs[:T].to(self.device)  # (T, N, share_obs_dim)
        actions = buffer.actions[:T].to(self.device)  # (T, N, act_dim)
        old_log_probs = buffer.log_probs[:T].to(self.device)  # (T, N, 1)
        advantages = buffer.advantages[:T].to(self.device)  # (T, N, 1)
        returns = buffer.returns[:T].to(self.device)  # (T, N, 1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten batch dimensions: (T, N, dim) -> (T*N, dim)
        num_agents = obs.shape[1]
        batch_size = T * num_agents
        obs = obs.reshape(batch_size, -1)
        share_obs = share_obs.reshape(batch_size, -1)
        actions = actions.reshape(batch_size, -1)
        old_log_probs = old_log_probs.reshape(batch_size)
        advantages = advantages.reshape(batch_size)
        returns = returns.reshape(batch_size)

        # PPO update
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        for _ in range(self._cfg.ppo_epochs):
            # Shuffle and create mini-batches
            indices = torch.randperm(batch_size)
            mini_batch_size = batch_size // self._cfg.num_mini_batch

            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]

                mb_obs = obs[mb_indices]
                mb_share_obs = share_obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Actor update
                log_probs, entropy = self.actor.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self._cfg.clip_param, 1 + self._cfg.clip_param) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self._cfg.entropy_coef * entropy.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self._cfg.max_grad_norm)
                self.actor_optimizer.step()

                # Critic update
                values = self.critic(mb_share_obs)
                critic_loss = F.mse_loss(values, mb_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self._cfg.max_grad_norm)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()

        num_updates = self._cfg.ppo_epochs * self._cfg.num_mini_batch
        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "cost_critic": self.cost_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "cost_critic" in checkpoint:
            self.cost_critic.load_state_dict(checkpoint["cost_critic"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def eval_mode(self):
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        self.cost_critic.eval()

    def train_mode(self):
        """Set networks to training mode."""
        self.actor.train()
        self.critic.train()
        self.cost_critic.train()

    def to(self, device: str) -> "MAPPO":
        """Move model to device."""
        self.device = device
        self.actor.to(device)
        self.critic.to(device)
        self.cost_critic.to(device)
        return self
