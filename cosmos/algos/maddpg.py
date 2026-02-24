"""
Multi-Agent DDPG (MADDPG) Algorithm

MADDPG extends DDPG to multi-agent settings with centralized critics
and decentralized actors.

Reference:
    Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
    Environments", NeurIPS 2017
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from cosmos.registry import ALGO_REGISTRY
from cosmos.algos.base import BaseMARLAlgo, AlgoConfig, OffPolicyAlgo


class Actor(nn.Module):
    """
    Actor network for continuous action space.

    Maps observations to deterministic actions.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        act_limit: float = 1.0
    ):
        super().__init__()

        self.act_limit = act_limit

        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns actions in [-act_limit, act_limit]."""
        x = self.net(obs)
        return torch.tanh(self.out(x)) * self.act_limit


class Critic(nn.Module):
    """
    Centralized critic for MADDPG.

    Takes all agents' observations and actions as input,
    outputs Q-value for one agent.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int,
        hidden_sizes: Tuple[int, ...] = (128, 128)
    ):
        super().__init__()

        # Input: all observations + all actions
        input_dim = (obs_dim + act_dim) * num_agents

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, 1)

    def forward(
        self,
        obs_all: torch.Tensor,
        acts_all: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            obs_all: All observations (batch, num_agents * obs_dim)
            acts_all: All actions (batch, num_agents * act_dim)

        Returns:
            Q-value (batch, 1)
        """
        x = torch.cat([obs_all, acts_all], dim=-1)
        return self.out(self.net(x))


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2
    ):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        """Reset noise state."""
        self.state = np.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Sample noise."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state


@ALGO_REGISTRY.register("maddpg", replace=True)
class MADDPG(OffPolicyAlgo):
    """
    Multi-Agent Deep Deterministic Policy Gradient.

    Features:
    - Centralized critics with decentralized actors
    - Deterministic policy gradient
    - Experience replay
    - Soft target updates

    Config options:
        actor_lr: Actor learning rate (default: 1e-4)
        critic_lr: Critic learning rate (default: 1e-3)
        gamma: Discount factor (default: 0.95)
        tau: Target network update rate (default: 0.01)
        hidden_sizes: Hidden layer sizes (default: (128, 128))
        buffer_size: Replay buffer capacity (default: 100000)
        batch_size: Training batch size (default: 256)
        noise_scale: Exploration noise scale (default: 0.1)
        noise_decay: Noise decay rate per episode (default: 0.9999)
    """

    def __init__(
        self,
        obs_dim: int,
        share_obs_dim: int,
        act_dim: int,
        num_agents: int,
        cfg: Optional[Union[AlgoConfig, Dict[str, Any]]] = None,
        device: str = "cpu"
    ):
        super().__init__(obs_dim, share_obs_dim, act_dim, num_agents, cfg, device)

        # Get config values
        self.actor_lr = getattr(self.cfg, 'actor_lr', 1e-4)
        self.critic_lr = getattr(self.cfg, 'critic_lr', 1e-3)
        self.gamma = getattr(self.cfg, 'gamma', 0.95)
        self.tau = getattr(self.cfg, 'tau', 0.01)
        hidden_sizes = getattr(self.cfg, 'hidden_sizes', (128, 128))
        self.batch_size = getattr(self.cfg, 'batch_size', 256)

        # Exploration noise
        self.noise_scale = getattr(cfg, 'noise_scale', 0.1) if isinstance(cfg, dict) else 0.1
        self.noise_decay = getattr(cfg, 'noise_decay', 0.9999) if isinstance(cfg, dict) else 0.9999

        # Create networks for each agent
        self.actors: List[Actor] = []
        self.critics: List[Critic] = []
        self.target_actors: List[Actor] = []
        self.target_critics: List[Critic] = []
        self.actor_optimizers: List[torch.optim.Adam] = []
        self.critic_optimizers: List[torch.optim.Adam] = []
        self.noises: List[OrnsteinUhlenbeckNoise] = []

        for i in range(num_agents):
            # Actor
            actor = Actor(obs_dim, act_dim, hidden_sizes).to(device)
            self.actors.append(actor)
            self.target_actors.append(copy.deepcopy(actor))
            self.actor_optimizers.append(torch.optim.Adam(actor.parameters(), lr=self.actor_lr))

            # Critic (centralized)
            critic = Critic(obs_dim, act_dim, num_agents, hidden_sizes).to(device)
            self.critics.append(critic)
            self.target_critics.append(copy.deepcopy(critic))
            self.critic_optimizers.append(torch.optim.Adam(critic.parameters(), lr=self.critic_lr))

            # Exploration noise
            self.noises.append(OrnsteinUhlenbeckNoise(act_dim))

        # Training step counter
        self.train_steps = 0

    def get_actions(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get actions for all agents.

        Args:
            obs: Observations (num_agents, obs_dim)
            deterministic: If True, no exploration noise

        Returns:
            actions: Actions (num_agents, act_dim)
            log_probs: Placeholder zeros (DDPG doesn't use log probs)
        """
        actions = []

        with torch.no_grad():
            for i in range(self.num_agents):
                obs_i = torch.FloatTensor(obs[i:i+1]).to(self.device)
                action = self.actors[i](obs_i).cpu().numpy()[0]

                if not deterministic:
                    noise = self.noises[i].sample() * self.noise_scale
                    action = np.clip(action + noise, -1.0, 1.0)

                actions.append(action)

        actions = np.array(actions)
        log_probs = np.zeros((self.num_agents, 1))  # DDPG doesn't use log probs

        return actions, log_probs

    def get_values(self, share_obs: np.ndarray) -> np.ndarray:
        """
        Get Q-values (not typically used in DDPG training).

        Returns placeholder zeros for interface compatibility.
        """
        return np.zeros((self.num_agents, 1))

    def update(self, buffer: Any) -> Dict[str, float]:
        """
        Update all agents from replay buffer.

        Args:
            buffer: ReplayBuffer with stored transitions

        Returns:
            Dict with loss values
        """
        if not buffer.can_sample(self.batch_size):
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        batch = buffer.sample(self.batch_size)

        # Unpack batch
        obs = batch["obs"]  # (batch, num_agents, obs_dim)
        actions = batch["actions"]  # (batch, num_agents, act_dim)
        rewards = batch["rewards"]  # (batch, num_agents, 1)
        next_obs = batch["next_obs"]  # (batch, num_agents, obs_dim)
        dones = batch["dones"]  # (batch, num_agents, 1)

        batch_size = obs.shape[0]

        # Flatten observations and actions for critic input
        obs_flat = obs.view(batch_size, -1)  # (batch, num_agents * obs_dim)
        actions_flat = actions.view(batch_size, -1)  # (batch, num_agents * act_dim)
        next_obs_flat = next_obs.view(batch_size, -1)

        # Get target actions for next state
        target_actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                target_act = self.target_actors[i](next_obs[:, i, :])
                target_actions.append(target_act)
        target_actions = torch.stack(target_actions, dim=1)  # (batch, num_agents, act_dim)
        target_actions_flat = target_actions.view(batch_size, -1)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for i in range(self.num_agents):
            # ============ Update Critic ============
            with torch.no_grad():
                target_q = self.target_critics[i](next_obs_flat, target_actions_flat)
                y = rewards[:, i, :] + self.gamma * (1 - dones[:, i, :]) * target_q

            current_q = self.critics[i](obs_flat, actions_flat)
            critic_loss = F.mse_loss(current_q, y)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critic_optimizers[i].step()

            total_critic_loss += critic_loss.item()

            # ============ Update Actor ============
            # Get current agent's action from its policy
            current_actions = []
            for j in range(self.num_agents):
                if j == i:
                    # Use policy output for agent i
                    act_j = self.actors[j](obs[:, j, :])
                else:
                    # Use sampled actions for other agents
                    act_j = actions[:, j, :]
                current_actions.append(act_j)
            current_actions = torch.stack(current_actions, dim=1)
            current_actions_flat = current_actions.view(batch_size, -1)

            # Actor loss: maximize Q-value
            actor_loss = -self.critics[i](obs_flat, current_actions_flat).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()

            total_actor_loss += actor_loss.item()

        # Soft update target networks
        self._soft_update_targets()

        # Decay noise
        self.noise_scale *= self.noise_decay

        self.train_steps += 1

        return {
            "actor_loss": total_actor_loss / self.num_agents,
            "critic_loss": total_critic_loss / self.num_agents,
            "noise_scale": self.noise_scale
        }

    def _soft_update_targets(self):
        """Soft update all target networks."""
        for i in range(self.num_agents):
            for param, target_param in zip(
                self.actors[i].parameters(),
                self.target_actors[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.critics[i].parameters(),
                self.target_critics[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def reset_noise(self):
        """Reset exploration noise for new episode."""
        for noise in self.noises:
            noise.reset()

    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "actors": [a.state_dict() for a in self.actors],
            "critics": [c.state_dict() for c in self.critics],
            "target_actors": [a.state_dict() for a in self.target_actors],
            "target_critics": [c.state_dict() for c in self.target_critics],
            "actor_optimizers": [o.state_dict() for o in self.actor_optimizers],
            "critic_optimizers": [o.state_dict() for o in self.critic_optimizers],
            "noise_scale": self.noise_scale,
            "train_steps": self.train_steps,
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint["actors"][i])
            self.critics[i].load_state_dict(checkpoint["critics"][i])
            self.target_actors[i].load_state_dict(checkpoint["target_actors"][i])
            self.target_critics[i].load_state_dict(checkpoint["target_critics"][i])
            self.actor_optimizers[i].load_state_dict(checkpoint["actor_optimizers"][i])
            self.critic_optimizers[i].load_state_dict(checkpoint["critic_optimizers"][i])

        self.noise_scale = checkpoint.get("noise_scale", 0.1)
        self.train_steps = checkpoint.get("train_steps", 0)

    def eval_mode(self):
        """Set all networks to evaluation mode."""
        for actor in self.actors:
            actor.eval()
        for critic in self.critics:
            critic.eval()

    def train_mode(self):
        """Set all networks to training mode."""
        for actor in self.actors:
            actor.train()
        for critic in self.critics:
            critic.train()

    def to(self, device: str) -> "MADDPG":
        """Move all networks to device."""
        self.device = device
        for i in range(self.num_agents):
            self.actors[i].to(device)
            self.critics[i].to(device)
            self.target_actors[i].to(device)
            self.target_critics[i].to(device)
        return self

    @property
    def actor(self):
        """Access first actor (for compatibility)."""
        return self.actors[0]

    @property
    def critic(self):
        """Access first critic (for compatibility)."""
        return self.critics[0]
