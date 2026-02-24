"""
Multi-Agent PPO (MAPPO) Algorithm

MAPPO with parameter sharing for cooperative multi-agent tasks.
Uses CTDE: centralized critic, decentralized actors.

Reference:
    Yu et al., "The Surprising Effectiveness of PPO in Cooperative
    Multi-Agent Games", NeurIPS 2022
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch

from cosmos.registry import ALGO_REGISTRY
from cosmos.algos.base import BaseMARLAlgo, AlgoConfig, OnPolicyAlgo

# Import original MAPPO
from formation_nav.algo.mappo import MAPPO as OriginalMAPPO
from formation_nav.config import AlgoConfig as OriginalAlgoConfig


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
        num_agents: int,
        cfg: Optional[Union[AlgoConfig, Dict[str, Any]]] = None,
        device: str = "cpu"
    ):
        super().__init__(obs_dim, share_obs_dim, act_dim, num_agents, cfg, device)

        # Convert to original config format
        original_cfg = OriginalAlgoConfig()
        if hasattr(self.cfg, 'actor_lr'):
            original_cfg.actor_lr = self.cfg.actor_lr
        if hasattr(self.cfg, 'critic_lr'):
            original_cfg.critic_lr = self.cfg.critic_lr
        if hasattr(self.cfg, 'clip_param'):
            original_cfg.clip_param = self.cfg.clip_param
        if hasattr(self.cfg, 'ppo_epochs'):
            original_cfg.ppo_epochs = self.cfg.ppo_epochs
        if hasattr(self.cfg, 'num_mini_batch'):
            original_cfg.num_mini_batch = self.cfg.num_mini_batch
        if hasattr(self.cfg, 'entropy_coef'):
            original_cfg.entropy_coef = self.cfg.entropy_coef
        if hasattr(self.cfg, 'gamma'):
            original_cfg.gamma = self.cfg.gamma
        if hasattr(self.cfg, 'gae_lambda'):
            original_cfg.gae_lambda = self.cfg.gae_lambda
        if hasattr(self.cfg, 'hidden_sizes'):
            original_cfg.hidden_sizes = list(self.cfg.hidden_sizes)

        # Create original MAPPO
        self._algo = OriginalMAPPO(
            obs_dim=obs_dim,
            share_obs_dim=share_obs_dim,
            act_dim=act_dim,
            cfg=original_cfg,
            device=device
        )

    def get_actions(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample actions from policy."""
        return self._algo.get_actions(obs, deterministic)

    def get_values(self, share_obs: np.ndarray) -> np.ndarray:
        """Estimate state values."""
        return self._algo.get_values(share_obs)

    def update(self, buffer: Any) -> Dict[str, float]:
        """Update policy from buffer."""
        return self._algo.update(buffer)

    def save(self, path: str):
        """Save model checkpoint."""
        self._algo.save(path)

    def load(self, path: str):
        """Load model checkpoint."""
        self._algo.load(path)

    def eval_mode(self):
        """Set networks to evaluation mode."""
        self._algo.actor.eval()
        self._algo.critic.eval()

    def train_mode(self):
        """Set networks to training mode."""
        self._algo.actor.train()
        self._algo.critic.train()

    def to(self, device: str) -> "MAPPO":
        """Move model to device."""
        self.device = device
        self._algo.actor.to(device)
        self._algo.critic.to(device)
        self._algo.cost_critic.to(device)
        self._algo.device = device
        return self

    @property
    def actor(self):
        """Access actor network."""
        return self._algo.actor

    @property
    def critic(self):
        """Access critic network."""
        return self._algo.critic

    @property
    def cost_critic(self):
        """Access cost critic network."""
        return self._algo.cost_critic
