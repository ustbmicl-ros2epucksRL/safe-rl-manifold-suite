"""
Base class for Multi-Agent RL Algorithms.

Provides a standardized interface for MARL algorithms:
- Action selection (exploration and exploitation)
- Value estimation
- Policy updates
- Model saving/loading
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class AlgoConfig:
    """Default algorithm configuration."""
    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # PPO specific
    clip_param: float = 0.2
    ppo_epochs: int = 10
    num_mini_batch: int = 4
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # General
    gamma: float = 0.99
    gae_lambda: float = 0.95
    hidden_sizes: Tuple[int, ...] = (128, 128)

    # Off-policy specific (for MADDPG, etc.)
    buffer_size: int = 100000
    batch_size: int = 256
    tau: float = 0.005

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlgoConfig":
        """Create config from dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


class BaseMARLAlgo(ABC):
    """
    Abstract base class for multi-agent RL algorithms.

    Provides a standardized interface for:
    - CTDE (Centralized Training, Decentralized Execution)
    - Parameter sharing across agents
    - On-policy and off-policy methods

    Subclasses must implement:
    - get_actions(): Sample actions from policy
    - get_values(): Estimate state values
    - update(): Update policy from experience
    - save(): Save model checkpoint
    - load(): Load model checkpoint
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
        """
        Args:
            obs_dim: Individual observation dimension.
            share_obs_dim: Shared observation dimension.
            act_dim: Action dimension.
            num_agents: Number of agents.
            cfg: Algorithm configuration.
            device: Torch device ("cpu" or "cuda").
        """
        self.obs_dim = obs_dim
        self.share_obs_dim = share_obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents
        self.device = device

        # Convert dict to config if needed
        if cfg is None:
            self.cfg = AlgoConfig()
        elif isinstance(cfg, dict):
            self.cfg = AlgoConfig.from_dict(cfg)
        else:
            self.cfg = cfg

    @abstractmethod
    def get_actions(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample actions from the policy.

        Args:
            obs: Observations, shape (num_agents, obs_dim)
            deterministic: If True, return mode of distribution

        Returns:
            actions: Actions, shape (num_agents, act_dim)
            log_probs: Log probabilities, shape (num_agents, 1)
        """
        pass

    @abstractmethod
    def get_values(self, share_obs: np.ndarray) -> np.ndarray:
        """
        Estimate state values.

        Args:
            share_obs: Shared observations, shape (num_agents, share_obs_dim)

        Returns:
            values: Value estimates, shape (num_agents, 1)
        """
        pass

    @abstractmethod
    def update(self, buffer: Any) -> Dict[str, float]:
        """
        Update policy from collected experience.

        Args:
            buffer: Experience buffer (RolloutBuffer for on-policy,
                   ReplayBuffer for off-policy)

        Returns:
            info: Dict with loss values and other metrics
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load model checkpoint.

        Args:
            path: Path to load checkpoint from.
        """
        pass

    # =========================================================================
    # Optional methods with default implementations
    # =========================================================================

    def eval_mode(self):
        """Set networks to evaluation mode."""
        pass

    def train_mode(self):
        """Set networks to training mode."""
        pass

    def to(self, device: str) -> "BaseMARLAlgo":
        """Move model to device."""
        self.device = device
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for serialization."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict."""
        pass

    def get_cost_values(self, share_obs: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate cost values (for constrained RL).

        Args:
            share_obs: Shared observations, shape (num_agents, share_obs_dim)

        Returns:
            cost_values: Cost estimates, shape (num_agents, 1), or None
        """
        return None


class OnPolicyAlgo(BaseMARLAlgo):
    """Base class for on-policy algorithms (PPO, A2C, etc.)."""

    def collect_rollout(self, env, buffer, num_steps: int):
        """
        Collect experience using current policy.

        Args:
            env: Environment to collect from.
            buffer: Buffer to store experience.
            num_steps: Number of steps to collect.
        """
        raise NotImplementedError


class OffPolicyAlgo(BaseMARLAlgo):
    """Base class for off-policy algorithms (MADDPG, SAC, etc.)."""

    def add_to_buffer(self, transition: Dict[str, np.ndarray]):
        """
        Add transition to replay buffer.

        Args:
            transition: Dict with obs, actions, rewards, next_obs, dones.
        """
        raise NotImplementedError

    def sample_from_buffer(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch from replay buffer.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Batch of transitions.
        """
        raise NotImplementedError
