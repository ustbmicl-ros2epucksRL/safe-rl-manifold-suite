"""
Base class for Multi-Agent Environments.

Extends Gymnasium's Env with multi-agent specific interfaces:
- Multiple agents with individual observations
- Shared observations for centralized training
- Constraint information for safety filters
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BaseMultiAgentEnv(gym.Env, ABC):
    """
    Abstract base class for multi-agent environments.

    Provides a standardized interface for:
    - CTDE (Centralized Training, Decentralized Execution)
    - Safety-constrained RL
    - Multi-agent coordination

    Subclasses must implement:
    - num_agents: Number of agents
    - observation_space: Individual agent observation space
    - action_space: Individual agent action space
    - share_observation_space: Shared observation for CTDE
    - reset(): Reset environment
    - step(): Execute actions
    - get_constraint_info(): Return safety constraint information
    """

    @property
    @abstractmethod
    def num_agents(self) -> int:
        """Number of agents in the environment."""
        pass

    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space:
        """
        Observation space for a single agent.
        Shape: (obs_dim,)
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space:
        """
        Action space for a single agent.
        Shape: (act_dim,)
        """
        pass

    @property
    @abstractmethod
    def share_observation_space(self) -> spaces.Space:
        """
        Shared observation space for centralized critic.
        Contains global state information visible to all agents.
        Shape: (share_obs_dim,)
        """
        pass

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            obs: Individual observations, shape (num_agents, obs_dim)
            share_obs: Shared observations, shape (num_agents, share_obs_dim)
            info: Additional information dict
        """
        pass

    @abstractmethod
    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, List[Dict], Any]:
        """
        Execute actions in the environment.

        Args:
            actions: Actions for all agents, shape (num_agents, act_dim)

        Returns:
            obs: Individual observations, shape (num_agents, obs_dim)
            share_obs: Shared observations, shape (num_agents, share_obs_dim)
            rewards: Rewards, shape (num_agents, 1)
            costs: Safety costs, shape (num_agents, 1)
            dones: Done flags, shape (num_agents,)
            infos: List of info dicts, one per agent
            truncated: Truncation flag (for Gymnasium compatibility)
        """
        pass

    @abstractmethod
    def get_constraint_info(self) -> Dict[str, Any]:
        """
        Return information needed by safety filters.

        This method should return all state information required
        to compute safety constraints.

        Returns:
            Dict containing constraint-relevant state, e.g.:
            {
                "positions": np.ndarray,      # Agent positions
                "velocities": np.ndarray,     # Agent velocities
                "obstacles": np.ndarray,      # Obstacle positions/radii
                "desired_distances": np.ndarray,  # Formation distances
                ...
            }
        """
        pass

    def render(self) -> Optional[np.ndarray]:
        """Render the environment (optional)."""
        return None

    def close(self):
        """Clean up resources."""
        pass

    # =========================================================================
    # Utility methods
    # =========================================================================

    def get_obs_dim(self) -> int:
        """Get observation dimension."""
        return self.observation_space.shape[0]

    def get_act_dim(self) -> int:
        """Get action dimension."""
        return self.action_space.shape[0]

    def get_share_obs_dim(self) -> int:
        """Get shared observation dimension."""
        return self.share_observation_space.shape[0]

    def seed(self, seed: int):
        """Set random seed (deprecated, use reset(seed=...) instead)."""
        self._np_random = np.random.RandomState(seed)


class VectorizedMultiAgentEnv:
    """
    Vectorized wrapper for running multiple environments in parallel.

    Useful for collecting experience from multiple environments simultaneously.
    """

    def __init__(self, env_fns: List[callable], num_envs: int = 1):
        """
        Args:
            env_fns: List of functions that create environments.
            num_envs: Number of parallel environments.
        """
        self.num_envs = num_envs
        self.envs = [fn() for fn in env_fns[:num_envs]]
        self._env = self.envs[0]

    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    @property
    def observation_space(self) -> spaces.Space:
        return self._env.observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._env.action_space

    @property
    def share_observation_space(self) -> spaces.Space:
        return self._env.share_observation_space

    def reset(self, seed: Optional[int] = None):
        """Reset all environments."""
        results = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            results.append(env.reset(seed=env_seed))

        obs = np.stack([r[0] for r in results])
        share_obs = np.stack([r[1] for r in results])
        infos = [r[2] for r in results]

        return obs, share_obs, infos

    def step(self, actions: np.ndarray):
        """Step all environments."""
        results = []
        for i, env in enumerate(self.envs):
            results.append(env.step(actions[i]))

        obs = np.stack([r[0] for r in results])
        share_obs = np.stack([r[1] for r in results])
        rewards = np.stack([r[2] for r in results])
        costs = np.stack([r[3] for r in results])
        dones = np.stack([r[4] for r in results])
        infos = [r[5] for r in results]

        return obs, share_obs, rewards, costs, dones, infos, None

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()
