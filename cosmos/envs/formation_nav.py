"""
Formation Navigation Environment

Multi-robot formation navigation with:
- Obstacle avoidance
- Formation maintenance
- Goal reaching

Wraps the original FormationNavEnv with the new base class interface.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from gymnasium import spaces

from cosmos.registry import ENV_REGISTRY
from cosmos.envs.base import BaseMultiAgentEnv

# Import original environment
from formation_nav.env.formation_env import FormationNavEnv as OriginalFormationNavEnv
from formation_nav.env.formations import FormationTopology
from formation_nav.config import EnvConfig, RewardConfig


@ENV_REGISTRY.register("formation_nav", aliases=["formation", "nav"])
class FormationNavEnv(BaseMultiAgentEnv):
    """
    Multi-robot formation navigation environment.

    Agents must navigate to a goal while maintaining formation
    and avoiding obstacles.

    Config options:
        num_agents: Number of robots (default: 4)
        formation_shape: Formation type (default: "square")
        arena_size: Arena size (default: 5.0)
        num_obstacles: Number of obstacles (default: 4)
        max_steps: Maximum episode length (default: 500)
        dt: Time step (default: 0.05)
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Args:
            cfg: Configuration dict or object with env parameters.
        """
        # Convert dict to EnvConfig if needed
        if cfg is None:
            env_cfg = EnvConfig()
            reward_cfg = RewardConfig()
        elif isinstance(cfg, dict):
            env_cfg = EnvConfig()
            reward_cfg = RewardConfig()
            # Apply dict values
            for key, value in cfg.items():
                if hasattr(env_cfg, key):
                    setattr(env_cfg, key, value)
                elif hasattr(reward_cfg, key):
                    setattr(reward_cfg, key, value)
        else:
            # Assume it's a config object
            env_cfg = getattr(cfg, 'env', cfg)
            reward_cfg = getattr(cfg, 'reward', RewardConfig())

        self._env = OriginalFormationNavEnv(env_cfg, reward_cfg)
        self._env_cfg = env_cfg
        self._topology = FormationTopology(env_cfg.num_agents, "complete")

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

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        return self._env.reset(seed=seed)

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, List[Dict], Any]:
        """Execute actions."""
        return self._env.step(actions)

    def get_constraint_info(self) -> Dict[str, Any]:
        """Return safety constraint information."""
        return {
            "positions": self._env.positions.copy(),
            "velocities": self._env.velocities.copy(),
            "obstacles": self._env.obstacles.copy(),
            "desired_distances": self._env.desired_distances.copy(),
            "topology_edges": self._topology.edges(),
            "arena_size": self._env_cfg.arena_size,
            "goal": self._env.goal.copy(),
        }

    def render(self) -> Optional[np.ndarray]:
        """Render is not implemented for this env."""
        return None

    def close(self):
        """Clean up."""
        pass

    # =========================================================================
    # Additional properties for convenience
    # =========================================================================

    @property
    def positions(self) -> np.ndarray:
        """Current agent positions."""
        return self._env.positions

    @property
    def velocities(self) -> np.ndarray:
        """Current agent velocities."""
        return self._env.velocities

    @property
    def obstacles(self) -> np.ndarray:
        """Obstacle positions and radii."""
        return self._env.obstacles

    @property
    def goal(self) -> np.ndarray:
        """Goal position."""
        return self._env.goal

    @property
    def desired_distances(self) -> np.ndarray:
        """Desired inter-agent distances for formation."""
        return self._env.desired_distances

    @property
    def topology(self) -> FormationTopology:
        """Formation topology."""
        return self._topology
