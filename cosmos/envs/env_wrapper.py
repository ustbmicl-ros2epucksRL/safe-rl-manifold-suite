"""
Environment Wrapper for Multi-Agent Environments

Provides a unified interface to wrap different environment sources:
- Internal environments (FormationNav)
- External environments (MPE, SMAC, etc.)
- Gymnasium-based environments

Similar to EPyMARL's GymWrapper but with safety filter support.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym

from cosmos.envs.base import BaseMultiAgentEnv


class MultiAgentEnvWrapper(BaseMultiAgentEnv):
    """
    Wrapper to convert various environment formats to our interface.

    Supports:
    - Gymnasium environments
    - PettingZoo environments (TODO)
    - SMAC environments (TODO)
    - MPE environments (TODO)
    """

    def __init__(
        self,
        env: Any,
        env_type: str = "auto",
        **kwargs
    ):
        """
        Args:
            env: The environment to wrap.
            env_type: Type of environment ("gymnasium", "pettingzoo", "smac", "auto").
        """
        self._env = env
        self._env_type = self._detect_env_type(env) if env_type == "auto" else env_type
        self._kwargs = kwargs

        # Cache spaces
        self._obs_space = None
        self._act_space = None
        self._share_obs_space = None

    def _detect_env_type(self, env: Any) -> str:
        """Detect environment type."""
        if hasattr(env, 'observation_space') and hasattr(env, 'action_space'):
            if hasattr(env, 'share_observation_space'):
                return "multi_agent_gym"
            return "gymnasium"
        if hasattr(env, 'agents'):
            return "pettingzoo"
        return "unknown"

    @property
    def num_agents(self) -> int:
        if hasattr(self._env, 'num_agents'):
            return self._env.num_agents
        if hasattr(self._env, 'n_agents'):
            return self._env.n_agents
        if hasattr(self._env, 'agents'):
            return len(self._env.agents)
        return 1

    @property
    def observation_space(self) -> gym.Space:
        if self._obs_space is not None:
            return self._obs_space
        if hasattr(self._env, 'observation_space'):
            self._obs_space = self._env.observation_space
        return self._obs_space

    @property
    def action_space(self) -> gym.Space:
        if self._act_space is not None:
            return self._act_space
        if hasattr(self._env, 'action_space'):
            self._act_space = self._env.action_space
        return self._act_space

    @property
    def share_observation_space(self) -> gym.Space:
        if self._share_obs_space is not None:
            return self._share_obs_space
        if hasattr(self._env, 'share_observation_space'):
            self._share_obs_space = self._env.share_observation_space
        else:
            # Create shared space by concatenating all observations
            obs_dim = self.observation_space.shape[0]
            share_dim = obs_dim * self.num_agents
            self._share_obs_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(share_dim,), dtype=np.float32
            )
        return self._share_obs_space

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        if self._env_type == "multi_agent_gym":
            return self._env.reset(seed=seed)
        elif self._env_type == "gymnasium":
            obs, info = self._env.reset(seed=seed)
            # Convert single-agent to multi-agent format
            obs = np.expand_dims(obs, 0)
            share_obs = obs.copy()
            return obs, share_obs, info
        else:
            raise NotImplementedError(f"Env type {self._env_type} not supported")

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, List[Dict], Any]:
        """Step environment."""
        if self._env_type == "multi_agent_gym":
            return self._env.step(actions)
        elif self._env_type == "gymnasium":
            # Single agent
            action = actions[0] if actions.ndim > 1 else actions
            obs, reward, terminated, truncated, info = self._env.step(action)
            obs = np.expand_dims(obs, 0)
            share_obs = obs.copy()
            rewards = np.array([[reward]])
            costs = np.array([[0.0]])  # No cost for generic gym envs
            dones = np.array([terminated or truncated])
            return obs, share_obs, rewards, costs, dones, [info], None
        else:
            raise NotImplementedError(f"Env type {self._env_type} not supported")

    def get_constraint_info(self) -> Dict[str, Any]:
        """Get constraint info if available."""
        if hasattr(self._env, 'get_constraint_info'):
            return self._env.get_constraint_info()
        return {}

    def render(self):
        if hasattr(self._env, 'render'):
            return self._env.render()
        return None

    def close(self):
        if hasattr(self._env, 'close'):
            self._env.close()


def make_env(
    env_name: str,
    env_type: str = "internal",
    **kwargs
) -> BaseMultiAgentEnv:
    """
    Create environment by name.

    Args:
        env_name: Environment name.
        env_type: "internal" (our envs), "gymnasium", "pettingzoo", "smac".
        **kwargs: Environment arguments.

    Returns:
        Wrapped environment.

    Example:
        >>> env = make_env("formation_nav", num_agents=4)
        >>> env = make_env("CartPole-v1", env_type="gymnasium")
    """
    if env_type == "internal":
        from cosmos.registry import ENV_REGISTRY
        return ENV_REGISTRY.build(env_name, cfg=kwargs)

    elif env_type == "gymnasium":
        gym_env = gym.make(env_name, **kwargs)
        return MultiAgentEnvWrapper(gym_env, env_type="gymnasium")

    elif env_type == "mpe":
        try:
            from pettingzoo.mpe import simple_spread_v3, simple_tag_v3
            env_map = {
                "simple_spread": simple_spread_v3,
                "simple_tag": simple_tag_v3,
            }
            if env_name in env_map:
                env = env_map[env_name].parallel_env(**kwargs)
                return MultiAgentEnvWrapper(env, env_type="pettingzoo")
        except ImportError:
            raise ImportError("PettingZoo not installed. Run: pip install pettingzoo[mpe]")

    elif env_type == "smac":
        try:
            from smac.env import StarCraft2Env
            env = StarCraft2Env(map_name=env_name, **kwargs)
            return MultiAgentEnvWrapper(env, env_type="smac")
        except ImportError:
            raise ImportError("SMAC not installed. See: https://github.com/oxwhirl/smac")

    else:
        raise ValueError(f"Unknown env_type: {env_type}")
