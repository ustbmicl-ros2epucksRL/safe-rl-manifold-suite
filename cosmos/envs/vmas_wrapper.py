"""
VMAS Environment Wrapper

Vectorized Multi-Agent Simulator integration.
VMAS is a PyTorch-based simulator for efficient batched MARL.

Installation:
    pip install vmas

Reference:
    https://github.com/proroklab/VectorizedMultiAgentSimulator
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from cosmos.registry import ENV_REGISTRY
from cosmos.envs.base import BaseMultiAgentEnv

try:
    import torch
    import vmas
    from vmas import make_env as vmas_make_env
    VMAS_AVAILABLE = True
except ImportError:
    VMAS_AVAILABLE = False


@ENV_REGISTRY.register("vmas", aliases=["vmas_navigation", "vmas_formation"])
class VMASWrapper(BaseMultiAgentEnv):
    """
    Wrapper for VMAS (Vectorized Multi-Agent Simulator).

    VMAS provides efficient vectorized simulation using PyTorch,
    enabling fast parallel environment execution on GPU.

    Supported scenarios:
    - navigation: Multi-agent navigation to goals
    - formation_control: Formation maintenance
    - transport: Cooperative transport
    - wheel: Differential drive robots
    - And many more...

    Config options:
        scenario: VMAS scenario name (default: "navigation")
        num_envs: Number of parallel environments (default: 32)
        device: "cpu" or "cuda" (default: "cpu")
        continuous_actions: Use continuous actions (default: True)
        max_steps: Maximum steps per episode (default: 100)
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        scenario: str = "navigation",
        num_agents: int = 4,
        num_envs: int = 32,
        device: str = "cpu",
        continuous_actions: bool = True,
        max_steps: int = 100,
        **kwargs
    ):
        if not VMAS_AVAILABLE:
            raise ImportError(
                "VMAS not installed. Install with: pip install vmas"
            )

        # Parse config
        if cfg:
            scenario = cfg.get("scenario", scenario)
            num_agents = cfg.get("num_agents", num_agents)
            num_envs = cfg.get("num_envs", num_envs)
            device = cfg.get("device", device)
            continuous_actions = cfg.get("continuous_actions", continuous_actions)
            max_steps = cfg.get("max_steps", max_steps)

        self._scenario = scenario
        self._num_agents = num_agents
        self._num_envs = num_envs
        self._device = device
        self._max_steps = max_steps

        # Create VMAS environment
        self._env = vmas_make_env(
            scenario=scenario,
            num_envs=num_envs,
            device=device,
            continuous_actions=continuous_actions,
            n_agents=num_agents,
            max_steps=max_steps,
            **kwargs
        )

        # Get space dimensions
        self._obs_dim = self._env.observation_space[0].shape[0]
        self._act_dim = self._env.action_space[0].shape[0] if continuous_actions else 1

        # Compute shared observation dimension
        self._share_obs_dim = self._obs_dim * num_agents

        self._step_count = 0

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def observation_space(self):
        return self._env.observation_space[0]

    @property
    def action_space(self):
        return self._env.action_space[0]

    @property
    def share_observation_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._share_obs_dim,),
            dtype=np.float32
        )

    def get_obs_dim(self) -> int:
        return self._obs_dim

    def get_act_dim(self) -> int:
        return self._act_dim

    def get_share_obs_dim(self) -> int:
        return self._share_obs_dim

    def reset(
        self,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        if seed is not None:
            torch.manual_seed(seed)

        obs_list = self._env.reset()
        self._step_count = 0

        # Convert to numpy: (num_agents, obs_dim)
        # VMAS returns list of (num_envs, obs_dim) tensors
        obs = torch.stack(obs_list, dim=1)[0].cpu().numpy()  # Take first env

        # Shared observation: concatenate all agents
        share_obs = obs.flatten()
        share_obs = np.tile(share_obs, (self._num_agents, 1))

        return obs, share_obs, {}

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict], bool]:
        """Step environment."""
        # Convert actions to VMAS format: list of (num_envs, act_dim) tensors
        actions_tensor = torch.FloatTensor(actions).to(self._device)

        # Expand for batch dimension and split by agent
        actions_list = [
            actions_tensor[i:i+1].expand(self._num_envs, -1)
            for i in range(self._num_agents)
        ]

        # Step VMAS
        obs_list, rewards_list, dones_list, infos = self._env.step(actions_list)

        self._step_count += 1

        # Convert outputs: take first environment
        obs = torch.stack(obs_list, dim=1)[0].cpu().numpy()
        rewards = torch.stack(rewards_list, dim=1)[0].cpu().numpy().reshape(-1, 1)
        dones = torch.stack(dones_list, dim=1)[0].cpu().numpy()

        # Shared observation
        share_obs = obs.flatten()
        share_obs = np.tile(share_obs, (self._num_agents, 1))

        # Costs (VMAS doesn't have built-in costs, use collisions if available)
        costs = np.zeros((self._num_agents, 1))

        # Info per agent
        info_list = [{} for _ in range(self._num_agents)]

        truncated = self._step_count >= self._max_steps

        return obs, share_obs, rewards, costs, dones, info_list, truncated

    def get_constraint_info(self) -> Dict[str, Any]:
        """Get constraint information for safety filters."""
        # Get agent positions from VMAS world
        positions = []
        velocities = []

        for agent in self._env.world.agents:
            pos = agent.state.pos[0].cpu().numpy()  # First env
            vel = agent.state.vel[0].cpu().numpy()
            positions.append(np.append(pos, 0))  # Add z=0
            velocities.append(np.append(vel, 0))

        return {
            "positions": np.array(positions),
            "velocities": np.array(velocities),
            "obstacles": np.zeros((0, 3)),
        }

    def render(self, mode: str = "human"):
        """Render environment."""
        return self._env.render(mode=mode)

    def close(self):
        """Close environment."""
        pass


# Convenience function
def make_vmas_env(scenario: str = "navigation", **kwargs) -> VMASWrapper:
    """Create VMAS environment."""
    return VMASWrapper(scenario=scenario, **kwargs)
