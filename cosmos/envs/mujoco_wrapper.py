"""
MuJoCo Multi-Agent Environment Wrapper

Integration with MuJoCo for high-fidelity physics simulation.

Installation:
    pip install mujoco
    pip install gymnasium[mujoco]

For multi-agent MuJoCo environments:
    pip install multiagent-mujoco  # MA-MuJoCo
    # or
    pip install mamujoco  # Alternative

Reference:
    https://mujoco.org/
    https://github.com/schroederdewitt/multiagent_mujoco
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from cosmos.registry import ENV_REGISTRY
from cosmos.envs.base import BaseMultiAgentEnv

# Check MuJoCo availability
try:
    import mujoco
    import gymnasium as gym
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

# Check Multi-Agent MuJoCo
try:
    from multiagent_mujoco.mujoco_multi import MujocoMulti
    MA_MUJOCO_AVAILABLE = True
except ImportError:
    MA_MUJOCO_AVAILABLE = False


@ENV_REGISTRY.register("mujoco", aliases=["mujoco_single"])
class MuJoCoWrapper(BaseMultiAgentEnv):
    """
    Wrapper for standard MuJoCo Gymnasium environments.

    Treats single-agent MuJoCo as multi-agent by splitting
    the action space among multiple "agents".

    Environments:
    - Ant-v4: 8-DOF quadruped
    - HalfCheetah-v4: 6-DOF running robot
    - Humanoid-v4: 17-DOF humanoid
    - Walker2d-v4: 6-DOF bipedal walker

    Config options:
        env_id: Gymnasium MuJoCo env ID (default: "Ant-v4")
        num_agents: Number of agents to split actions (default: 2)
        render_mode: "human" or "rgb_array" (default: None)
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        env_id: str = "Ant-v4",
        num_agents: int = 2,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        if not MUJOCO_AVAILABLE:
            raise ImportError(
                "MuJoCo not installed. Install with:\n"
                "  pip install mujoco gymnasium[mujoco]"
            )

        if cfg:
            env_id = cfg.get("env_id", env_id)
            num_agents = cfg.get("num_agents", num_agents)
            render_mode = cfg.get("render_mode", render_mode)

        self._env_id = env_id
        self._num_agents = num_agents

        # Create environment
        self._env = gym.make(env_id, render_mode=render_mode, **kwargs)

        # Get dimensions
        total_obs_dim = self._env.observation_space.shape[0]
        total_act_dim = self._env.action_space.shape[0]

        # Split action space among agents
        self._act_dim = total_act_dim // num_agents
        if total_act_dim % num_agents != 0:
            # Pad to make divisible
            self._act_dim = (total_act_dim + num_agents - 1) // num_agents

        self._obs_dim = total_obs_dim
        self._share_obs_dim = total_obs_dim
        self._total_act_dim = total_act_dim

        self._step_count = 0
        self._max_steps = 1000

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def observation_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32
        )

    @property
    def action_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._act_dim,),
            dtype=np.float32
        )

    @property
    def share_observation_space(self):
        return self.observation_space

    def get_obs_dim(self) -> int:
        return self._obs_dim

    def get_act_dim(self) -> int:
        return self._act_dim

    def get_share_obs_dim(self) -> int:
        return self._share_obs_dim

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        obs, info = self._env.reset(seed=seed)
        self._step_count = 0

        # Same observation for all agents
        obs_multi = np.tile(obs, (self._num_agents, 1))
        share_obs = obs_multi.copy()

        return obs_multi, share_obs, info

    def step(self, actions: np.ndarray):
        # Combine actions from all agents
        combined_action = actions.flatten()[:self._total_act_dim]

        # Pad if necessary
        if len(combined_action) < self._total_act_dim:
            combined_action = np.pad(
                combined_action,
                (0, self._total_act_dim - len(combined_action))
            )

        obs, reward, terminated, truncated, info = self._env.step(combined_action)
        self._step_count += 1

        # Same observation for all agents
        obs_multi = np.tile(obs, (self._num_agents, 1))
        share_obs = obs_multi.copy()

        # Equal reward split
        rewards = np.full((self._num_agents, 1), reward / self._num_agents)
        costs = np.zeros((self._num_agents, 1))
        dones = np.full(self._num_agents, terminated)

        info_list = [info.copy() for _ in range(self._num_agents)]

        return obs_multi, share_obs, rewards, costs, dones, info_list, truncated

    def get_constraint_info(self) -> Dict[str, Any]:
        # Extract position from MuJoCo data
        if hasattr(self._env, 'data'):
            qpos = self._env.data.qpos.copy()
            qvel = self._env.data.qvel.copy()
        else:
            qpos = np.zeros(3)
            qvel = np.zeros(3)

        # Use root position as agent position
        pos = qpos[:3] if len(qpos) >= 3 else np.zeros(3)
        vel = qvel[:3] if len(qvel) >= 3 else np.zeros(3)

        return {
            "positions": np.tile(pos, (self._num_agents, 1)),
            "velocities": np.tile(vel, (self._num_agents, 1)),
            "obstacles": np.zeros((0, 3)),
            "qpos": qpos,
            "qvel": qvel,
        }

    def render(self, mode: str = "human"):
        return self._env.render()

    def close(self):
        self._env.close()


@ENV_REGISTRY.register("ma_mujoco", aliases=["multiagent_mujoco", "mamujoco"])
class MultiAgentMuJoCoWrapper(BaseMultiAgentEnv):
    """
    Wrapper for Multi-Agent MuJoCo (MA-MuJoCo).

    MA-MuJoCo decomposes standard MuJoCo robots into
    multiple agents controlling different body parts.

    Supported configurations:
    - 2x4_ant: 2 agents, each controls 4 joints of Ant
    - 4x2_ant: 4 agents, each controls 2 joints of Ant
    - 2x3_hopper: 2 agents for Hopper
    - 3x1_humanoid: 3 agents for Humanoid

    Config options:
        env_id: MA-MuJoCo scenario (default: "2x4_ant")
        render_mode: "human" or "rgb_array"
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        env_id: str = "2x4_ant",
        render_mode: Optional[str] = None,
        **kwargs
    ):
        if not MA_MUJOCO_AVAILABLE:
            raise ImportError(
                "Multi-Agent MuJoCo not installed. Install with:\n"
                "  pip install multiagent-mujoco"
            )

        if cfg:
            env_id = cfg.get("env_id", env_id)
            render_mode = cfg.get("render_mode", render_mode)

        self._env_id = env_id

        # Parse env_id format: "NxM_robot"
        parts = env_id.split("_")
        agent_config = parts[0]  # e.g., "2x4"
        robot = "_".join(parts[1:])  # e.g., "ant"

        num_str = agent_config.split("x")
        self._num_agents = int(num_str[0])

        # Create MA-MuJoCo environment
        self._env = MujocoMulti(
            env_args={
                "scenario": robot,
                "agent_conf": agent_config,
                "agent_obsk": 1,
            }
        )

        # Get dimensions
        self._obs_dim = self._env.obs_size
        self._act_dim = self._env.n_actions
        self._share_obs_dim = self._env.share_obs_size

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def observation_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32
        )

    @property
    def action_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._act_dim,),
            dtype=np.float32
        )

    @property
    def share_observation_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._share_obs_dim,),
            dtype=np.float32
        )

    def get_obs_dim(self) -> int:
        return self._obs_dim

    def get_act_dim(self) -> int:
        return self._act_dim

    def get_share_obs_dim(self) -> int:
        return self._share_obs_dim

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        obs = self._env.reset()
        share_obs = self._env.get_state()

        obs = np.array(obs)
        share_obs = np.tile(share_obs, (self._num_agents, 1))

        return obs, share_obs, {}

    def step(self, actions: np.ndarray):
        reward, terminated, info = self._env.step(actions.tolist())

        obs = np.array(self._env.get_obs())
        share_obs = self._env.get_state()
        share_obs = np.tile(share_obs, (self._num_agents, 1))

        rewards = np.full((self._num_agents, 1), reward)
        costs = np.zeros((self._num_agents, 1))
        dones = np.full(self._num_agents, terminated)
        info_list = [info.copy() for _ in range(self._num_agents)]

        return obs, share_obs, rewards, costs, dones, info_list, False

    def get_constraint_info(self) -> Dict[str, Any]:
        return {
            "positions": np.zeros((self._num_agents, 3)),
            "velocities": np.zeros((self._num_agents, 3)),
            "obstacles": np.zeros((0, 3)),
        }

    def render(self, mode: str = "human"):
        return self._env.render(mode=mode)

    def close(self):
        self._env.close()
