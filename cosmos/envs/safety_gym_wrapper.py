"""
Safety-Gym Environment Wrapper

Integration with OpenAI Safety-Gym and Safety-Gymnasium.

Installation:
    pip install safety-gymnasium  # Recommended (maintained fork)
    # or
    pip install safety-gym  # Original (requires MuJoCo license)

Reference:
    https://github.com/PKU-Alignment/safety-gymnasium
    https://github.com/openai/safety-gym (original)
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from cosmos.registry import ENV_REGISTRY
from cosmos.envs.base import BaseMultiAgentEnv

# Try safety-gymnasium first (maintained), then safety-gym (original)
try:
    import safety_gymnasium as safety_gym
    from safety_gymnasium import make as sg_make
    SAFETY_GYM_AVAILABLE = True
    SAFETY_GYM_VERSION = "safety_gymnasium"
except ImportError:
    try:
        import safety_gym
        from safety_gym.envs.engine import Engine
        SAFETY_GYM_AVAILABLE = True
        SAFETY_GYM_VERSION = "safety_gym"
    except ImportError:
        SAFETY_GYM_AVAILABLE = False
        SAFETY_GYM_VERSION = None


@ENV_REGISTRY.register("safety_gym", aliases=["safetygym", "safe_gym"])
class SafetyGymWrapper(BaseMultiAgentEnv):
    """
    Wrapper for Safety-Gym / Safety-Gymnasium.

    Safety-Gym provides constrained RL environments with:
    - Cost functions for constraint violations
    - Various robot types (Point, Car, Doggo)
    - Multiple task types (Goal, Button, Push)
    - Configurable hazards and obstacles

    Environments:
    - SafetyPointGoal1-v0: Point robot navigating to goal
    - SafetyCarGoal1-v0: Car robot navigating to goal
    - SafetyPointButton1-v0: Point robot pressing buttons
    - SafetyPointPush1-v0: Point robot pushing objects

    Config options:
        env_id: Environment ID (default: "SafetyPointGoal1-v0")
        num_agents: Number of agents (default: 1, multi-agent via vectorization)
        render_mode: "human" or "rgb_array" (default: None)
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        env_id: str = "SafetyPointGoal1-v0",
        num_agents: int = 1,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        if not SAFETY_GYM_AVAILABLE:
            raise ImportError(
                "Safety-Gym not installed. Install with:\n"
                "  pip install safety-gymnasium  # Recommended\n"
                "  # or\n"
                "  pip install safety-gym mujoco-py  # Original"
            )

        # Parse config
        if cfg:
            env_id = cfg.get("env_id", env_id)
            num_agents = cfg.get("num_agents", num_agents)
            render_mode = cfg.get("render_mode", render_mode)

        self._env_id = env_id
        self._num_agents = num_agents

        # Create environment(s)
        if SAFETY_GYM_VERSION == "safety_gymnasium":
            self._env = sg_make(env_id, render_mode=render_mode, **kwargs)
        else:
            import gym
            self._env = gym.make(env_id, **kwargs)

        # Get dimensions
        self._obs_dim = self._env.observation_space.shape[0]
        self._act_dim = self._env.action_space.shape[0]
        self._share_obs_dim = self._obs_dim * num_agents

        self._step_count = 0
        self._max_steps = 1000

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

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
        if SAFETY_GYM_VERSION == "safety_gymnasium":
            obs, info = self._env.reset(seed=seed)
        else:
            self._env.seed(seed)
            obs = self._env.reset()
            info = {}

        self._step_count = 0

        # Expand for multi-agent format
        obs = np.tile(obs, (self._num_agents, 1))
        share_obs = obs.copy()

        return obs, share_obs, info

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict], bool]:
        """Step environment."""
        # Use first agent's action (single-agent env)
        action = actions[0] if actions.ndim > 1 else actions

        if SAFETY_GYM_VERSION == "safety_gymnasium":
            # safety-gymnasium returns 6 values: obs, reward, cost, terminated, truncated, info
            obs, reward, cost, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
        else:
            # Original safety-gym returns 4 values with cost in info
            obs, reward, done, info = self._env.step(action)
            cost = info.get("cost", 0.0)
            truncated = False

        self._step_count += 1

        # Expand for multi-agent format
        obs = np.tile(obs, (self._num_agents, 1))
        share_obs = obs.copy()
        rewards = np.full((self._num_agents, 1), reward)
        costs = np.full((self._num_agents, 1), cost)
        dones = np.full(self._num_agents, done)

        info_list = [info.copy() for _ in range(self._num_agents)]

        return obs, share_obs, rewards, costs, dones, info_list, truncated

    def get_constraint_info(self) -> Dict[str, Any]:
        """Get constraint information for safety filters."""
        # Extract robot position from environment
        if hasattr(self._env, 'robot_pos'):
            robot_pos = self._env.robot_pos()
        elif hasattr(self._env, 'world') and hasattr(self._env.world, 'robot_pos'):
            robot_pos = self._env.world.robot_pos()
        else:
            robot_pos = np.zeros(3)

        if hasattr(self._env, 'robot_vel'):
            robot_vel = self._env.robot_vel()
        else:
            robot_vel = np.zeros(3)

        # Get hazard positions
        hazards = []
        if hasattr(self._env, 'hazards_pos'):
            hazards = self._env.hazards_pos
        elif hasattr(self._env, 'world') and hasattr(self._env.world, 'hazards_pos'):
            hazards = self._env.world.hazards_pos

        positions = np.tile(robot_pos, (self._num_agents, 1))
        velocities = np.tile(robot_vel, (self._num_agents, 1))

        return {
            "positions": positions,
            "velocities": velocities,
            "obstacles": np.array(hazards) if len(hazards) > 0 else np.zeros((0, 3)),
            "hazards": hazards,
        }

    def render(self, mode: str = "human"):
        """Render environment."""
        return self._env.render()

    def close(self):
        """Close environment."""
        self._env.close()


# Multi-agent Safety-Gym (experimental)
@ENV_REGISTRY.register("ma_safety_gym", aliases=["multi_agent_safety_gym"])
class MultiAgentSafetyGymWrapper(BaseMultiAgentEnv):
    """
    Multi-Agent Safety-Gym wrapper.

    Runs multiple independent Safety-Gym environments in parallel,
    treating each as a separate agent.

    Config options:
        env_id: Base environment ID
        num_agents: Number of parallel agents/environments
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        env_id: str = "SafetyPointGoal1-v0",
        num_agents: int = 4,
        **kwargs
    ):
        if not SAFETY_GYM_AVAILABLE:
            raise ImportError("Safety-Gym not installed")

        if cfg:
            env_id = cfg.get("env_id", env_id)
            num_agents = cfg.get("num_agents", num_agents)

        self._num_agents = num_agents
        self._env_id = env_id

        # Create separate environment for each agent
        self._envs = []
        for _ in range(num_agents):
            if SAFETY_GYM_VERSION == "safety_gymnasium":
                env = sg_make(env_id, **kwargs)
            else:
                import gym
                env = gym.make(env_id, **kwargs)
            self._envs.append(env)

        # Get dimensions from first env
        self._obs_dim = self._envs[0].observation_space.shape[0]
        self._act_dim = self._envs[0].action_space.shape[0]
        self._share_obs_dim = self._obs_dim * num_agents

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def observation_space(self):
        return self._envs[0].observation_space

    @property
    def action_space(self):
        return self._envs[0].action_space

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
        obs_list = []
        for i, env in enumerate(self._envs):
            env_seed = seed + i if seed else None
            if SAFETY_GYM_VERSION == "safety_gymnasium":
                obs, _ = env.reset(seed=env_seed)
            else:
                if env_seed:
                    env.seed(env_seed)
                obs = env.reset()
            obs_list.append(obs)

        obs = np.stack(obs_list)
        share_obs = np.tile(obs.flatten(), (self._num_agents, 1))
        return obs, share_obs, {}

    def step(self, actions: np.ndarray):
        obs_list, reward_list, cost_list, done_list, info_list = [], [], [], [], []

        for i, (env, action) in enumerate(zip(self._envs, actions)):
            if SAFETY_GYM_VERSION == "safety_gymnasium":
                # safety-gymnasium returns 6 values: obs, reward, cost, terminated, truncated, info
                obs, reward, cost, term, trunc, info = env.step(action)
                done = term or trunc
            else:
                # Original safety-gym returns 4 values with cost in info
                obs, reward, done, info = env.step(action)
                cost = info.get("cost", 0.0)

            obs_list.append(obs)
            reward_list.append(reward)
            cost_list.append(cost)
            done_list.append(done)
            info_list.append(info)

        obs = np.stack(obs_list)
        share_obs = np.tile(obs.flatten(), (self._num_agents, 1))
        rewards = np.array(reward_list).reshape(-1, 1)
        costs = np.array(cost_list).reshape(-1, 1)
        dones = np.array(done_list)

        return obs, share_obs, rewards, costs, dones, info_list, False

    def get_constraint_info(self) -> Dict[str, Any]:
        positions = []
        velocities = []
        for env in self._envs:
            if hasattr(env, 'robot_pos'):
                positions.append(env.robot_pos())
            else:
                positions.append(np.zeros(3))
            if hasattr(env, 'robot_vel'):
                velocities.append(env.robot_vel())
            else:
                velocities.append(np.zeros(3))

        return {
            "positions": np.array(positions),
            "velocities": np.array(velocities),
            "obstacles": np.zeros((0, 3)),
        }

    def render(self, mode: str = "human"):
        return self._envs[0].render()

    def close(self):
        for env in self._envs:
            env.close()
