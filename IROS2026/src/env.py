"""
Environment wrapper for Safety-Gymnasium.

Provides unified interface for:
- SafetyPointGoal1-v0
- SafetyPointCircle1-v0
- SafetyPointPush1-v0
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any


class SimpleBox:
    """Simple Box space for MockEnv (no gymnasium dependency)."""

    def __init__(self, low: float, high: float, shape: Tuple[int, ...]):
        self.low = low
        self.high = high
        self.shape = shape

    def sample(self) -> np.ndarray:
        """Sample random action."""
        return np.random.uniform(self.low, self.high, self.shape)


@dataclass
class EnvConfig:
    """Environment configuration."""
    env_id: str = "SafetyPointGoal1-v0"
    max_episode_steps: int = 1000
    hazard_radius: float = 0.2
    safety_margin: float = 0.1


class SafetyGymEnv:
    """Wrapper for Safety-Gymnasium environments."""

    def __init__(self, env_id: str = "SafetyPointGoal1-v0", render_mode: str = None):
        self.env_id = env_id
        self.render_mode = render_mode
        self._env = None
        self._init_env()

    def _init_env(self):
        """Initialize the environment."""
        try:
            import safety_gymnasium
            self._env = safety_gymnasium.make(
                self.env_id,
                render_mode=self.render_mode
            )
            self._use_mock = False
        except Exception as e:
            print(f"Warning: Could not create {self.env_id}: {e}")
            print("Using MockEnv for testing.")
            self._env = MockEnv()
            self._use_mock = True

    @property
    def obs_dim(self) -> int:
        """Observation dimension."""
        if self._use_mock:
            return self._env.obs_dim
        return self._env.observation_space.shape[0]

    @property
    def act_dim(self) -> int:
        """Action dimension."""
        if self._use_mock:
            return self._env.act_dim
        return self._env.action_space.shape[0]

    @property
    def action_space(self):
        """Action space."""
        return self._env.action_space

    @property
    def observation_space(self):
        """Observation space."""
        return self._env.observation_space

    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        if self._use_mock:
            return self._env.reset(seed=seed)

        obs, info = self._env.reset(seed=seed)

        # Extract additional info
        info['robot_pos'] = self._get_robot_pos()
        info['robot_vel'] = self._get_robot_vel()
        info['hazards'] = self._get_hazards()
        info['goal_pos'] = self._get_goal_pos()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, float, bool, bool, Dict]:
        """
        Step environment.

        Returns:
            obs: observation
            reward: task reward
            cost: safety cost (1 if collision, 0 otherwise)
            terminated: episode done due to success/failure
            truncated: episode done due to time limit
            info: additional information
        """
        if self._use_mock:
            return self._env.step(action)

        obs, reward, cost, terminated, truncated, info = self._env.step(action)

        # Extract additional info
        info['robot_pos'] = self._get_robot_pos()
        info['robot_vel'] = self._get_robot_vel()
        info['hazards'] = self._get_hazards()
        info['goal_pos'] = self._get_goal_pos()

        return obs, reward, cost, terminated, truncated, info

    def _get_robot_pos(self) -> np.ndarray:
        """Get robot pose [x, y, theta]."""
        try:
            task = self._env.unwrapped.task
            agent = task.agent
            pos = agent.pos

            # Get heading from rotation matrix
            theta = 0.0
            if hasattr(agent, 'mat') and agent.mat is not None:
                mat = agent.mat
                theta = np.arctan2(mat[1, 0], mat[0, 0])

            return np.array([pos[0], pos[1], theta])
        except Exception:
            return np.zeros(3)

    def _get_robot_vel(self) -> np.ndarray:
        """Get robot velocity [vx, vy, omega]."""
        try:
            task = self._env.unwrapped.task
            agent = task.agent
            vel = agent.vel
            return np.array([vel[0], vel[1], 0.0])
        except Exception:
            return np.zeros(3)

    def _get_hazards(self) -> List[np.ndarray]:
        """Get hazard positions."""
        try:
            task = self._env.unwrapped.task
            hazards = []

            if hasattr(task, 'hazards'):
                hazards_obj = task.hazards
                if hasattr(hazards_obj, 'pos'):
                    for pos in hazards_obj.pos:
                        hazards.append(np.array([pos[0], pos[1]]))

            return hazards
        except Exception:
            return []

    def _get_goal_pos(self) -> np.ndarray:
        """Get goal position."""
        try:
            task = self._env.unwrapped.task
            if hasattr(task, 'goal') and hasattr(task.goal, 'pos'):
                pos = task.goal.pos
                return np.array([pos[0], pos[1]])
            return np.zeros(2)
        except Exception:
            return np.zeros(2)

    def close(self):
        """Close environment."""
        if self._env is not None:
            self._env.close()

    def render(self):
        """Render environment."""
        if hasattr(self._env, 'render'):
            return self._env.render()


class MockEnv:
    """Mock environment for testing without Safety-Gymnasium."""

    def __init__(self):
        self.obs_dim = 60
        self.act_dim = 2
        self._step = 0
        self._max_steps = 1000

        # Robot state
        self._robot_pos = np.zeros(3)  # [x, y, theta]
        self._robot_vel = np.zeros(3)  # [vx, vy, omega]

        # Environment
        self._hazards = [
            np.array([1.0, 0.5]),
            np.array([-0.5, 1.0]),
            np.array([0.5, -1.0]),
            np.array([-1.0, -0.5]),
        ]
        self._goal = np.array([2.0, 2.0])
        self._hazard_radius = 0.2

    @property
    def observation_space(self):
        """Observation space."""
        return SimpleBox(low=-np.inf, high=np.inf, shape=(self.obs_dim,))

    @property
    def action_space(self):
        """Action space."""
        return SimpleBox(low=-1.0, high=1.0, shape=(self.act_dim,))

    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)

        self._step = 0
        self._robot_pos = np.array([0.0, 0.0, 0.0])
        self._robot_vel = np.zeros(3)

        # Randomize goal
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(1.5, 2.5)
        self._goal = np.array([dist * np.cos(angle), dist * np.sin(angle)])

        obs = self._get_obs()
        info = {
            'robot_pos': self._robot_pos.copy(),
            'robot_vel': self._robot_vel.copy(),
            'hazards': [h.copy() for h in self._hazards],
            'goal_pos': self._goal.copy(),
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, float, bool, bool, Dict]:
        """Step environment."""
        action = np.clip(action, -1.0, 1.0)

        # Simple differential drive dynamics
        v = action[0]  # forward velocity
        omega = action[1]  # angular velocity

        dt = 0.1
        theta = self._robot_pos[2]

        # Update position
        self._robot_pos[0] += v * np.cos(theta) * dt
        self._robot_pos[1] += v * np.sin(theta) * dt
        self._robot_pos[2] += omega * dt

        # Update velocity
        self._robot_vel = np.array([v * np.cos(theta), v * np.sin(theta), omega])

        # Check collision
        cost = 0.0
        for hazard in self._hazards:
            dist = np.linalg.norm(self._robot_pos[:2] - hazard)
            if dist < self._hazard_radius:
                cost = 1.0
                break

        # Compute reward
        dist_to_goal = np.linalg.norm(self._robot_pos[:2] - self._goal)
        reward = -0.01 * dist_to_goal  # distance penalty

        # Check termination
        terminated = dist_to_goal < 0.3
        if terminated:
            reward += 10.0  # goal bonus

        self._step += 1
        truncated = self._step >= self._max_steps

        obs = self._get_obs()
        info = {
            'robot_pos': self._robot_pos.copy(),
            'robot_vel': self._robot_vel.copy(),
            'hazards': [h.copy() for h in self._hazards],
            'goal_pos': self._goal.copy(),
        }

        return obs, reward, cost, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get observation."""
        obs = np.zeros(self.obs_dim)

        # Robot state
        obs[0:3] = self._robot_pos
        obs[3:6] = self._robot_vel

        # Goal direction
        goal_dir = self._goal - self._robot_pos[:2]
        obs[6:8] = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)
        obs[8] = np.linalg.norm(goal_dir)

        # Hazard lidar (simplified)
        for i, hazard in enumerate(self._hazards[:8]):
            dist = np.linalg.norm(self._robot_pos[:2] - hazard)
            obs[10 + i * 2] = hazard[0] - self._robot_pos[0]
            obs[10 + i * 2 + 1] = hazard[1] - self._robot_pos[1]

        return obs

    def close(self):
        """Close environment."""
        pass
