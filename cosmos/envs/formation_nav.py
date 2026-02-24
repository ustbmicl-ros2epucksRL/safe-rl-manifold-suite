"""
Formation Navigation Environment

Multi-robot formation navigation with:
- Obstacle avoidance
- Formation maintenance
- Goal reaching

Self-contained implementation for the COSMOS framework.
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import numpy as np
from gymnasium import spaces

from cosmos.registry import ENV_REGISTRY
from cosmos.envs.base import BaseMultiAgentEnv
from cosmos.envs.formations import FormationTopology


@dataclass
class EnvConfig:
    """Environment configuration."""
    num_agents: int = 4
    num_obstacles: int = 4
    arena_size: float = 10.0
    dt: float = 0.05
    max_steps: int = 500
    formation_shape: str = "square"
    formation_radius: float = 1.0
    max_velocity: float = 2.0
    max_acceleration: float = 1.0
    obstacle_radius: float = 0.5
    goal_threshold: float = 0.5


@dataclass
class RewardConfig:
    """Reward configuration."""
    w_nav: float = 1.0
    w_formation: float = 0.1
    w_smooth: float = 0.01
    goal_bonus: float = 10.0


@ENV_REGISTRY.register("formation_nav", aliases=["formation", "nav"])
class FormationNavEnv(BaseMultiAgentEnv):
    """
    Multi-robot formation navigation environment.

    Agents must navigate to a goal while maintaining formation
    and avoiding obstacles.

    Config options:
        num_agents: Number of robots (default: 4)
        formation_shape: Formation type (default: "square")
        arena_size: Arena size (default: 10.0)
        num_obstacles: Number of obstacles (default: 4)
        max_steps: Maximum episode length (default: 500)
        dt: Time step (default: 0.05)
    """

    def __init__(self, cfg: Optional[Any] = None, reward_cfg: Optional[Any] = None):
        """
        Args:
            cfg: Configuration dict or EnvConfig object.
            reward_cfg: Reward configuration (optional).
        """
        # Parse configuration
        if cfg is None:
            self._cfg = EnvConfig()
        elif isinstance(cfg, dict):
            self._cfg = EnvConfig(**{k: v for k, v in cfg.items() if hasattr(EnvConfig, k)})
        else:
            self._cfg = cfg

        if reward_cfg is None:
            self._reward_cfg = RewardConfig()
        elif isinstance(reward_cfg, dict):
            self._reward_cfg = RewardConfig(**{k: v for k, v in reward_cfg.items() if hasattr(RewardConfig, k)})
        else:
            self._reward_cfg = reward_cfg

        self._num_agents = self._cfg.num_agents
        self._arena_size = self._cfg.arena_size
        self._dt = self._cfg.dt
        self._max_steps = self._cfg.max_steps

        # State variables
        self._positions = np.zeros((self._num_agents, 2))
        self._velocities = np.zeros((self._num_agents, 2))
        self._goal = np.zeros(2)
        self._obstacles = np.zeros((self._cfg.num_obstacles, 3))  # x, y, radius
        self._step_count = 0
        self._prev_actions = np.zeros((self._num_agents, 2))

        # Formation topology
        self._topology = FormationTopology(self._num_agents, "complete")
        self._desired_distances = self._compute_desired_distances()

        # Observation and action spaces
        # Observation: [pos_x, pos_y, vel_x, vel_y, goal_x, goal_y, rel_obs..., rel_agents...]
        obs_dim = 4 + 2 + self._cfg.num_obstacles * 3 + (self._num_agents - 1) * 4
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # Shared observation: concatenation of all agents' observations
        self._share_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._num_agents * obs_dim,), dtype=np.float32
        )

    def _compute_desired_distances(self) -> np.ndarray:
        """Compute desired inter-agent distances based on formation shape."""
        n = self._num_agents
        distances = np.zeros((n, n))
        radius = self._cfg.formation_radius

        if self._cfg.formation_shape == "square" and n == 4:
            # Square formation
            side = radius * np.sqrt(2)
            diag = radius * 2
            distances = np.array([
                [0, side, diag, side],
                [side, 0, side, diag],
                [diag, side, 0, side],
                [side, diag, side, 0]
            ])
        elif self._cfg.formation_shape == "triangle" and n == 3:
            # Equilateral triangle
            side = radius * np.sqrt(3)
            distances = np.array([
                [0, side, side],
                [side, 0, side],
                [side, side, 0]
            ])
        elif self._cfg.formation_shape == "line":
            # Line formation
            for i in range(n):
                for j in range(n):
                    distances[i, j] = abs(i - j) * radius
        else:
            # Default: regular polygon
            for i in range(n):
                for j in range(n):
                    if i != j:
                        angle_diff = 2 * np.pi * abs(i - j) / n
                        distances[i, j] = 2 * radius * np.sin(angle_diff / 2)

        return distances

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def share_observation_space(self) -> spaces.Space:
        return self._share_observation_space

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    @property
    def obstacles(self) -> np.ndarray:
        return self._obstacles

    @property
    def goal(self) -> np.ndarray:
        return self._goal

    @property
    def desired_distances(self) -> np.ndarray:
        return self._desired_distances

    @property
    def topology(self) -> FormationTopology:
        return self._topology

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        self._step_count = 0

        # Initialize agent positions in a cluster
        center = np.random.uniform(1, self._arena_size - 1, size=2)
        angles = np.linspace(0, 2 * np.pi, self._num_agents, endpoint=False)
        for i in range(self._num_agents):
            offset = 0.5 * np.array([np.cos(angles[i]), np.sin(angles[i])])
            self._positions[i] = center + offset

        # Initialize velocities to zero
        self._velocities = np.zeros((self._num_agents, 2))

        # Random goal position
        self._goal = np.random.uniform(2, self._arena_size - 2, size=2)

        # Random obstacles (away from agents and goal)
        self._obstacles = np.zeros((self._cfg.num_obstacles, 3))
        for i in range(self._cfg.num_obstacles):
            while True:
                pos = np.random.uniform(1, self._arena_size - 1, size=2)
                # Check distance from agents
                min_dist_agents = np.min(np.linalg.norm(self._positions - pos, axis=1))
                # Check distance from goal
                dist_goal = np.linalg.norm(self._goal - pos)
                if min_dist_agents > 1.5 and dist_goal > 1.5:
                    self._obstacles[i, :2] = pos
                    self._obstacles[i, 2] = self._cfg.obstacle_radius
                    break

        self._prev_actions = np.zeros((self._num_agents, 2))

        obs = self._get_observations()
        share_obs = self._get_share_observations(obs)
        info = {"goal": self._goal.copy()}

        return obs, share_obs, info

    def _get_observations(self) -> np.ndarray:
        """Get observations for all agents."""
        obs_list = []
        for i in range(self._num_agents):
            obs_i = self._get_agent_observation(i)
            obs_list.append(obs_i)
        return np.array(obs_list, dtype=np.float32)

    def _get_agent_observation(self, agent_id: int) -> np.ndarray:
        """Get observation for a single agent."""
        obs = []

        # Own position and velocity
        obs.extend(self._positions[agent_id])
        obs.extend(self._velocities[agent_id])

        # Relative goal position
        rel_goal = self._goal - self._positions[agent_id]
        obs.extend(rel_goal)

        # Relative obstacle positions and radii
        for j in range(self._cfg.num_obstacles):
            rel_obs = self._obstacles[j, :2] - self._positions[agent_id]
            obs.extend(rel_obs)
            obs.append(self._obstacles[j, 2])

        # Relative positions and velocities of other agents
        for j in range(self._num_agents):
            if j != agent_id:
                rel_pos = self._positions[j] - self._positions[agent_id]
                rel_vel = self._velocities[j] - self._velocities[agent_id]
                obs.extend(rel_pos)
                obs.extend(rel_vel)

        return np.array(obs, dtype=np.float32)

    def _get_share_observations(self, obs: np.ndarray) -> np.ndarray:
        """Get shared observations (concatenation of all observations)."""
        share_obs = obs.flatten()
        return np.tile(share_obs, (self._num_agents, 1))

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, List[Dict], Any]:
        """Execute actions."""
        self._step_count += 1

        # Scale actions to accelerations
        accelerations = actions * self._cfg.max_acceleration

        # Update velocities with acceleration
        self._velocities += accelerations * self._dt

        # Clip velocities
        speeds = np.linalg.norm(self._velocities, axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-8)
        scale = np.minimum(1.0, self._cfg.max_velocity / speeds)
        self._velocities *= scale

        # Update positions
        self._positions += self._velocities * self._dt

        # Clip positions to arena bounds
        self._positions = np.clip(self._positions, 0.1, self._arena_size - 0.1)

        # Compute rewards and costs
        rewards = self._compute_rewards(actions)
        costs = self._compute_costs()

        # Check termination
        goal_reached = self._check_goal_reached()
        truncated = self._step_count >= self._max_steps

        dones = np.array([goal_reached or truncated] * self._num_agents)

        # Get observations
        obs = self._get_observations()
        share_obs = self._get_share_observations(obs)

        # Info dict
        infos = []
        for i in range(self._num_agents):
            infos.append({
                "goal_reached": goal_reached,
                "formation_error": self._compute_formation_error(),
                "min_inter_dist": self._compute_min_inter_distance(),
                "collisions": int(costs[i] > 0),
            })

        self._prev_actions = actions.copy()

        return obs, share_obs, rewards, costs, dones, infos, truncated

    def _compute_rewards(self, actions: np.ndarray) -> np.ndarray:
        """Compute rewards for all agents."""
        rewards = np.zeros(self._num_agents)

        # Navigation reward: negative distance to goal
        centroid = np.mean(self._positions, axis=0)
        dist_to_goal = np.linalg.norm(centroid - self._goal)
        nav_reward = -dist_to_goal * self._reward_cfg.w_nav

        # Formation reward: negative formation error
        form_error = self._compute_formation_error()
        form_reward = -form_error * self._reward_cfg.w_formation

        # Smoothness reward: penalize action changes
        action_diff = np.linalg.norm(actions - self._prev_actions, axis=1)
        smooth_reward = -action_diff * self._reward_cfg.w_smooth

        # Goal bonus
        if self._check_goal_reached():
            rewards += self._reward_cfg.goal_bonus

        rewards += nav_reward + form_reward + smooth_reward

        return rewards.astype(np.float32)

    def _compute_costs(self) -> np.ndarray:
        """Compute safety costs (collision penalties)."""
        costs = np.zeros(self._num_agents)

        # Agent-agent collisions
        for i in range(self._num_agents):
            for j in range(i + 1, self._num_agents):
                dist = np.linalg.norm(self._positions[i] - self._positions[j])
                if dist < 0.3:  # collision threshold
                    costs[i] += 1.0
                    costs[j] += 1.0

        # Agent-obstacle collisions
        for i in range(self._num_agents):
            for j in range(self._cfg.num_obstacles):
                dist = np.linalg.norm(self._positions[i] - self._obstacles[j, :2])
                if dist < self._obstacles[j, 2] + 0.15:
                    costs[i] += 1.0

        return costs.astype(np.float32)

    def _compute_formation_error(self) -> float:
        """Compute formation maintenance error."""
        error = 0.0
        count = 0
        for i in range(self._num_agents):
            for j in range(i + 1, self._num_agents):
                actual_dist = np.linalg.norm(self._positions[i] - self._positions[j])
                desired_dist = self._desired_distances[i, j]
                error += abs(actual_dist - desired_dist)
                count += 1
        return error / max(count, 1)

    def _compute_min_inter_distance(self) -> float:
        """Compute minimum inter-agent distance."""
        min_dist = float('inf')
        for i in range(self._num_agents):
            for j in range(i + 1, self._num_agents):
                dist = np.linalg.norm(self._positions[i] - self._positions[j])
                min_dist = min(min_dist, dist)
        return min_dist

    def _check_goal_reached(self) -> bool:
        """Check if formation centroid reached the goal."""
        centroid = np.mean(self._positions, axis=0)
        return np.linalg.norm(centroid - self._goal) < self._cfg.goal_threshold

    def get_constraint_info(self) -> Dict[str, Any]:
        """Return safety constraint information."""
        return {
            "positions": self._positions.copy(),
            "velocities": self._velocities.copy(),
            "obstacles": self._obstacles.copy(),
            "desired_distances": self._desired_distances.copy(),
            "topology_edges": self._topology.edges(),
            "arena_size": self._arena_size,
            "goal": self._goal.copy(),
        }

    def render(self) -> Optional[np.ndarray]:
        """Render is not implemented for this env."""
        return None

    def close(self):
        """Clean up."""
        pass
