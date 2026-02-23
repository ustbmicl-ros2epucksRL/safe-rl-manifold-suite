import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple

from formation_nav.env.formations import FormationShape, FormationTopology
from formation_nav.config import EnvConfig, RewardConfig


class FormationNavEnv(gym.Env):
    """
    2D multi-robot formation navigation environment.

    Physics: double integrator (v += a*dt, p += v*dt) with velocity clipping.
    Observations are per-agent (decentralized).
    Safety is NOT enforced here — it is handled by the external ATACOM layer.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_cfg: EnvConfig = None, reward_cfg: RewardConfig = None):
        super().__init__()
        self.cfg = env_cfg or EnvConfig()
        self.rcfg = reward_cfg or RewardConfig()

        self.num_agents = self.cfg.num_agents
        self.dt = self.cfg.dt
        self.max_steps = self.cfg.max_steps
        self.arena_size = self.cfg.arena_size
        self.max_vel = self.cfg.max_velocity
        self.max_acc = self.cfg.max_acceleration

        # Formation
        self.formation_offsets = FormationShape.get_shape(
            self.cfg.formation_shape, self.num_agents, self.cfg.formation_radius
        )
        self.desired_distances = FormationShape.desired_distance_matrix(self.formation_offsets)
        self.topology = FormationTopology(self.num_agents, "complete")

        # Obstacles: (num_obstacles, 3) -> [x, y, radius]
        self.num_obstacles = self.cfg.num_obstacles
        self.obstacles = np.zeros((self.num_obstacles, 3))

        # State
        self.positions = np.zeros((self.num_agents, 2))
        self.velocities = np.zeros((self.num_agents, 2))
        self.goal = np.zeros(2)
        self.step_count = 0

        # Observation / action spaces
        # Per-agent obs: self_pos(2) + self_vel(2) + goal_rel(2) + goal_dist(1)
        #   + neighbors_rel_pos(2*(N-1)) + dist_errors(N-1)
        #   + obstacles_rel(2*K) + obstacles_dist(K)
        #   + formation_center_rel(2)
        n = self.num_agents
        k = self.num_obstacles
        self._obs_dim = 2 + 2 + 2 + 1 + 2 * (n - 1) + (n - 1) + 2 * k + k + 2
        self._share_obs_dim = n * 4 + 2 + k * 3  # all agents (pos+vel) + goal + obstacles

        # Action: 2D acceleration per agent
        self.action_space = spaces.Box(
            low=-self.max_acc * np.ones(2),
            high=self.max_acc * np.ones(2),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        self.share_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._share_obs_dim,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Reset and return (obs_all, share_obs_all, info)."""
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
        elif not hasattr(self, '_np_random'):
            self._np_random = np.random.RandomState()

        rng = self._np_random

        # Random start positions near origin
        center_start = rng.uniform(-2, 2, size=2)
        self.positions = center_start + self.formation_offsets + rng.randn(self.num_agents, 2) * 0.1
        self.velocities = np.zeros((self.num_agents, 2))

        # Random goal
        self.goal = rng.uniform(-self.arena_size * 0.6, self.arena_size * 0.6, size=2)
        while np.linalg.norm(self.goal - center_start) < 3.0:
            self.goal = rng.uniform(-self.arena_size * 0.6, self.arena_size * 0.6, size=2)

        # Random obstacles (avoid start and goal regions)
        self.obstacles = np.zeros((self.num_obstacles, 3))
        for i in range(self.num_obstacles):
            for _ in range(100):
                pos = rng.uniform(-self.arena_size * 0.7, self.arena_size * 0.7, size=2)
                r = self.cfg.obstacle_radius
                # Keep away from start and goal
                if (np.linalg.norm(pos - center_start) > 2.5 and
                        np.linalg.norm(pos - self.goal) > 2.5):
                    self.obstacles[i] = [pos[0], pos[1], r]
                    break

        self.step_count = 0
        self._prev_centroid_dist = np.linalg.norm(self._centroid() - self.goal)

        obs_all = self._get_all_obs()
        share_obs = self._get_share_obs()
        share_obs_all = np.tile(share_obs, (self.num_agents, 1))

        return obs_all, share_obs_all, {}

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, None]:
        """
        Step with actions shape (num_agents, 2).
        Returns: obs, share_obs, rewards, costs, dones, infos, None
        """
        actions = np.array(actions, dtype=np.float64).reshape(self.num_agents, 2)
        actions = np.clip(actions, -self.max_acc, self.max_acc)

        # Double integrator
        self.velocities += actions * self.dt
        self.velocities = np.clip(self.velocities, -self.max_vel, self.max_vel)
        self.positions += self.velocities * self.dt

        self.step_count += 1

        # Compute metrics
        centroid = self._centroid()
        centroid_dist = np.linalg.norm(centroid - self.goal)
        formation_error = self._formation_error()
        min_inter_dist = self._min_inter_agent_distance()
        collisions = self._count_collisions()
        out_of_bounds = self._count_out_of_bounds()

        # Reward (shared)
        nav_progress = self._prev_centroid_dist - centroid_dist
        smooth_penalty = np.mean(np.sum(actions ** 2, axis=1))
        reward = (self.rcfg.w_nav * nav_progress
                  - self.rcfg.w_formation * formation_error
                  - self.rcfg.w_smooth * smooth_penalty)

        reached = centroid_dist < self.cfg.goal_threshold
        if reached:
            reward += self.rcfg.goal_bonus

        self._prev_centroid_dist = centroid_dist

        # Cost signal (for logging, not used in training loss)
        cost = float(collisions > 0 or out_of_bounds > 0)

        # Done
        terminated = reached
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        # Per-agent outputs
        obs_all = self._get_all_obs()
        share_obs = self._get_share_obs()
        share_obs_all = np.tile(share_obs, (self.num_agents, 1))
        rewards = np.full((self.num_agents, 1), reward, dtype=np.float32)
        costs = np.full((self.num_agents, 1), cost, dtype=np.float32)
        dones = np.full((self.num_agents,), done, dtype=bool)

        info = {
            "formation_error": formation_error,
            "centroid_dist": centroid_dist,
            "min_inter_dist": min_inter_dist,
            "collisions": collisions,
            "out_of_bounds": out_of_bounds,
            "reached": reached,
        }
        infos = [info.copy() for _ in range(self.num_agents)]

        return obs_all, share_obs_all, rewards, costs, dones, infos, None

    # ---- Observation helpers ----

    def _centroid(self) -> np.ndarray:
        return self.positions.mean(axis=0)

    def _get_obs(self, agent_id: int) -> np.ndarray:
        """Per-agent observation."""
        pos = self.positions[agent_id]
        vel = self.velocities[agent_id]
        centroid = self._centroid()
        goal_rel = self.goal - centroid
        goal_dist = np.array([np.linalg.norm(goal_rel)])
        center_rel = centroid - pos

        # Neighbor relative positions and distance errors
        nbrs = self.topology.neighbors(agent_id)
        nbr_rel = []
        dist_err = []
        for j in nbrs:
            rel = self.positions[j] - pos
            nbr_rel.append(rel)
            actual_dist = np.linalg.norm(rel)
            desired_dist = self.desired_distances[agent_id, j]
            dist_err.append(actual_dist - desired_dist)
        nbr_rel = np.concatenate(nbr_rel) if nbr_rel else np.array([])
        dist_err = np.array(dist_err) if dist_err else np.array([])

        # Obstacle relative positions and distances
        obs_rel = []
        obs_dist = []
        for k in range(self.num_obstacles):
            rel = self.obstacles[k, :2] - pos
            obs_rel.append(rel)
            obs_dist.append(np.linalg.norm(rel))
        obs_rel = np.concatenate(obs_rel) if obs_rel else np.array([])
        obs_dist = np.array(obs_dist) if obs_dist else np.array([])

        return np.concatenate([
            pos, vel, goal_rel, goal_dist,
            nbr_rel, dist_err,
            obs_rel, obs_dist,
            center_rel,
        ]).astype(np.float32)

    def _get_all_obs(self) -> np.ndarray:
        """(num_agents, obs_dim)"""
        return np.array([self._get_obs(i) for i in range(self.num_agents)])

    def _get_share_obs(self) -> np.ndarray:
        """Global shared observation for critic."""
        parts = []
        for i in range(self.num_agents):
            parts.extend([self.positions[i], self.velocities[i]])
        parts.append(self.goal)
        for k in range(self.num_obstacles):
            parts.append(self.obstacles[k])
        return np.concatenate(parts).astype(np.float32)

    # ---- Metrics ----

    def _formation_error(self) -> float:
        """Mean squared distance error from desired formation."""
        errors = []
        for i, j in self.topology.edges():
            actual = np.linalg.norm(self.positions[i] - self.positions[j])
            desired = self.desired_distances[i, j]
            errors.append((actual - desired) ** 2)
        return np.mean(errors) if errors else 0.0

    def _min_inter_agent_distance(self) -> float:
        min_d = float('inf')
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                min_d = min(min_d, d)
        return min_d

    def _count_collisions(self) -> int:
        count = 0
        safety_r = 0.3  # collision detection radius
        # Agent-agent
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.linalg.norm(self.positions[i] - self.positions[j]) < safety_r:
                    count += 1
        # Agent-obstacle
        for i in range(self.num_agents):
            for k in range(self.num_obstacles):
                d = np.linalg.norm(self.positions[i] - self.obstacles[k, :2])
                if d < self.obstacles[k, 2]:
                    count += 1
        return count

    def _count_out_of_bounds(self) -> int:
        count = 0
        for i in range(self.num_agents):
            if np.any(np.abs(self.positions[i]) > self.arena_size):
                count += 1
        return count
