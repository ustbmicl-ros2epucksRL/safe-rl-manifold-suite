"""
Episode Runner for collecting complete episodes.

Similar to EPyMARL's EpisodeRunner, collects full episodes
and stores transitions in a batch for training.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import torch

from cosmos.envs.base import BaseMultiAgentEnv
from cosmos.algos.base import BaseMARLAlgo
from cosmos.safety.base import BaseSafetyFilter


class EpisodeBatch:
    """
    Batch of episode data for training.

    Stores transitions from one or more episodes in a structured format
    suitable for training MARL algorithms.
    """

    def __init__(
        self,
        num_agents: int,
        max_seq_length: int,
        obs_dim: int,
        share_obs_dim: int,
        act_dim: int,
        device: str = "cpu"
    ):
        self.num_agents = num_agents
        self.max_seq_length = max_seq_length
        self.obs_dim = obs_dim
        self.share_obs_dim = share_obs_dim
        self.act_dim = act_dim
        self.device = device

        # Pre-allocate tensors
        self.obs = torch.zeros(max_seq_length + 1, num_agents, obs_dim)
        self.share_obs = torch.zeros(max_seq_length + 1, num_agents, share_obs_dim)
        self.actions = torch.zeros(max_seq_length, num_agents, act_dim)
        self.rewards = torch.zeros(max_seq_length, num_agents, 1)
        self.costs = torch.zeros(max_seq_length, num_agents, 1)
        self.masks = torch.ones(max_seq_length + 1, num_agents, 1)
        self.dones = torch.zeros(max_seq_length, num_agents, 1)

        # For PPO
        self.log_probs = torch.zeros(max_seq_length, num_agents, 1)
        self.values = torch.zeros(max_seq_length + 1, num_agents, 1)

        # For Q-learning
        self.avail_actions = None  # Optional: available actions mask

        self.step = 0
        self.episode_length = 0

    def insert(
        self,
        obs: np.ndarray,
        share_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        costs: np.ndarray,
        dones: np.ndarray,
        log_probs: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
    ):
        """Insert one timestep of data."""
        t = self.step

        self.obs[t + 1] = torch.from_numpy(obs).float()
        self.share_obs[t + 1] = torch.from_numpy(share_obs).float()
        self.actions[t] = torch.from_numpy(actions).float()
        self.rewards[t] = torch.from_numpy(rewards.reshape(-1, 1)).float()
        self.costs[t] = torch.from_numpy(costs.reshape(-1, 1)).float()
        self.dones[t] = torch.from_numpy(dones.reshape(-1, 1)).float()
        self.masks[t + 1] = 1 - self.dones[t]

        if log_probs is not None:
            self.log_probs[t] = torch.from_numpy(log_probs.reshape(-1, 1)).float()
        if values is not None:
            self.values[t] = torch.from_numpy(values.reshape(-1, 1)).float()

        self.step += 1
        self.episode_length = self.step

    def set_first_obs(self, obs: np.ndarray, share_obs: np.ndarray):
        """Set initial observation."""
        self.obs[0] = torch.from_numpy(obs).float()
        self.share_obs[0] = torch.from_numpy(share_obs).float()

    def set_last_values(self, values: np.ndarray):
        """Set final value estimates for GAE."""
        self.values[self.step] = torch.from_numpy(values.reshape(-1, 1)).float()

    def to(self, device: str) -> "EpisodeBatch":
        """Move batch to device."""
        self.device = device
        self.obs = self.obs.to(device)
        self.share_obs = self.share_obs.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.costs = self.costs.to(device)
        self.masks = self.masks.to(device)
        self.dones = self.dones.to(device)
        self.log_probs = self.log_probs.to(device)
        self.values = self.values.to(device)
        return self

    def get_data(self) -> Dict[str, torch.Tensor]:
        """Get all data as dict, trimmed to actual episode length."""
        T = self.episode_length
        return {
            "obs": self.obs[:T + 1],
            "share_obs": self.share_obs[:T + 1],
            "actions": self.actions[:T],
            "rewards": self.rewards[:T],
            "costs": self.costs[:T],
            "masks": self.masks[:T + 1],
            "dones": self.dones[:T],
            "log_probs": self.log_probs[:T],
            "values": self.values[:T + 1],
        }


class EpisodeRunner:
    """
    Runs episodes and collects experience.

    Similar to EPyMARL's EpisodeRunner, handles:
    - Episode collection with safety filtering
    - Batch construction for training
    - Episode statistics tracking
    """

    def __init__(
        self,
        env: BaseMultiAgentEnv,
        algo: BaseMARLAlgo,
        safety_filter: Optional[BaseSafetyFilter] = None,
        max_episode_length: int = 500,
        device: str = "cpu"
    ):
        """
        Args:
            env: Multi-agent environment.
            algo: MARL algorithm.
            safety_filter: Optional safety filter.
            max_episode_length: Maximum episode length.
            device: Torch device.
        """
        self.env = env
        self.algo = algo
        self.safety_filter = safety_filter
        self.max_episode_length = max_episode_length
        self.device = device

        self.num_agents = env.num_agents
        self.obs_dim = env.get_obs_dim()
        self.share_obs_dim = env.get_share_obs_dim()
        self.act_dim = env.get_act_dim()

        # Statistics
        self.total_episodes = 0
        self.total_steps = 0

    def run_episode(
        self,
        seed: Optional[int] = None,
        deterministic: bool = False,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Run one episode.

        Args:
            seed: Random seed for reset.
            deterministic: Use deterministic actions.
            render: Render environment.

        Returns:
            Dict with episode statistics and batch.
        """
        # Create batch
        batch = EpisodeBatch(
            num_agents=self.num_agents,
            max_seq_length=self.max_episode_length,
            obs_dim=self.obs_dim,
            share_obs_dim=self.share_obs_dim,
            act_dim=self.act_dim,
            device=self.device
        )

        # Reset environment
        obs, share_obs, _ = self.env.reset(seed=seed)
        batch.set_first_obs(obs, share_obs)

        # Reset safety filter
        if self.safety_filter is not None:
            constraint_info = self.env.get_constraint_info()
            self.safety_filter.reset(constraint_info)
            self.safety_filter.update(constraint_info)

        # Episode metrics
        episode_reward = 0.0
        episode_cost = 0.0
        episode_collisions = 0
        episode_length = 0

        for step in range(self.max_episode_length):
            # Get actions
            actions, log_probs = self.algo.get_actions(obs, deterministic=deterministic)
            values = self.algo.get_values(share_obs)

            # Apply safety filter
            if self.safety_filter is not None:
                constraint_info = self.env.get_constraint_info()
                safe_actions = self.safety_filter.project(
                    actions, constraint_info, dt=getattr(self.env, '_env_cfg', None)
                    and getattr(self.env._env_cfg, 'dt', 0.05) or 0.05
                )
            else:
                safe_actions = actions

            # Step environment
            next_obs, next_share_obs, rewards, costs, dones, infos, _ = self.env.step(safe_actions)

            # Store transition
            batch.insert(
                obs=next_obs,
                share_obs=next_share_obs,
                actions=actions,  # Store original actions for learning
                rewards=rewards[:, 0],
                costs=costs[:, 0],
                dones=dones,
                log_probs=log_probs,
                values=values
            )

            # Update metrics
            episode_reward += rewards[0, 0]
            episode_cost += costs[0, 0]
            if infos and "collisions" in infos[0]:
                episode_collisions += infos[0]["collisions"]
            episode_length = step + 1

            # Render
            if render:
                self.env.render()

            obs, share_obs = next_obs, next_share_obs

            if dones.all():
                break

        # Set final values for GAE
        last_values = self.algo.get_values(share_obs)
        batch.set_last_values(last_values)

        # Update statistics
        self.total_episodes += 1
        self.total_steps += episode_length

        return {
            "batch": batch,
            "episode_reward": episode_reward,
            "episode_cost": episode_cost,
            "episode_collisions": episode_collisions,
            "episode_length": episode_length,
            "infos": infos
        }

    def collect_episodes(
        self,
        num_episodes: int,
        seed: Optional[int] = None,
        deterministic: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Collect multiple episodes.

        Args:
            num_episodes: Number of episodes to collect.
            seed: Base seed (incremented for each episode).
            deterministic: Use deterministic actions.

        Returns:
            List of episode results.
        """
        results = []
        for i in range(num_episodes):
            ep_seed = seed + i if seed is not None else None
            result = self.run_episode(seed=ep_seed, deterministic=deterministic)
            results.append(result)
        return results

    def get_stats(self) -> Dict[str, float]:
        """Get runner statistics."""
        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
        }
