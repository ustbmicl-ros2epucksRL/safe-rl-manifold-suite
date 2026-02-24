"""
Replay Buffer for Off-Policy MARL Algorithms.

Provides experience replay storage for algorithms like QMIX, MADDPG.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    """
    Standard replay buffer for off-policy algorithms.

    Stores transitions (obs, action, reward, next_obs, done) and
    samples random batches for training.
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int,
        share_obs_dim: int,
        act_dim: int,
        device: str = "cpu"
    ):
        """
        Args:
            capacity: Maximum number of transitions to store.
            num_agents: Number of agents.
            obs_dim: Observation dimension per agent.
            share_obs_dim: Shared observation dimension.
            act_dim: Action dimension per agent.
            device: Torch device.
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.share_obs_dim = share_obs_dim
        self.act_dim = act_dim
        self.device = device

        # Pre-allocate arrays
        self.obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.share_obs = np.zeros((capacity, num_agents, share_obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, num_agents, 1), dtype=np.float32)
        self.costs = np.zeros((capacity, num_agents, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.next_share_obs = np.zeros((capacity, num_agents, share_obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, num_agents, 1), dtype=np.float32)

        # For discrete action spaces (Q-learning)
        self.avail_actions = None
        self.next_avail_actions = None

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        share_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        costs: np.ndarray,
        next_obs: np.ndarray,
        next_share_obs: np.ndarray,
        dones: np.ndarray,
        avail_actions: Optional[np.ndarray] = None,
        next_avail_actions: Optional[np.ndarray] = None
    ):
        """Add a transition to the buffer."""
        self.obs[self.ptr] = obs
        self.share_obs[self.ptr] = share_obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards.reshape(self.num_agents, 1)
        self.costs[self.ptr] = costs.reshape(self.num_agents, 1)
        self.next_obs[self.ptr] = next_obs
        self.next_share_obs[self.ptr] = next_share_obs
        self.dones[self.ptr] = dones.reshape(self.num_agents, 1)

        if avail_actions is not None:
            if self.avail_actions is None:
                n_actions = avail_actions.shape[-1]
                self.avail_actions = np.zeros((self.capacity, self.num_agents, n_actions), dtype=np.float32)
                self.next_avail_actions = np.zeros((self.capacity, self.num_agents, n_actions), dtype=np.float32)
            self.avail_actions[self.ptr] = avail_actions
            self.next_avail_actions[self.ptr] = next_avail_actions

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions."""
        indices = np.random.choice(self.size, size=batch_size, replace=False)

        batch = {
            "obs": torch.from_numpy(self.obs[indices]).to(self.device),
            "share_obs": torch.from_numpy(self.share_obs[indices]).to(self.device),
            "actions": torch.from_numpy(self.actions[indices]).to(self.device),
            "rewards": torch.from_numpy(self.rewards[indices]).to(self.device),
            "costs": torch.from_numpy(self.costs[indices]).to(self.device),
            "next_obs": torch.from_numpy(self.next_obs[indices]).to(self.device),
            "next_share_obs": torch.from_numpy(self.next_share_obs[indices]).to(self.device),
            "dones": torch.from_numpy(self.dones[indices]).to(self.device),
        }

        if self.avail_actions is not None:
            batch["avail_actions"] = torch.from_numpy(self.avail_actions[indices]).to(self.device)
            batch["next_avail_actions"] = torch.from_numpy(self.next_avail_actions[indices]).to(self.device)

        return batch

    def __len__(self) -> int:
        return self.size

    def can_sample(self, batch_size: int) -> bool:
        """Check if we have enough samples."""
        return self.size >= batch_size


class EpisodeReplayBuffer:
    """
    Episode-based replay buffer for value decomposition methods.

    Stores complete episodes and samples episode batches.
    Useful for methods like QMIX that need temporal structure.
    """

    def __init__(
        self,
        capacity: int,
        max_episode_length: int,
        num_agents: int,
        obs_dim: int,
        share_obs_dim: int,
        act_dim: int,
        n_actions: Optional[int] = None,
        device: str = "cpu"
    ):
        """
        Args:
            capacity: Maximum number of episodes to store.
            max_episode_length: Maximum length of each episode.
            num_agents: Number of agents.
            obs_dim: Observation dimension per agent.
            share_obs_dim: Shared observation dimension.
            act_dim: Action dimension per agent.
            n_actions: Number of discrete actions (for Q-learning).
            device: Torch device.
        """
        self.capacity = capacity
        self.max_episode_length = max_episode_length
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.share_obs_dim = share_obs_dim
        self.act_dim = act_dim
        self.n_actions = n_actions
        self.device = device

        # Episode storage
        self.episodes: deque = deque(maxlen=capacity)

        # Current episode buffer
        self._current_episode = None
        self._reset_current_episode()

    def _reset_current_episode(self):
        """Reset the current episode buffer."""
        self._current_episode = {
            "obs": [],
            "share_obs": [],
            "actions": [],
            "rewards": [],
            "costs": [],
            "dones": [],
            "avail_actions": [],
            "filled": [],
        }

    def add(
        self,
        obs: np.ndarray,
        share_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        costs: np.ndarray,
        dones: np.ndarray,
        avail_actions: Optional[np.ndarray] = None
    ):
        """Add a timestep to the current episode."""
        self._current_episode["obs"].append(obs)
        self._current_episode["share_obs"].append(share_obs)
        self._current_episode["actions"].append(actions)
        self._current_episode["rewards"].append(rewards)
        self._current_episode["costs"].append(costs)
        self._current_episode["dones"].append(dones)
        self._current_episode["filled"].append(1.0)

        if avail_actions is not None:
            self._current_episode["avail_actions"].append(avail_actions)

    def end_episode(self):
        """Finalize and store the current episode."""
        if len(self._current_episode["obs"]) == 0:
            return

        episode_length = len(self._current_episode["obs"])

        # Pad to max_episode_length
        def pad_array(arr_list, shape_suffix, default=0.0):
            arr = np.array(arr_list)
            padded = np.full((self.max_episode_length,) + shape_suffix, default, dtype=np.float32)
            padded[:episode_length] = arr
            return padded

        episode = {
            "obs": pad_array(self._current_episode["obs"], (self.num_agents, self.obs_dim)),
            "share_obs": pad_array(self._current_episode["share_obs"], (self.num_agents, self.share_obs_dim)),
            "actions": pad_array(self._current_episode["actions"], (self.num_agents, self.act_dim)),
            "rewards": pad_array(self._current_episode["rewards"], (self.num_agents,)),
            "costs": pad_array(self._current_episode["costs"], (self.num_agents,)),
            "dones": pad_array(self._current_episode["dones"], (self.num_agents,)),
            "filled": pad_array(self._current_episode["filled"], ()),
            "episode_length": episode_length,
        }

        if len(self._current_episode["avail_actions"]) > 0:
            episode["avail_actions"] = pad_array(
                self._current_episode["avail_actions"],
                (self.num_agents, self.n_actions)
            )

        self.episodes.append(episode)
        self._reset_current_episode()

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of episodes."""
        indices = random.sample(range(len(self.episodes)), min(batch_size, len(self.episodes)))

        batch_episodes = [self.episodes[i] for i in indices]

        # Stack episodes
        batch = {
            "obs": torch.from_numpy(np.stack([ep["obs"] for ep in batch_episodes])).to(self.device),
            "share_obs": torch.from_numpy(np.stack([ep["share_obs"] for ep in batch_episodes])).to(self.device),
            "actions": torch.from_numpy(np.stack([ep["actions"] for ep in batch_episodes])).to(self.device),
            "rewards": torch.from_numpy(np.stack([ep["rewards"] for ep in batch_episodes])).to(self.device),
            "costs": torch.from_numpy(np.stack([ep["costs"] for ep in batch_episodes])).to(self.device),
            "dones": torch.from_numpy(np.stack([ep["dones"] for ep in batch_episodes])).to(self.device),
            "filled": torch.from_numpy(np.stack([ep["filled"] for ep in batch_episodes])).to(self.device),
        }

        if "avail_actions" in batch_episodes[0]:
            batch["avail_actions"] = torch.from_numpy(
                np.stack([ep["avail_actions"] for ep in batch_episodes])
            ).to(self.device)

        return batch

    def __len__(self) -> int:
        return len(self.episodes)

    def can_sample(self, batch_size: int) -> bool:
        """Check if we have enough episodes."""
        return len(self.episodes) >= batch_size
