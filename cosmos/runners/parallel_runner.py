"""
Parallel Runner for collecting experience from multiple environments.

Runs multiple environments in parallel for faster data collection.
"""

from typing import Dict, Any, Optional, List, Callable
import numpy as np
import torch
from multiprocessing import Process, Pipe
from functools import partial

from cosmos.envs.base import BaseMultiAgentEnv
from cosmos.algos.base import BaseMARLAlgo
from cosmos.safety.base import BaseSafetyFilter
from cosmos.runners.episode_runner import EpisodeBatch


def worker_process(remote, parent_remote, env_fn):
    """Worker process for parallel environment."""
    parent_remote.close()
    env = env_fn()

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            result = env.step(data)
            remote.send(result)
        elif cmd == "reset":
            result = env.reset(seed=data)
            remote.send(result)
        elif cmd == "get_constraint_info":
            result = env.get_constraint_info()
            remote.send(result)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send({
                "num_agents": env.num_agents,
                "obs_dim": env.get_obs_dim(),
                "share_obs_dim": env.get_share_obs_dim(),
                "act_dim": env.get_act_dim(),
            })
        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


class ParallelRunner:
    """
    Parallel runner for multiple environments.

    Collects experience from multiple environments simultaneously
    using multiprocessing for speed.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], BaseMultiAgentEnv]],
        algo: BaseMARLAlgo,
        safety_filter_fn: Optional[Callable[[], BaseSafetyFilter]] = None,
        max_episode_length: int = 500,
        device: str = "cpu"
    ):
        """
        Args:
            env_fns: List of functions that create environments.
            algo: MARL algorithm (shared across all envs).
            safety_filter_fn: Function to create safety filter for each env.
            max_episode_length: Maximum episode length.
            device: Torch device.
        """
        self.num_envs = len(env_fns)
        self.algo = algo
        self.safety_filter_fn = safety_filter_fn
        self.max_episode_length = max_episode_length
        self.device = device

        # Create worker processes
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        self.processes = []

        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = Process(target=worker_process, args=(work_remote, remote, env_fn))
            process.daemon = True
            process.start()
            self.processes.append(process)
            work_remote.close()

        # Get space info from first env
        self.remotes[0].send(("get_spaces", None))
        spaces = self.remotes[0].recv()
        self.num_agents = spaces["num_agents"]
        self.obs_dim = spaces["obs_dim"]
        self.share_obs_dim = spaces["share_obs_dim"]
        self.act_dim = spaces["act_dim"]

        # Create safety filters for each env
        if safety_filter_fn is not None:
            self.safety_filters = [safety_filter_fn() for _ in range(self.num_envs)]
        else:
            self.safety_filters = [None] * self.num_envs

        # Statistics
        self.total_episodes = 0
        self.total_steps = 0

    def reset(self, seeds: Optional[List[int]] = None):
        """Reset all environments."""
        if seeds is None:
            seeds = [None] * self.num_envs

        for remote, seed in zip(self.remotes, seeds):
            remote.send(("reset", seed))

        results = [remote.recv() for remote in self.remotes]
        obs = np.stack([r[0] for r in results])
        share_obs = np.stack([r[1] for r in results])
        infos = [r[2] for r in results]

        # Reset safety filters
        for i, sf in enumerate(self.safety_filters):
            if sf is not None:
                self.remotes[i].send(("get_constraint_info", None))
                constraint_info = self.remotes[i].recv()
                sf.reset(constraint_info)

        return obs, share_obs, infos

    def step(self, actions: np.ndarray):
        """Step all environments."""
        # Apply safety filters
        safe_actions = actions.copy()
        for i, sf in enumerate(self.safety_filters):
            if sf is not None:
                self.remotes[i].send(("get_constraint_info", None))
                constraint_info = self.remotes[i].recv()
                safe_actions[i] = sf.project(actions[i], constraint_info)

        # Step environments
        for remote, action in zip(self.remotes, safe_actions):
            remote.send(("step", action))

        results = [remote.recv() for remote in self.remotes]

        obs = np.stack([r[0] for r in results])
        share_obs = np.stack([r[1] for r in results])
        rewards = np.stack([r[2] for r in results])
        costs = np.stack([r[3] for r in results])
        dones = np.stack([r[4] for r in results])
        infos = [r[5] for r in results]

        return obs, share_obs, rewards, costs, dones, infos

    def collect_rollout(
        self,
        rollout_length: int,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Collect rollout from all environments.

        Args:
            rollout_length: Number of steps to collect.
            seed: Base seed for environment reset.

        Returns:
            Dict with batched data from all environments.
        """
        # Reset environments
        seeds = [seed + i if seed else None for i in range(self.num_envs)]
        obs, share_obs, _ = self.reset(seeds)

        # Allocate storage
        all_obs = np.zeros((rollout_length + 1, self.num_envs, self.num_agents, self.obs_dim))
        all_share_obs = np.zeros((rollout_length + 1, self.num_envs, self.num_agents, self.share_obs_dim))
        all_actions = np.zeros((rollout_length, self.num_envs, self.num_agents, self.act_dim))
        all_rewards = np.zeros((rollout_length, self.num_envs, self.num_agents, 1))
        all_costs = np.zeros((rollout_length, self.num_envs, self.num_agents, 1))
        all_dones = np.zeros((rollout_length, self.num_envs, self.num_agents))
        all_log_probs = np.zeros((rollout_length, self.num_envs, self.num_agents, 1))
        all_values = np.zeros((rollout_length + 1, self.num_envs, self.num_agents, 1))

        all_obs[0] = obs
        all_share_obs[0] = share_obs

        total_reward = np.zeros(self.num_envs)
        total_cost = np.zeros(self.num_envs)
        episode_lengths = np.zeros(self.num_envs)

        for t in range(rollout_length):
            # Flatten for batched inference
            obs_flat = obs.reshape(self.num_envs * self.num_agents, self.obs_dim)
            share_obs_flat = share_obs.reshape(self.num_envs * self.num_agents, self.share_obs_dim)

            # Get actions
            actions, log_probs = self.algo.get_actions(obs_flat)
            values = self.algo.get_values(share_obs_flat)

            actions = actions.reshape(self.num_envs, self.num_agents, self.act_dim)
            log_probs = log_probs.reshape(self.num_envs, self.num_agents, 1)
            values = values.reshape(self.num_envs, self.num_agents, 1)

            # Step environments
            next_obs, next_share_obs, rewards, costs, dones, infos = self.step(actions)

            # Store
            all_obs[t + 1] = next_obs
            all_share_obs[t + 1] = next_share_obs
            all_actions[t] = actions
            all_rewards[t] = rewards
            all_costs[t] = costs
            all_dones[t] = dones
            all_log_probs[t] = log_probs
            all_values[t] = values

            # Update metrics
            total_reward += rewards[:, 0, 0]
            total_cost += costs[:, 0, 0]
            episode_lengths += 1

            obs, share_obs = next_obs, next_share_obs

        # Final values
        obs_flat = obs.reshape(self.num_envs * self.num_agents, self.obs_dim)
        share_obs_flat = share_obs.reshape(self.num_envs * self.num_agents, self.share_obs_dim)
        last_values = self.algo.get_values(share_obs_flat)
        all_values[rollout_length] = last_values.reshape(self.num_envs, self.num_agents, 1)

        self.total_episodes += self.num_envs
        self.total_steps += rollout_length * self.num_envs

        return {
            "obs": torch.from_numpy(all_obs).float(),
            "share_obs": torch.from_numpy(all_share_obs).float(),
            "actions": torch.from_numpy(all_actions).float(),
            "rewards": torch.from_numpy(all_rewards).float(),
            "costs": torch.from_numpy(all_costs).float(),
            "dones": torch.from_numpy(all_dones).float(),
            "log_probs": torch.from_numpy(all_log_probs).float(),
            "values": torch.from_numpy(all_values).float(),
            "total_reward": total_reward,
            "total_cost": total_cost,
            "episode_lengths": episode_lengths,
        }

    def close(self):
        """Close all environments."""
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def __del__(self):
        if not self.closed:
            self.close()
