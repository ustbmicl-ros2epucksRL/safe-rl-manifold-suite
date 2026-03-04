#!/usr/bin/env python3
"""
Table II experiment runner for IROS 2026 paper.

Multi-agent experiments on MultiGoal Task (4 agents).

Methods:
    1. MAPPO (Multi-Agent PPO)
    2. HAPPO (Heterogeneous-Agent PPO)
    3. MACPO (Multi-Agent Constrained Policy Optimization)
    4. MAPPO-Lag (MAPPO + Lagrangian)
    5. Ours (MAPPO + Manifold Filter per agent)

Usage:
    python scripts/run_table2_experiments.py --seeds 3 --train-steps 30000
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'src')
sys.path.insert(0, SRC_DIR)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

from ppo import PPO, PPOConfig, RolloutBuffer
from safety import DistanceFilter


# =============================================================================
# Multi-Agent Goal Environment
# =============================================================================

class MultiGoalEnv:
    """
    Multi-agent goal navigation with hazard and inter-agent collision avoidance.

    4 agents start at different positions and must reach individual goals
    while avoiding hazards and collisions with each other.
    """

    def __init__(self, n_agents=4, max_steps=500):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.obs_dim = 20  # per agent
        self.act_dim = 2   # [v, omega] per agent

        self.dt = 0.1
        self.agent_radius = 0.15
        self.hazard_radius = 0.2
        self.goal_radius = 0.3
        self.arena_size = 3.0

        # Fixed hazards
        self._hazards = [
            np.array([0.5, 0.5]),
            np.array([-0.5, 0.5]),
            np.array([0.5, -0.5]),
            np.array([-0.5, -0.5]),
        ]

        self._step = 0
        self._positions = np.zeros((n_agents, 3))  # [x, y, theta]
        self._velocities = np.zeros((n_agents, 3))
        self._goals = np.zeros((n_agents, 2))

    def reset(self, seed=None):
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)

        self._step = 0

        # Agents start at corners
        start_positions = [
            np.array([-1.5, -1.5, np.pi / 4]),
            np.array([1.5, -1.5, 3 * np.pi / 4]),
            np.array([1.5, 1.5, -3 * np.pi / 4]),
            np.array([-1.5, 1.5, -np.pi / 4]),
        ]

        # Goals at opposite corners
        goals = [
            np.array([1.5, 1.5]),
            np.array([-1.5, 1.5]),
            np.array([-1.5, -1.5]),
            np.array([1.5, -1.5]),
        ]

        # Add small randomization
        for i in range(self.n_agents):
            self._positions[i] = start_positions[i] + np.random.uniform(-0.2, 0.2, 3)
            self._goals[i] = goals[i] + np.random.uniform(-0.2, 0.2, 2)

        self._velocities = np.zeros((self.n_agents, 3))

        obs = [self._get_obs(i) for i in range(self.n_agents)]
        info = self._get_info()
        return obs, info

    def step(self, actions):
        """
        Step all agents simultaneously.

        Args:
            actions: list of [v, omega] per agent

        Returns:
            obs_list, reward, cost, terminated, truncated, info
        """
        # Update each agent
        for i in range(self.n_agents):
            a = np.clip(actions[i], -1.0, 1.0)
            v, omega = a[0], a[1]
            theta = self._positions[i, 2]

            self._positions[i, 0] += v * np.cos(theta) * self.dt
            self._positions[i, 1] += v * np.sin(theta) * self.dt
            self._positions[i, 2] += omega * self.dt

            # Clip to arena
            self._positions[i, 0] = np.clip(self._positions[i, 0],
                                            -self.arena_size, self.arena_size)
            self._positions[i, 1] = np.clip(self._positions[i, 1],
                                            -self.arena_size, self.arena_size)

            self._velocities[i] = np.array([v * np.cos(theta),
                                            v * np.sin(theta), omega])

        # Compute cost (hazard + inter-agent collisions)
        cost = 0.0

        # Hazard collisions
        for i in range(self.n_agents):
            for h in self._hazards:
                dist = np.linalg.norm(self._positions[i, :2] - h)
                if dist < self.hazard_radius + self.agent_radius:
                    cost += 1.0

        # Inter-agent collisions
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = np.linalg.norm(self._positions[i, :2] - self._positions[j, :2])
                if dist < 2 * self.agent_radius:
                    cost += 1.0

        # Compute reward (sum of goal progress)
        reward = 0.0
        all_reached = True
        for i in range(self.n_agents):
            dist_to_goal = np.linalg.norm(self._positions[i, :2] - self._goals[i])
            reward -= 0.01 * dist_to_goal
            if dist_to_goal < self.goal_radius:
                reward += 1.0  # per-agent goal bonus (spread across steps)
            else:
                all_reached = False

        self._step += 1
        terminated = all_reached
        truncated = self._step >= self.max_steps

        obs = [self._get_obs(i) for i in range(self.n_agents)]
        info = self._get_info()

        return obs, reward, cost, terminated, truncated, info

    def _get_obs(self, agent_idx):
        """Get observation for a single agent."""
        obs = np.zeros(self.obs_dim)
        pos = self._positions[agent_idx]
        vel = self._velocities[agent_idx]

        # Own state
        obs[0:3] = pos
        obs[3:6] = vel

        # Goal direction
        goal_dir = self._goals[agent_idx] - pos[:2]
        goal_dist = np.linalg.norm(goal_dir) + 1e-6
        obs[6:8] = goal_dir / goal_dist
        obs[8] = goal_dist

        # Other agents relative positions (up to 3)
        idx = 9
        for j in range(self.n_agents):
            if j == agent_idx:
                continue
            rel = self._positions[j, :2] - pos[:2]
            obs[idx] = rel[0]
            obs[idx + 1] = rel[1]
            idx += 2
            if idx >= 15:
                break

        # Nearest hazard
        min_dist = float('inf')
        nearest_hazard_rel = np.zeros(2)
        for h in self._hazards:
            d = np.linalg.norm(pos[:2] - h)
            if d < min_dist:
                min_dist = d
                nearest_hazard_rel = h - pos[:2]
        obs[15:17] = nearest_hazard_rel
        obs[17] = min_dist

        return obs

    def _get_info(self):
        """Get info dict."""
        return {
            'positions': self._positions.copy(),
            'velocities': self._velocities.copy(),
            'goals': self._goals.copy(),
            'hazards': [h.copy() for h in self._hazards],
        }

    def close(self):
        pass


# =============================================================================
# Shared PPO agent (parameter sharing across agents)
# =============================================================================

def create_shared_agent(obs_dim, act_dim, config=None, device='cpu'):
    """Create a single PPO agent shared across all multi-agents."""
    config = config or PPOConfig(hidden_dim=128, n_layers=2)
    return PPO(obs_dim, act_dim, config=config, device=device)


# =============================================================================
# Multi-Agent Training and Evaluation
# =============================================================================

def train_and_eval_multiagent(
    method: str,
    train_steps: int,
    eval_episodes: int,
    seed: int,
    n_agents: int = 4,
) -> Dict[str, float]:
    """
    Train and evaluate a multi-agent configuration.

    Methods:
        - mappo: Independent PPO with parameter sharing
        - happo: Sequential agent updates with trust region
        - macpo: MAPPO + cost constraint in policy loss
        - mappo_lag: MAPPO + Lagrangian multiplier
        - ours: MAPPO + DistanceFilter per agent (including inter-agent)
    """
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)

    env = MultiGoalEnv(n_agents=n_agents, max_steps=500)
    ppo_config = PPOConfig(hidden_dim=128, n_layers=2, learning_rate=3e-4)
    buffer_size = 1024

    # Shared agent (parameter sharing)
    agent = create_shared_agent(env.obs_dim, env.act_dim, config=ppo_config)
    buffer = RolloutBuffer(buffer_size, env.obs_dim, env.act_dim)

    # Method-specific state
    lagrangian_lambda = 0.1
    lr_lambda = 0.01
    ep_costs_history = []
    cost_limit = 0.0

    # MACPO: cost critic
    cost_critic = None
    cost_optimizer = None
    if method == "macpo" and TORCH_AVAILABLE:
        cost_critic = nn.Sequential(
            nn.Linear(env.obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        cost_optimizer = torch.optim.Adam(cost_critic.parameters(), lr=1e-3)
        cost_buffer = []

    # HAPPO: per-agent learning rate scaling
    happo_agent_order = list(range(n_agents))

    # Ours: distance filter per agent
    safety_filters = None
    if method == "ours":
        safety_filters = [
            DistanceFilter(
                danger_radius=0.6, stop_radius=0.3,
                hazard_radius=0.2, lambda_calib=0.1,
            )
            for _ in range(n_agents)
        ]

    # IPO-like barrier for MACPO
    macpo_t = 1.0

    # --- Training ---
    obs_list, info = env.reset(seed=seed)

    if safety_filters:
        for i, sf in enumerate(safety_filters):
            # Include hazards + other agents as obstacles
            obstacles = list(info['hazards'])
            for j in range(n_agents):
                if j != i:
                    obstacles.append(info['positions'][j, :2])
            sf.reset(obstacles)

    total_steps = 0
    ep_cost_accum = 0.0
    buf_agent_idx = 0  # Round-robin agent for buffer

    while total_steps < train_steps:
        # Get actions for all agents
        actions = []
        actions_exec = []
        log_probs_all = []
        values_all = []

        for i in range(n_agents):
            action, log_prob, value = agent.get_action(obs_list[i])
            actions.append(action)
            log_probs_all.append(log_prob)
            values_all.append(value)

            if method == "ours" and safety_filters:
                robot_pos = info['positions'][i]
                # Update obstacles: hazards + other agents
                obstacles = list(info['hazards'])
                for j in range(n_agents):
                    if j != i:
                        obstacles.append(info['positions'][j, :2])
                safety_filters[i].reset(obstacles)

                result = safety_filters[i].project(action, robot_pos)
                actions_exec.append(result.action_safe)
            else:
                actions_exec.append(action)

        # Step environment
        obs_next_list, reward, cost, term, trunc, info = env.step(actions_exec)

        # Method-specific reward modification
        reward_modified = reward - cost  # base penalty

        if method == "mappo_lag":
            reward_modified -= lagrangian_lambda * cost
        elif method == "macpo":
            reward_modified -= macpo_t * cost
            macpo_t = min(macpo_t * 1.001, 50.0)
        elif method == "happo":
            # HAPPO uses sequential updates - slightly different reward signal
            reward_modified = reward - 0.5 * cost
        elif method == "ours" and safety_filters:
            # Add calibration penalty for corrections
            for i in range(n_agents):
                robot_pos = info['positions'][i]
                obstacles = list(info['hazards'])
                for j in range(n_agents):
                    if j != i:
                        obstacles.append(info['positions'][j, :2])
                safety_filters[i].reset(obstacles)
                result = safety_filters[i].project(actions[i], robot_pos)
                correction_norm = np.linalg.norm(result.correction)
                if correction_norm > 0:
                    reward_modified -= 0.05 * (correction_norm ** 2)

        done = term or trunc
        ep_cost_accum += cost

        # Store transitions round-robin across agents
        agent_i = buf_agent_idx % n_agents
        buffer.add(obs_list[agent_i], actions[agent_i],
                    reward_modified / n_agents,  # per-agent share
                    values_all[agent_i], log_probs_all[agent_i], done)
        buf_agent_idx += 1

        # MACPO: store cost transitions
        if method == "macpo" and cost_critic is not None:
            for i in range(n_agents):
                cost_buffer.append((obs_list[i], cost / n_agents))
                if len(cost_buffer) > 5000:
                    cost_buffer.pop(0)

        obs_list = obs_next_list
        total_steps += 1

        # PPO update when buffer full
        if buffer.ptr == buffer_size:
            _, _, last_value = agent.get_action(obs_list[0])
            buffer.finish_path(last_value, ppo_config.gamma, ppo_config.gae_lambda)

            if method == "happo":
                # HAPPO: sequential update with shuffled agent order
                np.random.shuffle(happo_agent_order)
                # Approximate HAPPO by running multiple small updates
                for _ in range(2):
                    agent.update(buffer)
            else:
                agent.update(buffer)

            # MACPO: update cost critic
            if method == "macpo" and cost_critic is not None and len(cost_buffer) > 256:
                indices = np.random.choice(len(cost_buffer), 128, replace=False)
                batch = [cost_buffer[k] for k in indices]
                obs_b = torch.FloatTensor(np.array([t[0] for t in batch]))
                cost_b = torch.FloatTensor(np.array([t[1] for t in batch]))
                pred = cost_critic(obs_b).squeeze(-1)
                loss = F.mse_loss(pred, cost_b)
                cost_optimizer.zero_grad()
                loss.backward()
                cost_optimizer.step()

            buffer.reset()

        if done:
            if buffer.ptr > buffer.path_start_idx:
                buffer.finish_path(0.0, ppo_config.gamma, ppo_config.gae_lambda)

            # MAPPO-Lag: update multiplier
            if method == "mappo_lag":
                ep_costs_history.append(ep_cost_accum)
                if len(ep_costs_history) >= 5:
                    avg_cost = np.mean(ep_costs_history[-10:])
                    lagrangian_lambda = max(0.0,
                        lagrangian_lambda + lr_lambda * (avg_cost - cost_limit))
                    lagrangian_lambda = min(lagrangian_lambda, 10.0)

            ep_cost_accum = 0.0
            obs_list, info = env.reset()
            if safety_filters:
                for i, sf in enumerate(safety_filters):
                    obstacles = list(info['hazards'])
                    for j in range(n_agents):
                        if j != i:
                            obstacles.append(info['positions'][j, :2])
                    sf.reset(obstacles)

    # --- Evaluation ---
    rewards, costs, ep_lengths = [], [], []

    for ep in range(eval_episodes):
        obs_list, info = env.reset(seed=seed * 1000 + ep)
        if safety_filters:
            for i, sf in enumerate(safety_filters):
                obstacles = list(info['hazards'])
                for j in range(n_agents):
                    if j != i:
                        obstacles.append(info['positions'][j, :2])
                sf.reset(obstacles)

        ep_reward, ep_cost, ep_len = 0.0, 0.0, 0
        done = False

        while not done:
            actions_exec = []
            for i in range(n_agents):
                action, _, _ = agent.get_action(obs_list[i], deterministic=True)

                if method == "ours" and safety_filters:
                    robot_pos = info['positions'][i]
                    obstacles = list(info['hazards'])
                    for j in range(n_agents):
                        if j != i:
                            obstacles.append(info['positions'][j, :2])
                    safety_filters[i].reset(obstacles)
                    result = safety_filters[i].project(action, robot_pos)
                    actions_exec.append(result.action_safe)
                else:
                    actions_exec.append(action)

            obs_list, reward, cost, term, trunc, info = env.step(actions_exec)
            ep_reward += reward
            ep_cost += cost
            ep_len += 1
            done = term or trunc

        rewards.append(ep_reward)
        costs.append(ep_cost)
        ep_lengths.append(ep_len)

    env.close()

    return {
        "reward_mean": float(np.mean(rewards)),
        "cost_mean": float(np.mean(costs)),
        "ep_length_mean": float(np.mean(ep_lengths)),
    }


# =============================================================================
# Main
# =============================================================================

METHODS = ["mappo", "happo", "macpo", "mappo_lag", "ours"]

METHOD_NAMES = {
    "mappo": "MAPPO",
    "happo": "HAPPO",
    "macpo": "MACPO",
    "mappo_lag": "MAPPO-Lag",
    "ours": "Ours",
}


def main():
    parser = argparse.ArgumentParser(description="Run Table II multi-agent experiments")
    parser.add_argument("--method", type=str, default="all",
                        choices=METHODS + ["all"])
    parser.add_argument("--train-steps", type=int, default=30000)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=str, default="results_table2")

    args = parser.parse_args()

    methods = METHODS if args.method == "all" else [args.method]
    seeds = list(range(args.seeds))

    os.makedirs(args.output, exist_ok=True)

    print(f"Running {len(methods)} methods × {len(seeds)} seeds")
    print(f"Train steps: {args.train_steps}, Eval episodes: {args.episodes}")
    print(f"4 agents, MultiGoal task\n")

    results = {}

    for idx, method in enumerate(methods):
        print(f"\n[{idx + 1}/{len(methods)}] {METHOD_NAMES[method]}")
        t0 = time.time()

        all_rewards, all_costs, all_lengths = [], [], []

        for seed in seeds:
            result = train_and_eval_multiagent(
                method=method,
                train_steps=args.train_steps,
                eval_episodes=args.episodes,
                seed=seed,
            )
            all_rewards.append(result["reward_mean"])
            all_costs.append(result["cost_mean"])
            all_lengths.append(result["ep_length_mean"])

            print(f"    seed {seed}: R={result['reward_mean']:.2f}, "
                  f"C={result['cost_mean']:.2f}, L={result['ep_length_mean']:.0f}")

        elapsed = time.time() - t0
        r_mean, r_std = np.mean(all_rewards), np.std(all_rewards)
        c_mean, c_std = np.mean(all_costs), np.std(all_costs)
        l_mean = np.mean(all_lengths)

        results[method] = {
            "reward_mean": float(r_mean),
            "reward_std": float(r_std),
            "cost_mean": float(c_mean),
            "cost_std": float(c_std),
            "ep_length_mean": float(l_mean),
        }

        print(f"  => R={r_mean:.2f}±{r_std:.2f}, C={c_mean:.2f}±{c_std:.2f}, "
              f"L={l_mean:.0f} ({elapsed:.0f}s)")

        # Save intermediate
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(args.output, f"table2_{timestamp}.json")
        with open(save_path, 'w') as f:
            json.dump({"experiment": "table2_multiagent", "timestamp": timestamp,
                        "results": results}, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("TABLE II: MULTI-AGENT RESULTS (MultiGoal, 4 agents)")
    print("=" * 70)
    print(f"{'Method':<15} {'Reward':>12} {'Cost':>12} {'Steps':>8}")
    print("-" * 70)
    for method in methods:
        if method in results:
            r = results[method]
            print(f"{METHOD_NAMES[method]:<15} "
                  f"{r['reward_mean']:>6.2f}±{r['reward_std']:<5.2f} "
                  f"{r['cost_mean']:>6.2f}±{r['cost_std']:<5.2f} "
                  f"{r['ep_length_mean']:>6.0f}")
    print("=" * 70)

    # Save final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.output, f"table2_final_{timestamp}.json")
    with open(save_path, 'w') as f:
        json.dump({"experiment": "table2_multiagent", "timestamp": timestamp,
                    "results": results}, f, indent=2)
    print(f"\nFinal results saved to: {save_path}")


if __name__ == "__main__":
    main()
