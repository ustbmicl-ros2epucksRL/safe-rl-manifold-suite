#!/usr/bin/env python3
"""
Table I experiment runner for IROS 2026 paper.

Runs 5 methods × 3 tasks × 3 collision penalties × N seeds.

Methods:
    1. PPO (baseline)
    2. IPO (Interior Point Optimization)
    3. PPO-Lagrangian
    4. Recovery-RL
    5. Ours (PPO + Manifold Filter)

Usage:
    python scripts/run_table1_experiments.py --seeds 3 --train-steps 30000
    python scripts/run_table1_experiments.py --seeds 5 --train-steps 50000
    python scripts/run_table1_experiments.py --method ours --task goal  # single run
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

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

from env import SafetyGymEnv
from safety import DistanceFilter
from ppo import PPO, PPOConfig, RolloutBuffer


# =============================================================================
# Configuration
# =============================================================================

ENV_MAP = {
    "goal": "SafetyPointGoal1-v0",
    "circle": "SafetyPointCircle1-v0",
    "push": "SafetyPointPush1-v0",
}

R_COL_VALUES = [1.0, 0.5, 0.1]

METHODS = ["ppo", "ipo", "ppo_lag", "recovery_rl", "ours"]


# =============================================================================
# Baseline: PPO-Lagrangian
# =============================================================================

class PPOLagrangian:
    """
    PPO with Lagrangian constraint optimization.

    Optimizes: max_pi min_{lambda>=0} J(pi) - lambda * (J_c(pi) - d)

    The Lagrangian multiplier lambda is updated based on cost constraint violation:
        lambda <- max(0, lambda + lr_lambda * (avg_cost - cost_limit))
    """

    def __init__(self, obs_dim, act_dim, cost_limit=0.0, lr_lambda=0.01,
                 lambda_init=0.1, config=None, device='cpu'):
        self.ppo = PPO(obs_dim, act_dim, config=config, device=device)
        self.cost_limit = cost_limit
        self.lr_lambda = lr_lambda
        self.lam = lambda_init  # Lagrangian multiplier
        self.ep_costs = []

    def get_action(self, obs, deterministic=False):
        return self.ppo.get_action(obs, deterministic)

    def update(self, buffer):
        return self.ppo.update(buffer)

    def update_lambda(self, episode_cost):
        """Update Lagrangian multiplier after episode."""
        self.ep_costs.append(episode_cost)
        if len(self.ep_costs) >= 5:
            avg_cost = np.mean(self.ep_costs[-10:])
            self.lam = max(0.0, self.lam + self.lr_lambda * (avg_cost - self.cost_limit))
            self.lam = min(self.lam, 10.0)  # Clip to prevent explosion

    def get_cost_penalty(self, cost):
        """Get Lagrangian cost penalty for reward shaping."""
        return self.lam * cost


# =============================================================================
# Baseline: IPO (Interior Point Optimization)
# =============================================================================

class IPO:
    """
    Interior Point Optimization for safe RL.

    Uses log-barrier penalty on cost constraints:
        reward_ipo = reward - (1/t) * penalty(cost)

    where t increases over training (barrier becomes sharper).
    """

    def __init__(self, obs_dim, act_dim, t_init=1.0, t_max=50.0,
                 t_growth=1.001, config=None, device='cpu'):
        self.ppo = PPO(obs_dim, act_dim, config=config, device=device)
        self.t = t_init
        self.t_max = t_max
        self.t_growth = t_growth

    def get_action(self, obs, deterministic=False):
        return self.ppo.get_action(obs, deterministic)

    def update(self, buffer):
        result = self.ppo.update(buffer)
        # Increase barrier parameter
        self.t = min(self.t * self.t_growth, self.t_max)
        return result

    def get_cost_penalty(self, cost):
        """IPO barrier penalty."""
        if cost > 0:
            return self.t * cost  # Direct penalty when violated
        return 0.0


# =============================================================================
# Baseline: Recovery-RL
# =============================================================================

class SafetyCritic(nn.Module if TORCH_AVAILABLE else object):
    """Safety critic Q_safe(s,a) predicting future cumulative cost."""

    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


class RecoveryRL:
    """
    Recovery RL: uses a safety critic to detect unsafe states.

    When Q_safe(s,a) > threshold, switches to recovery action
    (reverse direction from nearest obstacle).
    """

    def __init__(self, obs_dim, act_dim, cost_threshold=0.3,
                 config=None, device='cpu'):
        self.ppo = PPO(obs_dim, act_dim, config=config, device=device)
        self.device = torch.device(device)
        self.cost_threshold = cost_threshold

        # Safety critic
        self.safety_critic = SafetyCritic(obs_dim, act_dim).to(self.device)
        self.safety_optimizer = torch.optim.Adam(
            self.safety_critic.parameters(), lr=1e-3
        )

        # Experience buffer for safety critic
        self.safety_buffer = []
        self.safety_buffer_max = 10000

    def get_action(self, obs, deterministic=False):
        return self.ppo.get_action(obs, deterministic)

    def is_unsafe(self, obs, action):
        """Check if action is predicted unsafe by safety critic."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            act_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            q_safe = self.safety_critic(obs_t, act_t).item()
        return q_safe > self.cost_threshold

    def get_recovery_action(self, obs):
        """Simple recovery: slow down and reverse."""
        return np.array([-0.3, 0.0])

    def store_safety_transition(self, obs, action, cost, next_obs, done):
        """Store transition for safety critic update."""
        self.safety_buffer.append((obs, action, cost, next_obs, done))
        if len(self.safety_buffer) > self.safety_buffer_max:
            self.safety_buffer.pop(0)

    def update_safety_critic(self, gamma=0.99):
        """Update safety critic from buffer."""
        if len(self.safety_buffer) < 256:
            return

        indices = np.random.choice(len(self.safety_buffer), 128, replace=False)
        batch = [self.safety_buffer[i] for i in indices]

        obs_b = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        act_b = torch.FloatTensor(np.array([t[1] for t in batch])).to(self.device)
        cost_b = torch.FloatTensor(np.array([t[2] for t in batch])).to(self.device)
        next_obs_b = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        done_b = torch.FloatTensor(np.array([t[4] for t in batch])).to(self.device)

        # Target: cost + gamma * Q_safe(s', a') where a' = current policy
        with torch.no_grad():
            next_act, _, _ = self.ppo.ac.get_action(next_obs_b, deterministic=True)
            q_next = self.safety_critic(next_obs_b, next_act)
            target = cost_b + gamma * (1 - done_b) * q_next

        q_pred = self.safety_critic(obs_b, act_b)
        loss = F.mse_loss(q_pred, target)

        self.safety_optimizer.zero_grad()
        loss.backward()
        self.safety_optimizer.step()

    def update(self, buffer):
        return self.ppo.update(buffer)


# =============================================================================
# Core training loop (shared across methods)
# =============================================================================

def train_and_eval(
    env_id: str,
    method: str,
    r_col: float,
    train_steps: int,
    eval_episodes: int,
    seed: int,
) -> Dict[str, float]:
    """
    Train and evaluate a single configuration.

    Returns: {reward_mean, cost_mean, ep_length_mean}
    """
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)

    env = SafetyGymEnv(env_id)
    ppo_config = PPOConfig()
    buffer_size = 2048
    buffer = RolloutBuffer(buffer_size, env.obs_dim, env.act_dim)

    # --- Create method-specific agent ---
    if method == "ppo":
        agent = PPO(env.obs_dim, env.act_dim, config=ppo_config, device='cpu')
        safety_filter = None
    elif method == "ipo":
        ipo = IPO(env.obs_dim, env.act_dim, config=ppo_config, device='cpu')
        agent = ipo
        safety_filter = None
    elif method == "ppo_lag":
        lag = PPOLagrangian(env.obs_dim, env.act_dim, config=ppo_config, device='cpu')
        agent = lag
        safety_filter = None
    elif method == "recovery_rl":
        rec = RecoveryRL(env.obs_dim, env.act_dim, config=ppo_config, device='cpu')
        agent = rec
        safety_filter = None
    elif method == "ours":
        agent = PPO(env.obs_dim, env.act_dim, config=ppo_config, device='cpu')
        safety_filter = DistanceFilter(
            danger_radius=0.6,
            stop_radius=0.3,
            hazard_radius=0.2,
            lambda_calib=0.1,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # --- Training ---
    obs, info = env.reset(seed=seed)
    if safety_filter:
        safety_filter.reset(info.get('hazards', []))

    ep_cost_accum = 0.0

    for step in range(train_steps):
        action, log_prob, value = agent.get_action(obs)

        # Method-specific action modification
        if method == "ours" and safety_filter:
            robot_pos = info.get('robot_pos', np.zeros(3))
            result = safety_filter.project(action, robot_pos)
            action_exec = result.action_safe
        elif method == "recovery_rl":
            if agent.is_unsafe(obs, action):
                action_exec = agent.get_recovery_action(obs)
            else:
                action_exec = action
        else:
            action_exec = action

        obs_next, reward, cost, term, trunc, info = env.step(action_exec)

        # Method-specific reward modification
        reward_modified = reward - r_col * cost

        if method == "ours" and safety_filter:
            correction_norm = np.linalg.norm(result.correction)
            if correction_norm > 0:
                reward_modified -= 0.1 * (correction_norm ** 2)
        elif method == "ppo_lag":
            reward_modified -= agent.get_cost_penalty(cost)
        elif method == "ipo":
            reward_modified -= agent.get_cost_penalty(cost)

        # Store in buffer (use original action for PPO update, not modified)
        done = term or trunc
        buffer.add(obs, action, reward_modified, value, log_prob, done)

        # Recovery-RL: store safety transition
        if method == "recovery_rl":
            agent.store_safety_transition(obs, action, cost, obs_next, float(done))

        obs = obs_next
        ep_cost_accum += cost

        # PPO update when buffer full
        if buffer.ptr == buffer_size:
            _, _, last_value = agent.get_action(obs)
            buffer.finish_path(last_value, ppo_config.gamma, ppo_config.gae_lambda)
            agent.update(buffer)
            buffer.reset()

            # Recovery-RL: update safety critic
            if method == "recovery_rl":
                agent.update_safety_critic()

        if done:
            if buffer.ptr > buffer.path_start_idx:
                buffer.finish_path(0.0, ppo_config.gamma, ppo_config.gae_lambda)

            # PPO-Lag: update multiplier
            if method == "ppo_lag":
                agent.update_lambda(ep_cost_accum)

            ep_cost_accum = 0.0
            obs, info = env.reset()
            if safety_filter:
                safety_filter.reset(info.get('hazards', []))

    # --- Evaluation ---
    rewards, costs, ep_lengths = [], [], []

    for ep in range(eval_episodes):
        obs, info = env.reset(seed=seed * 1000 + ep)
        if safety_filter:
            safety_filter.reset(info.get('hazards', []))

        ep_reward, ep_cost, ep_len = 0.0, 0.0, 0
        done = False

        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)

            if method == "ours" and safety_filter:
                robot_pos = info.get('robot_pos', np.zeros(3))
                result = safety_filter.project(action, robot_pos)
                action = result.action_safe
            elif method == "recovery_rl":
                if agent.is_unsafe(obs, action):
                    action = agent.get_recovery_action(obs)

            obs, reward, cost, term, trunc, info = env.step(action)
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
# Run all seeds for one configuration
# =============================================================================

def run_config(
    method: str,
    task: str,
    r_col: float,
    train_steps: int,
    eval_episodes: int,
    seeds: List[int],
) -> Dict[str, float]:
    """Run a single (method, task, r_col) across all seeds."""
    env_id = ENV_MAP[task]

    all_rewards = []
    all_costs = []
    all_lengths = []

    for seed in seeds:
        result = train_and_eval(
            env_id=env_id,
            method=method,
            r_col=r_col,
            train_steps=train_steps,
            eval_episodes=eval_episodes,
            seed=seed,
        )
        all_rewards.append(result["reward_mean"])
        all_costs.append(result["cost_mean"])
        all_lengths.append(result["ep_length_mean"])

        print(f"    seed {seed}: R={result['reward_mean']:.2f}, "
              f"C={result['cost_mean']:.2f}, L={result['ep_length_mean']:.0f}")

    return {
        "reward_mean": float(np.mean(all_rewards)),
        "reward_std": float(np.std(all_rewards)),
        "cost_mean": float(np.mean(all_costs)),
        "cost_std": float(np.std(all_costs)),
        "ep_length_mean": float(np.mean(all_lengths)),
        "ep_length_std": float(np.std(all_lengths)),
    }


# =============================================================================
# Main
# =============================================================================

METHOD_NAMES = {
    "ppo": "PPO",
    "ipo": "IPO",
    "ppo_lag": "PPO-Lag",
    "recovery_rl": "Recovery-RL",
    "ours": "Ours",
}


def print_table1(results: Dict):
    """Print Table I format."""
    print("\n" + "=" * 100)
    print("TABLE I: COMPARISON OF SAFE RL METHODS")
    print("=" * 100)

    header = f"{'Task':<8} {'r_col':<6}"
    for m in METHODS:
        header += f" | {'Rew':>7} {'Cost':>7} {'Steps':>5}"
    print(header)
    print("-" * 100)

    for task in ["goal", "circle", "push"]:
        for r_col in R_COL_VALUES:
            row = f"{task:<8} {r_col:<6.1f}"
            for method in METHODS:
                key = f"{method}_{task}_{r_col}"
                if key in results:
                    r = results[key]
                    row += f" | {r['reward_mean']:>7.2f} {r['cost_mean']:>7.2f} {r['ep_length_mean']:>5.0f}"
                else:
                    row += f" |    N/A     N/A   N/A"
            print(row)
        print("-" * 100)

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Run Table I experiments")
    parser.add_argument("--method", type=str, default="all",
                        choices=METHODS + ["all"])
    parser.add_argument("--task", type=str, default="all",
                        choices=list(ENV_MAP.keys()) + ["all"])
    parser.add_argument("--r-col", type=float, default=None,
                        help="Specific r_col value (default: all)")
    parser.add_argument("--train-steps", type=int, default=30000)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=str, default="results_table1")

    args = parser.parse_args()

    methods = METHODS if args.method == "all" else [args.method]
    tasks = list(ENV_MAP.keys()) if args.task == "all" else [args.task]
    r_cols = R_COL_VALUES if args.r_col is None else [args.r_col]
    seeds = list(range(args.seeds))

    total_configs = len(methods) * len(tasks) * len(r_cols)
    print(f"Running {total_configs} configs × {len(seeds)} seeds "
          f"= {total_configs * len(seeds)} total runs")
    print(f"Train steps: {args.train_steps}, Eval episodes: {args.episodes}")

    results = {}
    config_idx = 0

    for method in methods:
        for task in tasks:
            for r_col in r_cols:
                config_idx += 1
                key = f"{method}_{task}_{r_col}"

                print(f"\n[{config_idx}/{total_configs}] "
                      f"{METHOD_NAMES.get(method, method)} / {task} / r_col={r_col}")

                t0 = time.time()
                results[key] = run_config(
                    method=method,
                    task=task,
                    r_col=r_col,
                    train_steps=args.train_steps,
                    eval_episodes=args.episodes,
                    seeds=seeds,
                )
                elapsed = time.time() - t0
                r = results[key]
                print(f"  => R={r['reward_mean']:.2f}±{r['reward_std']:.2f}, "
                      f"C={r['cost_mean']:.2f}±{r['cost_std']:.2f}, "
                      f"L={r['ep_length_mean']:.0f} ({elapsed:.0f}s)")

                # Save intermediate results
                _save_results(results, args.output)

    print_table1(results)
    filepath = _save_results(results, args.output)
    print(f"\nFinal results saved to: {filepath}")


def _save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"table1_{timestamp}.json")
    with open(filepath, 'w') as f:
        json.dump({"experiment": "table1", "timestamp": timestamp,
                    "results": results}, f, indent=2)
    return filepath


if __name__ == "__main__":
    main()
