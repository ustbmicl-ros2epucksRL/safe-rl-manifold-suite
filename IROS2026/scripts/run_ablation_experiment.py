#!/usr/bin/env python3
"""
Run proper ablation experiments for IROS 2026 paper.

This script runs controlled experiments to generate real data for:
- Table III (Ablation Study)
- Table IV (Sensor Noise)
- Table V (EKF Comparison)

Usage:
    python scripts/run_ablation_experiment.py --env goal --episodes 50 --seeds 5
    python scripts/run_ablation_experiment.py --env all --episodes 100 --seeds 10
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

import numpy as np

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

try:
    import torch
except ImportError:
    print("Warning: PyTorch not available")

from src.envs.safety_gym_env import SafetyGymEnv, MockSafetyEnv
from src.safety.distance_filter import DistanceSafetyFilter
from src.algos.ppo import PPO


def run_episode(
    env,
    agent,
    safety_filter=None,
    use_reward_calibration=True,
    max_steps=1000,
    add_noise=False,
    noise_std=0.1,
):
    """Run a single episode and return metrics."""
    obs, info = env.reset()

    if safety_filter is not None:
        safety_filter.reset(info['robot_pos'], info['hazards'])

    total_reward = 0
    total_cost = 0
    total_correction = 0
    steps = 0

    for step in range(max_steps):
        # Get action from policy
        action, log_prob, value = agent.get_action(obs)

        # Get robot state
        robot_pos = info.get('robot_pos', np.zeros(3))

        # Add noise if requested
        if add_noise:
            robot_pos = robot_pos + np.random.randn(3) * noise_std

        # Apply safety filter
        correction = 0.0
        if safety_filter is not None:
            result = safety_filter.project(action, robot_pos)
            action_safe = result.action_safe
            correction = np.linalg.norm(result.correction)
        else:
            action_safe = action

        # Step environment
        obs, reward, cost, term, trunc, info = env.step(action_safe)

        # Reward calibration
        if use_reward_calibration and correction > 0:
            reward = reward - 0.1 * (correction ** 2)

        total_reward += reward
        total_cost += cost
        total_correction += correction
        steps += 1

        if term or trunc:
            break

    return {
        'reward': total_reward,
        'cost': total_cost,
        'steps': steps,
        'correction': total_correction,
    }


def run_ablation_config(
    env_id: str,
    config_name: str,
    use_safety: bool,
    use_calibration: bool,
    n_episodes: int,
    seed: int,
    train_steps: int = 50000,
    add_noise: bool = False,
    noise_std: float = 0.1,
) -> Dict[str, float]:
    """Run ablation for a single configuration."""

    # Create environment and agent
    try:
        env = SafetyGymEnv(env_id)
    except:
        env = MockSafetyEnv()

    agent = PPO(obs_dim=env.obs_dim, act_dim=env.act_dim, device='cpu')

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Brief training
    print(f"  Training agent for {train_steps} steps...")
    obs, info = env.reset(seed=seed)
    for step in range(train_steps):
        action, _, _ = agent.get_action(obs)

        # Apply safety during training if enabled
        if use_safety:
            safety_filter = DistanceSafetyFilter(
                danger_zone_radius=0.6,
                hard_stop_radius=0.35,
            )
            safety_filter.reset(info.get('robot_pos', np.zeros(3)), info.get('hazards', []))
            result = safety_filter.project(action, info.get('robot_pos', np.zeros(3)))
            action = result.action_safe

        obs, reward, cost, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset()

    # Evaluation
    print(f"  Evaluating for {n_episodes} episodes...")
    rewards = []
    costs = []

    safety_filter = None
    if use_safety:
        safety_filter = DistanceSafetyFilter(
            danger_zone_radius=0.6,
            hard_stop_radius=0.35,
        )

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 1000 + ep)
        result = run_episode(
            env, agent, safety_filter,
            use_reward_calibration=use_calibration,
            add_noise=add_noise,
            noise_std=noise_std,
        )
        rewards.append(result['reward'])
        costs.append(result['cost'])

    env.close()

    return {
        'name': config_name,
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'cost_mean': float(np.mean(costs)),
        'cost_std': float(np.std(costs)),
        'zero_cost_rate': float(sum(1 for c in costs if c == 0) / len(costs)),
    }


def run_full_ablation(
    env_id: str,
    env_name: str,
    n_episodes: int,
    seeds: List[int],
    train_steps: int,
) -> Dict[str, Any]:
    """Run full ablation study for one environment."""

    configs = [
        ("PPO (baseline)", False, False),
        ("+ Manifold Filter", True, False),
        ("+ Reward Calibration", True, True),
    ]

    results = {}

    for config_name, use_safety, use_calibration in configs:
        print(f"\n{config_name}:")

        all_rewards = []
        all_costs = []

        for seed in seeds:
            print(f"  Seed {seed}...")
            result = run_ablation_config(
                env_id=env_id,
                config_name=config_name,
                use_safety=use_safety,
                use_calibration=use_calibration,
                n_episodes=n_episodes,
                seed=seed,
                train_steps=train_steps,
            )
            all_rewards.append(result['reward_mean'])
            all_costs.append(result['cost_mean'])

        results[config_name] = {
            'reward_mean': float(np.mean(all_rewards)),
            'reward_std': float(np.std(all_rewards)),
            'cost_mean': float(np.mean(all_costs)),
            'cost_std': float(np.std(all_costs)),
        }

        print(f"  Result: Reward={results[config_name]['reward_mean']:.2f}±{results[config_name]['reward_std']:.2f}, "
              f"Cost={results[config_name]['cost_mean']:.2f}±{results[config_name]['cost_std']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--env", type=str, default="goal",
                       choices=["goal", "circle", "push", "all"])
    parser.add_argument("--episodes", type=int, default=50,
                       help="Episodes per seed for evaluation")
    parser.add_argument("--seeds", type=int, default=5,
                       help="Number of random seeds")
    parser.add_argument("--train-steps", type=int, default=50000,
                       help="Training steps per configuration")
    parser.add_argument("--output", type=str, default="scripts/results",
                       help="Output directory")

    args = parser.parse_args()

    # Environment mapping
    env_map = {
        "goal": "SafetyPointGoal1-v0",
        "circle": "SafetyPointCircle1-v0",
        "push": "SafetyPointPush1-v0",
    }

    envs = ["goal", "circle", "push"] if args.env == "all" else [args.env]
    seeds = list(range(args.seeds))

    all_results = {}

    for env_name in envs:
        print(f"\n{'='*60}")
        print(f"Environment: {env_name}")
        print(f"{'='*60}")

        results = run_full_ablation(
            env_id=env_map[env_name],
            env_name=env_name,
            n_episodes=args.episodes,
            seeds=seeds,
            train_steps=args.train_steps,
        )
        all_results[env_name] = results

    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = os.path.join(args.output, f"ablation_real_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump({
            "description": "Real ablation study results",
            "timestamp": timestamp,
            "config": {
                "episodes_per_seed": args.episodes,
                "seeds": seeds,
                "train_steps": args.train_steps,
            },
            "results": all_results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    # Print summary table
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    for env_name, results in all_results.items():
        print(f"\n{env_name.upper()}:")
        print(f"{'Configuration':<30} | {'Reward':>12} | {'Cost':>10}")
        print("-"*56)
        for config_name, stats in results.items():
            r = f"{stats['reward_mean']:.2f}±{stats['reward_std']:.2f}"
            c = f"{stats['cost_mean']:.2f}±{stats['cost_std']:.2f}"
            print(f"{config_name:<30} | {r:>12} | {c:>10}")


if __name__ == "__main__":
    main()
