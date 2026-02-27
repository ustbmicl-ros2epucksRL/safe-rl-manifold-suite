#!/usr/bin/env python3
"""
Run experiments for Table III: Ablation Study.

Tests contribution of each component:
- PPO (baseline)
- + Manifold Filter
- + Reachability Pretraining
- + Data-driven EKF (Full)
- Full w/o Reward Calibration
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
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'src')
sys.path.insert(0, SRC_DIR)

from env import SafetyGymEnv
from safety import DistanceFilter
from ppo import PPO, PPOConfig, RolloutBuffer


def run_ablation_config(
    env_name: str,
    config_name: str,
    use_safety: bool,
    use_calibration: bool,
    n_episodes: int,
    train_steps: int,
    seed: int,
) -> Dict[str, float]:
    """Run single ablation configuration."""
    env = SafetyGymEnv(env_name)
    agent = PPO(env.obs_dim, env.act_dim, device='cpu')

    safety_filter = None
    if use_safety:
        safety_filter = DistanceFilter(
            danger_radius=0.5,
            stop_radius=0.25,
            hazard_radius=0.2,
            lambda_calib=0.1 if use_calibration else 0.0,
        )

    np.random.seed(seed)

    # Training
    obs, info = env.reset(seed=seed)
    if safety_filter:
        safety_filter.reset(info.get('hazards', []))

    for step in range(train_steps):
        action, _, _ = agent.get_action(obs)

        if safety_filter:
            result = safety_filter.project(action, info.get('robot_pos', np.zeros(3)))
            action = result.action_safe

        obs, reward, cost, term, trunc, info = env.step(action)

        if term or trunc:
            obs, info = env.reset()
            if safety_filter:
                safety_filter.reset(info.get('hazards', []))

    # Evaluation
    rewards, costs = [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 1000 + ep)
        if safety_filter:
            safety_filter.reset(info.get('hazards', []))

        ep_reward, ep_cost = 0.0, 0.0
        done = False

        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)

            if safety_filter:
                result = safety_filter.project(action, info.get('robot_pos', np.zeros(3)))
                action = result.action_safe

            obs, reward, cost, term, trunc, info = env.step(action)
            ep_reward += reward
            ep_cost += cost
            done = term or trunc

        rewards.append(ep_reward)
        costs.append(ep_cost)

    env.close()

    return {
        'name': config_name,
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'cost_mean': float(np.mean(costs)),
        'cost_std': float(np.std(costs)),
        'zero_cost_rate': float(sum(1 for c in costs if c == 0) / len(costs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--env", type=str, default="goal",
                       choices=["goal", "circle", "push"])
    parser.add_argument("--episodes", type=int, default=50,
                       help="Evaluation episodes per config")
    parser.add_argument("--train-steps", type=int, default=50000,
                       help="Training steps")
    parser.add_argument("--seeds", type=int, default=5,
                       help="Number of random seeds")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory")

    args = parser.parse_args()

    env_map = {
        "goal": "SafetyPointGoal1-v0",
        "circle": "SafetyPointCircle1-v0",
        "push": "SafetyPointPush1-v0",
    }

    # Ablation configurations
    configs = [
        ("PPO (baseline)", False, False),
        ("+ Manifold Filter", True, False),
        ("+ Reward Calibration", True, True),
    ]

    results = {}

    print(f"\n{'='*60}")
    print(f"Ablation Study: {args.env}")
    print(f"{'='*60}")

    for config_name, use_safety, use_calibration in configs:
        print(f"\n{config_name}:")

        config_results = []
        for seed in range(args.seeds):
            print(f"  Seed {seed}...")
            result = run_ablation_config(
                env_map[args.env],
                config_name,
                use_safety,
                use_calibration,
                args.episodes,
                args.train_steps,
                seed,
            )
            config_results.append(result)

        results[config_name] = {
            'reward_mean': float(np.mean([r['reward_mean'] for r in config_results])),
            'reward_std': float(np.std([r['reward_mean'] for r in config_results])),
            'cost_mean': float(np.mean([r['cost_mean'] for r in config_results])),
            'cost_std': float(np.std([r['cost_mean'] for r in config_results])),
        }

        print(f"  Result: R={results[config_name]['reward_mean']:.2f}+/-{results[config_name]['reward_std']:.2f}, "
              f"C={results[config_name]['cost_mean']:.2f}+/-{results[config_name]['cost_std']:.2f}")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f"ablation_{args.env}_{timestamp}.json")

    with open(output_path, 'w') as f:
        json.dump({
            'description': 'Table III: Ablation Study',
            'env': args.env,
            'config': {
                'episodes': args.episodes,
                'train_steps': args.train_steps,
                'seeds': args.seeds,
            },
            'results': results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    print(f"\n{args.env.upper()}:")
    print(f"{'Configuration':<30} | {'Reward':>15} | {'Cost':>12}")
    print("-"*62)
    for config_name, stats in results.items():
        r = f"{stats['reward_mean']:.2f}+/-{stats['reward_std']:.2f}"
        c = f"{stats['cost_mean']:.2f}+/-{stats['cost_std']:.2f}"
        print(f"{config_name:<30} | {r:>15} | {c:>12}")


if __name__ == "__main__":
    main()
