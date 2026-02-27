#!/usr/bin/env python3
"""
Run experiments for Table V: EKF Comparison.

Compares different state estimation methods:
- No filtering
- Standard EKF (fixed R)
- Data-driven EKF (learned R)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'src')
sys.path.insert(0, SRC_DIR)

from env import SafetyGymEnv
from safety import DistanceFilter
from ppo import PPO
from ekf import DataDrivenEKF, StandardEKF, EKFConfig


def run_ekf_comparison(
    env_name: str,
    config_name: str,
    ekf_type: str,  # 'none', 'standard', 'learned'
    noise_std: float,
    n_episodes: int,
    train_steps: int,
    seed: int,
) -> Dict[str, float]:
    """Run single EKF comparison experiment."""
    env = SafetyGymEnv(env_name)
    agent = PPO(env.obs_dim, env.act_dim, device='cpu')

    safety_filter = DistanceFilter(
        danger_radius=0.5,
        stop_radius=0.25,
        hazard_radius=0.2,
    )

    # Create EKF based on type
    ekf = None
    if ekf_type == 'standard':
        ekf = StandardEKF(EKFConfig())
    elif ekf_type == 'learned':
        ekf = DataDrivenEKF(EKFConfig())

    np.random.seed(seed)

    # Training
    obs, info = env.reset(seed=seed)
    safety_filter.reset(info.get('hazards', []))
    if ekf:
        ekf.reset(info.get('robot_pos', np.zeros(3)))

    for step in range(train_steps):
        action, _, _ = agent.get_action(obs)

        # Get noisy measurement
        true_pos = info.get('robot_pos', np.zeros(3))
        noisy_pos = true_pos + np.random.randn(3) * noise_std

        # Get position estimate
        if ekf:
            robot_pos = ekf.get_position()
        else:
            robot_pos = noisy_pos

        result = safety_filter.project(action, robot_pos)
        action = result.action_safe

        obs, reward, cost, term, trunc, info = env.step(action)

        # Update EKF
        if ekf:
            measurement = info.get('robot_pos', np.zeros(3)) + np.random.randn(3) * noise_std
            ekf.predict(action)
            ekf.update(measurement)

        if term or trunc:
            obs, info = env.reset()
            safety_filter.reset(info.get('hazards', []))
            if ekf:
                ekf.reset(info.get('robot_pos', np.zeros(3)))

    # Evaluation
    rewards, costs, pos_errors = [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 1000 + ep)
        safety_filter.reset(info.get('hazards', []))
        if ekf:
            ekf.reset(info.get('robot_pos', np.zeros(3)))

        ep_reward, ep_cost = 0.0, 0.0
        ep_pos_errors = []
        done = False

        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)

            true_pos = info.get('robot_pos', np.zeros(3))
            noisy_pos = true_pos + np.random.randn(3) * noise_std

            if ekf:
                robot_pos = ekf.get_position()
            else:
                robot_pos = noisy_pos

            # Track position error
            pos_error = np.linalg.norm(robot_pos[:2] - true_pos[:2])
            ep_pos_errors.append(pos_error)

            result = safety_filter.project(action, robot_pos)
            action = result.action_safe

            obs, reward, cost, term, trunc, info = env.step(action)

            if ekf:
                measurement = info.get('robot_pos', np.zeros(3)) + np.random.randn(3) * noise_std
                ekf.predict(action)
                ekf.update(measurement)

            ep_reward += reward
            ep_cost += cost
            done = term or trunc

        rewards.append(ep_reward)
        costs.append(ep_cost)
        pos_errors.append(np.mean(ep_pos_errors))

    env.close()

    return {
        'name': config_name,
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'cost_mean': float(np.mean(costs)),
        'cost_std': float(np.std(costs)),
        'pos_error_mean': float(np.mean(pos_errors)),
        'pos_error_std': float(np.std(pos_errors)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run EKF comparison experiments")
    parser.add_argument("--env", type=str, default="goal")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--train-steps", type=int, default=50000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="results")

    args = parser.parse_args()

    env_map = {
        "goal": "SafetyPointGoal1-v0",
        "circle": "SafetyPointCircle1-v0",
        "push": "SafetyPointPush1-v0",
    }

    # EKF configurations
    configs = [
        ("No filtering", "none"),
        ("Standard EKF (fixed R)", "standard"),
        ("Data-driven EKF (learned R)", "learned"),
    ]

    results = {}

    print(f"\n{'='*60}")
    print(f"EKF Comparison: {args.env}")
    print(f"Noise std: {args.noise_std}")
    print(f"{'='*60}")

    for config_name, ekf_type in configs:
        print(f"\n{config_name}:")

        config_results = []
        for seed in range(args.seeds):
            print(f"  Seed {seed}...")
            result = run_ekf_comparison(
                env_map[args.env],
                config_name,
                ekf_type,
                args.noise_std,
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
            'pos_error_mean': float(np.mean([r['pos_error_mean'] for r in config_results])),
            'pos_error_std': float(np.std([r['pos_error_mean'] for r in config_results])),
        }

        print(f"  Result: Pos Error={results[config_name]['pos_error_mean']:.3f}m, "
              f"R={results[config_name]['reward_mean']:.2f}, "
              f"C={results[config_name]['cost_mean']:.2f}")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f"ekf_compare_{args.env}_{timestamp}.json")

    with open(output_path, 'w') as f:
        json.dump({
            'description': 'Table V: EKF Comparison',
            'env': args.env,
            'noise_std': args.noise_std,
            'config': {
                'episodes': args.episodes,
                'train_steps': args.train_steps,
                'seeds': args.seeds,
            },
            'results': results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("EKF COMPARISON SUMMARY (Table V)")
    print("="*70)
    print(f"{'Method':<30} | {'Pos. Error (m)':>14} | {'Reward':>10} | {'Cost':>8}")
    print("-"*70)
    for config_name, stats in results.items():
        e = f"{stats['pos_error_mean']:.2f}+/-{stats['pos_error_std']:.2f}"
        r = f"{stats['reward_mean']:.2f}"
        c = f"{stats['cost_mean']:.2f}"
        print(f"{config_name:<30} | {e:>14} | {r:>10} | {c:>8}")


if __name__ == "__main__":
    main()
