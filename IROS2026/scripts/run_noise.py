#!/usr/bin/env python3
"""
Run experiments for Table IV: Sensor Noise Experiments.

Tests robustness to perception uncertainty:
- PPO (no noise)
- PPO (with noise)
- PPO + Manifold Filter
- PPO + Manifold + EKF
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
from ekf import DataDrivenEKF, EKFConfig


def run_noise_experiment(
    env_name: str,
    config_name: str,
    add_noise: bool,
    use_safety: bool,
    use_ekf: bool,
    noise_std: float,
    n_episodes: int,
    train_steps: int,
    seed: int,
) -> Dict[str, float]:
    """Run single noise experiment."""
    env = SafetyGymEnv(env_name)
    agent = PPO(env.obs_dim, env.act_dim, device='cpu')

    safety_filter = None
    if use_safety:
        safety_filter = DistanceFilter(
            danger_radius=0.5,
            stop_radius=0.25,
            hazard_radius=0.2,
        )

    ekf = None
    if use_ekf:
        ekf_config = EKFConfig()
        ekf = DataDrivenEKF(ekf_config)

    np.random.seed(seed)

    # Training
    obs, info = env.reset(seed=seed)
    if safety_filter:
        safety_filter.reset(info.get('hazards', []))
    if ekf:
        ekf.reset(info.get('robot_pos', np.zeros(3)))

    for step in range(train_steps):
        action, _, _ = agent.get_action(obs)

        # Get robot position (with optional noise)
        robot_pos = info.get('robot_pos', np.zeros(3))
        if add_noise:
            robot_pos = robot_pos + np.random.randn(3) * noise_std

        # Use EKF estimate
        if ekf:
            robot_pos = ekf.get_position()

        if safety_filter:
            result = safety_filter.project(action, robot_pos)
            action = result.action_safe

        obs, reward, cost, term, trunc, info = env.step(action)

        # Update EKF
        if ekf:
            measurement = info.get('robot_pos', np.zeros(3))
            if add_noise:
                measurement = measurement + np.random.randn(3) * noise_std
            ekf.predict(action)
            ekf.update(measurement)

        if term or trunc:
            obs, info = env.reset()
            if safety_filter:
                safety_filter.reset(info.get('hazards', []))
            if ekf:
                ekf.reset(info.get('robot_pos', np.zeros(3)))

    # Evaluation
    rewards, costs = [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 1000 + ep)
        if safety_filter:
            safety_filter.reset(info.get('hazards', []))
        if ekf:
            ekf.reset(info.get('robot_pos', np.zeros(3)))

        ep_reward, ep_cost = 0.0, 0.0
        done = False

        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)

            robot_pos = info.get('robot_pos', np.zeros(3))
            if add_noise:
                robot_pos = robot_pos + np.random.randn(3) * noise_std

            if ekf:
                robot_pos = ekf.get_position()

            if safety_filter:
                result = safety_filter.project(action, robot_pos)
                action = result.action_safe

            obs, reward, cost, term, trunc, info = env.step(action)

            if ekf:
                measurement = info.get('robot_pos', np.zeros(3))
                if add_noise:
                    measurement = measurement + np.random.randn(3) * noise_std
                ekf.predict(action)
                ekf.update(measurement)

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
    }


def main():
    parser = argparse.ArgumentParser(description="Run noise experiments")
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

    # Noise experiment configurations
    configs = [
        ("PPO (no noise)", False, False, False),
        ("PPO (with noise)", True, False, False),
        ("PPO + Manifold Filter", True, True, False),
        ("PPO + Manifold + EKF", True, True, True),
    ]

    results = {}

    print(f"\n{'='*60}")
    print(f"Sensor Noise Experiments: {args.env}")
    print(f"Noise std: {args.noise_std}")
    print(f"{'='*60}")

    for config_name, add_noise, use_safety, use_ekf in configs:
        print(f"\n{config_name}:")

        config_results = []
        for seed in range(args.seeds):
            print(f"  Seed {seed}...")
            result = run_noise_experiment(
                env_map[args.env],
                config_name,
                add_noise,
                use_safety,
                use_ekf,
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
        }

        print(f"  Result: R={results[config_name]['reward_mean']:.2f}, "
              f"C={results[config_name]['cost_mean']:.2f}")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f"noise_{args.env}_{timestamp}.json")

    with open(output_path, 'w') as f:
        json.dump({
            'description': 'Table IV: Sensor Noise Experiments',
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
    print("NOISE EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Configuration':<25} | {'Reward':>12} | {'Cost':>10}")
    print("-"*52)
    for config_name, stats in results.items():
        r = f"{stats['reward_mean']:.2f}"
        c = f"{stats['cost_mean']:.2f}"
        print(f"{config_name:<25} | {r:>12} | {c:>10}")


if __name__ == "__main__":
    main()
