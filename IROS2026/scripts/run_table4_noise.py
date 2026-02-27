#!/usr/bin/env python3
"""
Run experiments for Table IV: Sensor Noise Impact.

Configurations:
1. PPO (no noise)              <-- MISSING in original IROS, need to add
2. PPO (with noise)            <-- Exists: 1.29, 11.33
3. PPO + Manifold Filter       <-- Exists: 1.23, 1.91
4. PPO + Manifold + EKF        <-- Exists: 1.93, 0.00

Usage:
    python scripts/run_table4_noise.py --env goal --seeds 5
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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from env import SafetyGymEnv
from safety import DistanceFilter
from ppo import PPO
from ekf import DataDrivenEKF, EKFConfig


ENV_MAP = {
    "goal": "SafetyPointGoal1-v0",
    "circle": "SafetyPointCircle1-v0",
    "push": "SafetyPointPush1-v0",
}


def run_noise_config(
    env_id: str,
    config_name: str,
    add_noise: bool,
    use_safety: bool,
    use_ekf: bool,
    noise_std: float,
    train_steps: int,
    eval_episodes: int,
    seeds: List[int],
) -> Dict[str, float]:
    """Run noise experiment configuration."""
    all_rewards = []
    all_costs = []

    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)

        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)

        env = SafetyGymEnv(env_id)
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
            ekf = DataDrivenEKF(EKFConfig())

        # Training
        obs, info = env.reset(seed=seed)
        if safety_filter:
            safety_filter.reset(info.get('hazards', []))
        if ekf:
            ekf.reset(info.get('robot_pos', np.zeros(3)))

        for step in range(train_steps):
            action, _, _ = agent.get_action(obs)

            # Get position (with optional noise)
            true_pos = info.get('robot_pos', np.zeros(3))
            if add_noise:
                noisy_pos = true_pos + np.random.randn(3) * noise_std
            else:
                noisy_pos = true_pos

            # Use EKF estimate if available
            if ekf:
                robot_pos = ekf.get_position()
            else:
                robot_pos = noisy_pos

            # Apply safety filter
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

        for ep in range(eval_episodes):
            obs, info = env.reset(seed=seed * 1000 + ep)
            if safety_filter:
                safety_filter.reset(info.get('hazards', []))
            if ekf:
                ekf.reset(info.get('robot_pos', np.zeros(3)))

            ep_reward, ep_cost = 0.0, 0.0
            done = False

            while not done:
                action, _, _ = agent.get_action(obs, deterministic=True)

                true_pos = info.get('robot_pos', np.zeros(3))
                if add_noise:
                    noisy_pos = true_pos + np.random.randn(3) * noise_std
                else:
                    noisy_pos = true_pos

                if ekf:
                    robot_pos = ekf.get_position()
                else:
                    robot_pos = noisy_pos

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

        avg_r = np.mean(rewards)
        avg_c = np.mean(costs)
        all_rewards.append(avg_r)
        all_costs.append(avg_c)
        print(f"R={avg_r:.2f}, C={avg_c:.2f}")

    return {
        "reward_mean": float(np.mean(all_rewards)),
        "reward_std": float(np.std(all_rewards)),
        "cost_mean": float(np.mean(all_costs)),
        "cost_std": float(np.std(all_costs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Table IV noise experiments")
    parser.add_argument("--env", type=str, default="goal")
    parser.add_argument("--train-steps", type=int, default=50000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="results")

    args = parser.parse_args()

    env_id = ENV_MAP.get(args.env, args.env)
    seeds = list(range(args.seeds))

    # Table IV configurations
    configs = [
        # (name, add_noise, use_safety, use_ekf)
        ("PPO (no noise)", False, False, False),
        ("PPO (with noise)", True, False, False),
        ("PPO + Manifold Filter", True, True, False),
        ("PPO + Manifold + EKF", True, True, True),
    ]

    results = {}

    print(f"\n{'='*70}")
    print(f"TABLE IV: SENSOR NOISE EXPERIMENTS - {args.env.upper()}")
    print(f"Noise std: {args.noise_std}m")
    print(f"{'='*70}")

    for config_name, add_noise, use_safety, use_ekf in configs:
        print(f"\n{config_name}:")

        result = run_noise_config(
            env_id=env_id,
            config_name=config_name,
            add_noise=add_noise,
            use_safety=use_safety,
            use_ekf=use_ekf,
            noise_std=args.noise_std,
            train_steps=args.train_steps,
            eval_episodes=args.episodes,
            seeds=seeds,
        )

        results[config_name] = result

    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f"table4_noise_{args.env}_{timestamp}.json")

    with open(output_path, 'w') as f:
        json.dump({
            "description": "Table IV: Sensor Noise Impact",
            "env": args.env,
            "noise_std": args.noise_std,
            "config": {
                "train_steps": args.train_steps,
                "episodes": args.episodes,
                "seeds": seeds,
            },
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "="*70)
    print("TABLE IV: SENSOR NOISE IMPACT")
    print("="*70)
    print(f"{'Configuration':<30} | {'Reward':>15} | {'Cost':>12}")
    print("-"*62)
    for name, stats in results.items():
        r = f"{stats['reward_mean']:.2f} ± {stats['reward_std']:.2f}"
        c = f"{stats['cost_mean']:.2f} ± {stats['cost_std']:.2f}"
        print(f"{name:<30} | {r:>15} | {c:>12}")
    print("="*70)

    # Print LaTeX format
    print("\n% LaTeX Table IV")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Impact of Sensor Noise on Goal Task (Gaussian noise $\\sigma = 0.1$m)}")
    print("\\begin{tabular}{l|cc}")
    print("\\toprule")
    print("Configuration & Reward & Cost \\\\")
    print("\\midrule")
    for name, stats in results.items():
        r = f"${stats['reward_mean']:.2f} \\pm {stats['reward_std']:.2f}$"
        c = f"${stats['cost_mean']:.2f} \\pm {stats['cost_std']:.2f}$"
        if name == "PPO + Manifold + EKF":
            print(f"{name} & \\textbf{{{r}}} & \\textbf{{{c}}} \\\\")
        else:
            print(f"{name} & {r} & {c} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
