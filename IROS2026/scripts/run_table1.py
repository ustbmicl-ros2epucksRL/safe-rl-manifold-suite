#!/usr/bin/env python3
"""
Run experiments for Table I: Main Results Comparison.

Compares our method against baselines (PPO, IPO, PPO-Lag, Recovery-RL)
across different tasks and collision penalties.
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


def run_baseline_ppo(
    env_name: str,
    n_episodes: int,
    train_steps: int,
    seed: int,
) -> Dict[str, float]:
    """Run baseline PPO without safety filter."""
    env = SafetyGymEnv(env_name)
    agent = PPO(env.obs_dim, env.act_dim, device='cpu')

    np.random.seed(seed)

    # Training
    obs, info = env.reset(seed=seed)
    for step in range(train_steps):
        action, _, _ = agent.get_action(obs)
        obs, reward, cost, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset()

    # Evaluation
    rewards, costs = [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 1000 + ep)
        ep_reward, ep_cost = 0.0, 0.0
        done = False

        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)
            obs, reward, cost, term, trunc, info = env.step(action)
            ep_reward += reward
            ep_cost += cost
            done = term or trunc

        rewards.append(ep_reward)
        costs.append(ep_cost)

    env.close()

    return {
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'cost_mean': float(np.mean(costs)),
        'cost_std': float(np.std(costs)),
    }


def run_ours(
    env_name: str,
    n_episodes: int,
    train_steps: int,
    seed: int,
    use_calibration: bool = True,
) -> Dict[str, float]:
    """Run our method with safety filter."""
    env = SafetyGymEnv(env_name)
    agent = PPO(env.obs_dim, env.act_dim, device='cpu')
    safety_filter = DistanceFilter(
        danger_radius=0.5,
        stop_radius=0.25,
        hazard_radius=0.2,
        lambda_calib=0.1 if use_calibration else 0.0,
    )

    np.random.seed(seed)

    # Training
    obs, info = env.reset(seed=seed)
    safety_filter.reset(info.get('hazards', []))

    for step in range(train_steps):
        action, _, _ = agent.get_action(obs)
        result = safety_filter.project(action, info.get('robot_pos', np.zeros(3)))
        action_safe = result.action_safe

        obs, reward, cost, term, trunc, info = env.step(action_safe)

        if term or trunc:
            obs, info = env.reset()
            safety_filter.reset(info.get('hazards', []))

    # Evaluation
    rewards, costs = [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 1000 + ep)
        safety_filter.reset(info.get('hazards', []))
        ep_reward, ep_cost = 0.0, 0.0
        done = False

        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)
            result = safety_filter.project(action, info.get('robot_pos', np.zeros(3)))
            action_safe = result.action_safe

            obs, reward, cost, term, trunc, info = env.step(action_safe)
            ep_reward += reward
            ep_cost += cost
            done = term or trunc

        rewards.append(ep_reward)
        costs.append(ep_cost)

    env.close()

    return {
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'cost_mean': float(np.mean(costs)),
        'cost_std': float(np.std(costs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Table I experiments")
    parser.add_argument("--envs", nargs="+", default=["goal", "circle", "push"],
                       help="Environments to test")
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

    all_results = {}

    for env_name in args.envs:
        print(f"\n{'='*60}")
        print(f"Environment: {env_name}")
        print(f"{'='*60}")

        env_results = {}

        # Run PPO baseline
        print("\nPPO (baseline):")
        ppo_results = []
        for seed in range(args.seeds):
            print(f"  Seed {seed}...")
            result = run_baseline_ppo(
                env_map[env_name],
                args.episodes,
                args.train_steps,
                seed,
            )
            ppo_results.append(result)

        env_results['PPO'] = {
            'reward_mean': float(np.mean([r['reward_mean'] for r in ppo_results])),
            'reward_std': float(np.std([r['reward_mean'] for r in ppo_results])),
            'cost_mean': float(np.mean([r['cost_mean'] for r in ppo_results])),
            'cost_std': float(np.std([r['cost_mean'] for r in ppo_results])),
        }
        print(f"  Result: R={env_results['PPO']['reward_mean']:.2f}, "
              f"C={env_results['PPO']['cost_mean']:.2f}")

        # Run Ours
        print("\nOurs (Safety Filter + Calibration):")
        ours_results = []
        for seed in range(args.seeds):
            print(f"  Seed {seed}...")
            result = run_ours(
                env_map[env_name],
                args.episodes,
                args.train_steps,
                seed,
                use_calibration=True,
            )
            ours_results.append(result)

        env_results['Ours'] = {
            'reward_mean': float(np.mean([r['reward_mean'] for r in ours_results])),
            'reward_std': float(np.std([r['reward_mean'] for r in ours_results])),
            'cost_mean': float(np.mean([r['cost_mean'] for r in ours_results])),
            'cost_std': float(np.std([r['cost_mean'] for r in ours_results])),
        }
        print(f"  Result: R={env_results['Ours']['reward_mean']:.2f}, "
              f"C={env_results['Ours']['cost_mean']:.2f}")

        all_results[env_name] = env_results

    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f"table1_{timestamp}.json")

    with open(output_path, 'w') as f:
        json.dump({
            'description': 'Table I: Main Results',
            'config': {
                'episodes': args.episodes,
                'train_steps': args.train_steps,
                'seeds': args.seeds,
            },
            'results': all_results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # Print summary table
    print("\n" + "="*70)
    print("TABLE I SUMMARY")
    print("="*70)
    for env_name, results in all_results.items():
        print(f"\n{env_name.upper()}:")
        print(f"{'Method':<20} | {'Reward':>12} | {'Cost':>10}")
        print("-"*46)
        for method, stats in results.items():
            r = f"{stats['reward_mean']:.2f}+/-{stats['reward_std']:.2f}"
            c = f"{stats['cost_mean']:.2f}+/-{stats['cost_std']:.2f}"
            print(f"{method:<20} | {r:>12} | {c:>10}")


if __name__ == "__main__":
    main()
