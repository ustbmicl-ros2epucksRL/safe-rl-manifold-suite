#!/usr/bin/env python3
"""
Clean EKF evaluation for Table V.

Key insight: Train ONE agent (with safety filter, no noise), then deploy
with different EKF configurations under velocity-dependent noise.
This isolates the EKF contribution from training dynamics.

Usage:
    python scripts/run_ekf_eval.py --seeds 5 --train-steps 50000 --episodes 50
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

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
from ppo import PPO, PPOConfig, RolloutBuffer
from ekf import DataDrivenEKF, StandardEKF, EKFConfig, NoiseAdapter, train_noise_adapter


def velocity_dependent_noise(base_std, vel, vel_scale=5.0):
    speed = np.linalg.norm(vel[:2]) if len(vel) >= 2 else 0.0
    sigma = base_std * (1.0 + vel_scale * speed)
    return np.random.randn(3) * sigma


def synthesize_imu(true_vel, prev_vel, dt=0.1):
    accel = (true_vel - prev_vel) / dt
    omega = np.array([0.0, 0.0, true_vel[2] if len(true_vel) > 2 else 0.0])
    imu = np.concatenate([omega, accel])
    imu += np.random.randn(6) * 0.05
    return imu


def collect_imu_data(env_id, noise_std=0.1, noise_vel_scale=5.0,
                     n_episodes=30, window_length=10, seed=0):
    np.random.seed(seed)
    env = SafetyGymEnv(env_id)
    data = []
    dt = 0.1

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        prev_vel = np.zeros(3)
        imu_buffer = []

        while not done:
            action = env.action_space.sample()
            obs, reward, cost, term, trunc, info = env.step(action)
            done = term or trunc

            true_pos = info.get('robot_pos', np.zeros(3))
            true_vel = info.get('robot_vel', np.zeros(3))

            accel = (true_vel - prev_vel) / dt
            omega = np.array([0.0, 0.0, true_vel[2] if len(true_vel) > 2 else 0.0])
            imu_reading = np.concatenate([omega, accel]) + np.random.randn(6) * 0.05
            imu_buffer.append(imu_reading)
            prev_vel = true_vel.copy()

            if len(imu_buffer) >= window_length:
                imu_window = np.array(imu_buffer[-window_length:])
                noisy_meas = true_pos + velocity_dependent_noise(
                    noise_std, true_vel, noise_vel_scale)
                data.append((imu_window, noisy_meas, true_pos.copy()))

    env.close()
    return data


def train_agent(env_id, train_steps, seed):
    """Train a clean PPO agent with safety filter (NO noise)."""
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)

    env = SafetyGymEnv(env_id)
    ppo_config = PPOConfig()
    agent = PPO(env.obs_dim, env.act_dim, config=ppo_config, device='cpu')
    buffer = RolloutBuffer(2048, env.obs_dim, env.act_dim)

    safety_filter = DistanceFilter(
        danger_radius=0.6,
        stop_radius=0.3,
        hazard_radius=0.2,
        lambda_calib=0.1,
    )

    obs, info = env.reset(seed=seed)
    safety_filter.reset(info.get('hazards', []))

    for step in range(train_steps):
        action, log_prob, value = agent.get_action(obs)

        robot_pos = info.get('robot_pos', np.zeros(3))
        result = safety_filter.project(action, robot_pos)
        action_safe = result.action_safe
        correction_norm = np.linalg.norm(result.correction)

        obs_next, reward, cost, term, trunc, info = env.step(action_safe)

        if correction_norm > 0:
            reward = reward - 0.1 * (correction_norm ** 2)

        done = term or trunc
        buffer.add(obs, action, reward, value, log_prob, done)
        obs = obs_next

        if buffer.ptr == 2048:
            _, _, last_value = agent.get_action(obs)
            buffer.finish_path(last_value, ppo_config.gamma, ppo_config.gae_lambda)
            agent.update(buffer)
            buffer.reset()

        if done:
            if buffer.ptr > buffer.path_start_idx:
                buffer.finish_path(0.0, ppo_config.gamma, ppo_config.gae_lambda)
            obs, info = env.reset()
            safety_filter.reset(info.get('hazards', []))

    env.close()
    return agent


def evaluate_with_ekf(agent, env_id, ekf_type, noise_adapter,
                      eval_episodes, seed, noise_std=0.1, noise_vel_scale=5.0):
    """Evaluate a trained agent with a specific EKF configuration under noise."""
    np.random.seed(seed * 10000 + hash(ekf_type) % 1000)

    env = SafetyGymEnv(env_id)
    safety_filter = DistanceFilter(
        danger_radius=0.6,
        stop_radius=0.3,
        hazard_radius=0.2,
    )

    ekf = None
    if ekf_type == "standard":
        ekf = StandardEKF(EKFConfig())
    elif ekf_type == "learned":
        ekf = DataDrivenEKF(EKFConfig(), noise_adapter=noise_adapter)

    rewards, costs, pos_errors = [], [], []

    for ep in range(eval_episodes):
        obs, info = env.reset(seed=seed * 1000 + ep)
        safety_filter.reset(info.get('hazards', []))
        if ekf:
            ekf.reset(info.get('robot_pos', np.zeros(3)))

        ep_reward, ep_cost = 0.0, 0.0
        ep_pos_errors = []
        done = False
        prev_vel = np.zeros(3)

        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)

            true_pos = info.get('robot_pos', np.zeros(3))
            true_vel = info.get('robot_vel', np.zeros(3))
            noisy_pos = true_pos + velocity_dependent_noise(
                noise_std, true_vel, noise_vel_scale)

            if ekf:
                robot_pos = ekf.get_position()
            else:
                robot_pos = noisy_pos

            pos_error = np.linalg.norm(robot_pos[:2] - true_pos[:2])
            ep_pos_errors.append(pos_error)

            result = safety_filter.project(action, robot_pos)
            action_eval = result.action_safe

            obs, reward, cost, term, trunc, info = env.step(action_eval)

            if ekf:
                true_vel_post = info.get('robot_vel', np.zeros(3))
                measurement = info.get('robot_pos', np.zeros(3)) + velocity_dependent_noise(
                    noise_std, true_vel_post, noise_vel_scale)
                true_vel = info.get('robot_vel', np.zeros(3))
                imu_data = synthesize_imu(true_vel, prev_vel)
                prev_vel = true_vel.copy()

                ekf.predict(action_eval)
                ekf.update(measurement, imu_data=imu_data)

            ep_reward += reward
            ep_cost += cost
            done = term or trunc

        rewards.append(ep_reward)
        costs.append(ep_cost)
        pos_errors.append(np.mean(ep_pos_errors))

    env.close()
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "pos_error_mean": float(np.mean(pos_errors)),
        "pos_error_std": float(np.std(pos_errors)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="goal")
    parser.add_argument("--train-steps", type=int, default=50000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--noise-vel-scale", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="results_ekf_v4")
    args = parser.parse_args()

    env_map = {"goal": "SafetyPointGoal1-v0"}
    env_id = env_map[args.env]

    all_results = {"No filtering": [], "Standard EKF (fixed R)": [], "Data-driven EKF (learned R)": []}
    metrics = ["pos_error_mean", "reward_mean", "cost_mean"]

    for seed in range(args.seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        # Step 1: Train agent (clean, no noise)
        print(f"  Training PPO agent ({args.train_steps} steps)...")
        agent = train_agent(env_id, args.train_steps, seed)

        # Step 2: Pretrain NoiseAdapter for Data-driven EKF
        print(f"  Pretraining NoiseAdapter...")
        imu_data = collect_imu_data(
            env_id, noise_std=args.noise_std,
            noise_vel_scale=args.noise_vel_scale,
            n_episodes=30, seed=seed)
        adapter = NoiseAdapter(window_length=10)
        losses = train_noise_adapter(adapter, imu_data, n_epochs=100, learning_rate=1e-3)
        print(f"    {len(imu_data)} samples, final loss={losses[-1]:.4f}")

        # Step 3: Evaluate same agent under 3 EKF configs
        for ekf_type, label in [("none", "No filtering"),
                                ("standard", "Standard EKF (fixed R)"),
                                ("learned", "Data-driven EKF (learned R)")]:
            print(f"  Evaluating [{label}]...", end=" ", flush=True)
            result = evaluate_with_ekf(
                agent, env_id, ekf_type, adapter,
                args.episodes, seed,
                args.noise_std, args.noise_vel_scale)
            all_results[label].append(result)
            print(f"Err={result['pos_error_mean']:.3f}, "
                  f"R={result['reward_mean']:.2f}, "
                  f"C={result['cost_mean']:.2f}")

    # Aggregate results
    final = {}
    for label, runs in all_results.items():
        final[label] = {
            "pos_error_mean": float(np.mean([r["pos_error_mean"] for r in runs])),
            "pos_error_std": float(np.std([r["pos_error_mean"] for r in runs])),
            "reward_mean": float(np.mean([r["reward_mean"] for r in runs])),
            "reward_std": float(np.std([r["reward_mean"] for r in runs])),
            "cost_mean": float(np.mean([r["cost_mean"] for r in runs])),
            "cost_std": float(np.std([r["cost_mean"] for r in runs])),
        }

    # Print table
    print(f"\n{'='*80}")
    print("TABLE V: EKF COMPARISON (same agent, different deployment EKF)")
    print(f"{'='*80}")
    print(f"{'Method':<30} | {'Pos. Error (m)':>15} | {'Reward':>10} | {'Cost':>8}")
    print("-"*80)
    for label, stats in final.items():
        e = f"{stats['pos_error_mean']:.3f} ± {stats['pos_error_std']:.3f}"
        r = f"{stats['reward_mean']:.2f}"
        c = f"{stats['cost_mean']:.2f}"
        print(f"{label:<30} | {e:>15} | {r:>10} | {c:>8}")
    print(f"{'='*80}")

    # Save
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(args.output, f"table5_ekf_clean_{timestamp}.json")
    with open(filepath, 'w') as f:
        json.dump({"experiment": "table5_ekf_clean", "timestamp": timestamp, "results": final}, f, indent=2)
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
