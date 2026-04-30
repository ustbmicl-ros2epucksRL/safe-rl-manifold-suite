#!/usr/bin/env python3
"""
Table III: Ablation study reproduction.

5 configurations on Goal task:
1. PPO (baseline)
2. + Manifold Filter
3. + Reachability Pretraining
4. + EKF (Full, with noise)
5. Full w/o Reward Calibration

Usage:
    python experiments/table3_ablation/run.py
    python experiments/table3_ablation/run.py --train-steps 100000 --seeds 3
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import IROS_DIR, SRC_DIR, ENV_MAP, PAPER_VALUES

sys.path.insert(0, SRC_DIR)

from env import SafetyGymEnv
from safety import DistanceFilter
from safety.distance_filter import AdaptiveDistanceFilter
from safety.reachability import collect_offline_data
from ppo import PPO, PPOConfig, RolloutBuffer
from ekf import DataDrivenEKF, StandardEKF, EKFConfig, NoiseAdapter, train_noise_adapter

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
    """Collect IMU pretraining data."""
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
                noisy_meas = true_pos + velocity_dependent_noise(noise_std, true_vel, noise_vel_scale)
                data.append((imu_window, noisy_meas, true_pos.copy()))
    env.close()
    return data


def pretrain_noise_adapter(env_id, noise_std=0.1, noise_vel_scale=5.0, seed=0):
    """Pretrain NoiseAdapter CNN."""
    print("    Collecting IMU data...", end=" ", flush=True)
    data = collect_imu_data(env_id, noise_std, noise_vel_scale, n_episodes=10, seed=seed)
    print(f"{len(data)} samples")
    adapter = NoiseAdapter(window_length=10)
    print(f"    Training NoiseAdapter...", end=" ", flush=True)
    losses = train_noise_adapter(adapter, data, n_epochs=50, learning_rate=1e-3)
    print(f"loss={losses[-1]:.4f}")
    return adapter


# ---- Ablation configurations ----

CONFIGS = [
    {  # 1. baseline
        "name": "PPO (baseline)",
        "use_safety": False, "use_reachability": False,
        "use_ekf": False, "use_calibration": False, "add_noise": False,
    },
    {  # 2. + safety filter (small margin)
        "name": "+ Manifold Filter",
        "use_safety": True, "use_reachability": False,
        "use_ekf": False, "use_calibration": False, "add_noise": False,
    },
    {  # 3. + reachability margin only (calib still off, isolates radius effect)
        "name": "+ Reachability Pretraining",
        "use_safety": True, "use_reachability": True,
        "use_ekf": False, "use_calibration": False, "add_noise": False,
    },
    {  # 4. + reward calibration only (no noise yet, isolates shaping effect)
        "name": "+ Reward Calibration",
        "use_safety": True, "use_reachability": True,
        "use_ekf": False, "use_calibration": True, "add_noise": False,
    },
    {  # 5. + sensor noise + EKF (full system)
        "name": "+ EKF (Full, with noise)",
        "use_safety": True, "use_reachability": True,
        "use_ekf": True, "use_calibration": True, "add_noise": True,
    },
    {  # 6. ablate calibration from full system
        "name": "Full w/o Reward Calibration",
        "use_safety": True, "use_reachability": True,
        "use_ekf": True, "use_calibration": False, "add_noise": True,
    },
]


def run_single_seed(env_id, cfg, train_steps, eval_episodes, seed,
                    noise_std=0.1, noise_vel_scale=5.0, lambda_calib=0.1):
    """Run single ablation config for one seed."""
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)

    env = SafetyGymEnv(env_id)
    ppo_config = PPOConfig()
    agent = PPO(env.obs_dim, env.act_dim, config=ppo_config, device='cpu')
    buffer = RolloutBuffer(2048, env.obs_dim, env.act_dim)

    # Safety filter.
    # A1: when "reachability" is enabled, use the velocity-adaptive filter
    # (short-horizon BRT proxy) with the SAME base radii as the no-reachability
    # filter.  The margin only expands when the robot is fast --- this avoids
    # the previous problem where a statically larger margin (0.5 -> 0.6)
    # over-saturated PPO action interventions and hurt training.
    safety_filter = None
    if cfg["use_safety"]:
        if cfg["use_reachability"]:
            safety_filter = AdaptiveDistanceFilter(
                base_danger_radius=0.5,
                base_stop_radius=0.25,
                velocity_scaling=0.3,
                hazard_radius=0.2,
                lambda_calib=lambda_calib if cfg["use_calibration"] else 0.0,
            )
        else:
            safety_filter = DistanceFilter(
                danger_radius=0.5,
                stop_radius=0.25,
                hazard_radius=0.2,
                lambda_calib=lambda_calib if cfg["use_calibration"] else 0.0,
            )

    # EKF
    ekf = None
    if cfg["use_ekf"]:
        adapter = pretrain_noise_adapter(env_id, noise_std, noise_vel_scale, seed)
        ekf = DataDrivenEKF(EKFConfig(), noise_adapter=adapter)

    # Training
    obs, info = env.reset(seed=seed)
    if safety_filter:
        safety_filter.reset(info.get('hazards', []))
    if ekf:
        ekf.reset(info.get('robot_pos', np.zeros(3)))

    prev_vel = np.zeros(3)

    for step in range(train_steps):
        action, log_prob, value = agent.get_action(obs)

        robot_pos = info.get('robot_pos', np.zeros(3))
        robot_vel = info.get('robot_vel', np.zeros(3))
        if cfg["add_noise"]:
            robot_pos = robot_pos + velocity_dependent_noise(noise_std, robot_vel, noise_vel_scale)
        if ekf:
            robot_pos = ekf.get_position()

        correction_norm = 0.0
        min_dist_to_hazard = float('inf')
        scale_factor = 1.0
        if safety_filter:
            result = safety_filter.project(action, robot_pos, current_velocity=robot_vel)
            action_safe = result.action_safe
            correction_norm = np.linalg.norm(result.correction)
            min_dist_to_hazard = result.min_distance
            scale_factor = result.scale_factor
        else:
            action_safe = action

        obs_next, reward, cost, term, trunc, info = env.step(action_safe)

        # B1: filter-intervention reward calibration.  Penalty is monotone in
        # filter activity (1 - scale_factor in [0, 1]): zero when the filter
        # passes the action through unchanged, full when the filter forces
        # emergency escape.  This replaces the previous distance-shaped
        # proximity penalty, which created a dual-attractor problem:
        # tangential approaches incurred penalty without filter activity,
        # leaving an inconsistent gradient for PPO.
        if cfg["use_calibration"] and safety_filter is not None:
            intervention = max(0.0, 1.0 - scale_factor)
            reward = reward - lambda_calib * intervention

        if ekf:
            measurement = info.get('robot_pos', np.zeros(3))
            if cfg["add_noise"]:
                true_vel_post = info.get('robot_vel', np.zeros(3))
                measurement = measurement + velocity_dependent_noise(noise_std, true_vel_post, noise_vel_scale)
            true_vel = info.get('robot_vel', np.zeros(3))
            imu_data = synthesize_imu(true_vel, prev_vel)
            prev_vel = true_vel.copy()
            ekf.predict(action_safe)
            ekf.update(measurement, imu_data=imu_data)

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
            prev_vel = np.zeros(3)
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
        ep_r, ep_c = 0.0, 0.0
        done = False
        prev_vel = np.zeros(3)
        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)
            robot_pos = info.get('robot_pos', np.zeros(3))
            robot_vel = info.get('robot_vel', np.zeros(3))
            if cfg["add_noise"]:
                robot_pos = robot_pos + velocity_dependent_noise(noise_std, robot_vel, noise_vel_scale)
            if ekf:
                robot_pos = ekf.get_position()
            if safety_filter:
                result = safety_filter.project(action, robot_pos, current_velocity=robot_vel)
                action = result.action_safe
            obs, reward, cost, term, trunc, info = env.step(action)
            if ekf:
                measurement = info.get('robot_pos', np.zeros(3))
                if cfg["add_noise"]:
                    measurement = measurement + velocity_dependent_noise(
                        noise_std, info.get('robot_vel', np.zeros(3)), noise_vel_scale)
                true_vel = info.get('robot_vel', np.zeros(3))
                imu_data = synthesize_imu(true_vel, prev_vel)
                prev_vel = true_vel.copy()
                ekf.predict(action)
                ekf.update(measurement, imu_data=imu_data)
            ep_r += reward
            ep_c += cost
            done = term or trunc
        rewards.append(ep_r)
        costs.append(ep_c)

    env.close()
    return {"reward": float(np.mean(rewards)), "cost": float(np.mean(costs))}


def compare_with_paper(results):
    paper = PAPER_VALUES["table3"]
    # New 6-row strict-additive structure; some rows have no paper counterpart yet (None)
    paper_keys = [
        "ppo_baseline", "manifold_filter", "reachability",
        None,  # "+ Reward Calibration" — new row, no paper value yet
        "full_ekf", "no_calibration",
    ]
    print("\n" + "=" * 70)
    print("COMPARISON WITH PAPER VALUES")
    print(f"{'Config':<30} {'Metric':<8} {'Paper':>8} {'Ours':>8} {'Diff':>8}")
    print("-" * 70)
    for cfg, pk in zip(CONFIGS, paper_keys):
        name = cfg["name"]
        if name not in results or pk is None:
            continue
        r = results[name]
        p = paper[pk]
        for metric in ["reward", "cost"]:
            pv = p[metric]
            rv = r[f"{metric}_mean"]
            diff = rv - pv
            print(f"{name:<30} {metric:<8} {pv:>8.2f} {rv:>8.2f} {diff:>+8.2f}")


def main():
    parser = argparse.ArgumentParser(description="Table III ablation study")
    parser.add_argument("--train-steps", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--config-index", type=int, default=None,
                        help="Run single config (0-4)")
    parser.add_argument("--lambda-calib", type=float, default=0.1,
                        help="Reward calibration coefficient (default 0.1)")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional tag appended to results filename")
    args = parser.parse_args()

    env_id = ENV_MAP["goal"]
    seeds = list(range(args.seeds))
    configs_to_run = [CONFIGS[args.config_index]] if args.config_index is not None else CONFIGS

    print("=" * 60)
    print("TABLE III: ABLATION STUDY")
    print(f"Task: Goal, Seeds: {seeds}, Steps: {args.train_steps}")
    print("=" * 60)

    all_results = {}
    for i, cfg in enumerate(configs_to_run):
        print(f"\n[{i+1}/{len(configs_to_run)}] {cfg['name']}")
        seed_rewards, seed_costs = [], []
        for seed in seeds:
            print(f"  seed={seed} ...", end=" ", flush=True)
            r = run_single_seed(env_id, cfg, args.train_steps, args.eval_episodes, seed,
                                lambda_calib=args.lambda_calib)
            seed_rewards.append(r["reward"])
            seed_costs.append(r["cost"])
            print(f"R={r['reward']:.2f} C={r['cost']:.2f}")

        all_results[cfg["name"]] = {
            "reward_mean": float(np.mean(seed_rewards)),
            "reward_std": float(np.std(seed_rewards)),
            "cost_mean": float(np.mean(seed_costs)),
            "cost_std": float(np.std(seed_costs)),
            "raw": [{"reward": r, "cost": c} for r, c in zip(seed_rewards, seed_costs)],
        }

    # Save
    output_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    output_path = os.path.join(output_dir, f"results_{timestamp}{suffix}.json")
    with open(output_path, 'w') as f:
        json.dump({
            "config": {"train_steps": args.train_steps, "eval_episodes": args.eval_episodes,
                       "seeds": seeds, "lambda_calib": args.lambda_calib},
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    compare_with_paper(all_results)


if __name__ == "__main__":
    main()
