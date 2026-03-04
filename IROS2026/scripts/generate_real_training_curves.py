#!/usr/bin/env python3
"""
Generate training_curves.png from REAL ablation experiment data.

Runs 4 ablation configs (PPO, +Filter, +Reach, +Full) with per-episode
reward/cost logging, then plots actual learning curves.

Usage:
    cd IROS2026
    python scripts/generate_real_training_curves.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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
from safety.reachability import collect_offline_data
from ppo import PPO, PPOConfig, RolloutBuffer
from ekf import DataDrivenEKF, StandardEKF, EKFConfig, NoiseAdapter, train_noise_adapter

FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results_curves')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def _velocity_dependent_noise(base_std, vel, vel_scale=5.0):
    speed = np.linalg.norm(vel[:2]) if len(vel) >= 2 else 0.0
    sigma = base_std * (1.0 + vel_scale * speed)
    return np.random.randn(3) * sigma


def _synthesize_imu(true_vel, prev_vel, dt=0.1):
    accel = (true_vel - prev_vel) / dt
    omega = np.array([0.0, 0.0, true_vel[2] if len(true_vel) > 2 else 0.0])
    imu = np.concatenate([omega, accel])
    imu += np.random.randn(6) * 0.05
    return imu


def collect_imu_pretraining_data(env_id, noise_std=0.1, noise_vel_scale=5.0,
                                  n_episodes=10, window_length=10, seed=0):
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
                noisy_meas = true_pos + _velocity_dependent_noise(noise_std, true_vel, noise_vel_scale)
                data.append((imu_window, noisy_meas, true_pos.copy()))
    env.close()
    return data


def run_config_with_logging(config_name, env_id, train_steps, seed,
                             use_safety, use_reachability, use_ekf,
                             use_calibration, add_noise,
                             noise_std=0.1, noise_vel_scale=5.0):
    """Run one config and return per-episode reward/cost lists."""
    print(f"  [{config_name}] seed={seed}, steps={train_steps}")
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)

    env = SafetyGymEnv(env_id)
    ppo_config = PPOConfig()
    agent = PPO(env.obs_dim, env.act_dim, config=ppo_config, device='cpu')

    buffer_size = 2048
    buffer = RolloutBuffer(buffer_size, env.obs_dim, env.act_dim)

    # Safety filter
    safety_filter = None
    if use_safety:
        danger_r = 0.6 if use_reachability else 0.5
        safety_filter = DistanceFilter(
            danger_radius=danger_r,
            stop_radius=0.3 if use_reachability else 0.25,
            hazard_radius=0.2,
            lambda_calib=0.1 if use_calibration else 0.0,
        )

    # EKF
    ekf = None
    if use_ekf:
        print(f"    Pretraining NoiseAdapter...", end=" ", flush=True)
        imu_data_list = collect_imu_pretraining_data(
            env_id, noise_std=noise_std, noise_vel_scale=noise_vel_scale,
            n_episodes=10, seed=seed)
        adapter = NoiseAdapter(window_length=10)
        losses = train_noise_adapter(adapter, imu_data_list, n_epochs=50, learning_rate=1e-3)
        print(f"done (loss={losses[-1]:.4f})")
        ekf = DataDrivenEKF(EKFConfig(), noise_adapter=adapter)

    # Training with per-episode logging
    episode_rewards = []
    episode_costs = []
    ep_reward = 0.0
    ep_cost = 0.0

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
        if add_noise:
            robot_pos = robot_pos + _velocity_dependent_noise(noise_std, robot_vel, noise_vel_scale)
        if ekf:
            robot_pos = ekf.get_position()

        correction_norm = 0.0
        if safety_filter:
            result = safety_filter.project(action, robot_pos)
            action_safe = result.action_safe
            correction_norm = np.linalg.norm(result.correction)
        else:
            action_safe = action

        obs_next, reward, cost, term, trunc, info = env.step(action_safe)

        if use_calibration and correction_norm > 0:
            reward = reward - 0.1 * (correction_norm ** 2)

        if ekf:
            measurement = info.get('robot_pos', np.zeros(3))
            if add_noise:
                true_vel_post = info.get('robot_vel', np.zeros(3))
                measurement = measurement + _velocity_dependent_noise(noise_std, true_vel_post, noise_vel_scale)
            true_vel = info.get('robot_vel', np.zeros(3))
            imu_data = _synthesize_imu(true_vel, prev_vel)
            prev_vel = true_vel.copy()
            ekf.predict(action_safe)
            ekf.update(measurement, imu_data=imu_data)

        ep_reward += reward
        ep_cost += cost

        done = term or trunc
        buffer.add(obs, action, reward, value, log_prob, done)
        obs = obs_next

        if buffer.ptr == buffer_size:
            _, _, last_value = agent.get_action(obs)
            buffer.finish_path(last_value, ppo_config.gamma, ppo_config.gae_lambda)
            agent.update(buffer)
            buffer.reset()

        if done:
            episode_rewards.append(ep_reward)
            episode_costs.append(ep_cost)
            ep_reward = 0.0
            ep_cost = 0.0

            if buffer.ptr > buffer.path_start_idx:
                buffer.finish_path(0.0, ppo_config.gamma, ppo_config.gae_lambda)

            obs, info = env.reset()
            prev_vel = np.zeros(3)
            if safety_filter:
                safety_filter.reset(info.get('hazards', []))
            if ekf:
                ekf.reset(info.get('robot_pos', np.zeros(3)))

        if (step + 1) % 5000 == 0:
            n_ep = len(episode_rewards)
            avg_r = np.mean(episode_rewards[-10:]) if n_ep > 0 else 0
            avg_c = np.mean(episode_costs[-10:]) if n_ep > 0 else 0
            print(f"    step {step+1}/{train_steps}, episodes={n_ep}, "
                  f"avg_r={avg_r:.2f}, avg_c={avg_c:.2f}")

    env.close()
    print(f"  [{config_name}] Done: {len(episode_rewards)} episodes")
    return episode_rewards, episode_costs


def smooth(y, window=15):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')


def plot_training_curves(all_data, output_path):
    """Plot training curves from real data."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    colors = {'PPO (baseline)': '#e74c3c',
              '+ Manifold Filter': '#3498db',
              '+ Reachability': '#2ecc71',
              'Full (+ EKF)': '#9b59b6'}

    for name, (rewards, costs) in all_data.items():
        color = colors.get(name, 'black')

        # Reward
        sr = smooth(rewards, window=15)
        episodes_r = np.arange(len(sr))
        axes[0].plot(episodes_r, sr, color=color, label=name, linewidth=1.8, alpha=0.85)

        # Cost
        sc = smooth(costs, window=15)
        episodes_c = np.arange(len(sc))
        axes[1].plot(episodes_c, sc, color=color, label=name, linewidth=1.8, alpha=0.85)

    axes[0].set_xlabel('Episode', fontsize=13)
    axes[0].set_ylabel('Episode Reward', fontsize=13)
    axes[0].set_title('(a) Reward', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=9.5, framealpha=0.9)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=11)

    axes[1].set_xlabel('Episode', fontsize=13)
    axes[1].set_ylabel('Episode Cost', fontsize=13)
    axes[1].set_title('(b) Cost (Safety Violations)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9.5, framealpha=0.9)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[1].tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved training curves to: {output_path}")
    plt.close()


def main():
    env_id = "SafetyPointGoal1-v0"
    train_steps = 100000
    seed = 42

    print("=" * 70)
    print("GENERATING REAL TRAINING CURVES")
    print(f"Env: {env_id}, Steps: {train_steps}, Seed: {seed}")
    print("=" * 70)

    all_data = {}

    # Config 1: PPO baseline
    print("\n[1/4] PPO (baseline)")
    r, c = run_config_with_logging(
        "PPO (baseline)", env_id, train_steps, seed,
        use_safety=False, use_reachability=False,
        use_ekf=False, use_calibration=False, add_noise=False)
    all_data["PPO (baseline)"] = (r, c)

    # Config 2: + Manifold Filter
    print("\n[2/4] + Manifold Filter")
    r, c = run_config_with_logging(
        "+ Manifold Filter", env_id, train_steps, seed,
        use_safety=True, use_reachability=False,
        use_ekf=False, use_calibration=True, add_noise=False)
    all_data["+ Manifold Filter"] = (r, c)

    # Config 3: + Reachability
    print("\n[3/4] + Reachability")
    r, c = run_config_with_logging(
        "+ Reachability", env_id, train_steps, seed,
        use_safety=True, use_reachability=True,
        use_ekf=False, use_calibration=True, add_noise=False)
    all_data["+ Reachability"] = (r, c)

    # Config 4: Full (+ EKF with noise)
    print("\n[4/4] Full (+ EKF)")
    r, c = run_config_with_logging(
        "Full (+ EKF)", env_id, train_steps, seed,
        use_safety=True, use_reachability=True,
        use_ekf=True, use_calibration=True, add_noise=True)
    all_data["Full (+ EKF)"] = (r, c)

    # Save raw data
    save_data = {}
    for name, (rewards, costs) in all_data.items():
        save_data[name] = {"rewards": rewards, "costs": costs}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = os.path.join(RESULTS_DIR, f"training_curves_{timestamp}.json")
    with open(data_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved raw data to: {data_path}")

    # Plot
    plot_training_curves(all_data, os.path.join(FIGURES_DIR, 'training_curves.png'))

    print("\n" + "=" * 70)
    print("DONE - training_curves.png generated from real experiment data")
    print("=" * 70)


if __name__ == "__main__":
    main()
