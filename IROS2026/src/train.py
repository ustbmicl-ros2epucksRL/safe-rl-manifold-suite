#!/usr/bin/env python3
"""
Training script for Safe RL with Constraint Manifold Projection.

Usage:
    python -m src.train --env goal --episodes 500 --use-safety
    python -m src.train --env circle --episodes 500 --use-safety --use-ekf
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np

# Ensure src is in path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

from env import SafetyGymEnv, MockEnv
from safety import DistanceFilter, ManifoldFilter
from ppo import PPO, PPOConfig, RolloutBuffer


# Environment mapping
ENV_MAP = {
    "goal": "SafetyPointGoal1-v0",
    "circle": "SafetyPointCircle1-v0",
    "push": "SafetyPointPush1-v0",
}


def train(
    env_name: str = "goal",
    n_episodes: int = 500,
    steps_per_episode: int = 1000,
    use_safety: bool = True,
    use_calibration: bool = True,
    use_ekf: bool = False,
    seed: int = 0,
    save_dir: str = "results",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Train PPO agent with optional safety filter.

    Args:
        env_name: Environment name (goal, circle, push)
        n_episodes: Number of training episodes
        steps_per_episode: Maximum steps per episode
        use_safety: Whether to use safety filter
        use_calibration: Whether to calibrate reward
        use_ekf: Whether to use data-driven EKF
        seed: Random seed
        save_dir: Directory to save results
        device: PyTorch device

    Returns:
        Training results dictionary
    """
    # Set seed
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)

    # Create environment
    env_id = ENV_MAP.get(env_name, env_name)
    env = SafetyGymEnv(env_id)

    print(f"Environment: {env_id}")
    print(f"Observation dim: {env.obs_dim}")
    print(f"Action dim: {env.act_dim}")

    # Create agent
    ppo_config = PPOConfig(
        hidden_dim=256,
        n_layers=2,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
    )
    agent = PPO(env.obs_dim, env.act_dim, ppo_config, device)

    # Create safety filter
    safety_filter: Optional[DistanceFilter] = None
    if use_safety:
        safety_filter = DistanceFilter(
            danger_radius=0.5,
            stop_radius=0.25,
            hazard_radius=0.2,
            lambda_calib=0.1 if use_calibration else 0.0,
        )

    # Create EKF if requested
    ekf = None
    if use_ekf:
        from ekf import DataDrivenEKF, EKFConfig
        ekf_config = EKFConfig()
        ekf = DataDrivenEKF(ekf_config, device=device)

    # Training loop
    buffer_size = 2048
    buffer = RolloutBuffer(buffer_size, env.obs_dim, env.act_dim)

    history = {
        'episode_rewards': [],
        'episode_costs': [],
        'episode_lengths': [],
        'policy_losses': [],
        'value_losses': [],
    }

    total_steps = 0
    update_count = 0

    print(f"\nTraining with:")
    print(f"  Safety filter: {use_safety}")
    print(f"  Reward calibration: {use_calibration}")
    print(f"  Data-driven EKF: {use_ekf}")
    print(f"  Episodes: {n_episodes}")
    print()

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed * 10000 + episode)

        # Initialize safety filter with hazard positions
        if safety_filter is not None:
            safety_filter.reset(info.get('hazards', []))

        # Initialize EKF
        if ekf is not None:
            ekf.reset(info.get('robot_pos', np.zeros(3)))

        episode_reward = 0.0
        episode_cost = 0.0
        episode_length = 0

        for step in range(steps_per_episode):
            # Get action from policy
            action, log_prob, value = agent.get_action(obs)

            # Apply safety filter
            correction = 0.0
            if safety_filter is not None:
                robot_pos = info.get('robot_pos', np.zeros(3))

                # Use EKF estimate if available
                if ekf is not None:
                    robot_pos = ekf.get_position()

                result = safety_filter.project(action, robot_pos)
                action_safe = result.action_safe
                correction = np.linalg.norm(result.correction)
            else:
                action_safe = action

            # Step environment
            next_obs, reward, cost, terminated, truncated, next_info = env.step(action_safe)

            # Update EKF
            if ekf is not None:
                measurement = next_info.get('robot_pos', np.zeros(3))
                ekf.predict(action_safe)
                ekf.update(measurement)

            # Calibrate reward
            if use_calibration and correction > 0:
                reward = reward - 0.1 * (correction ** 2)

            # Store transition
            done = terminated or truncated
            buffer.add(obs, action, reward, value, log_prob, done)

            episode_reward += reward
            episode_cost += cost
            episode_length += 1
            total_steps += 1

            obs = next_obs
            info = next_info

            # Update policy when buffer is full
            if buffer.ptr == buffer_size:
                # Get last value for GAE
                _, _, last_value = agent.get_action(obs)
                buffer.finish_path(last_value, ppo_config.gamma, ppo_config.gae_lambda)

                # Update
                losses = agent.update(buffer)
                history['policy_losses'].append(losses['policy_loss'])
                history['value_losses'].append(losses['value_loss'])

                buffer.reset()
                update_count += 1

            if done:
                break

        # Finish partial episode
        if buffer.ptr > buffer.path_start_idx:
            buffer.finish_path(0.0, ppo_config.gamma, ppo_config.gae_lambda)

        # Record episode stats
        history['episode_rewards'].append(episode_reward)
        history['episode_costs'].append(episode_cost)
        history['episode_lengths'].append(episode_length)

        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(history['episode_rewards'][-10:])
            avg_cost = np.mean(history['episode_costs'][-10:])
            avg_length = np.mean(history['episode_lengths'][-10:])

            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Cost: {avg_cost:5.2f} | "
                  f"Length: {avg_length:6.1f} | "
                  f"Updates: {update_count}")

    # Final update with remaining data
    if buffer.ptr > 0:
        buffer.finish_path(0.0, ppo_config.gamma, ppo_config.gae_lambda)
        # Pad buffer if needed (simplified)

    env.close()

    # Compute summary statistics
    results = {
        'env': env_name,
        'env_id': env_id,
        'n_episodes': n_episodes,
        'total_steps': total_steps,
        'use_safety': use_safety,
        'use_calibration': use_calibration,
        'use_ekf': use_ekf,
        'seed': seed,
        'reward_mean': float(np.mean(history['episode_rewards'])),
        'reward_std': float(np.std(history['episode_rewards'])),
        'cost_mean': float(np.mean(history['episode_costs'])),
        'cost_std': float(np.std(history['episode_costs'])),
        'final_reward': float(np.mean(history['episode_rewards'][-50:])),
        'final_cost': float(np.mean(history['episode_costs'][-50:])),
        'history': history,
    }

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config_str = f"{'safe' if use_safety else 'unsafe'}"
    if use_calibration:
        config_str += "_calib"
    if use_ekf:
        config_str += "_ekf"

    result_path = os.path.join(save_dir, f"{env_name}_{config_str}_{timestamp}.json")

    # Save without full history to reduce file size
    save_results = {k: v for k, v in results.items() if k != 'history'}
    save_results['history_summary'] = {
        'n_episodes': len(history['episode_rewards']),
        'n_updates': len(history['policy_losses']),
    }

    with open(result_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {result_path}")
    print(f"\nFinal Results:")
    print(f"  Reward: {results['final_reward']:.2f}")
    print(f"  Cost:   {results['final_cost']:.2f}")

    # Save model
    if TORCH_AVAILABLE:
        model_path = os.path.join(save_dir, f"{env_name}_{config_str}_{timestamp}.pt")
        agent.save(model_path)
        print(f"  Model:  {model_path}")

    return results


def evaluate(
    env_name: str,
    model_path: str,
    n_episodes: int = 50,
    use_safety: bool = True,
    seed: int = 0,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate trained model.

    Returns:
        Evaluation metrics
    """
    # Create environment
    env_id = ENV_MAP.get(env_name, env_name)
    env = SafetyGymEnv(env_id)

    # Load agent
    agent = PPO(env.obs_dim, env.act_dim, device=device)
    agent.load(model_path)

    # Create safety filter
    safety_filter = None
    if use_safety:
        safety_filter = DistanceFilter(
            danger_radius=0.5,
            stop_radius=0.25,
            hazard_radius=0.2,
        )

    rewards = []
    costs = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 10000 + ep)

        if safety_filter is not None:
            safety_filter.reset(info.get('hazards', []))

        episode_reward = 0.0
        episode_cost = 0.0
        done = False

        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)

            if safety_filter is not None:
                robot_pos = info.get('robot_pos', np.zeros(3))
                result = safety_filter.project(action, robot_pos)
                action = result.action_safe

            obs, reward, cost, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_cost += cost
            done = terminated or truncated

        rewards.append(episode_reward)
        costs.append(episode_cost)

    env.close()

    return {
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'cost_mean': float(np.mean(costs)),
        'cost_std': float(np.std(costs)),
        'zero_cost_rate': float(sum(1 for c in costs if c == 0) / len(costs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Safe RL agent")
    parser.add_argument("--env", type=str, default="goal",
                       choices=["goal", "circle", "push"],
                       help="Environment name")
    parser.add_argument("--episodes", type=int, default=500,
                       help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=1000,
                       help="Max steps per episode")
    parser.add_argument("--use-safety", action="store_true",
                       help="Use safety filter")
    parser.add_argument("--use-calibration", action="store_true",
                       help="Use reward calibration")
    parser.add_argument("--use-ekf", action="store_true",
                       help="Use data-driven EKF")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--save-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="cpu",
                       help="PyTorch device")

    args = parser.parse_args()

    train(
        env_name=args.env,
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        use_safety=args.use_safety,
        use_calibration=args.use_calibration,
        use_ekf=args.use_ekf,
        seed=args.seed,
        save_dir=args.save_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
