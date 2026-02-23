"""
Main training script for multi-robot formation navigation with ATACOM safety.

Training loop:
  obs = env.reset()
  for step:
      alphas = mappo.get_actions(obs)           # RL policy output
      safe_actions = atacom.project(alphas, ...) # ATACOM safety projection
      obs, rewards, ... = env.step(safe_actions) # Execute safe actions
      buffer.insert(obs, alphas, ...)            # Store raw actions
  mappo.update()                                  # PPO update
"""

import argparse
import csv
import os
import time
import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from formation_nav.config import Config, EnvConfig, SafetyConfig, AlgoConfig, RewardConfig, TrainConfig
from formation_nav.env.formation_env import FormationNavEnv
from formation_nav.env.formations import FormationTopology
from formation_nav.safety.atacom import AtacomSafetyFilter
from formation_nav.algo.mappo import MAPPO
from formation_nav.algo.buffer import RolloutBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Formation Navigation Training")
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-obstacles", type=int, default=4)
    parser.add_argument("--formation", type=str, default="square",
                        choices=["polygon", "square", "triangle", "hexagon",
                                 "circle", "line", "v"])
    parser.add_argument("--formation-radius", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-episodes", type=int, default=5000)
    parser.add_argument("--rmp-blend", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    return parser.parse_args()


def make_config(args) -> Config:
    cfg = Config()
    cfg.env.num_agents = args.num_agents
    cfg.env.num_obstacles = args.num_obstacles
    cfg.env.formation_shape = args.formation
    cfg.env.formation_radius = args.formation_radius
    cfg.safety.rmp_formation_blend = args.rmp_blend
    cfg.train.total_episodes = args.total_episodes
    cfg.train.seed = args.seed
    cfg.train.device = args.device
    cfg.train.log_dir = args.log_dir
    cfg.train.save_dir = args.save_dir
    cfg.train.log_interval = args.log_interval
    cfg.train.save_interval = args.save_interval
    return cfg


def train(cfg: Config):
    # Seed
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    # Environment
    env = FormationNavEnv(cfg.env, cfg.reward)

    # Topology for formation
    topology = FormationTopology(cfg.env.num_agents, "complete")
    topology_edges = topology.edges()

    # ATACOM safety filter (initialized with dummy obstacles, updated per episode)
    atacom = AtacomSafetyFilter(
        env_cfg=cfg.env,
        safety_cfg=cfg.safety,
        desired_distances=env.desired_distances,
        topology_edges=topology_edges,
        obstacle_positions=env.obstacles,
    )

    # MAPPO
    obs_dim = env.observation_space.shape[0]
    share_obs_dim = env.share_observation_space.shape[0]
    act_dim = 2  # ATACOM null-space dim is always 2
    mappo = MAPPO(obs_dim, share_obs_dim, act_dim, cfg.algo, cfg.train.device)

    # Buffer
    buffer = RolloutBuffer(
        episode_length=cfg.env.max_steps,
        num_agents=cfg.env.num_agents,
        obs_dim=obs_dim,
        share_obs_dim=share_obs_dim,
        act_dim=act_dim,
        gamma=cfg.algo.gamma,
        gae_lambda=cfg.algo.gae_lambda,
        device=cfg.train.device,
    )

    # Logging
    log_path = os.path.join(cfg.train.log_dir,
                            f"formation_{cfg.env.formation_shape}_n{cfg.env.num_agents}_s{cfg.train.seed}")
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path) if HAS_TENSORBOARD else None

    os.makedirs(cfg.train.save_dir, exist_ok=True)

    # CSV log for plotting
    csv_path = os.path.join(log_path, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "cost", "formation_error",
                         "min_inter_dist", "collisions", "reached",
                         "policy_loss", "value_loss", "entropy"])

    # Training loop
    total_steps = 0
    start_time = time.time()

    for episode in range(cfg.train.total_episodes):
        obs_all, share_obs_all, _ = env.reset(seed=cfg.train.seed + episode)

        # Update ATACOM with new obstacle positions
        atacom.update_obstacles(env.obstacles)
        atacom.reset(env.positions)

        buffer.set_first_obs(obs_all, share_obs_all)

        episode_reward = 0.0
        episode_cost = 0.0
        episode_formation_error = 0.0
        episode_min_dist = float('inf')
        episode_collisions = 0

        for step in range(cfg.env.max_steps):
            # Get raw policy actions
            alphas, log_probs = mappo.get_actions(obs_all)
            values = mappo.get_values(share_obs_all)

            # Safety projection
            safe_actions = atacom.project(
                alphas, env.positions, env.velocities)

            # Environment step
            next_obs, next_share_obs, rewards, costs, dones, infos, _ = env.step(safe_actions)

            # Masks (1 = not done)
            masks = (~dones).astype(np.float32).reshape(-1, 1)

            # Store raw alphas (not safe_actions) for consistent log_probs
            buffer.insert(
                obs=next_obs,
                share_obs=next_share_obs,
                actions=alphas,
                log_probs=log_probs,
                values=values,
                rewards=rewards,
                costs=costs,
                masks=masks,
            )

            obs_all = next_obs
            share_obs_all = next_share_obs
            total_steps += cfg.env.num_agents

            # Accumulate metrics
            episode_reward += rewards[0, 0]
            episode_cost += costs[0, 0]
            info = infos[0]
            episode_formation_error += info["formation_error"]
            episode_min_dist = min(episode_min_dist, info["min_inter_dist"])
            episode_collisions += info["collisions"]

            if dones.all():
                break

        # Compute returns and update
        last_values = mappo.get_values(share_obs_all)
        buffer.compute_returns_and_advantages(last_values)
        update_info = mappo.update(buffer)
        buffer.after_update()

        # Logging
        steps_in_episode = step + 1
        avg_formation_error = episode_formation_error / steps_in_episode

        if (episode + 1) % cfg.train.log_interval == 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed if elapsed > 0 else 0

            if writer is not None:
                writer.add_scalar("Reward/episode", episode_reward, episode)
                writer.add_scalar("Cost/episode", episode_cost, episode)
                writer.add_scalar("Formation/error", avg_formation_error, episode)
                writer.add_scalar("Safety/min_inter_dist", episode_min_dist, episode)
                writer.add_scalar("Safety/collisions", episode_collisions, episode)
                writer.add_scalar("Loss/policy", update_info["policy_loss"], episode)
                writer.add_scalar("Loss/value", update_info["value_loss"], episode)
                writer.add_scalar("Loss/entropy", update_info["entropy"], episode)
                writer.add_scalar("Train/fps", fps, episode)

            reached = infos[0].get("reached", False)
            csv_writer.writerow([
                episode + 1, episode_reward, episode_cost,
                avg_formation_error, episode_min_dist, episode_collisions,
                int(reached), update_info["policy_loss"],
                update_info["value_loss"], update_info["entropy"],
            ])
            csv_file.flush()

            print(f"Episode {episode + 1:5d} | "
                  f"R={episode_reward:7.2f} | "
                  f"Cost={episode_cost:4.1f} | "
                  f"FormErr={avg_formation_error:.4f} | "
                  f"MinDist={episode_min_dist:.3f} | "
                  f"Coll={episode_collisions:2d} | "
                  f"Reached={reached} | "
                  f"FPS={fps:.0f}")

        # Save
        if (episode + 1) % cfg.train.save_interval == 0:
            save_path = os.path.join(
                cfg.train.save_dir,
                f"mappo_formation_{cfg.env.formation_shape}_ep{episode + 1}.pt")
            mappo.save(save_path)

    # Final save
    final_path = os.path.join(cfg.train.save_dir, "mappo_formation_final.pt")
    mappo.save(final_path)
    csv_file.close()
    if writer is not None:
        writer.close()
    print(f"Training complete. Model saved to {final_path}")
    print(f"Metrics CSV saved to {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    cfg = make_config(args)
    train(cfg)
