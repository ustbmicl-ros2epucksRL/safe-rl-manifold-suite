#!/usr/bin/env python3
"""
Quick visualization demo for formation navigation.

Shows:
1. Pure RMPflow baseline (geometric control, no learning)
2. COSMOS + MAPPO trained model (if available)

Usage:
    python formation_nav/demo_visualization.py
    python formation_nav/demo_visualization.py --mode rmp     # RMPflow only
    python formation_nav/demo_visualization.py --mode mappo   # Trained model only
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from pathlib import Path

from cosmos.apps.formation_nav.config import Config
from cosmos.envs.formation_nav import FormationNavEnv
from cosmos.envs.formations import FormationTopology
from cosmos.safety.rmp_tree import RMPRoot, RMPNode
from cosmos.safety.rmp_policies import (
    GoalAttractorUni, CollisionAvoidance, FormationDecentralized, Damper, MultiRobotRMPForest
)


def compute_rmp_action(pos_i, vel_i, goal, all_positions, obstacles, desired_distances, agent_id):
    """Compute RMPflow-style action for a single agent using geometric policies."""
    num_agents = len(all_positions)

    # Goal attraction (toward formation centroid goal)
    centroid = all_positions.mean(axis=0)
    goal_dir = goal - centroid
    goal_dist = np.linalg.norm(goal_dir)
    if goal_dist > 0.1:
        goal_force = goal_dir / goal_dist * min(2.0, goal_dist * 0.5)
    else:
        goal_force = np.zeros(2)

    # Formation maintenance (spring-damper to neighbors)
    form_force = np.zeros(2)
    for j in range(num_agents):
        if agent_id != j:
            rel = all_positions[j] - pos_i
            dist = np.linalg.norm(rel)
            desired = desired_distances[agent_id, j]
            if dist > 0.01:
                error = dist - desired
                form_force += error * rel / dist * 0.8

    # Collision avoidance with other agents
    avoid_force = np.zeros(2)
    for j in range(num_agents):
        if agent_id != j:
            rel = pos_i - all_positions[j]
            dist = np.linalg.norm(rel)
            if dist < 1.0 and dist > 0.01:
                repel = (1.0 - dist) / 1.0
                avoid_force += rel / dist * repel * 3.0

    # Obstacle avoidance
    obs_force = np.zeros(2)
    for obs in obstacles:
        rel = pos_i - obs[:2]
        dist = np.linalg.norm(rel)
        safe_dist = obs[2] + 0.6
        if dist < safe_dist and dist > 0.01:
            repel = (safe_dist - dist) / safe_dist
            obs_force += rel / dist * repel * 4.0

    # Damping
    damping = -0.8 * vel_i

    return goal_force + form_force + avoid_force + obs_force + damping


def run_rmp_episode(env, cfg, seed=42):
    """Run one episode with pure RMPflow-style geometric control."""
    obs, share_obs, _ = env.reset(seed=seed)

    trajectory = [env.positions.copy()]

    for step in range(cfg.env.max_steps):
        actions = np.zeros((cfg.env.num_agents, 2))

        for i in range(cfg.env.num_agents):
            actions[i] = compute_rmp_action(
                env.positions[i],
                env.velocities[i],
                env.goal,
                env.positions,
                env.obstacles,
                env.desired_distances,
                i
            )

        # Clip actions
        actions = np.clip(actions, -cfg.env.max_acceleration, cfg.env.max_acceleration)

        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)
        trajectory.append(env.positions.copy())

        if dones.all():
            break

    return np.array(trajectory), env.obstacles, env.goal, infos[0]


def run_mappo_episode(env, cfg, model_path, seed=42):
    """Run one episode with trained MAPPO model."""
    import torch
    from formation_nav.algo.mappo import MAPPO
    from formation_nav.safety import COSMOS, COSMOSMode

    obs, share_obs, _ = env.reset(seed=seed)
    topology = FormationTopology(cfg.env.num_agents, "complete")

    # Load model
    obs_dim = env.observation_space.shape[0]
    share_obs_dim = env.share_observation_space.shape[0]
    mappo = MAPPO(obs_dim, share_obs_dim, act_dim=2, cfg=cfg.algo, device="cpu")

    checkpoint = torch.load(model_path, map_location="cpu")
    mappo.actor.load_state_dict(checkpoint['actor'])
    mappo.critic.load_state_dict(checkpoint['critic'])

    # COSMOS safety filter
    cosmos = COSMOS(
        env_cfg=cfg.env,
        safety_cfg=cfg.safety,
        desired_distances=env.desired_distances,
        topology_edges=topology.edges(),
        obstacle_positions=env.obstacles,
        mode=COSMOSMode.DECENTRALIZED
    )
    cosmos.update_obstacles(env.obstacles)
    cosmos.reset(env.positions)

    trajectory = [env.positions.copy()]

    for step in range(cfg.env.max_steps):
        alphas, _ = mappo.get_actions(obs, deterministic=True)
        safe_actions = cosmos.project(alphas, env.positions, env.velocities, dt=cfg.env.dt)
        obs, share_obs, rewards, costs, dones, infos, _ = env.step(safe_actions)
        trajectory.append(env.positions.copy())

        if dones.all():
            break

    return np.array(trajectory), env.obstacles, env.goal, infos[0]


def plot_comparison(traj_rmp, traj_mappo, obstacles, goal, cfg, save_path=None):
    """Plot side-by-side comparison of RMPflow vs MAPPO."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, cfg.env.num_agents))
    topology = FormationTopology(cfg.env.num_agents, "complete")

    for ax, traj, title in [(axes[0], traj_rmp, "RMPflow Baseline (Geometric)"),
                            (axes[1], traj_mappo, "COSMOS + MAPPO (Learned)")]:
        # Arena boundary
        rect = patches.Rectangle(
            (-cfg.env.arena_size, -cfg.env.arena_size),
            2 * cfg.env.arena_size, 2 * cfg.env.arena_size,
            linewidth=2, edgecolor='black', facecolor='white', linestyle='--')
        ax.add_patch(rect)

        # Obstacles
        for obs in obstacles:
            circle = patches.Circle(
                (obs[0], obs[1]), obs[2],
                facecolor='gray', edgecolor='darkgray', alpha=0.6)
            ax.add_patch(circle)

        # Goal
        ax.plot(goal[0], goal[1], 'r*', markersize=25, label='Goal', zorder=10)

        # Trajectories
        for i in range(cfg.env.num_agents):
            traj_i = traj[:, i, :]
            ax.plot(traj_i[:, 0], traj_i[:, 1], '-', color=colors[i],
                   alpha=0.7, linewidth=2, label=f'Agent {i}')
            ax.plot(traj_i[0, 0], traj_i[0, 1], 'o', color=colors[i], markersize=12)
            ax.plot(traj_i[-1, 0], traj_i[-1, 1], 's', color=colors[i], markersize=12)

        # Final formation edges
        final_pos = traj[-1]
        for (i, j) in topology.edges():
            ax.plot([final_pos[i, 0], final_pos[j, 0]],
                   [final_pos[i, 1], final_pos[j, 1]],
                   'k--', alpha=0.4, linewidth=1.5)

        ax.set_xlim(-cfg.env.arena_size * 1.1, cfg.env.arena_size * 1.1)
        ax.set_ylim(-cfg.env.arena_size * 1.1, cfg.env.arena_size * 1.1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)


def create_animation(traj, obstacles, goal, cfg, title="Formation Navigation", save_path=None):
    """Create animation of formation navigation."""
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, cfg.env.num_agents))
    topology = FormationTopology(cfg.env.num_agents, "complete")
    T = len(traj)

    def init():
        ax.clear()
        rect = patches.Rectangle(
            (-cfg.env.arena_size, -cfg.env.arena_size),
            2 * cfg.env.arena_size, 2 * cfg.env.arena_size,
            linewidth=2, edgecolor='black', facecolor='white', linestyle='--')
        ax.add_patch(rect)
        for obs in obstacles:
            circle = patches.Circle(
                (obs[0], obs[1]), obs[2],
                facecolor='gray', edgecolor='darkgray', alpha=0.6)
            ax.add_patch(circle)
        ax.plot(goal[0], goal[1], 'r*', markersize=25)
        ax.set_xlim(-cfg.env.arena_size * 1.1, cfg.env.arena_size * 1.1)
        ax.set_ylim(-cfg.env.arena_size * 1.1, cfg.env.arena_size * 1.1)
        ax.set_aspect('equal')
        return []

    def update(frame):
        ax.clear()
        init()
        pos = traj[frame]

        # Trails
        trail_start = max(0, frame - 50)
        for i in range(cfg.env.num_agents):
            trail = traj[trail_start:frame + 1, i, :]
            ax.plot(trail[:, 0], trail[:, 1], '-', color=colors[i], alpha=0.5, linewidth=2)
            ax.plot(pos[i, 0], pos[i, 1], 'o', color=colors[i], markersize=14)

        # Formation edges
        for (i, j) in topology.edges():
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                   'k-', alpha=0.4, linewidth=1.5)

        ax.set_title(f"{title} - Step {frame}/{T-1}", fontsize=14)
        return []

    frames = list(range(0, T, max(1, T // 150)))
    anim = FuncAnimation(fig, update, init_func=init, frames=frames, blit=False, interval=50)

    if save_path:
        try:
            anim.save(save_path, writer='pillow', fps=20)
            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Could not save animation: {e}")

    plt.close(fig)
    return anim


def main():
    parser = argparse.ArgumentParser(description="Formation Navigation Visualization Demo")
    parser.add_argument("--mode", choices=["both", "rmp", "mappo"], default="both",
                       help="Which mode to visualize")
    parser.add_argument("--model-path", type=str, default="artifacts/demo_output/cosmos_mappo_model.pt",
                       help="Path to trained MAPPO model")
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="artifacts/demo_output")
    args = parser.parse_args()

    cfg = Config()
    cfg.env.num_agents = args.num_agents
    cfg.env.num_obstacles = 4
    cfg.env.arena_size = 10.0
    cfg.env.formation_radius = 1.0

    env = FormationNavEnv(cfg.env, cfg.reward)

    print("=" * 60)
    print("Formation Navigation Visualization Demo")
    print("=" * 60)
    print(f"Agents:     {cfg.env.num_agents}")
    print(f"Obstacles:  {cfg.env.num_obstacles}")
    print(f"Arena:      {cfg.env.arena_size} x {cfg.env.arena_size}")
    print(f"Mode:       {args.mode}")
    print("=" * 60)

    # Run RMPflow baseline
    traj_rmp = None
    if args.mode in ["both", "rmp"]:
        print("\nRunning RMPflow baseline...")
        traj_rmp, obstacles, goal, info_rmp = run_rmp_episode(env, cfg, seed=args.seed)
        print(f"  Steps: {len(traj_rmp)}")
        print(f"  Formation Error: {info_rmp['formation_error']:.4f}")
        print(f"  Reached Goal: {info_rmp['reached']}")

    # Run MAPPO
    traj_mappo = None
    model_path = Path(args.model_path)
    if args.mode in ["both", "mappo"] and model_path.exists():
        print(f"\nRunning COSMOS + MAPPO (model: {model_path})...")
        traj_mappo, obstacles, goal, info_mappo = run_mappo_episode(
            env, cfg, str(model_path), seed=args.seed)
        print(f"  Steps: {len(traj_mappo)}")
        print(f"  Formation Error: {info_mappo['formation_error']:.4f}")
        print(f"  Reached Goal: {info_mappo['reached']}")
    elif args.mode in ["both", "mappo"]:
        print(f"\nWarning: Model not found at {model_path}")
        print("  Run training first or specify --model-path")

    # Visualize
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    if traj_rmp is not None and traj_mappo is not None:
        print("\nCreating comparison plot...")
        plot_comparison(traj_rmp, traj_mappo, obstacles, goal, cfg,
                       save_path=save_dir / "comparison.png")
    elif traj_rmp is not None:
        print("\nCreating RMPflow animation...")
        create_animation(traj_rmp, obstacles, goal, cfg,
                        title="RMPflow Baseline",
                        save_path=save_dir / "rmp_demo.gif")
    elif traj_mappo is not None:
        print("\nCreating MAPPO animation...")
        create_animation(traj_mappo, obstacles, goal, cfg,
                        title="COSMOS + MAPPO",
                        save_path=save_dir / "mappo_demo.gif")

    print("\nDone!")


if __name__ == "__main__":
    main()
