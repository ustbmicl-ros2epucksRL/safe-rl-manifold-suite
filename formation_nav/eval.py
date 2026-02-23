"""
Evaluation and visualization for trained formation navigation models.

Loads a trained MAPPO model, runs evaluation episodes, and produces:
  - Matplotlib animation of trajectories
  - Summary metrics: success rate, formation error, collisions, path efficiency
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

from formation_nav.config import Config, EnvConfig, SafetyConfig, AlgoConfig, RewardConfig
from formation_nav.env.formation_env import FormationNavEnv
from formation_nav.env.formations import FormationTopology
from formation_nav.safety.atacom import AtacomSafetyFilter
from formation_nav.algo.mappo import MAPPO


def parse_args():
    parser = argparse.ArgumentParser(description="Formation Navigation Evaluation")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-obstacles", type=int, default=4)
    parser.add_argument("--formation", type=str, default="square")
    parser.add_argument("--formation-radius", type=float, default=1.0)
    parser.add_argument("--rmp-blend", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    return parser.parse_args()


def evaluate_episode(env, mappo, atacom, cfg, seed=0):
    """Run one evaluation episode. Returns trajectory and metrics."""
    obs_all, share_obs_all, _ = env.reset(seed=seed)
    atacom.update_obstacles(env.obstacles)
    atacom.reset(env.positions)

    trajectory = [env.positions.copy()]
    total_reward = 0.0
    total_cost = 0.0
    total_collisions = 0
    formation_errors = []
    min_dists = []

    for step in range(cfg.env.max_steps):
        alphas, _ = mappo.get_actions(obs_all, deterministic=True)
        safe_actions = atacom.project(alphas, env.positions, env.velocities)
        next_obs, next_share_obs, rewards, costs, dones, infos, _ = env.step(safe_actions)

        trajectory.append(env.positions.copy())
        total_reward += rewards[0, 0]
        total_cost += costs[0, 0]
        info = infos[0]
        formation_errors.append(info["formation_error"])
        min_dists.append(info["min_inter_dist"])
        total_collisions += info["collisions"]

        obs_all = next_obs
        share_obs_all = next_share_obs

        if dones.all():
            break

    reached = infos[0].get("reached", False)
    path_length = sum(
        np.linalg.norm(trajectory[t + 1] - trajectory[t], axis=1).mean()
        for t in range(len(trajectory) - 1)
    )
    straight_line = np.linalg.norm(trajectory[-1].mean(axis=0) - trajectory[0].mean(axis=0))
    path_efficiency = straight_line / max(path_length, 1e-8)

    metrics = {
        "reward": total_reward,
        "cost": total_cost,
        "collisions": total_collisions,
        "reached": reached,
        "avg_formation_error": np.mean(formation_errors),
        "min_inter_dist": np.min(min_dists),
        "path_efficiency": path_efficiency,
        "steps": len(trajectory) - 1,
    }

    return np.array(trajectory), metrics, env.obstacles, env.goal


def plot_trajectories(trajectory, obstacles, goal, num_agents, arena_size,
                      topology_edges, desired_distances=None,
                      title="Formation Navigation", save_path=None):
    """Plot agent trajectories, obstacles, and goal."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

    # Arena boundary
    rect = patches.Rectangle(
        (-arena_size, -arena_size), 2 * arena_size, 2 * arena_size,
        linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    # Obstacles
    for obs in obstacles:
        circle = patches.Circle(
            (obs[0], obs[1]), obs[2],
            facecolor='gray', edgecolor='black', alpha=0.5)
        ax.add_patch(circle)

    # Goal
    ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal')

    # Trajectories
    T = trajectory.shape[0]
    for i in range(num_agents):
        traj_i = trajectory[:, i, :]
        ax.plot(traj_i[:, 0], traj_i[:, 1], '-', color=colors[i],
                alpha=0.6, linewidth=1)
        ax.plot(traj_i[0, 0], traj_i[0, 1], 'o', color=colors[i],
                markersize=8, label=f'Agent {i} start')
        ax.plot(traj_i[-1, 0], traj_i[-1, 1], 's', color=colors[i],
                markersize=8)

    # Final formation edges
    final_pos = trajectory[-1]
    for (i, j) in topology_edges:
        ax.plot([final_pos[i, 0], final_pos[j, 0]],
                [final_pos[i, 1], final_pos[j, 1]],
                'k--', alpha=0.3, linewidth=1)

    ax.set_xlim(-arena_size * 1.1, arena_size * 1.1)
    ax.set_ylim(-arena_size * 1.1, arena_size * 1.1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close(fig)


def create_animation(trajectory, obstacles, goal, num_agents, arena_size,
                     topology_edges, save_path="formation_nav.mp4"):
    """Create animation of the formation navigation."""
    try:
        from matplotlib.animation import FuncAnimation, FFMpegWriter
    except ImportError:
        print("FFMpeg not available, skipping animation.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

    T = trajectory.shape[0]

    def init():
        ax.clear()
        rect = patches.Rectangle(
            (-arena_size, -arena_size), 2 * arena_size, 2 * arena_size,
            linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        for obs in obstacles:
            circle = patches.Circle(
                (obs[0], obs[1]), obs[2],
                facecolor='gray', edgecolor='black', alpha=0.5)
            ax.add_patch(circle)
        ax.plot(goal[0], goal[1], 'r*', markersize=20)
        ax.set_xlim(-arena_size * 1.1, arena_size * 1.1)
        ax.set_ylim(-arena_size * 1.1, arena_size * 1.1)
        ax.set_aspect('equal')
        return []

    def update(frame):
        ax.clear()
        init()
        pos = trajectory[frame]
        # Trails
        trail_start = max(0, frame - 50)
        for i in range(num_agents):
            trail = trajectory[trail_start:frame + 1, i, :]
            ax.plot(trail[:, 0], trail[:, 1], '-', color=colors[i], alpha=0.4)
            ax.plot(pos[i, 0], pos[i, 1], 'o', color=colors[i], markersize=10)
        # Formation edges
        for (i, j) in topology_edges:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    'k-', alpha=0.3, linewidth=1)
        ax.set_title(f"Step {frame}/{T - 1}")
        return []

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=range(0, T, max(1, T // 200)),
                         blit=False, interval=50)

    try:
        writer = FFMpegWriter(fps=20)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    except Exception:
        # Fallback to gif
        gif_path = save_path.replace('.mp4', '.gif')
        anim.save(gif_path, writer='pillow', fps=20)
        print(f"Animation saved to {gif_path}")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = Config()
    cfg.env.num_agents = args.num_agents
    cfg.env.num_obstacles = args.num_obstacles
    cfg.env.formation_shape = args.formation
    cfg.env.formation_radius = args.formation_radius
    cfg.safety.rmp_formation_blend = args.rmp_blend

    env = FormationNavEnv(cfg.env, cfg.reward)
    topology = FormationTopology(cfg.env.num_agents, "complete")
    topology_edges = topology.edges()

    atacom = AtacomSafetyFilter(
        env_cfg=cfg.env,
        safety_cfg=cfg.safety,
        desired_distances=env.desired_distances,
        topology_edges=topology_edges,
        obstacle_positions=env.obstacles,
    )

    obs_dim = env.observation_space.shape[0]
    share_obs_dim = env.share_observation_space.shape[0]
    act_dim = 2

    mappo = MAPPO(obs_dim, share_obs_dim, act_dim, cfg.algo, "cpu")
    mappo.load(args.model_path)
    print(f"Model loaded from {args.model_path}")

    # Run evaluation episodes
    all_metrics = []
    for ep in range(args.num_episodes):
        trajectory, metrics, obstacles, goal = evaluate_episode(
            env, mappo, atacom, cfg, seed=args.seed + ep)
        all_metrics.append(metrics)

        # Plot first episode
        if ep == 0:
            plot_trajectories(
                trajectory, obstacles, goal,
                cfg.env.num_agents, cfg.env.arena_size, topology_edges,
                title=f"Episode {ep + 1}",
                save_path=os.path.join(args.output_dir, "trajectory.png"),
            )
            if args.save_video:
                create_animation(
                    trajectory, obstacles, goal,
                    cfg.env.num_agents, cfg.env.arena_size, topology_edges,
                    save_path=os.path.join(args.output_dir, "formation_nav.mp4"),
                )

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    success = sum(m["reached"] for m in all_metrics)
    print(f"Success rate:        {success}/{args.num_episodes} "
          f"({100 * success / args.num_episodes:.1f}%)")
    print(f"Avg reward:          {np.mean([m['reward'] for m in all_metrics]):.2f}")
    print(f"Avg cost:            {np.mean([m['cost'] for m in all_metrics]):.2f}")
    print(f"Total collisions:    {sum(m['collisions'] for m in all_metrics)}")
    print(f"Avg formation error: {np.mean([m['avg_formation_error'] for m in all_metrics]):.4f}")
    print(f"Min inter-agent dist:{np.min([m['min_inter_dist'] for m in all_metrics]):.3f}")
    print(f"Avg path efficiency: {np.mean([m['path_efficiency'] for m in all_metrics]):.3f}")
    print(f"Avg steps:           {np.mean([m['steps'] for m in all_metrics]):.1f}")


if __name__ == "__main__":
    main()
