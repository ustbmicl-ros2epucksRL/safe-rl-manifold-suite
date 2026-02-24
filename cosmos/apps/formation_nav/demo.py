#!/usr/bin/env python3
"""
COSMOS + RMPflow + MAPPO 编队导航演示脚本

功能：
  1. 训练 MAPPO 策略（使用 COSMOS 安全滤波器）
  2. 生成轨迹可视化图
  3. 生成动画视频
  4. 输出训练指标

用法：
  PYTHONPATH=. python formation_nav/demo.py [选项]

选项：
  --episodes    训练轮数 (默认 300)
  --num-agents  智能体数量 (默认 4)
  --formation   编队形状 (默认 square)
  --output-dir  输出目录 (默认 artifacts/demo_output)
  --no-video    不生成视频
  --seed        随机种子 (默认 42)
"""

import argparse
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from cosmos.apps.formation_nav.config import Config
from cosmos.envs.formation_nav import FormationNavEnv
from cosmos.envs.formations import FormationTopology
from cosmos.safety.cosmos_filter import COSMOSFilter as COSMOS, COSMOSMode
from cosmos.algos.mappo import MAPPO
from cosmos.buffers.rollout_buffer import RolloutBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="COSMOS Formation Navigation Demo")
    parser.add_argument("--episodes", type=int, default=300, help="训练轮数")
    parser.add_argument("--num-agents", type=int, default=4, help="智能体数量")
    parser.add_argument("--formation", type=str, default="square",
                        choices=["square", "triangle", "circle", "line", "hexagon"])
    parser.add_argument("--output-dir", type=str, default="artifacts/demo_output", help="输出目录")
    parser.add_argument("--no-video", action="store_true", help="不生成视频")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def print_banner():
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██████╗ ██████╗ ███████╗███╗   ███╗ ██████╗ ███████╗                        ║
║  ██╔════╝██╔═══██╗██╔════╝████╗ ████║██╔═══██╗██╔════╝                        ║
║  ██║     ██║   ██║███████╗██╔████╔██║██║   ██║███████╗                        ║
║  ██║     ██║   ██║╚════██║██║╚██╔╝██║██║   ██║╚════██║                        ║
║  ╚██████╗╚██████╔╝███████║██║ ╚═╝ ██║╚██████╔╝███████║                        ║
║   ╚═════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝                        ║
║                                                                               ║
║   COordinated Safety On Manifold for multi-agent Systems                     ║
║   + RMPflow + MAPPO 多机器人编队导航演示                                      ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def train(cfg, env, cosmos, mappo, buffer, num_episodes, log_interval=20):
    """训练循环"""
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    metrics_history = {
        'episode': [], 'reward': [], 'cost': [],
        'formation_error': [], 'min_dist': [], 'collisions': []
    }

    total_steps = 0
    start_time = time.time()

    for episode in range(num_episodes):
        obs, share_obs, _ = env.reset(seed=cfg.train.seed + episode)
        cosmos.update_obstacles(env.obstacles)
        cosmos.reset(env.positions)
        buffer.set_first_obs(obs, share_obs)

        ep_reward = 0.0
        ep_cost = 0.0
        ep_form_err = 0.0
        ep_min_dist = float('inf')
        ep_collisions = 0

        for step in range(cfg.env.max_steps):
            alphas, log_probs = mappo.get_actions(obs)
            values = mappo.get_values(share_obs)

            safe_actions = cosmos.project(alphas, env.positions, env.velocities, dt=cfg.env.dt)
            next_obs, next_share_obs, rewards, costs, dones, infos, _ = env.step(safe_actions)

            masks = (~dones).astype(np.float32).reshape(-1, 1)
            buffer.insert(next_obs, next_share_obs, alphas, log_probs, values, rewards, costs, masks)

            obs, share_obs = next_obs, next_share_obs
            total_steps += cfg.env.num_agents

            ep_reward += rewards[0] if rewards.ndim == 1 else rewards[0, 0]
            ep_cost += costs[0] if costs.ndim == 1 else costs[0, 0]
            ep_form_err += infos[0]["formation_error"]
            ep_min_dist = min(ep_min_dist, infos[0]["min_inter_dist"])
            ep_collisions += infos[0]["collisions"]

            if dones.all():
                break

        # PPO 更新
        last_values = mappo.get_values(share_obs)
        buffer.compute_returns_and_advantages(last_values)
        mappo.update(buffer)
        buffer.after_update()

        # 记录指标
        steps_done = step + 1
        avg_form_err = ep_form_err / steps_done

        metrics_history['episode'].append(episode + 1)
        metrics_history['reward'].append(ep_reward)
        metrics_history['cost'].append(ep_cost)
        metrics_history['formation_error'].append(avg_form_err)
        metrics_history['min_dist'].append(ep_min_dist)
        metrics_history['collisions'].append(ep_collisions)

        # 打印进度
        if (episode + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            print(f"Episode {episode+1:4d}/{num_episodes} | "
                  f"R={ep_reward:7.2f} | "
                  f"Cost={ep_cost:4.0f} | "
                  f"FormErr={avg_form_err:.4f} | "
                  f"MinDist={ep_min_dist:.3f} | "
                  f"Coll={ep_collisions:2d} | "
                  f"FPS={fps:.0f}")

    print("\n训练完成!")
    return metrics_history


def evaluate_episode(env, cosmos, mappo, cfg, seed=0):
    """评估一个 episode，返回轨迹"""
    obs, share_obs, _ = env.reset(seed=seed)
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


def plot_trajectory(trajectory, obstacles, goal, num_agents, arena_size,
                    topology_edges, save_path):
    """绘制轨迹图"""
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

    # 场地边界
    rect = patches.Rectangle((-arena_size, -arena_size), 2*arena_size, 2*arena_size,
                              linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    # 障碍物
    for obs in obstacles:
        circle = patches.Circle((obs[0], obs[1]), obs[2],
                                 facecolor='gray', edgecolor='black', alpha=0.5)
        ax.add_patch(circle)

    # 目标
    ax.plot(goal[0], goal[1], 'r*', markersize=25, label='Goal', zorder=10)

    # 轨迹
    T = trajectory.shape[0]
    for i in range(num_agents):
        traj = trajectory[:, i, :]
        ax.plot(traj[:, 0], traj[:, 1], '-', color=colors[i], alpha=0.7, linewidth=2)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=colors[i], markersize=12,
                label=f'Agent {i} start', zorder=5)
        ax.plot(traj[-1, 0], traj[-1, 1], 's', color=colors[i], markersize=12, zorder=5)

    # 最终编队连线
    final_pos = trajectory[-1]
    for (i, j) in topology_edges:
        ax.plot([final_pos[i, 0], final_pos[j, 0]],
                [final_pos[i, 1], final_pos[j, 1]], 'k--', alpha=0.5, linewidth=1.5)

    ax.set_xlim(-arena_size * 1.1, arena_size * 1.1)
    ax.set_ylim(-arena_size * 1.1, arena_size * 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('COSMOS + RMPflow + MAPPO Formation Navigation', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"轨迹图已保存: {save_path}")


def create_animation(trajectory, obstacles, goal, num_agents, arena_size,
                     topology_edges, save_path):
    """创建动画"""
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("动画库不可用，跳过动画生成")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    T = trajectory.shape[0]

    def init():
        ax.clear()
        rect = patches.Rectangle((-arena_size, -arena_size), 2*arena_size, 2*arena_size,
                                  linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        for obs in obstacles:
            circle = patches.Circle((obs[0], obs[1]), obs[2],
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

        # 轨迹尾迹
        trail_start = max(0, frame - 30)
        for i in range(num_agents):
            trail = trajectory[trail_start:frame+1, i, :]
            ax.plot(trail[:, 0], trail[:, 1], '-', color=colors[i], alpha=0.5, linewidth=2)
            ax.plot(pos[i, 0], pos[i, 1], 'o', color=colors[i], markersize=15)

        # 编队连线
        for (i, j) in topology_edges:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 'k-', alpha=0.3, linewidth=1)

        ax.set_title(f'COSMOS Formation Navigation - Step {frame}/{T-1}', fontsize=12)
        return []

    # 采样帧以加速
    frame_step = max(1, T // 150)
    frames = list(range(0, T, frame_step))

    anim = FuncAnimation(fig, update, init_func=init, frames=frames, blit=False, interval=50)

    # 保存为 GIF
    gif_path = save_path.replace('.mp4', '.gif')
    try:
        anim.save(gif_path, writer=PillowWriter(fps=20))
        print(f"动画已保存: {gif_path}")
    except Exception as e:
        print(f"动画保存失败: {e}")

    plt.close()


def plot_training_curves(metrics, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    episodes = metrics['episode']

    # Reward
    axes[0, 0].plot(episodes, metrics['reward'], 'b-', alpha=0.7)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].grid(True, alpha=0.3)

    # Cost
    axes[0, 1].plot(episodes, metrics['cost'], 'r-', alpha=0.7)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Cost')
    axes[0, 1].set_title('Episode Cost (Collisions)')
    axes[0, 1].grid(True, alpha=0.3)

    # Formation Error
    axes[0, 2].plot(episodes, metrics['formation_error'], 'g-', alpha=0.7)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Formation Error')
    axes[0, 2].set_title('Average Formation Error')
    axes[0, 2].grid(True, alpha=0.3)

    # Min Distance
    axes[1, 0].plot(episodes, metrics['min_dist'], 'm-', alpha=0.7)
    axes[1, 0].axhline(y=0.5, color='r', linestyle='--', label='Safety Radius')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Min Inter-Agent Distance')
    axes[1, 0].set_title('Minimum Inter-Agent Distance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Collisions
    axes[1, 1].plot(episodes, metrics['collisions'], 'c-', alpha=0.7)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Collisions')
    axes[1, 1].set_title('Collisions per Episode')
    axes[1, 1].grid(True, alpha=0.3)

    # Smoothed Reward (移动平均)
    window = min(20, len(episodes) // 5) if len(episodes) > 5 else 1
    if window > 1:
        smoothed = np.convolve(metrics['reward'], np.ones(window)/window, mode='valid')
        axes[1, 2].plot(episodes[window-1:], smoothed, 'b-', linewidth=2)
    else:
        axes[1, 2].plot(episodes, metrics['reward'], 'b-', linewidth=2)
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Reward (Smoothed)')
    axes[1, 2].set_title(f'Smoothed Reward (window={window})')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def main():
    args = parse_args()
    print_banner()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 配置
    print("\n" + "=" * 60)
    print("配置参数")
    print("=" * 60)

    cfg = Config()
    cfg.env.num_agents = args.num_agents
    cfg.env.formation_shape = args.formation
    cfg.train.seed = args.seed

    print(f"  智能体数量:   {cfg.env.num_agents}")
    print(f"  编队形状:     {cfg.env.formation_shape}")
    print(f"  障碍物数量:   {cfg.env.num_obstacles}")
    print(f"  训练轮数:     {args.episodes}")
    print(f"  安全半径:     {cfg.safety.safety_radius}")
    print(f"  输出目录:     {args.output_dir}")

    # 创建环境
    env = FormationNavEnv(cfg.env, cfg.reward)
    topology = FormationTopology(cfg.env.num_agents, "complete")

    # 创建 COSMOS 安全滤波器
    cosmos = COSMOS(
        env_cfg=cfg.env,
        safety_cfg=cfg.safety,
        desired_distances=env.desired_distances,
        topology_edges=topology.edges(),
        obstacle_positions=env.obstacles,
        mode=COSMOSMode.DECENTRALIZED
    )

    # 创建 MAPPO
    obs_dim = env.observation_space.shape[0]
    share_obs_dim = env.share_observation_space.shape[0]
    mappo = MAPPO(obs_dim, share_obs_dim, act_dim=2, cfg=cfg.algo, device="cpu")

    # 创建 Buffer
    buffer = RolloutBuffer(
        episode_length=cfg.env.max_steps,
        num_agents=cfg.env.num_agents,
        obs_dim=obs_dim,
        share_obs_dim=share_obs_dim,
        act_dim=2,
        gamma=cfg.algo.gamma,
        gae_lambda=cfg.algo.gae_lambda,
        device="cpu"
    )

    print(f"\n  COSMOS 模式:  {COSMOSMode.DECENTRALIZED.value}")
    print(f"  观测维度:     {obs_dim}")
    print(f"  动作维度:     2")

    # 训练
    metrics = train(cfg, env, cosmos, mappo, buffer, args.episodes)

    # 保存模型
    model_path = os.path.join(args.output_dir, "cosmos_mappo_model.pt")
    mappo.save(model_path)
    print(f"\n模型已保存: {model_path}")

    # 评估并可视化
    print("\n" + "=" * 60)
    print("生成可视化")
    print("=" * 60)

    trajectory, obstacles, goal, final_info = evaluate_episode(
        env, cosmos, mappo, cfg, seed=args.seed + 1000
    )

    print(f"\n评估结果:")
    print(f"  轨迹长度:     {len(trajectory)} 步")
    print(f"  编队误差:     {final_info['formation_error']:.4f}")
    print(f"  最小距离:     {final_info['min_inter_dist']:.3f}")
    print(f"  碰撞次数:     {final_info['collisions']}")

    # 绘制轨迹
    traj_path = os.path.join(args.output_dir, "trajectory.png")
    plot_trajectory(trajectory, obstacles, goal, cfg.env.num_agents,
                    cfg.env.arena_size, topology.edges(), traj_path)

    # 绘制训练曲线
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_curves(metrics, curves_path)

    # 生成动画
    if not args.no_video:
        anim_path = os.path.join(args.output_dir, "formation_nav.gif")
        create_animation(trajectory, obstacles, goal, cfg.env.num_agents,
                         cfg.env.arena_size, topology.edges(), anim_path)

    # 汇总
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - {model_path}")
    print(f"  - {traj_path}")
    print(f"  - {curves_path}")
    if not args.no_video:
        print(f"  - {os.path.join(args.output_dir, 'formation_nav.gif')}")

    # 安全统计
    total_collisions = sum(metrics['collisions'])
    safe_episodes = sum(1 for c in metrics['collisions'] if c == 0)
    print(f"\n安全统计:")
    print(f"  总碰撞次数:   {total_collisions}")
    print(f"  零碰撞轮数:   {safe_episodes}/{args.episodes} ({100*safe_episodes/args.episodes:.1f}%)")
    print(f"  最终编队误差: {metrics['formation_error'][-1]:.4f}")


if __name__ == "__main__":
    main()
