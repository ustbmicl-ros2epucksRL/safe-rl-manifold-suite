#!/usr/bin/env python3
"""
COSMOS + RMPflow + MAPPO 本地训练脚本

功能：
  - COSMOS 安全滤波器
  - WandB 日志记录 (可选)
  - 断点续训
  - 定期保存检查点
  - 多模型保存 (best, final, periodic)

用法：
  PYTHONPATH=. python formation_nav/train.py [选项]

示例：
  # 基础训练
  PYTHONPATH=. python formation_nav/train.py --episodes 200

  # 使用 WandB
  PYTHONPATH=. python formation_nav/train.py --episodes 200 --use-wandb

  # 断点续训
  PYTHONPATH=. python formation_nav/train.py --resume outputs/cosmos_exp/xxx/checkpoints/checkpoint_ep100.pt

  # GPU 训练
  PYTHONPATH=. python formation_nav/train.py --device cuda --episodes 500
"""

import argparse
import os
import json
import time
from datetime import datetime
import numpy as np
import torch

from formation_nav.config import Config
from formation_nav.env.formation_env import FormationNavEnv
from formation_nav.env.formations import FormationTopology
from formation_nav.safety import COSMOS, COSMOSMode
from formation_nav.algo.mappo import MAPPO
from formation_nav.algo.buffer import RolloutBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="COSMOS Formation Navigation Training")

    # 训练参数
    parser.add_argument("--episodes", type=int, default=200, help="训练轮数")
    parser.add_argument("--num-agents", type=int, default=4, help="智能体数量")
    parser.add_argument("--formation", type=str, default="square",
                        choices=["square", "triangle", "circle", "line", "hexagon"])
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 输出和日志
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--exp-name", type=str, default="cosmos_exp", help="实验名称")
    parser.add_argument("--use-wandb", action="store_true", help="启用 WandB 日志")

    # 检查点
    parser.add_argument("--resume", type=str, default="", help="从检查点恢复")
    parser.add_argument("--save-every", type=int, default=20, help="检查点保存间隔")

    # 设备
    parser.add_argument("--device", type=str, default="auto", help="设备 (cpu/cuda/auto)")

    # 环境参数
    parser.add_argument("--arena-size", type=float, default=5.0, help="场地大小")
    parser.add_argument("--num-obstacles", type=int, default=4, help="障碍物数量")
    parser.add_argument("--max-steps", type=int, default=500, help="最大步数")

    # 安全参数
    parser.add_argument("--safety-radius", type=float, default=0.5, help="安全半径")
    parser.add_argument("--cosmos-mode", type=str, default="decentralized",
                        choices=["centralized", "decentralized"])

    return parser.parse_args()


def save_checkpoint(episode, mappo, metrics, config, path):
    """保存检查点"""
    checkpoint = {
        'episode': episode,
        'actor': mappo.actor.state_dict(),
        'critic': mappo.critic.state_dict(),
        'cost_critic': mappo.cost_critic.state_dict(),
        'metrics': metrics,
        'config': config,
    }
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path, mappo, device):
    """加载检查点"""
    checkpoint = torch.load(path, map_location=device)
    mappo.actor.load_state_dict(checkpoint['actor'])
    mappo.critic.load_state_dict(checkpoint['critic'])
    if 'cost_critic' in checkpoint:
        mappo.cost_critic.load_state_dict(checkpoint['cost_critic'])
    return checkpoint.get('episode', 0), checkpoint.get('metrics', None)


def main():
    args = parse_args()

    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 创建运行 ID 和目录
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, args.exp_name, run_id)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 配置
    cfg = Config()
    cfg.env.num_agents = args.num_agents
    cfg.env.formation_shape = args.formation
    cfg.env.arena_size = args.arena_size
    cfg.env.num_obstacles = args.num_obstacles
    cfg.env.max_steps = args.max_steps
    cfg.safety.safety_radius = args.safety_radius
    cfg.train.seed = args.seed

    config_dict = {
        "experiment_name": args.exp_name,
        "run_id": run_id,
        "num_episodes": args.episodes,
        "num_agents": args.num_agents,
        "formation": args.formation,
        "seed": args.seed,
        "arena_size": args.arena_size,
        "safety_radius": args.safety_radius,
        "cosmos_mode": args.cosmos_mode,
        "device": device,
    }

    # 保存配置
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # 打印配置
    print("=" * 60)
    print("COSMOS Formation Navigation Training")
    print("=" * 60)
    print(f"实验名称:     {args.exp_name}")
    print(f"运行 ID:      {run_id}")
    print(f"设备:         {device}")
    print(f"智能体数量:   {args.num_agents}")
    print(f"编队形状:     {args.formation}")
    print(f"训练轮数:     {args.episodes}")
    print(f"安全半径:     {args.safety_radius}")
    print(f"COSMOS 模式:  {args.cosmos_mode}")
    print(f"输出目录:     {output_dir}")
    print("=" * 60)

    # 初始化 WandB
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="cosmos-formation-nav",
                name=f"{args.exp_name}_{run_id}",
                id=run_id,
                resume="allow",
                config=config_dict,
                tags=[args.formation, f"n{args.num_agents}", args.cosmos_mode],
            )
            print(f"\nWandB 已启用: {wandb_run.get_url()}")
        except ImportError:
            print("\n警告: WandB 未安装，跳过日志记录")
            args.use_wandb = False

    # 创建环境
    env = FormationNavEnv(cfg.env, cfg.reward)
    topology = FormationTopology(cfg.env.num_agents, "complete")

    # COSMOS 模式
    cosmos_mode = COSMOSMode.CENTRALIZED if args.cosmos_mode == "centralized" else COSMOSMode.DECENTRALIZED

    # 创建 COSMOS
    cosmos = COSMOS(
        env_cfg=cfg.env,
        safety_cfg=cfg.safety,
        desired_distances=env.desired_distances,
        topology_edges=topology.edges(),
        obstacle_positions=env.obstacles,
        mode=cosmos_mode
    )

    # 创建 MAPPO
    obs_dim = env.observation_space.shape[0]
    share_obs_dim = env.share_observation_space.shape[0]
    mappo = MAPPO(obs_dim, share_obs_dim, act_dim=2, cfg=cfg.algo, device=device)

    # 创建 Buffer
    buffer = RolloutBuffer(
        episode_length=cfg.env.max_steps,
        num_agents=cfg.env.num_agents,
        obs_dim=obs_dim,
        share_obs_dim=share_obs_dim,
        act_dim=2,
        gamma=cfg.algo.gamma,
        gae_lambda=cfg.algo.gae_lambda,
        device=device
    )

    # 断点续训
    start_episode = 0
    metrics = {
        'episode': [], 'reward': [], 'cost': [],
        'formation_error': [], 'min_dist': [], 'collisions': []
    }

    if args.resume and os.path.exists(args.resume):
        print(f"\n从检查点恢复: {args.resume}")
        start_episode, loaded_metrics = load_checkpoint(args.resume, mappo, device)
        if loaded_metrics:
            metrics = loaded_metrics
        print(f"已恢复到 Episode {start_episode}")

    print(f"\n观测维度: {obs_dim}, 共享观测: {share_obs_dim}")
    print(f"起始轮数: {start_episode}")

    # 训练循环
    print("\n开始训练...")
    print("=" * 70)

    start_time = time.time()
    total_steps = 0
    log_interval = max(1, args.episodes // 10)
    best_reward = float('-inf')

    for episode in range(start_episode, args.episodes):
        obs, share_obs, _ = env.reset(seed=cfg.train.seed + episode)
        cosmos.update_obstacles(env.obstacles)
        cosmos.reset(env.positions)
        buffer.set_first_obs(obs, share_obs)

        ep_reward, ep_cost, ep_form_err = 0.0, 0.0, 0.0
        ep_min_dist, ep_collisions = float('inf'), 0

        for step in range(cfg.env.max_steps):
            alphas, log_probs = mappo.get_actions(obs)
            values = mappo.get_values(share_obs)
            safe_actions = cosmos.project(alphas, env.positions, env.velocities, dt=cfg.env.dt)
            next_obs, next_share_obs, rewards, costs, dones, infos, _ = env.step(safe_actions)

            masks = (~dones).astype(np.float32).reshape(-1, 1)
            buffer.insert(next_obs, next_share_obs, alphas, log_probs, values, rewards, costs, masks)

            obs, share_obs = next_obs, next_share_obs
            total_steps += cfg.env.num_agents

            ep_reward += rewards[0, 0]
            ep_cost += costs[0, 0]
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
        avg_form_err = ep_form_err / (step + 1)
        metrics['episode'].append(episode + 1)
        metrics['reward'].append(ep_reward)
        metrics['cost'].append(ep_cost)
        metrics['formation_error'].append(avg_form_err)
        metrics['min_dist'].append(ep_min_dist)
        metrics['collisions'].append(ep_collisions)

        # WandB 日志
        if args.use_wandb:
            import wandb
            wandb.log({
                "train/reward": ep_reward,
                "train/cost": ep_cost,
                "train/formation_error": avg_form_err,
                "train/min_inter_dist": ep_min_dist,
                "train/collisions": ep_collisions,
                "train/episode_length": step + 1,
                "episode": episode + 1,
            })

        # 保存检查点
        if (episode + 1) % args.save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode+1}.pt")
            save_checkpoint(episode + 1, mappo, metrics, config_dict, ckpt_path)
            print(f"  [检查点已保存: checkpoint_ep{episode+1}.pt]")

        # 保存最佳模型
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            save_checkpoint(episode + 1, mappo, metrics, config_dict, best_path)

        # 打印进度
        if (episode + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            print(f"Ep {episode+1:4d}/{args.episodes} | R={ep_reward:7.2f} | "
                  f"Cost={ep_cost:4.0f} | FormErr={avg_form_err:.4f} | "
                  f"MinDist={ep_min_dist:.3f} | Coll={ep_collisions:2d} | FPS={fps:.0f}")

    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    save_checkpoint(args.episodes, mappo, metrics, config_dict, final_path)

    # 保存指标
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    elapsed = time.time() - start_time
    total_collisions = sum(metrics['collisions'])
    safe_episodes = sum(1 for c in metrics['collisions'] if c == 0)

    print("=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"用时:         {elapsed:.1f}s")
    print(f"总碰撞次数:   {total_collisions}")
    print(f"零碰撞轮数:   {safe_episodes}/{args.episodes} ({100*safe_episodes/args.episodes:.1f}%)")
    print(f"\n保存的文件:")
    print(f"  - {final_path}")
    print(f"  - {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print(f"  - {metrics_path}")

    # 完成 WandB
    if args.use_wandb:
        import wandb
        wandb.run.summary["train/total_collisions"] = total_collisions
        wandb.run.summary["train/safe_episode_rate"] = safe_episodes / args.episodes
        wandb.finish()
        print("\nWandB 运行已完成")


if __name__ == "__main__":
    main()
