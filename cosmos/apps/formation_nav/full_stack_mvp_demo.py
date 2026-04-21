#!/usr/bin/env python3
"""
chap5 § 5.1.3 完整分层控制栈的 pure-python MVP 动画 demo.

不依赖 Webots, 不依赖 MAPPO checkpoint, 只用 matplotlib + cosmos.safety.
用 double-integrator 动力学近似 E-puck, 演示:

    MAPPO (= 零 RL 意图) → chap4 GCPL 软协调 → chap3 ATACOM 硬投影 → (v, ω)

场景对齐 chap4 实验:
    * 3 机楔形编队, 目标间距 0.5 m
    * 左右两面墙 @ x=±1, 形成沿 y 轴的狭窄通道
    * 初始位置: 下方区域随机采样 (y≈-0.8)
    * 目标:   (0, 2)

使用:
    python -m cosmos.apps.formation_nav.full_stack_mvp_demo --mode fusion
    python -m cosmos.apps.formation_nav.full_stack_mvp_demo --mode rmp_only
    python -m cosmos.apps.formation_nav.full_stack_mvp_demo --mode mappo_only

    --save out.mp4   保存动画为 mp4
    --save out.gif   保存为 gif
    --headless       不开窗口, 仅打印指标 + 保存轨迹图

对比模式:
    fusion     : chap4 软层 + chap3 硬层  (完整 GCPL 栈)
    rmp_only   : 只有 chap4 软层           (近似 RMPflow 基线)
    mappo_only : 只有裸 RL 动作 (这里是零) (近似 MAPPO 基线, 会撞墙)
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np

_THIS = os.path.abspath(os.path.dirname(__file__))
_SUITE = os.path.abspath(os.path.join(_THIS, "../../.."))
if _SUITE not in sys.path:
    sys.path.insert(0, _SUITE)

from cosmos.safety.rmp_multi_agent import MultiAgentRMPAdapter
from cosmos.safety.atacom import AtacomSafetyFilter
from cosmos.apps.formation_nav.config import EnvConfig, SafetyConfig


# =============================================================================
# 场景与动力学
# =============================================================================
NUM_AGENTS = 3
FORM_DIST = 0.5
SOFT_R = 0.3           # chap4 软避碰半径
HARD_R = 0.25          # chap3 硬安全半径
WALL_CENTERS = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
WALL_HARD_R = HARD_R + 0.10   # 硬半径 = 0.35 (agent 应停在墙外 > 0.35)
GOAL = np.array([0.0, 2.0], dtype=np.float64)
DT = 0.05
MAX_STEPS = 600
ARENA_HALF = 1.5       # 绘图范围
# demo 用的速度/加速度 > E-puck 实物 (0.129 m/s), 便于在有限 step 内看清差异.
# 真实硬件参数见 cosmos/ros2/epuck_formation/gcpl_full_stack_supervisor.py
V_MAX_SI = 0.5         # demo 速度上限 (m/s)
A_MAX = 3.0            # 加速度饱和 (m/s²)


def init_positions(rng: np.random.Generator) -> np.ndarray:
    """通道入口下方随机采样 3 机初始位置 (楔形大致形状).

    demo 把起点靠近通道 (y≈-0.5) 以便在 MAX_STEPS 内展示穿越过程; 对应论文
    4.6.1 实验场景里的 y=-0.8 初始区更保守.
    """
    base_y = rng.uniform(-0.55, -0.45)
    x_spread = rng.uniform(0.30, 0.45)
    return np.array([
        [rng.uniform(-0.1, 0.1), base_y + 0.15],   # 0 顶点, 稍前
        [-x_spread, base_y],                         # 1 左后
        [ x_spread, base_y],                         # 2 右后
    ], dtype=np.float64)


# =============================================================================
# 策略: a_RL = 真实 MAPPO 推理 或 fallback 到简单 goal-attract
# =============================================================================
def goal_attract_policy(positions: np.ndarray, k: float = 0.8) -> np.ndarray:
    """Fallback 策略: 指向 goal 的简单吸引力, 用于没提供 MAPPO checkpoint 时."""
    centroid = positions.mean(axis=0)
    direction = GOAL - centroid
    dist = float(np.linalg.norm(direction))
    if dist > 1e-6:
        direction = direction / dist
    a_intent = k * direction
    return np.tile(a_intent, (len(positions), 1))


def make_mappo_policy(ckpt_dir: str):
    """从 safepo run 目录加载 MAPPOPolicyLoader, 返回 callable(positions, velocities) -> a_rl.

    Args:
        ckpt_dir: 指向算法目录 (如 .../mappo_rmp) 或具体 seed run 目录.
    Returns:
        policy_fn: (positions(N,2), velocities(N,2)) -> a_rl(N,2)
    """
    from cosmos.policies.mappo_loader import MAPPOPolicyLoader, safetygym_obs_mirror
    from pathlib import Path
    p = Path(ckpt_dir)
    if (p / "config.json").is_file():
        loader = MAPPOPolicyLoader(p)
    else:
        loader = MAPPOPolicyLoader.latest(p)
    assert loader.num_agents == NUM_AGENTS, (
        f"checkpoint 是 {loader.num_agents} 机器人, 但 demo 场景是 {NUM_AGENTS} 机"
    )

    hazards = np.asarray(WALL_CENTERS, dtype=np.float64)

    def policy_fn(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        obs = safetygym_obs_mirror(
            task_name="SafetyPointMultiFormationGoal0-v0",
            num_agents=NUM_AGENTS,
            formation_shape="wedge",
            positions=positions,
            velocities=velocities,
            goal=GOAL.copy(),
            hazards=hazards,
        )
        return loader.act(obs, deterministic=True)

    print(f"[mappo] loaded {loader.num_agents} actors from {loader.models_dir}"
          f" (algo={loader.config.get('algorithm_name')}, obs_dim={loader.obs_dim})")
    return policy_fn


# =============================================================================
# 控制栈构造
# =============================================================================
def build_stack(mode: str):
    """构造指定模式的控制栈, 返回可调用对象 step(pos, vel, heading, a_rl)→a_safe."""

    if mode == "mappo_only":
        # 只用 RL 动作, 不做任何修正 (裸 MAPPO)
        def step(pos, vel, hdg, a_rl):
            return np.clip(a_rl, -A_MAX, A_MAX)
        return step, None, None

    # chap4 软层 (关 diff-drive 映射, 保持世界加速度)
    rmp = MultiAgentRMPAdapter(
        num_agents=NUM_AGENTS, formation_shape='wedge',
        formation_target_distance=FORM_DIST, collision_safety_radius=SOFT_R,
        fusion_mode='leaf', rl_leaf_weight=10.0,   # 模式都用 GCPL 权重; mode=rmp_only 通过 a_rl=0 模拟
        use_diff_drive_mapping=False,
    )

    if mode == "rmp_only":
        def step(pos, vel, hdg, a_rl):
            # RMP-only: 忽略 RL 动作 (传零)
            a_soft = rmp.step(pos, vel, hdg, np.zeros_like(a_rl))
            return np.clip(a_soft, -A_MAX, A_MAX)
        return step, rmp, None

    # mode == "fusion": chap4 软 + chap3 硬
    env_cfg = EnvConfig(num_agents=NUM_AGENTS, arena_size=3.0, dt=DT,
                         formation_shape='wedge', formation_radius=FORM_DIST)
    safety_cfg = SafetyConfig(
        safety_radius=HARD_R, K_c=20.0,
        slack_type='softcorner', slack_beta=10.0, slack_threshold=0.02,
        rmp_formation_blend=0.0, dq_max=V_MAX_SI,
        boundary_margin=0.15, eps_pinv=1e-4,
    )
    dist_mat = np.full((NUM_AGENTS, NUM_AGENTS), FORM_DIST)
    np.fill_diagonal(dist_mat, 0.0)
    edges = [(0, 1), (0, 2)]   # wedge
    walls = np.hstack([WALL_CENTERS, np.full((len(WALL_CENTERS), 1), WALL_HARD_R)])
    atacom = AtacomSafetyFilter(env_cfg, safety_cfg, dist_mat, edges, walls)

    def step(pos, vel, hdg, a_rl):
        a_soft = rmp.step(pos, vel, hdg, a_rl)
        a_safe = atacom.project(a_soft, pos, vel, dt=DT)
        return np.clip(a_safe, -A_MAX, A_MAX)

    return step, rmp, atacom


# =============================================================================
# 仿真一次 rollout
# =============================================================================
def rollout(mode: str, seed: int = 0, policy_fn=None):
    """单次 rollout. policy_fn=None 时回退到 goal_attract_policy."""
    rng = np.random.default_rng(seed)
    positions = init_positions(rng)
    velocities = np.zeros((NUM_AGENTS, 2))
    headings = np.full(NUM_AGENTS, np.pi / 2.0)    # 初始朝上

    step_fn, _, _ = build_stack(mode)

    traj = [positions.copy()]
    min_dists = []          # 每步记录最小对内/对墙距离 (越小越危险)
    wall_collisions = 0
    inter_collisions = 0
    reached = False

    for t in range(MAX_STEPS):
        if policy_fn is not None:
            a_rl = policy_fn(positions, velocities)
        else:
            a_rl = goal_attract_policy(positions, k=0.8)
        a = step_fn(positions, velocities, headings, a_rl)

        # double-integrator 更新
        velocities = velocities + a * DT
        # 速度饱和
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        cap = np.clip(V_MAX_SI / np.maximum(speeds, 1e-6), 0.0, 1.0)
        velocities = velocities * cap
        positions = positions + velocities * DT

        # 航向跟随速度方向 (模拟差速转向)
        for i in range(NUM_AGENTS):
            v = velocities[i]
            if np.linalg.norm(v) > 1e-3:
                headings[i] = float(np.arctan2(v[1], v[0]))

        traj.append(positions.copy())

        # 统计
        pair_dists = []
        for i in range(NUM_AGENTS):
            for j in range(i + 1, NUM_AGENTS):
                pair_dists.append(np.linalg.norm(positions[i] - positions[j]))
        wall_dists = []
        for i in range(NUM_AGENTS):
            for w in WALL_CENTERS:
                wall_dists.append(np.linalg.norm(positions[i] - w))
        min_pair = float(min(pair_dists)) if pair_dists else float('inf')
        min_wall = float(min(wall_dists))
        min_dists.append((min_pair, min_wall))
        if min_pair < HARD_R:
            inter_collisions += 1
        if min_wall < HARD_R:
            wall_collisions += 1

        # 到达 goal 判定: 编队质心在 goal 的 0.3 m 范围内
        if np.linalg.norm(positions.mean(axis=0) - GOAL) < 0.3:
            reached = True
            break

    traj = np.stack(traj, axis=0)       # (T, N, 2)
    metrics = dict(
        mode=mode,
        seed=seed,
        steps=len(traj) - 1,
        reached=reached,
        inter_collision_steps=inter_collisions,
        wall_collision_steps=wall_collisions,
        min_pair_dist=float(np.min([m[0] for m in min_dists])),
        min_wall_dist=float(np.min([m[1] for m in min_dists])),
    )
    return traj, metrics


# =============================================================================
# 可视化
# =============================================================================
def plot_static(trajs: dict, save_path: str | None = None):
    """多模式静态轨迹对比图 (3 个 panel)."""
    import matplotlib
    matplotlib.use("Agg" if save_path else matplotlib.get_backend())
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, axes = plt.subplots(1, len(trajs), figsize=(5 * len(trajs), 5), squeeze=False)
    axes = axes[0]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for ax, (label, (traj, m)) in zip(axes, trajs.items()):
        # 墙
        for w in WALL_CENTERS:
            ax.add_patch(patches.Circle(w, WALL_HARD_R, fill=False, ec='red', lw=1.5))
            ax.add_patch(patches.Circle(w, 0.08, color='black'))
        # 硬安全圆范围 (软)
        for w in WALL_CENTERS:
            ax.add_patch(patches.Circle(w, SOFT_R, fill=False, ec='orange', ls='--', lw=0.8))
        # goal
        ax.scatter([GOAL[0]], [GOAL[1]], marker='*', s=300, c='gold', ec='black', zorder=5)

        # 轨迹
        T = traj.shape[0]
        for i in range(NUM_AGENTS):
            ax.plot(traj[:, i, 0], traj[:, i, 1], color=colors[i], lw=1.5, label=f'agent {i}')
            ax.scatter(traj[0, i, 0], traj[0, i, 1], color=colors[i], marker='o', s=60)
            ax.scatter(traj[-1, i, 0], traj[-1, i, 1], color=colors[i], marker='s', s=60, ec='black')

        ax.set_xlim(-ARENA_HALF, ARENA_HALF); ax.set_ylim(-1.2, 2.3)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        title = (f"{label}\n"
                 f"reached={m['reached']}  steps={m['steps']}\n"
                 f"min_pair={m['min_pair_dist']:.3f}  min_wall={m['min_wall_dist']:.3f}\n"
                 f"coll(pair/wall)={m['inter_collision_steps']}/{m['wall_collision_steps']}")
        ax.set_title(title, fontsize=9)
        ax.legend(loc='lower left', fontsize=7)

    fig.suptitle("chap5 分层控制栈对比 · 3 机楔形穿通道", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)


def animate_run(traj: np.ndarray, metrics: dict, save_path: str | None = None):
    """单个 rollout 的动画 (matplotlib FuncAnimation)."""
    import matplotlib
    matplotlib.use("Agg" if (save_path and save_path.endswith(('.mp4', '.gif'))) else matplotlib.get_backend())
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6, 6))
    for w in WALL_CENTERS:
        ax.add_patch(patches.Circle(w, WALL_HARD_R, fill=False, ec='red', lw=1.5))
        ax.add_patch(patches.Circle(w, 0.08, color='black'))
    ax.scatter([GOAL[0]], [GOAL[1]], marker='*', s=300, c='gold', ec='black', zorder=5)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    # 初始化 artists
    robots = [ax.plot([], [], 'o', ms=12, color=colors[i])[0] for i in range(NUM_AGENTS)]
    tracks = [ax.plot([], [], '-', color=colors[i], lw=1.0, alpha=0.6)[0] for i in range(NUM_AGENTS)]
    title = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    ax.set_xlim(-ARENA_HALF, ARENA_HALF); ax.set_ylim(-1.2, 2.3)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_title(f"{metrics['mode']} · seed {metrics['seed']}")

    def update(frame: int):
        for i in range(NUM_AGENTS):
            robots[i].set_data([traj[frame, i, 0]], [traj[frame, i, 1]])
            tracks[i].set_data(traj[:frame + 1, i, 0], traj[:frame + 1, i, 1])
        title.set_text(f"step {frame}/{traj.shape[0] - 1}  "
                       f"reached={metrics['reached']}  "
                       f"coll={metrics['inter_collision_steps']}/{metrics['wall_collision_steps']}")
        return robots + tracks + [title]

    anim = FuncAnimation(fig, update, frames=traj.shape[0], interval=50, blit=False)

    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer=PillowWriter(fps=20))
        else:
            anim.save(save_path, writer='ffmpeg', fps=20)
        print(f"[saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)


# =============================================================================
# 入口
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["fusion", "rmp_only", "mappo_only", "all"],
                    help="all = 对比三种模式")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None,
                    help="静态对比图/动画文件名 (png/mp4/gif). 默认仅展示")
    ap.add_argument("--headless", action="store_true",
                    help="不开窗口, 适合服务器跑; 隐含 --save 必须给")
    ap.add_argument(
        "--ckpt-mappo",
        default=None,
        help="裸 MAPPO checkpoint 目录 (用于 mappo_only). 可是算法目录或具体 run dir. "
             "例如 runs/Base/<env>/mappo. 未提供则 fallback 到 goal_attract_policy.",
    )
    ap.add_argument(
        "--ckpt-rmp",
        default=None,
        help="mappo_rmp checkpoint 目录 (用于 fusion). 未提供则 fallback 到 goal_attract_policy.",
    )
    args = ap.parse_args()

    if args.headless and not args.save:
        args.save = "chap5_mvp_demo.png"

    # ---- 初始化可选的 MAPPO 策略 ----
    mappo_only_policy = None
    fusion_policy = None
    if args.ckpt_mappo:
        try:
            mappo_only_policy = make_mappo_policy(args.ckpt_mappo)
        except Exception as e:
            print(f"[mappo] 加载 --ckpt-mappo 失败: {e}; fallback 到 goal_attract")
    if args.ckpt_rmp:
        try:
            fusion_policy = make_mappo_policy(args.ckpt_rmp)
        except Exception as e:
            print(f"[mappo] 加载 --ckpt-rmp 失败: {e}; fallback 到 goal_attract")

    # rmp_only 永远不用 RL (传零给 rmp.step)
    policy_for = {
        "mappo_only": mappo_only_policy,
        "rmp_only":   None,           # 用 goal_attract 生成"假意图", 但 build_stack 里会忽略
        "fusion":     fusion_policy,
    }

    if args.mode == "all":
        trajs = {}
        for m in ["mappo_only", "rmp_only", "fusion"]:
            src = ("MAPPO" if policy_for[m] is not None
                   else ("ZERO" if m == "rmp_only" else "goal_attract"))
            print(f"[rollout] mode={m}  policy={src}")
            traj, metrics = rollout(m, seed=args.seed, policy_fn=policy_for[m])
            trajs[m] = (traj, metrics)
            print(f"  reached={metrics['reached']}  steps={metrics['steps']}  "
                  f"min_pair={metrics['min_pair_dist']:.3f}  "
                  f"min_wall={metrics['min_wall_dist']:.3f}  "
                  f"coll(pair/wall)={metrics['inter_collision_steps']}/"
                  f"{metrics['wall_collision_steps']}")
        if args.save and args.save.endswith(('.mp4', '.gif')):
            print("[warn] --mode all 不支持动画输出, 退化为静态对比 png")
            save = args.save.rsplit('.', 1)[0] + '.png'
        else:
            save = args.save
        plot_static(trajs, save_path=save)
    else:
        src = ("MAPPO" if policy_for[args.mode] is not None
               else ("ZERO" if args.mode == "rmp_only" else "goal_attract"))
        print(f"[rollout] mode={args.mode}  policy={src}")
        traj, metrics = rollout(args.mode, seed=args.seed, policy_fn=policy_for[args.mode])
        print(f"  {metrics}")
        if args.save and args.save.endswith(('.mp4', '.gif')):
            animate_run(traj, metrics, save_path=args.save)
        else:
            plot_static({args.mode: (traj, metrics)}, save_path=args.save)


if __name__ == "__main__":
    main()
