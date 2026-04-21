#!/usr/bin/env python3
"""
Webots Supervisor controller for chap5 §5.1.3 完整分层控制栈:
    MAPPO → chap4 GCPL (soft) → chap3 ATACOM (hard) → 差速映射 → 电机

与同目录 gcpl_supervisor.py 的区别:
    * gcpl_supervisor.py  : 只跑 chap4 软协调 (适合 §5.2.4 单独验证)
    * 本文件              : chap4 软 + chap3 硬 完整栈 (适合 §5.1.3 / §5.2.5 / §5.3)

融合设计参考: paper-safeMARL/chap5-fusion-design.md §1-§4
关键点:
    1. chap4 adapter 关闭 diff-drive 映射 (use_diff_drive_mapping=False),
       输出世界坐标加速度 (ax, ay).
    2. chap3 ATACOM 在世界加速度级做零空间投影, 输出 (ax, ay)_safe.
    3. 最后统一 diff-drive 映射到 (v, omega) 发给电机.

前置:
    * Webots R2023b+
    * MULTI_ROBOT_RMPFLOW_PATH 指向 algorithms/multi-robot-rmpflow/
    * 场景 .wbt 中 E-puck DEF 命名为 EPUCK_0, EPUCK_1, ...
    * Supervisor 机器人附带 emitter (频道 1)

环境变量覆盖:
    GCPL_NUM_AGENTS=3            机器人数
    GCPL_SHAPE=wedge             编队 (line | wedge | circle | mesh)
    GCPL_DIST=0.5                编队目标间距 (m)
    GCPL_SAFETY_R=0.3            软避碰半径 (m, 用于 chap4 碰撞叶)
    GCPL_HARD_SAFETY_R=0.25      硬安全半径 (m, 用于 chap3, 应 < GCPL_SAFETY_R)
    GCPL_FUSION_MODE=leaf        RL 融合模式
    GCPL_RL_WEIGHT=0.01          RL 叶权重 (0.01=rmp_only; 10=GCPL)
    GCPL_ENABLE_ATACOM=1         0 = 关闭 chap3 投影, 退化为只有 chap4
    GCPL_MAPPO_CKPT_DIR=         MAPPO checkpoint 目录; 为空则 a_RL=0
    EPUCK_DEF_FMT=EPUCK_{i}      场景中 E-puck DEF 名格式
    GCPL_TEST_REPLAY=            若非空, 跳过 MAPPO/GCPL/ATACOM, 直接下发脚本化
                                 cmd_vel (供平台演示/论文截图使用). 取值:
                                   wedge_channel : 3 机同速直行 10 s 后停车
"""
from __future__ import annotations

import os
import sys
import numpy as np

_THIS = os.path.abspath(os.path.dirname(__file__))
_SUITE = os.path.abspath(os.path.join(_THIS, "../../.."))
if _SUITE not in sys.path:
    sys.path.insert(0, _SUITE)

from cosmos.safety.rmp_multi_agent import MultiAgentRMPAdapter

# chap3 ATACOM 投影 + 构造所需的 config 模板
from cosmos.safety.atacom import AtacomSafetyFilter
from cosmos.apps.formation_nav.config import EnvConfig, SafetyConfig

try:
    from controller import Supervisor  # type: ignore
except ImportError as e:
    raise ImportError(
        "Webots controller module not found. Run this script under Webots's Python "
        "environment, or set WEBOTS_HOME and append $WEBOTS_HOME/lib/controller/python "
        "to PYTHONPATH."
    ) from e


# ---- 配置从环境变量读取 ----
NUM_AGENTS = int(os.environ.get("GCPL_NUM_AGENTS", 3))
SHAPE = os.environ.get("GCPL_SHAPE", "wedge").lower()
FORM_DIST = float(os.environ.get("GCPL_DIST", 0.5))
SOFT_R = float(os.environ.get("GCPL_SAFETY_R", 0.3))
HARD_R = float(os.environ.get("GCPL_HARD_SAFETY_R", 0.25))
FUSION = os.environ.get("GCPL_FUSION_MODE", "leaf").lower()
W_RL = float(os.environ.get("GCPL_RL_WEIGHT", 0.01))
ENABLE_ATACOM = os.environ.get("GCPL_ENABLE_ATACOM", "1") == "1"
MAPPO_CKPT = os.environ.get("GCPL_MAPPO_CKPT_DIR", "").strip()
EPUCK_FMT = os.environ.get("EPUCK_DEF_FMT", "EPUCK_{i}")
# GCPL_TEST_REPLAY: 启动时的 env var 只作为默认值, 运行时每 tick 还会读
# TEST_REPLAY_FLAG_FILE (默认 /tmp/gcpl_test_replay.flag) 以支持前端按钮动态开关.
TEST_REPLAY_DEFAULT = os.environ.get("GCPL_TEST_REPLAY", "").strip().lower()
TEST_REPLAY_FLAG_FILE = os.environ.get(
    "GCPL_TEST_REPLAY_FLAG_FILE", "/tmp/gcpl_test_replay.flag"
)


def read_test_replay_flag(default: str = "") -> str:
    """Read replay pattern from flag file each tick.

    文件不存在 或 读失败 → 返回 default (通常来自启动时的 env var).
    文件内容空 → 关闭回放 (返回 "").
    内容非空 → 使用该 pattern (如 'wedge_channel'). 写入时会被 strip+lower.
    """
    try:
        with open(TEST_REPLAY_FLAG_FILE, "r", encoding="utf-8") as f:
            return f.read().strip().lower()
    except FileNotFoundError:
        return default
    except OSError:
        return default

# E-puck 硬件常数
MAX_WHEEL_SPEED = 6.28
WHEEL_RADIUS = 0.0205
AXLE_LENGTH = 0.052
MAX_V_SI = WHEEL_RADIUS * MAX_WHEEL_SPEED  # ≈ 0.129 m/s
K_THETA = 1.0


# =============================================================================
# 辅助函数
# =============================================================================

def build_formation_topology(num_agents: int, shape: str, d: float):
    """Return (desired_distances: (N, N), topology_edges: list[(i, j)]).

    与 chap4 RMPCorrector 的 formation_spec 对齐,  chap3 ATACOM 也需要知道
    哪些对是编队邻居用于硬约束 (本项目不启用编队硬约束, 但保留接口).
    """
    N = num_agents
    dist = np.full((N, N), float(d), dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    edges = []
    if shape == "line":
        edges = [(i, i + 1) for i in range(N - 1)]
    elif shape == "circle":
        edges = [(i, (i + 1) % N) for i in range(N)]
    elif shape == "wedge":
        # 0 是顶点, 其余两侧
        edges = [(0, i) for i in range(1, N)]
    else:  # mesh 或未知
        edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    return dist, edges


def diff_drive_split(v_si: float, omega_si: float) -> tuple[float, float]:
    """(v, omega) in SI units → (left_wheel_speed, right_wheel_speed) rad/s.
    E-puck 差速运动学; 轮速饱和到 ±MAX_WHEEL_SPEED.
    """
    v_L = (v_si - omega_si * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    v_R = (v_si + omega_si * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    v_L = float(np.clip(v_L, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
    v_R = float(np.clip(v_R, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
    return v_L, v_R


def world_accel_to_diff(a_xy: np.ndarray, heading: float) -> tuple[float, float]:
    """把世界坐标加速度 (ax, ay) 映射为差速指令 (v, omega) in SI units.

    对应 chap4 §4.4.4 的差速映射:
        v    = cos θ * ax + sin θ * ay   (机体前向分量)
        θ_d  = atan2(ay, ax)              (期望朝向)
        ω    = k_θ * wrap(θ_d - θ)

    a_xy 的范数被视为"加速度意图", 此处退化为速度级指令: v 为 [-MAX_V_SI, MAX_V_SI] clip.
    更严格的做法应在积分器里累加, 但考虑 Webots 32 ms timestep 以及 E-puck 的动力学近似,
    当前直接作为速度指令送出 (与 cosmos/envs/webots_wrapper.py 一致).
    """
    ax, ay = float(a_xy[0]), float(a_xy[1])
    v = np.cos(heading) * ax + np.sin(heading) * ay
    if ax * ax + ay * ay > 1e-8:
        theta_d = float(np.arctan2(ay, ax))
        theta_err = (theta_d - heading + np.pi) % (2.0 * np.pi) - np.pi
    else:
        theta_err = 0.0
    omega = K_THETA * theta_err
    # Clip 到 E-puck 动力学包络
    v_si = float(np.clip(v, -MAX_V_SI, MAX_V_SI))
    omega_si = float(np.clip(omega, -6.0, 6.0))  # 角速度软限 6 rad/s
    return v_si, omega_si


# =============================================================================
# 脚本化测试路径 (用于 §5.2 平台截图; 不参与论文方法实验)
# =============================================================================
def scripted_cmd(t_sec: float, agent_idx: int, pattern: str) -> tuple[float, float]:
    """Return (v_si, omega_si) for a hard-coded replay path.

    Motivation: RoboticsAcademy 前端演示 / 论文 §5.2.2 双通道可视化截图时, 需要
    机器人有可见运动; 当 MAPPO 未加载或策略输出为零会导致静止. 本函数提供完全
    绕过 RL/GCPL/ATACOM 的脚本化指令, 仅用于平台本身的视觉验证.

    所有 agent 目前共享同一命令序列 (楔形编队平移), 方便 WebGUI 的 3 条轨迹曲线
    平行上升, 同时 noVNC 窗口里可以看到 E-puck 穿过 Sigwall 通道.
    """
    if pattern == "wedge_channel":
        # 0 .. 10 s : 直行 0.10 m/s  →  约推进 1 m, 穿过 Sigwall 通道
        # 10 .. 12 s: 线性减速
        # >12 s     : 停车, 保持画面可截图
        V_CRUISE = 0.10
        if t_sec < 10.0:
            return V_CRUISE, 0.0
        if t_sec < 12.0:
            return V_CRUISE * (1.0 - (t_sec - 10.0) / 2.0), 0.0
        return 0.0, 0.0
    # 未知模式 → 静止
    return 0.0, 0.0


# =============================================================================
# 可选: MAPPO checkpoint 加载 (stub - 需要时实现)
# =============================================================================
def try_load_mappo_policy(ckpt_dir: str, num_agents: int, shape: str, goal_xy: tuple):
    """尝试加载 safepo mappo_rmp 保存的 actor_agent*.pt + 配一个 obs mirror.

    返回字典:
        {
            "act": callable(positions, velocities) -> a_rl (N, 2),
            "loader": MAPPOPolicyLoader,
        }
    或失败时返回 None.

    Obs 管线 (对齐 chap5-fusion-design.md §2 融合点③):
        Webots positions/velocities (世界坐标)
          → safetygym_obs_mirror (teleport + env.task.obs() + one-hot + z-score)
          → MAPPO actor.forward (deterministic=True)
          → a_RL (N, 2) 归一化世界加速度意图
    """
    if not ckpt_dir:
        return None
    try:
        # 延迟 import, 避免在未装 torch 环境下直接 import 文件失败
        import numpy as _np
        from cosmos.policies.mappo_loader import MAPPOPolicyLoader, safetygym_obs_mirror

        # 支持两种路径: 直接指到 seed-xxx run dir, 或指到算法目录 (取最新 seed)
        from pathlib import Path as _P
        p = _P(ckpt_dir)
        if (p / "config.json").is_file():
            loader = MAPPOPolicyLoader(p)
        else:
            loader = MAPPOPolicyLoader.latest(p)

        if loader.num_agents != num_agents:
            raise RuntimeError(
                f"checkpoint 是 {loader.num_agents} 机器人, 但 supervisor 要求 {num_agents}"
            )

        goal_arr = _np.asarray(goal_xy, dtype=_np.float64)
        # chap4 实验场景的静态 sigwall 中心 (对齐 gcpl_full_stack_supervisor.py 默认值)
        hazards = _np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=_np.float64)

        def act_fn(positions: _np.ndarray, velocities: _np.ndarray) -> _np.ndarray:
            obs = safetygym_obs_mirror(
                task_name="SafetyPointMultiFormationGoal0-v0",
                num_agents=num_agents,
                formation_shape=shape,
                positions=positions,
                velocities=velocities,
                goal=goal_arr,
                hazards=hazards,
            )
            return loader.act(obs, deterministic=True)

        print(f"[full_stack] MAPPO loaded OK: N={loader.num_agents}, obs_dim={loader.obs_dim}")
        return {"act": act_fn, "loader": loader}
    except Exception as e:
        print(f"[full_stack] 加载 MAPPO failed: {type(e).__name__}: {e}; 使用零 a_RL")
        return None


# =============================================================================
# 主循环
# =============================================================================
def main():
    # ---- Webots 初始化 ----
    robot = Supervisor()
    dt_ms = int(robot.getBasicTimeStep())
    dt = dt_ms / 1000.0

    epuck_nodes = []
    for i in range(NUM_AGENTS):
        name = EPUCK_FMT.format(i=i)
        node = robot.getFromDef(name)
        if node is None:
            raise RuntimeError(
                f"未找到场景节点 DEF={name}. 请在 .wbt 中给每个 E-puck 加 DEF={name}."
            )
        epuck_nodes.append(node)

    emitter = robot.getDevice("emitter")
    if emitter is None:
        print("[full_stack] WARN: 没有 emitter, 将只读状态, 不下发控制")

    # ---- chap4 软协调层 ----
    rmp = MultiAgentRMPAdapter(
        num_agents=NUM_AGENTS,
        formation_shape=SHAPE,
        formation_target_distance=FORM_DIST,
        collision_safety_radius=SOFT_R,
        fusion_mode=FUSION,
        rl_leaf_weight=W_RL,
        use_diff_drive_mapping=False,   # 关键: 留给 chap3 后处理
        diff_drive_k_theta=K_THETA,
    )
    print(
        f"[full_stack] chap4 软层: N={NUM_AGENTS} shape={SHAPE} d={FORM_DIST} "
        f"R_soft={SOFT_R} fusion={FUSION} w_RL={W_RL} (diff_drive=OFF, 输出 world accel)"
    )

    # ---- chap3 硬安全层 ----
    atacom = None
    if ENABLE_ATACOM:
        env_cfg = EnvConfig(
            num_agents=NUM_AGENTS,
            arena_size=3.0,           # chap4 实验场景: 3x3 平面
            dt=dt,
            formation_shape=SHAPE,
            formation_radius=FORM_DIST,
        )
        safety_cfg = SafetyConfig(
            safety_radius=HARD_R,
            K_c=20.0,                 # 硬约束增益 (偏保守, Webots 仿真稳定)
            slack_type="softcorner",
            slack_beta=10.0,
            slack_threshold=0.02,
            rmp_formation_blend=0.0,   # 硬层不叠编队, 编队交给 chap4
            dq_max=MAX_V_SI,
            boundary_margin=0.15,
            eps_pinv=1e-4,
        )
        desired_distances, topology_edges = build_formation_topology(
            NUM_AGENTS, SHAPE, FORM_DIST
        )
        # chap4 实验场景里的 Sigwalls 中心 (x, y, r): r 取硬安全半径 + 10 cm
        walls = np.array([
            [-1.0, 0.0, HARD_R + 0.10],
            [ 1.0, 0.0, HARD_R + 0.10],
        ], dtype=np.float64)
        atacom = AtacomSafetyFilter(
            env_cfg=env_cfg,
            safety_cfg=safety_cfg,
            desired_distances=desired_distances,
            topology_edges=topology_edges,
            obstacle_positions=walls,
        )
        print(
            f"[full_stack] chap3 硬层: R_hard={HARD_R} K_c={safety_cfg.K_c} "
            f"walls={len(walls)} dq_max={MAX_V_SI:.4f}"
        )
    else:
        print("[full_stack] chap3 硬层: 已禁用 (GCPL_ENABLE_ATACOM=0)")

    # ---- 可选 MAPPO 策略 ----
    # MAPPO 策略 + obs mirror 打包回调 (接入 chap4 RL 叶节点的 a_RL 意图)
    # 目标位置对齐 chap4 实验: (0, 2.0)
    GOAL_XY = (0.0, 2.0)
    policy_pack = try_load_mappo_policy(MAPPO_CKPT, NUM_AGENTS, SHAPE, GOAL_XY)

    # ---- 主循环 ----
    prev_positions = None
    step_count = 0
    log_interval = max(1, int(500 / dt_ms))  # ~500ms 打印一次

    # 运行时 replay 状态跟踪 (开关由 flag 文件控制, 支持前端按钮)
    current_replay = TEST_REPLAY_DEFAULT  # 启动时默认值
    replay_start_step = step_count if current_replay else None
    if current_replay:
        print(f"[full_stack] !! TEST_REPLAY_DEFAULT='{current_replay}' 启动即启用, "
              f"跳过 MAPPO/GCPL/ATACOM, 直接下发脚本化 cmd_vel")
    print(f"[full_stack] replay flag file: {TEST_REPLAY_FLAG_FILE} "
          f"(写入 pattern 名启用, 写空关闭)")

    while robot.step(dt_ms) != -1:
        step_count += 1

        # 0. 每 tick 刷新 replay 开关 (文件开关, 供前端按钮使用)
        new_replay = read_test_replay_flag(default=TEST_REPLAY_DEFAULT)
        if new_replay != current_replay:
            # 状态切换: 重置 replay 起点, 这样脚本的 t_sec 总是从 0 开始
            if new_replay:
                replay_start_step = step_count
                print(f"[full_stack] >>> replay ENABLE  pattern='{new_replay}' "
                      f"(at step {step_count})")
            else:
                replay_start_step = None
                print(f"[full_stack] <<< replay DISABLE (back to MAPPO/GCPL/ATACOM)")
            current_replay = new_replay

        # 1. 读取状态
        positions = np.zeros((NUM_AGENTS, 2))
        headings = np.zeros(NUM_AGENTS)
        for i, node in enumerate(epuck_nodes):
            pos3 = node.getField("translation").getSFVec3f()
            positions[i] = [pos3[0], pos3[1]]
            rot = node.getField("rotation").getSFRotation()
            headings[i] = rot[3] if abs(rot[2]) > 0.9 else 0.0

        # 2. 估计速度 (有限差分)
        if prev_positions is None:
            velocities = np.zeros((NUM_AGENTS, 2))
        else:
            velocities = (positions - prev_positions) / dt
        prev_positions = positions.copy()

        # ★ 测试回放模式: 绕过整条 RL/安全栈, 直接下发脚本化指令后跳到日志
        if current_replay:
            t_sec = (step_count - replay_start_step) * dt
            if emitter is not None:
                for i in range(NUM_AGENTS):
                    v_si, omega_si = scripted_cmd(t_sec, i, current_replay)
                    v_L, v_R = diff_drive_split(v_si, omega_si)
                    msg = f"{i} {v_L:.4f} {v_R:.4f}".encode("utf-8")
                    emitter.send(msg)
            if step_count % log_interval == 0:
                print(f"[replay t={t_sec:5.2f}s] pos={positions.round(3).tolist()}")
            continue

        # 3. MAPPO 策略 (可选)
        if policy_pack is not None:
            try:
                a_rl = policy_pack["act"](positions, velocities)
            except Exception as e:
                print(f"[full_stack] MAPPO 推理错误 (step {step_count}): {e}; fallback 零")
                a_rl = np.zeros((NUM_AGENTS, 2))
        else:
            a_rl = np.zeros((NUM_AGENTS, 2))

        # 4. chap4 软协调层 → 世界加速度
        try:
            a_soft = rmp.step(
                positions=positions,
                velocities=velocities,
                headings=headings,
                rl_actions=a_rl,
            )
        except Exception as e:
            print(f"[full_stack] chap4 软层错误 (step {step_count}): {e}")
            continue

        # 5. chap3 硬投影 (可选)
        if atacom is not None:
            try:
                a_safe = atacom.project(
                    alphas=a_soft,
                    positions=positions,
                    velocities=velocities,
                    dt=dt,
                )
            except Exception as e:
                print(f"[full_stack] chap3 硬层错误 (step {step_count}): {e}")
                a_safe = a_soft  # 降级: 只用软层
        else:
            a_safe = a_soft

        # 6. 统一差速映射 + 下发
        if emitter is not None:
            for i in range(NUM_AGENTS):
                v_si, omega_si = world_accel_to_diff(a_safe[i], headings[i])
                v_L, v_R = diff_drive_split(v_si, omega_si)
                msg = f"{i} {v_L:.4f} {v_R:.4f}".encode("utf-8")
                emitter.send(msg)

        # 7. 日志
        if step_count % log_interval == 0:
            pair_dists = [
                float(np.linalg.norm(positions[i] - positions[j]))
                for i in range(NUM_AGENTS) for j in range(i + 1, NUM_AGENTS)
            ]
            min_dist = min(pair_dists) if pair_dists else float("inf")
            print(
                f"[step {step_count:5d}] "
                f"pos={positions.round(3).tolist()} "
                f"min_dist={min_dist:.3f} "
                f"a_soft_norm={np.linalg.norm(a_soft, axis=1).round(3).tolist()} "
                f"a_safe_norm={np.linalg.norm(a_safe, axis=1).round(3).tolist()}"
            )


if __name__ == "__main__":
    main()
