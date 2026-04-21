#!/usr/bin/env python3
"""
Webots Supervisor controller for chap5 §5.2.4:
    集中式运行 chap4 GCPL 多机编队控制.

职责:
    1. 用 Supervisor API 读取场景中所有 E-puck 的位置/朝向
    2. 调用 MultiAgentRMPAdapter 得到每机 (v_cmd, omega_cmd)
    3. 把 cmd 分发给各机器人 (通过 emitter / receiver 或设置关节速度)
    4. 周期性输出日志便于绘图

前置:
    * Webots R2023b+ (python controller)
    * MULTI_ROBOT_RMPFLOW_PATH 环境变量指向 algorithms/multi-robot-rmpflow/
    * safepo 已 editable install (cosmos 会自动 path 注入)
    * 场景 .wbt 文件里 E-puck 节点 DEF 名为 EPUCK_0, EPUCK_1, ...

使用:
    Webots 场景树中添加一个 Robot:
        Robot {
          name  "gcpl_supervisor"
          controller "gcpl_supervisor"
          supervisor TRUE
        }
    把本文件拷贝为 controllers/gcpl_supervisor/gcpl_supervisor.py

    或通过 extern controller:
        WEBOTS_CONTROLLER_URL=ipc://1234 python gcpl_supervisor.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

# 让 cosmos 能被 import (用 extern controller 时 sys.path 干净)
_THIS = os.path.abspath(os.path.dirname(__file__))
_SUITE = os.path.abspath(os.path.join(_THIS, "../../.."))
if _SUITE not in sys.path:
    sys.path.insert(0, _SUITE)

from cosmos.safety.rmp_multi_agent import MultiAgentRMPAdapter

# Webots controller API (需要在 Webots 提供的 Python 环境下运行)
try:
    from controller import Supervisor  # type: ignore
except ImportError as e:
    raise ImportError(
        "Webots controller module not found. Run this script under Webots's Python "
        "environment (e.g., `webots-controller python gcpl_supervisor.py`), or set "
        "WEBOTS_HOME and append $WEBOTS_HOME/lib/controller/python to PYTHONPATH."
    ) from e


# =============================================================================
# 配置
# =============================================================================
NUM_AGENTS = int(os.environ.get("GCPL_NUM_AGENTS", 3))
FORMATION_SHAPE = os.environ.get("GCPL_SHAPE", "wedge")
FORMATION_DIST = float(os.environ.get("GCPL_DIST", 0.5))    # meters
COLLISION_R = float(os.environ.get("GCPL_SAFETY_R", 0.3))    # meters
FUSION_MODE = os.environ.get("GCPL_FUSION_MODE", "leaf")     # leaf | additive
W_RL = float(os.environ.get("GCPL_RL_WEIGHT", 0.01))         # 0.01 = RMPflow-only approx
TIMESTEP_MS = 32   # Webots basic timestep

EPUCK_DEF_FMT = os.environ.get("EPUCK_DEF_FMT", "EPUCK_{i}")  # 场景里的 DEF 命名约定
MAX_WHEEL_SPEED = 6.28   # rad/s (E-puck spec)
WHEEL_RADIUS = 0.0205    # m
AXLE_LENGTH = 0.052      # m


def diff_drive_from_v_omega(v: float, omega: float) -> tuple[float, float]:
    """把 (v, omega) (m/s, rad/s) 转成 (left_wheel, right_wheel) 角速度 (rad/s).

    E-puck 差速运动学:
        v_L = (v - omega * L/2) / r
        v_R = (v + omega * L/2) / r
    """
    # 适配器返回的 v ∈ [-1, 1] 归一化, 先映射回 m/s
    # E-puck 最大线速度 ≈ wheel_radius * max_wheel_speed ≈ 0.0205 * 6.28 ≈ 0.129 m/s
    v_si = v * (WHEEL_RADIUS * MAX_WHEEL_SPEED)
    omega_si = omega   # omega 已是 rad/s
    v_L = (v_si - omega_si * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    v_R = (v_si + omega_si * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    # 饱和到硬件极限
    v_L = float(np.clip(v_L, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
    v_R = float(np.clip(v_R, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
    return v_L, v_R


def main():
    # ---- 初始化 Supervisor ----
    robot = Supervisor()
    dt = int(robot.getBasicTimeStep())

    # 获取每个 E-puck 的 Node 和它的 wheel motor (供 Supervisor 控制)
    epuck_nodes = []
    left_motors = []
    right_motors = []
    for i in range(NUM_AGENTS):
        name = EPUCK_DEF_FMT.format(i=i)
        node = robot.getFromDef(name)
        if node is None:
            raise RuntimeError(
                f"未找到场景节点 DEF={name}. 请在 .wbt 文件中给 E-puck 加 DEF."
            )
        epuck_nodes.append(node)

        # Supervisor 不能直接控制别的机器人的 motor; 需要 emitter/receiver 协议.
        # 这里假定各 E-puck 自己跑一个 cmd_receiver 控制器, supervisor 通过 emitter 发指令.
        # 简化: 直接用 node.getField("translation") 读位置, 用 emitter 广播命令.

    emitter = robot.getDevice("emitter")
    if emitter is None:
        print("[warn] Supervisor 没有 emitter 设备, 将只读状态不下发指令")

    # ---- 构造 RMP 适配器 ----
    adapter = MultiAgentRMPAdapter(
        num_agents=NUM_AGENTS,
        formation_shape=FORMATION_SHAPE,
        formation_target_distance=FORMATION_DIST,
        collision_safety_radius=COLLISION_R,
        fusion_mode=FUSION_MODE,
        rl_leaf_weight=W_RL,
        use_diff_drive_mapping=True,
        diff_drive_k_theta=1.0,
    )
    print(
        f"[GCPL supervisor] N={NUM_AGENTS} shape={FORMATION_SHAPE} "
        f"d={FORMATION_DIST} R={COLLISION_R} fusion={FUSION_MODE} w_RL={W_RL}"
    )

    # ---- 主循环 ----
    prev_positions = None
    log_interval = max(1, int(500 / TIMESTEP_MS))  # 每 ~500ms 打一次日志
    step_count = 0

    while robot.step(dt) != -1:
        step_count += 1

        # 1. 读取所有机器人状态
        positions = np.zeros((NUM_AGENTS, 2))
        headings = np.zeros(NUM_AGENTS)
        for i, node in enumerate(epuck_nodes):
            pos3 = node.getField("translation").getSFVec3f()
            positions[i] = [pos3[0], pos3[1]]
            # 从 rotation [x, y, z, angle] 抽 yaw (假设轴向为 z)
            rot = node.getField("rotation").getSFRotation()
            headings[i] = rot[3] if abs(rot[2]) > 0.9 else 0.0

        # 2. 估计速度 (有限差分)
        if prev_positions is None:
            velocities = np.zeros((NUM_AGENTS, 2))
        else:
            velocities = (positions - prev_positions) / (dt / 1000.0)
        prev_positions = positions.copy()

        # 3. 调用 RMP 适配器 (零 RL action = 纯几何)
        try:
            cmd = adapter.step(
                positions=positions,
                velocities=velocities,
                headings=headings,
                rl_actions=np.zeros((NUM_AGENTS, 2)),  # TODO: 接入 MAPPO checkpoint
            )
        except Exception as e:
            print(f"[GCPL err] step {step_count}: {e}")
            continue

        # 4. 分发指令
        if emitter is not None:
            for i in range(NUM_AGENTS):
                v, omega = float(cmd[i, 0]), float(cmd[i, 1])
                v_L, v_R = diff_drive_from_v_omega(v, omega)
                # 约定消息格式: b"i vL vR"
                msg = f"{i} {v_L:.4f} {v_R:.4f}".encode("utf-8")
                emitter.send(msg)

        # 5. 日志
        if step_count % log_interval == 0:
            pairwise = []
            for i in range(NUM_AGENTS):
                for j in range(i + 1, NUM_AGENTS):
                    pairwise.append(np.linalg.norm(positions[i] - positions[j]))
            print(
                f"[step {step_count}] "
                f"pos={positions.flatten().round(3).tolist()} "
                f"dists={[round(d, 3) for d in pairwise]} "
                f"cmd_v={cmd[:, 0].round(3).tolist()}"
            )


if __name__ == "__main__":
    main()
