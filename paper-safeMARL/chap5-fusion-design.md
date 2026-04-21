# 第 5 章 · chap3 + chap4 融合设计（硬-软分层控制栈）

> 创建日期: 2026-04-19
> 对应论文: chap5 §5.1.3 (执行流水线), §5.2.3-5.2.4 (Webots 部署), §5.3 (实物部署)
> 上游文档: `chap4-rq-deliverables.md`, `chap5-revision-plan.md`

---

## 1. 核心架构：串联式硬-软分层

两章提供**不同语义层次**的安全保证，在 chap5 中通过**串联管道**合成：

```
┌────────────────────────────────────────────────────────────────────┐
│                  MAPPO policy  π_θ(o_i,t)                          │
│                          ↓  a_RL ∈ R²                              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  chap4 软协调层 · MultiAgentRMPAdapter                     │    │
│  │    • 编队距离叶 + 方向叶 (刚性保持)                         │    │
│  │    • 碰撞软排斥叶 (速度势垒)                                │    │
│  │    • RL 叶 (w_RL 度量加权, 把策略意图纳入 pullback)         │    │
│  │    • 差速映射 (ax, ay) → (v*, ω*)                          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                          ↓  a* = (v*, ω*)                          │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  chap3 硬安全层 · 约束流形投影                              │    │
│  │    • 约束集 c(q) ≤ 0 : 墙体、机器人间最小距离、HJ 可行集     │    │
│  │    • 状态来自 data-driven EKF (odom+IMU+邻居+lidar)         │    │
│  │    • 零空间投影 + 约束修正:                                │    │
│  │        dq = N_c α + (-K_c J_c^† c(q))                       │    │
│  │    • 加速度级修正后再做差速映射                             │    │
│  └────────────────────────────────────────────────────────────┘    │
│                          ↓  a_safe = (v_s, ω_s)                    │
│                  wheel split → 电机执行                            │
└────────────────────────────────────────────────────────────────────┘
```

**关键设计原则**：

- chap4 **在上**，负责"目标协调 + 几何一致性"（软）
- chap3 **在下**，负责"硬约束不可违反"（硬）
- chap3 **不修改** chap4 的优化意图——只把违约的方向投影掉
- 任何算法（MAPPO、MAPPO_Lag、ATACOM、GCPL...）都可以作为上游被 chap3 兜底

---

## 2. 三个关键融合点

### 融合点 ① · 动作空间对齐

| 层 | 原生动作空间 | 单位 |
|---|---|---|
| MAPPO | $a_{RL} \in \mathbb{R}^2$ 加速度意图 | m/s² (归一化) |
| chap4 软层输出 | $(v^*, \omega^*)$ 差速指令 | m/s, rad/s |
| chap3 硬层输入 | $\dot q \in \mathbb{R}^{2N}$ 构型空间速度 | m/s |
| chap3 硬层输出 | $\dot q_{\text{safe}}$ | m/s |
| 最终 | $(v_s, \omega_s)$ | m/s, rad/s |

**融合方式**：chap4 软层的 `apply_correction` **不**启用 `use_diff_drive_mapping`，输出保持世界坐标加速度 $(a_x, a_y)$；chap3 在加速度级做零空间投影，得到 $(a_x^{\text{safe}}, a_y^{\text{safe}})$；最后由 **统一的差速映射器**生成 $(v_s, \omega_s)$。

```python
# 伪代码 (chap5 §5.1.3)
a_rl = policy(obs)                                  # (N, 2) world-frame accel
a_soft = rmp_adapter.step(                          # (N, 2) world-frame accel
    positions, velocities, headings, a_rl,
    use_diff_drive_mapping=False,                    # 关闭, 留给 chap3 后处理
)
a_safe = atacom.project(a_soft, positions,          # (N, 2) null-space 投影后
                        velocities, neighbors, dt)
v_cmd, omega_cmd = diff_drive_map(a_safe, headings) # (N, 2)
```

---

### 融合点 ② · 约束集划分（软 vs 硬）

| 约束类别 | 放在软层 (chap4) | 放在硬层 (chap3) |
|---|:---:|:---:|
| 编队距离 $\|p_i-p_j\| = d^\star$ | ✅ | ❌ (允许临时压缩) |
| 编队方向 $\hat r_{ij} \cdot \hat r^\star = 1$ | ✅ | ❌ |
| 目标到达 $\|p_i - p_{\text{goal}}\| \to 0$ | 奖励 / 隐式 | ❌ |
| 机器人间**最小**安全距离 $\|p_i-p_j\| \ge d_S^{\text{hard}}$ | 软层有速度势垒 | ✅ **不可违反** |
| 墙体 / 静态障碍物距离 $\ge d_W^{\text{hard}}$ | 软层有碰撞叶 | ✅ |
| HJ 可行集 $V(x) \ge 0$ | ❌ | ✅ |
| 速度/加速度饱和 $\|v\| \le v_{\max}$ | ❌ | ✅ (简单限幅) |

**原则**：**同一物理约束在两层可以都存在，但语义不同**。例如 "机器人间避碰"：
- 软层的碰撞叶 $w(z) = z^{-4}$ 是**训练期软引导**，让策略学会远离危险区
- 硬层的约束 $\|p_i - p_j\| \ge d_S^{\text{hard}}$ 是**部署期硬兜底**，防止数值/传感器噪声突破

这种"双层冗余"是分层架构的核心优势。

---

### 融合点 ③ · 状态共享与 EKF

chap5 实际部署时，**状态估计**是共享基础：

```
        ┌─────────────────────────────────────────┐
        │  Per-agent Data-Driven EKF (chap3 §3.4)  │
        │   输入: 本机 odom + IMU + lidar hazard   │
        │         + 邻居广播的位姿 (ROS2 topic)    │
        │   输出: ĥat x_i, Σ_i                    │
        └─────────────────────────────────────────┘
                       │
         ┌─────────────┴──────────────┐
         ↓                            ↓
    chap4 软层                   chap3 硬层
    需要: 邻居 p, v              需要: 本机 + 邻居 p, v, Σ (用于鲁棒投影)
```

**关键细节**：
- 每机运行独立的 EKF（chap3 §3.4）
- 邻居位置通过 ROS2 话题 `/agent_j/state_estimate` 广播
- chap3 硬层可吸收 EKF 协方差做鲁棒投影：$c(q) + \beta \sqrt{\text{tr}(J_c \Sigma J_c^\top)} \le 0$（β = 保守度系数）

---

## 3. 分布式 vs 集中式

对实物部署的**工程抉择**：

### 方案 A · 全集中式（Supervisor 统一调度）

```
一台 ROS2 节点 (GCPL supervisor):
  ├─ sub /agent_i/odom * N
  ├─ pub /agent_i/cmd_vel * N
  └─ 运行 chap4 + chap3 统一优化
```

- ✅ 简单；chap4 的 RMPflow 树本身就是全局的
- ❌ 单点故障；通信延迟高时降级
- 适合 Webots 仿真（通信零延迟）

### 方案 B · 全分布式（每机一个节点）

```
每个机器人节点 i:
  ├─ 订阅 /agent_j/state_estimate (all j≠i)  ← 邻居状态
  ├─ 本地 EKF (chap3 §3.4)
  ├─ 本地 chap4: 只看邻居的编队叶 + 本机的碰撞/RL 叶 (decentralized RMP)
  ├─ 本地 chap3: 投影只用邻居最近估计
  └─ 发 /agent_i/cmd_vel
```

- ✅ 无单点故障；可扩展到大 N
- ❌ 邻居状态过时会破坏一致性
- 适合 E-puck 实物（WiFi 延迟 10-50ms 可接受）

### 方案 C · 半分布式（推荐 chap5 实物）

- chap4 软协调走**中心式**（supervisor 收集所有位置 → 运行一次 RMP → 分发命令）
- chap3 硬投影走**本地**（每机自己看本机 lidar + 订阅到的邻居位姿，本地投影）

**理由**：
- chap4 的 RMP 树本身就是全局（需要所有位置才能 pullback），集中算比分布算更干净
- chap3 对通信不敏感：硬投影只依赖瞬时邻居位姿（可容忍旧邻居估计，因为是"不违约"兜底）
- 通信带宽低：supervisor 下发 $(v^*, \omega^*)$ 是 2 个 float，本机投影只耗本机算力

---

## 4. chap5 正文可直接用的执行伪代码

### 4.1 Webots Supervisor 版（§5.2.4）

```python
# cosmos/ros2/epuck_formation/gcpl_full_stack_supervisor.py
from cosmos.safety.rmp_multi_agent import MultiAgentRMPAdapter
from cosmos.safety.atacom import COSMOS       # chap3 约束流形投影
from cosmos.safety.constraints import WallConstraint, InterRobotConstraint
# (可选) 本地 EKF
from cosmos.estimation.ekf import DataDrivenEKF

# 一次性初始化
rmp = MultiAgentRMPAdapter(
    num_agents=N, formation_shape="wedge",
    formation_target_distance=0.5,
    use_diff_drive_mapping=False,   # 关, 交给 chap3 后处理
)
atacom = COSMOS(
    num_agents=N,
    constraints=[WallConstraint(walls=[(-1.0,0), (1.0,0)], d_hard=0.25),
                 InterRobotConstraint(d_hard=0.18)],
)
policy = load_mappo_checkpoint("...path.../actor_agent*.pt")

# 每个控制周期:
while supervisor.step(dt) != -1:
    # 1. 读位姿
    pos, vel, heading = read_from_webots()

    # 2. 每机的观测 -> MAPPO 策略
    obs = build_obs(pos, vel, goal, neighbors)         # (N, obs_dim)
    a_rl = policy.act(obs, deterministic=True)         # (N, 2) accel intent

    # 3. chap4 软协调 (世界坐标加速度)
    a_soft = rmp.step(pos, vel, heading, a_rl)         # (N, 2) world accel

    # 4. chap3 硬投影 (不违约)
    a_safe = atacom.project(a_soft, pos, vel, dt=dt)   # (N, 2) world accel

    # 5. 统一差速映射
    for i in range(N):
        v_i, omega_i = diff_drive_map(a_safe[i], heading[i])
        send_cmd(agent_id=i, v=v_i, omega=omega_i)
```

### 4.2 E-puck 实物分布式版（§5.3.4，每机节点）

```python
# cosmos/ros2/epuck_formation/agent_node.py
# 每个 E-puck 运行一个这样的实例
class AgentNode(Node):
    def __init__(self, agent_id, total_agents):
        self.id = agent_id
        self.N = total_agents
        self.ekf = DataDrivenEKF(...)
        self.atacom = COSMOS(num_agents=1,          # 本机只需本机的硬投影
                              constraints=[WallConstraint(...),
                                            InterRobotConstraint(d_hard=0.18)])
        # rmp 是全局的, 通过 /rmp_cmd_vel/agent_i 订阅 supervisor 结果
        self.sub_rmp = self.create_subscription(Twist,
            f'/rmp_cmd_vel/agent_{agent_id}', self.on_rmp_cmd, 10)
        self.pub_cmd = self.create_publisher(Twist,
            f'/agent_{agent_id}/cmd_vel', 10)
        # 订阅邻居估计
        self.neighbors = {}   # j -> (p, v, t)
        for j in range(total_agents):
            if j != agent_id:
                self.create_subscription(PoseStamped,
                    f'/agent_{j}/state_estimate', self.on_neighbor, 10)

    def on_odom(self, msg):
        # chap3 本地 EKF
        self.ekf.update(odom=msg, lidar=self.last_lidar)
        self.pub_state_estimate.publish(self.ekf.state)

    def on_rmp_cmd(self, msg):
        # supervisor 发来的 a_soft (已经过 chap4), 本机做 chap3 投影
        a_soft = np.array([msg.linear.x, msg.linear.y])
        x_i = self.ekf.position
        v_i = self.ekf.velocity
        # 本机视野: 本机 + 最近邻居快照
        a_safe = self.atacom.project(
            a_soft.reshape(1, 2),
            positions=np.vstack([x_i, list_neighbor_p()]),
            velocities=np.vstack([v_i, list_neighbor_v()]),
            dt=0.05,
        )[0]
        v_cmd, omega_cmd = diff_drive_map(a_safe, self.ekf.heading)
        self.pub_cmd.publish(Twist(linear=..., angular=...))
```

---

## 5. 陷阱与对策

| 陷阱 | 症状 | 对策 |
|------|------|------|
| chap3 投影把 chap4 的编队调整全部抹掉 | 永远停在离墙 $d_{\text{hard}}$ 处不动 | 把硬约束设得比 chap4 的软屏障**更保守**（例如硬 0.25 vs 软 0.30） |
| chap4 软层已经收敛，chap3 无事可做但空耗 | 计算开销 | 硬层加"bypass if c(q) < -margin" 分支 |
| 邻居位姿传过来已经过时 (ROS2 延迟) | chap3 投影不一致，机器人抖动 | chap3 吸收 EKF Σ 做鲁棒投影；限制通信间隔 ≤ 50ms |
| chap4 用世界加速度、chap3 用构型空间加速度 | 单位/坐标系不匹配 | 在两层之间统一到"世界平面加速度 $(a_x, a_y)$"，最后统一差速映射 |
| HJ 可达值函数只在训练分布上准确 | 真实环境分布外，V(x) 失效 | 跨平台重新离线训练一遍 HJ；或用 conservative margin |
| 扩展到 N>3 时 chap4 矩阵伪逆变慢 | 控制频率下降 | 利用 RMPflow 分布式 pullback；或降频到 10 Hz |

---

## 6. 验证计划（对接 chap5 实验）

chap5 不再重复 chap3/chap4 本章的消融，而是做**集成与跨平台验证**：

### 6.1 消融：是否需要分层？

| 档位 | 组件 | 预期结果 |
|------|------|---------|
| S1 (chap4 only) | GCPL | 软协调达成，但可能偶发碰撞 |
| S2 (chap3 only) | ATACOM + 零编队 | 安全但无编队 |
| **S3 (完整分层)** | **GCPL + ATACOM** | **零违约且编队一致** |

**表** `tab:chap5_layered_ablation`：在 Webots 中 3 档 × 20 episodes 的 success/collision/formation_err。

### 6.2 跨平台一致性

| 平台 | 算法 | 指标 |
|------|------|------|
| Safety-Gym | 完整分层 | 基准 |
| Webots | 完整分层 | 与基准 gap |
| E-puck 实物 | 完整分层 | 与 Webots gap |

**表** `tab:sim2real_gap`：同一策略 + 同一分层栈在 3 平台的 4 指标数值，体现 sim-to-real 衰减但不失效。

---

## 7. 相关代码现状

| 组件 | 路径 | 状态 |
|------|------|------|
| chap4 MultiAgentRMPAdapter | `cosmos/safety/rmp_multi_agent.py` | ✅ **今天完成** |
| Webots supervisor demo (RMP-only) | `cosmos/ros2/epuck_formation/gcpl_supervisor.py` | ✅ **今天完成** |
| chap3 ATACOM 投影 | `cosmos/safety/atacom.py` | ✅ 已有 |
| chap3 约束定义 | `cosmos/safety/constraints.py` | ✅ 已有 |
| chap3 EKF | `cosmos/estimation/ekf.py` | ⚠ 需确认 |
| **融合栈 supervisor (chap4+chap3)** | `cosmos/ros2/epuck_formation/gcpl_full_stack_supervisor.py` | ❌ **待写** |
| 每机分布式节点 (实物) | `cosmos/ros2/epuck_formation/agent_node.py` | ❌ **待写** |
| MAPPO checkpoint 加载 helper | `cosmos/policies/load_checkpoint.py` | ❌ 待写 |

---

## 8. 对接 chap5 正文章节

| chap5 小节 | 本文对应 §  | 字数估计 |
|-----------|-----------|---------|
| §5.1.3 执行流水线 | §1 + §4.1 伪代码 | ~1 页 |
| §5.2.3 chap3 在 Webots | §2 约束划分 + §4.1 部分 | ~1.5 页 |
| §5.2.4 chap4 在 Webots | §4.1 Webots 版 + §7 现状 | ~1.5 页 |
| §5.3.1 ROS2 架构 | §3 方案 C + §4.2 实物版 | ~1 页 |
| §5.3.4 实物实验 | §6 跨平台验证 | ~2 页 |
| §5.5 讨论 | §5 陷阱 | ~0.5 页 |

## 9. TODO

- [ ] 写 `cosmos/ros2/epuck_formation/gcpl_full_stack_supervisor.py`（融合 chap3 投影）
- [ ] 写 pure-python MVP demo（matplotlib 动画，无 Webots 依赖）用于答辩演示
- [ ] 确认 `cosmos.estimation.ekf` 是否已有可用 EKF；否则参考 IROS2026 代码移植
- [ ] 为 chap5 §5.1 绘 `fig:deploy_pipeline` 融合架构图（三栏）
- [ ] 跑 chap5 S3 完整分层消融（要先有上两个实现）
