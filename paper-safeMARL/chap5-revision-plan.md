# 第 5 章修订方案 · 从仿真到实物的分层安全验证

> 创建日期: 2026-04-19
> 目标: 把第 5 章从"通用仿真平台"改造为**"chap3 硬安全层 + chap4 软协调层在 Webots 和 E-puck 实物上的端到端验证"**
> 参考: `contents/chap5.tex` (184 行), `cosmos/ros2/`, `IROS2026/` 已有 Webots 资产

---

## 0. 5 天实施排期（执行版）

> 最后更新: 2026-04-19
> 用途: 作为 chap5 落地的最小可执行路线图，后续直接在本节迭代

| Day | 任务 | 产出 |
|---|---|---|
| 1 | 目录骨架 + 单 E-puck `.wbt` + Docker volume 打通 | 容器内 `import cosmos` 成功 |
| 2 | 3 机 `.wbt` + slave controller + ROS2 话题联通 | supervisor 可下发 `cmd_vel` |
| 3 | 4 档批量实验 + 出图 | chap5 §5.5 的表/图 |
| 4 | React 前端组件（可选） | 可选，无硬性产出 |
| 5 | `chap5.tex` 写作与收敛 | 第 5 章正文 |

### 0.1 核心判断（本轮结论）

- 改动集中在"新增一个 exercise"，不改 Django/React/DB/Gazebo/RAM 等上游大模块。
- chap4 的 MVP 代码已就绪，可直接复用：
  - `cosmos/ros2/epuck_formation/gcpl_full_stack_supervisor.py`
  - `cosmos/safety/rmp_multi_agent.py`
  - `cosmos/policies/mappo_loader.py`
- 当前不需要重构 Docker/ROS2 主架构，仅在 compose 增加 volume 即可支撑 Webots 容器内调用。

### 0.2 维护约定

- 本节只记录"5天内可交付"的事项；超出范围的重构移到文档末尾长期 TODO。
- 每日结束更新 3 项：`完成`、`阻塞`、`次日目标`。
- 如发生方案切换（例如从集中式改半分布式），先更新本节再改代码。

---

## 1. 现状诊断

### 1.1 目前 chap5.tex 写了什么

```
第5章 机器人安全验证平台
├─ 5.1 平台层次架构 (5 层分层)
├─ 5.2 核心机制 (容器化调度 / 双通道可视化 / 安全评估接口)
├─ 5.3 系统实现 (仿真 / 前端 / 后端 / 通信)
└─ 5.4 本章小结
```

核心内容是**一个 Web 化 ROS2 仿真平台**，类似 RoboticsAcademy 的扩展版。包括前端 (React)、后端 (Django)、容器化、noVNC + WebGUI、Gazebo/Webots 集成。

### 1.2 现有 chap5 的两个根本问题

**问题 A · 跟 chap3/chap4 脱节**
- 完全没有提到 chap3 的约束流形/HJ 可达/EKF 方法
- 完全没有提到 chap4 的 GCPL/RMPflow/编队叶节点
- 仅作为"通用工具平台"存在，**本章工作无法体现前 4 章的方法学主线**

**问题 B · sim-to-real 没有实质内容**
- 只是说"支持硬件扩展"，没有具体的迁移步骤、接口对齐、标定、验证数据
- 没有 chap3/chap4 算法在 Webots 上跑通的证据
- 没有 E-puck 实物部署的任何实验数据

**问题 C · 结构比重失调**
- 前端/后端细节占了 ~60% 篇幅（React 组件、Django RESTful、PostgreSQL 表结构），这些是**工具级工程细节**，不适合硕士论文方法章
- 真正应作为重头戏的"如何把 chap3+chap4 落到实物"完全缺失

---

## 2. 代码资产盘点（哪些能直接用）

| 资产 | 路径 | 成熟度 | 能支撑哪部分 |
|---|---|---|---|
| Webots env wrapper | `cosmos/envs/webots_wrapper.py` | ✅ 完整 | chap5 Webots 迁移基础 |
| Webots world 示例 | `cosmos/ros2/worlds/epuck_arena.wbt` | ✅ 场景已建 | chap3 单机走廊 |
| E-puck 可视化器 | `cosmos/envs/epuck_visualizer.py` | ✅ | 实验图生成 |
| Webots demo 视频 | `IROS2026/webots_demo_corridor.mp4`、`webots_demo_dense.mp4`、`webots_demo_corridor_2x.mp4`、`webots_demo_dense_2x.mp4` | ✅ 已录 | chap3 sim-to-sim 证据 |
| Webots 实验数据 | `IROS2026/results_webots_real/webots_run*.json` | ✅ 多次 run | chap3 参数敏感性表 |
| ROS2 epuck_formation pkg | `cosmos/ros2/epuck_formation/` | ⚠ 骨架（CMakeLists + package.xml + worlds + scripts, 核心 Python 只有 `__init__.py`） | chap5 ROS2 多机部署待补 |
| E-puck 实物硬件 | 物理机器人 | ❓ 未知是否可用 | chap5 实物验证 |

### 关键空缺

- ❌ **chap4 (GCPL) 在 Webots 上无代码**（Webots wrapper 是单机的，未与 RMPCorrector 多机对接）
- ❌ **E-puck 实物部署代码**（只有骨架，没有真正的 ros2 node）
- ❌ **任何实物验证数据**

---

## 3. 建议新大纲

### 3.1 核心定位调整

**旧定位**（工具论）:
> "提供一个 Web 化平台来跑 RL 实验"

**新定位**（方法论）:
> "把 chap3 硬安全层 + chap4 软协调层的完整分层控制栈，从 Safety-Gymnasium 仿真迁移到 Webots 高保真仿真和 E-puck 实物系统，验证 sim-to-real 可行性与分层架构的跨平台一致性。"

### 3.2 推荐 6 节结构

```
第5章 从仿真到实物的分层安全验证
├─ 引言 (sim-to-real gap + 分层控制栈的落地需求)
├─ 5.1 分层控制栈的端到端部署框架
│   ├─ 5.1.1 Safety-Gym → Webots → E-puck 的三阶段迁移思路
│   ├─ 5.1.2 接口对齐: 状态/动作/安全约束三件套统一
│   └─ 5.1.3 分层控制栈在实物上的执行流水线
│       (策略 → chap4 GCPL 软协调 → chap3 约束流形硬投影 → 差速底盘)
├─ 5.2 Webots 高保真仿真迁移
│   ├─ 5.2.1 差速 E-puck 机器人建模 (轮距/最大速度/噪声)
│   ├─ 5.2.2 传感器与通信接口对齐 (lidar/odom/cmd_vel)
│   ├─ 5.2.3 chap3 约束流形层在 Webots 中的部署
│   ├─ 5.2.4 chap4 GCPL 多机协调层在 Webots 中的部署
│   └─ 5.2.5 Webots 验证实验
│       - 单机狭窄通道避障 (chap3 方法, fig:webots_chap3)
│       - 3 机楔形编队过通道 (chap4 方法, fig:webots_chap4)
│       - 与 Safety-Gym 基准的一致性对比 (tab:webots_vs_gym)
├─ 5.3 E-puck 实物部署
│   ├─ 5.3.1 ROS2 集成架构 (话题映射、控制周期、参数标定)
│   ├─ 5.3.2 域随机化: 从 Webots 参数分布到实物标定
│   ├─ 5.3.3 安全保障机制 (通信看门狗/速度饱和/人工急停)
│   ├─ 5.3.4 实物验证实验
│       - 单机避障 (chap3 实物 run)
│       - 多机协同 (chap4 实物 run, 可 N=2)
│   └─ 5.3.5 sim-to-real gap 度量与讨论
├─ 5.4 平台工程基础 (原 5.1-5.3 浓缩为一节, 或直接下放附录)
│   ├─ 5.4.1 容器化运行环境
│   ├─ 5.4.2 双通道可视化 (WebGUI + noVNC)
│   └─ 5.4.3 批量实验与评估接口
└─ 5.5 本章小结
```

### 3.3 节间比重（目标 15-20 页）

| 节 | 目标页数 | 比重 |
|---|---|---|
| 引言 | 0.5 | — |
| 5.1 部署框架 | 2-3 | 方法串联 |
| **5.2 Webots 迁移** | **4-5** | **重头戏** |
| **5.3 E-puck 实物** | **3-4** | **重头戏** |
| 5.4 平台基础 | 2-3 | 工程支撑 (可压缩) |
| 5.5 小结 | 0.5 | — |

主线：方法**落地**占主导，平台工程降格为背景。

---

## 4. 每节内容 spec

### 5.1 分层控制栈的端到端部署框架

**目标**：把 chap3（硬安全）+ chap4（软协调）在跨平台部署时的**接口契约**讲清。

**关键图**（新增）：
- `fig:deploy_pipeline` — 三栏并列示意图（Safety-Gym / Webots / E-puck），每栏都画出"MAPPO → GCPL → 约束流形 → 机器人"的执行链，标出三栏之间的对应接口。

**关键表**（新增）：
- `tab:interface_alignment` — 三平台的状态/动作/约束接口对照
  ```
  Safety-Gym       Webots            E-puck
  state.agent.pos  supervisor.get()  /odom.pose
  state.hazards    static WB box     lidar cluster
  action=Box(2)    wheel speed       cmd_vel
  cost=lidar.hit   collision flag    estop + bumper
  ```

### 5.2 Webots 迁移

**5.2.1 机器人建模**：贴 `epuck_arena.wbt` 关键参数（轮距 53mm、轮半径 20.5mm、最大速度 6.28 rad/s）；给出 URDF 或 PROTO 摘要代码 (~20 行)。

**5.2.2 传感器接口对齐**：
- lidar: Safety-Gym 的抽象 hazard_obs → Webots 的 8 个近距红外传感器 + 聚类成 hazard 中心
- odom: `supervisor.getSelf().getPosition()` 取 xyz → 填 `(x, y)`
- cmd_vel: `(v, ω)` → `wheel_left_speed` / `wheel_right_speed` 差速分解

**5.2.3 chap3 在 Webots 中的部署**：
- 约束流形层：复用 `cosmos.safety.atacom` 模块，**只需改状态源**（从 Gym state → Webots supervisor state）
- EKF：基于 wheel odom + 噪声模型（chap5 自己估计 Webots 的噪声 σ）
- HJ 可达值函数：离线训练好的表（chap3 离线模块产出），直接加载
- 给出 1 张走廊避障的**轨迹对比图**：裸 MAPPO vs MAPPO+chap3 约束流形

**5.2.4 chap4 在 Webots 中的部署**（现在没代码，要补）：
- RMPCorrector 是纯 Python numpy，无 CUDA 依赖 → **可以直接放到 Webots controller 里**
- 需要一个 supervisor node 收集 N 个 E-puck 位置 → 传入 RMPCorrector → 分发差速指令
- 或每个 E-puck 自己跑 RMPCorrector 拿到邻居位置（分布式）
- 3 机楔形过通道的**轨迹叠加图 + 编队误差时序图**

**5.2.5 Webots 验证实验结果**：
- 给出 3 张图（单机避障轨迹、3 机楔形编队轨迹、安全代价对比柱状图）
- `tab:webots_vs_gym` 表：相同算法在 Safety-Gym vs Webots 的 success/collision/formation_completion 差异（预期 Webots 结果略差但一致趋势）

### 5.3 E-puck 实物部署

**5.3.1 ROS2 集成**：
- 节点图（`fig:ros2_graph`）：`policy_node` / `rmp_node` / `manifold_node` / `cmd_vel_bridge` / `odom_filter`
- 话题约定（`/agent_0/odom`, `/agent_0/cmd_vel`, `/formation/state`, ...）
- 控制周期 20 Hz（Webots 对齐）

**5.3.2 域随机化与标定**：
- 里程计 bias 标定步骤
- 角速度限幅（max 2.5 rad/s）
- 电池电压-速度关系（降压导致延迟）
- 表：Webots 模型 vs 实测参数的 delta

**5.3.3 安全保障**：
- 通信看门狗：30ms 无心跳 → 停车
- 速度饱和与角速度限幅
- 人工急停按钮 + bumper 接触停车
- 软件硬停 + 硬件急停两层

**5.3.4 实物实验**：
- **至少做** 单机避障（chap3）：走廊 + 随机障碍
- **建议做** 2 机或 3 机编队通过（chap4）：直线或楔形
- 给出：实物轨迹照片/截图、成功率、平均耗时、碰撞次数、与 Webots 结果的 gap 数字

**5.3.5 sim-to-real gap 讨论**：
- 量化表：`tab:sim2real_gap` — 同一策略在 Gym / Webots / 实物上的 success/collision/formation_err
- 定性：哪些 gap 可以归因给何种模型失配（摩擦、电池、延迟、传感器噪声）

### 5.4 平台工程基础（浓缩）

把原 5.1-5.3 的内容压缩：
- 保留：容器化 + 双通道可视化 + API 接口
- 删除：前端 React 组件细节、后端 Django ORM 表结构、WebSocket 协议
- 目标长度：2-3 页。若答辩评审说"平台细节太少"，再选部分细节回补附录。

---

## 5. 关键修改清单（可执行 TODO）

### 5.1 现有 chap5.tex 改造 ✏️

- [ ] 章节标题：`机器人安全验证平台` → `从仿真到实物的分层安全验证`
- [ ] 引言段改写：去"RoboticsAcademy 改进"口吻，改为"sim-to-real 分层部署" 叙事
- [ ] 删除当前 5.3.2（前端）、5.3.3（后端 Django/PostgreSQL 大段），下放附录或整体删
- [ ] 新增 5.1 部署框架 + fig:deploy_pipeline + tab:interface_alignment
- [ ] 新增 5.2 Webots 迁移（5 小节）
- [ ] 新增 5.3 E-puck 实物部署（5 小节）
- [ ] 保留但压缩原核心机制到 5.4
- [ ] 本章小结改写对接 chap6 的结论

### 5.2 代码工作 💻（按完成度分级）

**级别 1 · 最小可写论文**
- [x] chap3 Webots 代码（IROS2026 已有）
- [ ] 补：chap3 Webots 轨迹图与数据重新绘制到 chap5 用的格式
- [ ] **写 chap4 RMPflow on Webots**（~1-2 天）：
  - 改 `cosmos/envs/webots_wrapper.py` 支持多 agent
  - 写 Webots supervisor 调用 `RMPCorrector`
  - 跑一次 3 机楔形 demo，录轨迹图

**级别 2 · 论文完整性**
- [ ] E-puck 实物 chap3 部署（~2-3 天）：
  - 完善 `cosmos/ros2/epuck_formation/` 的 Python node
  - 单机避障实物 run（至少 3 次 trial）
- [ ] 实物数据采集脚本（记录 odom/cmd_vel 时序到 rosbag）

**级别 3 · 完整性 + chap4 实物**（挑战级）
- [ ] E-puck 实物 chap4 部署（~5-7 天）：
  - 多机 ROS2 节点协调
  - 室内定位方案（如果没有外部定位系统，用 odometry 或 aruco 标签）
  - 多机 chap4 实物 run

### 5.3 配图与数据 📊

| 类型 | 新增图表 ID | 来源 |
|---|---|---|
| 图 | `fig:deploy_pipeline` | 手绘 (draw.io / tikz) |
| 表 | `tab:interface_alignment` | 手写 |
| 图 | `fig:webots_chap3_trajectory` | IROS2026 已有 + 重绘 |
| 图 | `fig:webots_chap4_trajectory` | **需要跑** |
| 图 | `fig:webots_formation_error` | **需要跑** |
| 表 | `tab:webots_vs_gym` | 跑完后整理 |
| 图 | `fig:ros2_graph` | 手绘 |
| 表 | `tab:webots_to_real_params` | 实物标定数据 |
| 图 | `fig:realrobot_trajectory` | 实物照片 + rosbag 绘图 |
| 表 | `tab:sim2real_gap` | 三平台对比数据 |

---

## 6. 两种完成度方案对比

### 方案 A · 保守（只证明可行性）

- 只写 chap3 + chap4 在 Webots 上的完整验证
- 实物部分只写**架构设计** + chap3 单机一次实物 run（现有代码骨架够用）
- chap4 实物留作 future work

**工作量**: ~4-5 天（主要是写 chap4 Webots 代码 + 跑实验）
**风险**: 实物部分只有单机可能被评审挑"多机实物缺失"

### 方案 B · 完整（承诺的全部做）

- chap3 + chap4 在 Webots 全跑
- chap3 + chap4 在 E-puck 实物全跑

**工作量**: ~10-14 天（多机实物是最大不确定）
**风险**: 多机实物时间控制差，可能影响论文整体进度

### 🎯 推荐

基于当前 S1 还在跑 + 时间成本，建议 **方案 A 加强版**：
- Webots 上 chap3 + chap4 完整跑通（把 chap4 移植到 Webots 是重点）
- 实物上 chap3 单机 + **2 机**编队（N=2 比 N=3 简单，少一个机器人的定位问题）
- chap4 N=3 实物作为扩展讨论

---

## 7. 对接当前工作

- **chap5 这一重写不急于立刻做**：S1 medium sweep 在跑，先等 chap4 数据稳定
- **修改时机**：S1 + S2 完成、chap4 图表定稿之后（预计 2026-04-22 之后）
- **章节间同步点**：
  - chap3 末尾补"→ 第 5.2.3 节在 Webots 上部署"
  - chap4 末尾补"→ 第 5.2.4 节在 Webots 上部署"
  - chap5 引言回引 chap3+chap4

---

## 8. 相关文件

```
safe-rl-manifold-suite/paper-safeMARL/
├─ contents/chap5.tex                      ← 要改的文件
├─ chap5-revision-plan.md                  ← 本文件
├─ chap5-design.md (在 chap4-local)         ← 既有设计笔记, 可参考
├─ chap4-rq-deliverables.md                ← chap4 RQ 映射
└─ images/
   ├─ platform_framework.png                ← 保留
   ├─ platform data stream.png               ← 保留
   ├─ (新增) deploy_pipeline.png             ← 待绘
   ├─ (新增) webots_chap3_trajectory.png    ← 待跑
   ├─ (新增) webots_chap4_trajectory.png    ← 待跑
   └─ ...

safe-rl-manifold-suite/cosmos/
├─ envs/webots_wrapper.py                  ← 单机 ok, 多机要补
├─ envs/epuck_visualizer.py                ← 能用
└─ ros2/
   ├─ worlds/epuck_arena.wbt                ← 场景已建
   └─ epuck_formation/                       ← 骨架, 要补核心 node
```
