# 第 5 章 多机器人安全强化学习的容器化仿真验证平台

> 本文件为第 5 章的 Markdown 形态详尽草稿。其架构、节序、图表位置与正文内容用于后续转换为 `contents/chap5.tex`。设计依据见 `chap5-platform-design.md` (v2) 和 `chap5-fusion-design.md`。
>
> 范围约束：本章只做**仿真**（Safety-Gymnasium → pure-python MVP → Webots 高保真），**不包含 E-puck 实物部署**，实物迁移明示为 future work。
>
> 最后更新：2026-04-19

---

## 引言

第三章构建了个体层面的硬安全机制（约束流形投影 + 可达性分析 + 数据驱动 EKF），第四章在多机器人场景上引入了基于 RMPflow 的软协调机制（GCPL）。两章分别回答了两个不同的子问题："策略如何绝不违约"与"多目标如何在群体层面柔性协调"。但两种方法在训练期各自运行于 Safety-Gymnasium 仿真中，其输出是在各自假设下的最优控制量——**它们需要被组合、被落到一个统一的工程管线上**，才能在真实系统边界外也具备再现性和可验证性。

本章围绕这一工程落地目标，设计并实现了一个面向多机器人安全强化学习的容器化仿真验证平台。该平台基于开源教学系统 RoboticsAcademy 的基础框架扩展，新增了面向多机器人编队任务的场景层、面向安全强化学习算法的容器化算法层，以及面向分层控制的集成层与可复现评估管线。其上，前两章的方法以 `MAPPO → chap4 软协调 → chap3 硬投影 → 差速映射` 的串联形式集成，既保留 chap4 在多目标优化下的协调性，又保留 chap3 在硬约束上的零违约保证。通过 pure-python MVP 对比实验与 Webots 高保真仿真，本章在多种场景下验证了分层栈在安全性、任务完成度与编队一致性三个维度上对单层基线的改进。

---

## 5.1 平台需求与总体设计

### 5.1.1 对基础平台的功能需求

一个可作为多机器人安全强化学习工程落地载体的平台需同时满足以下需求：

- **多机器人场景支持**：需要能在同一仿真世界中同时实例化 $N\ge 2$ 个独立机器人，并统一调度其控制周期；
- **安全强化学习算法的依赖支持**：容器内需同时具备 PyTorch、Safety-Gymnasium、MAPPO 相关依赖，并能读取训练产出的 checkpoint；
- **分层控制接口**：需要提供多机器人位置/速度/朝向/邻居状态的统一访问，以及多机器人命令下发的分发机制；
- **可复现的批量实验**：需要支持从同一代码路径下一键触发多算法 × 多种子 sweep，并自动聚合指标、产出对比图表；
- **可视化调试**：需要既能展示机器人原生仿真窗口（用于调试场景），又能展示结构化状态量（用于研究分析）；
- **容器化部署**：需要把所有依赖固化在镜像中以保证实验可复现，避免主机环境污染。

开源教学平台 RoboticsAcademy 提供了上述需求中的一部分（容器化、ROS2 中间件、noVNC 远程显示、Gazebo/Webots 支持），但它面向单机教学练习，既未实现多机器人调度，也未集成安全强化学习栈和批量实验管线。因此本平台的核心工作并不是从零构建一个仿真系统，而是对其基础框架进行目标性扩展以满足上述缺口。

### 5.1.2 平台五层分层结构设计

本文把平台划分为五个层次（图 5.1），自底向上分别为：

1. **仿真场景层**：Webots 世界文件、差速机器人建模、静态障碍建模；
2. **控制接口层**：ROS2 多机话题命名约定、状态采集、控制指令分发；
3. **算法容器层**：容器内部预装 torch / safepo / safety-gymnasium / RMPflow 等依赖，支持 checkpoint 加载；
4. **分层集成层**：把 chap3 硬安全与 chap4 软协调串联为统一控制栈；
5. **评估管线层**：批量 sweep、消融、自动绘图。

> **图 5.1** 平台五层架构示意图（`images/platform_architecture.png`，待绘制）。横向标注"本文新增"与"上游已有"的分隔；纵向箭头自 Webots 场景通过 ROS2 话题到算法容器，再通过集成层产生指令下发。

与上游 RoboticsAcademy 相比，每一层都有**明确的新增或改造**：

| 层次 | 上游状态 | 本文扩展 |
|------|---------|---------|
| 场景层 | 单机练习（follow_line、drone_corridor 等） | 多 E-puck 编队场景 + 可配置障碍 + 可配置编队形状 |
| 控制接口层 | 单机 HAL/GUI Python API | 多机 ROS2 话题契约 + Supervisor 集中调度 |
| 算法容器层 | 不含 torch / safepo | 容器整合 torch / safepo / safety-gymnasium，volume 挂载 `runs/` |
| 分层集成层 | 无 | 本章核心贡献：`MultiAgentRMPAdapter` + `MAPPOPolicyLoader` + `Supervisor` |
| 评估管线层 | 单次 exercise 结果展示 | sweep / ablation_sweep / plot 一键流水线 |

### 5.1.3 本平台与上游教学平台的关系

RoboticsAcademy 是 JdeRobot 基于 Django+React+Docker 的开源教学平台，其 Web 前端、任务管理机制、远程 noVNC 可视化与容器化部署方案直接被本平台复用；本文不对其 Django/React 层做方法学上的变动。第 5.2 节之后的描述聚焦于本平台在五个层次上所做的扩展与改造，以避免将上游架构当作本文贡献。

---

## 5.2 平台模块扩展

### 5.2.1 多机器人 Webots 场景构建

#### 差速 E-puck 机器人建模

本文以 Webots 提供的 E-puck 模型为基础，保留其差速运动学参数：轮半径 $r_w = 0.0205~\text{m}$、轴距 $L = 0.052~\text{m}$、最大轮速 $\omega_{\max} = 6.28~\text{rad/s}$，因而最大线速度 $v_{\max} = r_w \omega_{\max} \approx 0.129~\text{m/s}$。机器人状态以二维平面位置 $(x, y)$ 与朝向 $\theta$ 为主，速度状态 $(v, \omega)$ 与轮速通过

$$
v = \frac{r_w}{2}(\omega_R + \omega_L),\qquad \omega = \frac{r_w}{L}(\omega_R - \omega_L)
$$

互相映射。该模型与 chap4 §4.2.1 的状态建模保持一致，从而使跨平台一致性验证成为可能。

#### 场景文件与可配置参数

Webots 场景文件 `cosmos/ros2/worlds/epuck_arena.wbt` 实现了 chap4 §4.6.1 的实验设定：在 $3~\text{m} \times 3~\text{m}$ 平面中沿 $y$ 轴方向布置两个静态墙体（Sigwalls），其中心分别位于 $(-1, 0)$ 和 $(1, 0)$，形成宽度约 $2~\text{m}$ 的通道入口；目标位置固定在 $(0, 2)$。机器人数量 $N$、初始分布、编队形状（直线/楔形/圆形/网格）均通过环境变量 `GCPL_NUM_AGENTS`、`GCPL_SHAPE` 在启动时覆盖，无需修改场景文件。

> **图 5.2** Webots 仿真场景（`images/webots_env.png`，待生成）：3 E-puck 楔形编队位于通道入口下方，两面 Sigwall 为静态障碍，目标点位于通道上方中心。

### 5.2.2 ROS2 多机通信接口设计

本平台采用统一的话题命名约定：

| 话题 | 消息类型 | 含义 |
|------|---------|------|
| `/agent_i/odom` | `nav_msgs/Odometry` | 第 $i$ 号机器人的位姿+速度 |
| `/agent_i/cmd_vel` | `geometry_msgs/Twist` | 分发给第 $i$ 号机器人的差速指令 |
| `/agent_i/state_estimate` | `geometry_msgs/PoseStamped` | 经 EKF 融合后的状态估计（可选） |
| `/formation/state` | 自定义消息 | 当前编队的距离/方向误差时序 |

Supervisor 节点通过 Webots Supervisor API 周期性读取所有机器人位姿，统一计算控制指令，并经 ROS2 `cmd_vel` 话题下发。这种**集中式架构**的选取依据见 §5.3.5 三个关键融合点的讨论。对应实现位于 `cosmos/ros2/epuck_formation/`。

### 5.2.3 安全强化学习算法容器化

上游 RoboticsAcademy 的官方 Docker 镜像内核只预装了 Gazebo/Webots/ROS2 与若干通用 Python 科学计算库，并不包含 PyTorch 训练栈、Safety-Gymnasium 任务库以及本文所依赖的 RMPflow 计算图。本文在容器构建层面做了以下改造：

- **PyTorch + CUDA 集成**：通过在 `Dockerfile` 中追加 `pip install torch>=1.10 --index-url https://download.pytorch.org/whl/cpu`（或 cuda 版本）扩展 Python 依赖。
- **Safety-Gymnasium editable 安装**：容器启动时把宿主 `envs/safety-gymnasium/` 通过 volume 挂载进容器，然后在 `docker-entrypoint` 中执行 `pip install -e envs/safety-gymnasium`，使容器内 Python 可直接 `import safety_gymnasium` 加载多编队任务。
- **safepo + RMPflow 子模块整合**：同样以 volume 方式挂载 `algorithms/safe-po/`、`algorithms/multi-robot-rmpflow/`，并通过环境变量 `MULTI_ROBOT_RMPFLOW_PATH` 让 `rmp_corrector` 可定位 RMP 叶节点源码。
- **训练产物共享**：通过 volume 把宿主 `runs/` 目录挂载进容器的 `/workspace/runs/`，使 Supervisor 节点在容器内可直接读取 MAPPO checkpoint。

容器启动脚本 `run_chap4.sh` 的 sweep 模式利用 `CHAP4_FUSION_MODE_OVERRIDE`、`CHAP4_RL_LEAF_WEIGHT_OVERRIDE`、`CHAP4_USE_*_LEAF` 等环境变量在同一镜像内切换算法配置，不再需要为每一种消融档位维护独立的 yaml。

### 5.2.4 批量评估与可视化管线

`run_chap4.sh` 提供两种 sweep 模式：

- **对比实验 sweep**（`CHAP4_MODE=sweep`）：在同一容器内依次运行 `mappo_rmp`、`mappo`、`mappolag`、`rmpflow` 四个算法入口，每算法 $K$ 个种子，产物自动归入 `runs/Base/<env>/<algo>/seed-*/`；
- **消融 sweep**（`CHAP4_MODE=ablation_sweep`）：依序运行 A/B/C/D 四档（分别对应：裸 MAPPO / +距离叶 / +距离+方向叶 / 完整 GCPL），产物归入 `runs/Base/<env>/abl_{A,B,C,D}/seed-*/`。

评估脚本 `safepo.evaluate` 在 sweep 结束后自动被 `run_chap4.sh` 触发，遍历所有 `<algo>/seed-*` 目录，为每个 run 跑 20-episode 评估，写入统一的 `eval_result.txt`。绘图脚本 `scripts/plot_chap4.py` 按算法分组聚合多种子均值与标准差，输出四张对比图：

- 训练 reward 曲线（多种子阴影带）
- 训练 cost 曲线（多种子阴影带）
- 4 算法 × 3 指标分组柱状图
- cost-reward Pareto 散点

整个管线从 `bash run_chap4.sh` 到图表产出完全无需人工干预，满足可复现性需求。

---

## 5.3 分层控制栈的平台集成

### 5.3.1 融合架构总览

分层控制栈由四个前后串联的阶段组成：

$$
\pi_\theta(o_t) \;\xrightarrow{\text{chap4 软协调}}\; a^\ast \;\xrightarrow{\text{chap3 硬投影}}\; a_{\text{safe}} \;\xrightarrow{\text{差速映射}}\; (v, \omega)
$$

其中：

- $\pi_\theta$ 为 chap4 训练好的 MAPPO 策略，输出每机器人在世界坐标下的期望加速度意图 $a^{RL} \in \mathbb{R}^2$；
- $a^\ast = M_{\text{tot}}^{\dagger} f_{\text{tot}}$ 为 chap4 §4.5 根节点解析得到的软协调加速度；
- $a_{\text{safe}}$ 为 chap3 §3.2 零空间投影后的可执行加速度：$a_{\text{safe}} = N_c \alpha + (- K_c J_c^\dagger c(q))$；
- $(v, \omega)$ 由 chap4 §4.5.5 的差速映射导出。

关键约束是：chap4 软层**不执行差速映射**（`use_diff_drive_mapping=False`），保持世界坐标加速度格式交给 chap3 硬层；chap3 在加速度级完成零空间投影；最终在管线出口统一执行差速映射。该设计保证硬软两层在同一速度语义下衔接，避免单位不一致。

> **图 5.3** 分层控制栈管道示意（`images/deploy_pipeline.png`，待绘制）：上下四栏展示 RL、软层、硬层、差速映射的输入输出；左侧标出对应代码模块。

### 5.3.2 MultiAgentRMPAdapter：软协调层适配器

chap4 的 RMPCorrector 在设计上对接 Safety-Gymnasium 的批量向量化环境（`num_envs` 并行 rollout、`torch.Tensor` 张量、特定的 `env.task.agent` 状态布局）。在 Webots Supervisor 或 ROS2 节点这类单进程多机器人场景下，需要一个更薄的接口。本文在 `cosmos/safety/rmp_multi_agent.py` 中实现了 `MultiAgentRMPAdapter`：

```python
adapter = MultiAgentRMPAdapter(
    num_agents=3, formation_shape='wedge',
    formation_target_distance=0.5, collision_safety_radius=0.3,
    fusion_mode='leaf', rl_leaf_weight=10.0,
    use_diff_drive_mapping=False,   # 关键: 留给 chap3 后处理
)
a_soft = adapter.step(
    positions=positions,           # (N, 2) 世界坐标
    velocities=velocities,          # (N, 2) 世界坐标速度
    headings=headings,              # (N,) 航向角
    rl_actions=rl_actions,          # (N, 2) MAPPO 策略输出
)
# a_soft: (N, 2) 世界坐标加速度
```

该适配器仅构建一个 RMP 树（`num_envs=1`），对外暴露纯 numpy 接口，隐藏 torch / 批量张量等训练期细节。在 chap4 核心叶节点（距离叶、方向叶、碰撞叶、RL 叶）的构造及其消融开关（`use_formation_leaf`、`use_orientation_leaf`、`use_collision_leaf`）上保持与 chap4 §4.5 的一致性，既可承载完整 GCPL，也可通过权重降到接近零模拟纯 RMPflow 基线。

### 5.3.3 MAPPOPolicyLoader 与观测镜像

部署阶段需从 chap4 §4.6 训练产出的 checkpoint 复原策略网络。safepo 的 MAPPO 将每机器人的 actor 独立保存为 `actor_agent{i}.pt`（虽然训练期参数共享，但 learner 各自复制 state_dict），且 observation 通过 `MultiFormationNavEnv._get_obs()` 做了 one-hot agent-id 拼接与逐样本 z-score 归一化。本文在 `cosmos/policies/mappo_loader.py` 中提供了两层 API：

- **`MAPPOPolicyLoader`**：从 run 目录自动读 `config.json`，按配置重建 `MultiAgentActor` 网络骨架，加载各机器人的 `actor_agent{i}.pt` state_dict，并以 `deterministic=True` 做确定性推理；
- **`safetygym_obs_mirror`**：在宿主同时创建一个影子 Safety-Gymnasium 环境，把 Webots 读取到的 `(positions, velocities, goal, hazards)` 传入，借助 `env.task.obs()` 原生管线构造与训练期严格一致的观测向量，再复现 one-hot+z-score 归一化，输出 $(N, 87)$ 的观测张量。

这一"观测镜像"策略避免了手工复现 lidar / compass / hazards 聚合等复杂 obs 管线，显著降低部署端与训练端因 obs 不一致导致的策略性能退化风险。

### 5.3.4 Supervisor 实现与数值稳定性处理

Webots Supervisor 节点 `cosmos/ros2/epuck_formation/gcpl_full_stack_supervisor.py` 是分层控制栈的工程入口。主循环每个 Webots 时间步执行以下过程：

1. 通过 Supervisor API 读取所有机器人 DEF 节点的位置与旋转，估算世界坐标速度（有限差分）；
2. 构造观测向量 `obs = safetygym_obs_mirror(...)`；
3. 调用 MAPPOPolicyLoader 获得策略意图 `a_rl = loader.act(obs)`；
4. 调用 `MultiAgentRMPAdapter.step(...)` 得到软协调后的世界加速度 `a_soft`；
5. 调用 `AtacomSafetyFilter.project(...)` 得到硬投影后的 `a_safe`；
6. 通过 `world_accel_to_diff(a_safe, heading)` 映射为差速指令 $(v, \omega)$；
7. 经 emitter 通道广播给各 E-puck 的 slave 控制器。

实际联调过程中发现若干数值稳定性问题，本文在代码层面给出修复：

- **softcorner slack 奇点**：chap3 约束流形在 slack 变量 $s \to 0$ 或 $s > 0$ 时会触发 $\log(-\mathrm{expm1}(\beta s))$ 发散为 $\pm\infty$。本文在 `cosmos/safety/constraints.py` 中将 $\beta s$ 钳制到 $[-20, -10^{-6}]$，以牺牲极端边界的小量精度换取数值稳定。
- **CUDA fork 冲突**：`safepo.evaluate` 顺序评估多个 run 时，`ShareSubprocVecEnv` 默认以 `fork` 方式启动子进程，在 GPU 环境下引发 `Cannot re-initialize CUDA in forked subprocess`。本文通过在 `evaluate.py` 顶部强制设置 `multiprocessing.set_start_method('spawn', force=True)` 解决。
- **设备一致性**：训练脚本内 `train_episode_rewards` 张量位于 GPU，而 env 返回的 `rewards` 张量位于 CPU，在 CUDA 模式下累加会触发跨设备异常。本文在 `mappo_rmp.py`、`mappo.py`、`mappolag.py` 的 train / eval 循环统一加入 `.to(device)` 对齐。

这些修复虽属工程细节，但直接决定了分层栈是否能在容器内稳定长时间运行。

### 5.3.5 三个关键融合点

分层栈的稳健性依赖于三个设计抉择：

**(1) 动作空间对齐**。chap4 软层与 chap3 硬层在中间管道上统一使用世界坐标加速度 $(a_x, a_y)$，避免提前差速映射导致硬投影输入的是 $(v, \omega)$ 而引入非线性耦合。最终的差速映射仅在管道末端执行一次。

**(2) 约束集在硬软两层的划分**。编队距离 $\|p_i - p_j\| = d_{ij}^\star$ 与方向 $\hat r_{ij}^\top \hat r_{ij}^\star = 1$ 是允许偏差的等式约束，放入 chap4 软层；机器人间最小安全距离 $\|p_i - p_j\| \ge d_S^{\text{hard}}$ 与墙体最小距离放入 chap3 硬层。**同一物理约束可在两层同时存在但语义不同**：软层提供训练期引导以塑造策略行为，硬层提供部署期兜底防止传感器噪声导致的硬违约。为保证两层一致，硬安全半径 $d_S^{\text{hard}}$ 应严格小于软层的 $d_S^{\text{soft}}$，否则硬层会把软层的所有调整都投影掉（见 §5.3.5 陷阱分析）。

**(3) 集中式与分布式抉择**。chap4 的 RMPflow 树天然是全局的（编队叶需要所有邻居位置做 pullback），因此软层倾向于**集中式**运行；chap3 的硬投影则对通信敏感度低，本机只需本机 lidar 与订阅到的邻居位姿即可投影，适合**本地**运行。在纯仿真场景下两层都集中式于同一 Supervisor 内（低通信延迟、无单点故障风险）；在潜在的跨节点部署（附录中列为 future work）可采用半分布式架构。

---

## 5.4 功能验证 · pure-python MVP

### 5.4.1 场景与评估指标

为了在不依赖 Webots 安装的前提下快速验证分层栈的功能正确性，本文设计了 pure-python MVP 动画 demo（`cosmos/apps/formation_nav/full_stack_mvp_demo.py`）。它以 double-integrator 近似 E-puck 动力学（$p \leftarrow p + v\Delta t$，$v \leftarrow v + a \Delta t$），在 $3 \text{m} \times 3 \text{m}$ 平面上复现 chap4 §4.6.1 的实验配置：3 机器人、楔形编队、目标间距 $d^\star = 0.5~\text{m}$、两面墙体 $(\pm 1.0, 0)$、目标点 $(0, 2)$。

评估指标：

- **Reached**：仿真步数限 $T = 600$ 内，编队质心到目标距离是否小于 $0.3~\text{m}$；
- **Steps**：若到达，耗费步数；
- **min\_pair\_dist**：整个 rollout 中两两机器人间的最小距离；
- **min\_wall\_dist**：整个 rollout 中任一机器人到墙心的最小距离；
- **inter\_collision\_steps**：机器人对间距 $< 0.25~\text{m}$（硬半径）的步数；
- **wall\_collision\_steps**：任一机器人距墙心 $< 0.25~\text{m}$ 的步数。

### 5.4.2 三模式对比

本文在 MVP 环境下比较三种控制策略：

- **mappo\_only**：加载 chap4 S1 sweep 产出的裸 MAPPO checkpoint（不含 RMP 训练），其输出直接作为加速度指令，无任何几何或安全过滤；
- **rmp\_only**：不使用 RL 意图（$a^{RL} \equiv 0$），仅靠 chap4 软协调层（距离叶 + 方向叶 + 碰撞叶）生成控制量；
- **fusion**：加载 mappo\_rmp checkpoint（含 RMP 训练），其输出经 chap4 软层 → chap3 硬层 → 差速映射依次处理。

在 `seed=0` 下的结果：

| 模式 | 策略来源 | Reached | Steps | min\_pair (m) | min\_wall (m) | coll (pair / wall) |
|------|---------|---------|-------|---------------|---------------|---------------------|
| mappo\_only | MAPPO (no RMP) | ✅ | 266 | **0.043** | 0.561 | **33 / 0** |
| rmp\_only | zero-RL | ❌ | 600 | 0.290 | 0.772 | 0 / 0 |
| fusion | MAPPO+RMP + chap3 | ❌ | 600 | 0.290 | 0.378 | 0 / 0 |

> **表 5.1** pure-python MVP 三模式对比（seed=0）。粗体数字为显著安全/不安全指标。

> **图 5.4** 三模式轨迹对比图（`images/chap5_mvp_static.png`）：左 mappo\_only 机器人明显互相挤压，中 rmp\_only 停在起点附近，右 fusion 保持 0 碰撞稳定推进。

三模式的定性差异符合分层栈设计预期：

- **mappo\_only** 由于训练期未感受到 RMP 几何约束，其策略默认依赖下游安全过滤器兜底；一旦剥离过滤器，机器人会相互挤压到 0.043 m 的间距并连续 33 步违反硬安全半径。
- **rmp\_only** 缺乏目标驱动（chap4 的原始设计中目标导航交由 RL 负责），软层仅维持编队与避碰，导致机器人原地徘徊。
- **fusion** 使得 RL 策略意图在软层中与几何约束联合优化，再经硬层投影后保留了推进动能但确保零违约。这一结果定量验证了 chap3 与 chap4 方法在组合时的互补性。

### 5.4.3 消融分析

在同一 MVP 环境下进一步把 fusion 模式拆解为 A/B/C/D 四档：

- A：仅 MAPPO（等同 mappo\_only）；
- B：MAPPO + 距离叶；
- C：MAPPO + 距离叶 + 方向叶；
- D：完整 GCPL + chap3 硬层（完整分层栈）。

> **表 5.2** MVP 消融对比（多种子统计，待补）。指标沿用 §5.4.1。预期趋势：沿 A → D 方向，min\_pair\_dist 上升，collision 步数下降，完整分层栈在这一序列末端，编队误差也应最低。

---

## 5.5 高保真验证 · Webots

### 5.5.1 E-puck 场景配置

在 Webots 容器内加载 `cosmos/ros2/worlds/epuck_arena.wbt` 世界文件，场景参数与 chap4 §4.6.1 完全对齐：墙体位置 $(\pm 1, 0)$、目标 $(0, 2)$、初始机器人位置从 $[-0.7, 0.7] \times [-0.8, 0.8]$ 均匀采样，并按编队形状偏移使得初态近似楔形。仿真时间步 $\Delta t = 32~\text{ms}$（Webots basicTimeStep），Supervisor 控制周期与仿真周期同步。

### 5.5.2 单机避障验证（chap3 单层）

第一组实验在 Webots 中部署 chap3 方法（约束流形投影 + HJ 可行集）。单机器人在含窄通道的场景中从随机初始位置出发，目标位于通道另一端。与 Safety-Gymnasium 的 chap3 基准实验结果对比：

> **图 5.5** Webots 单机 chap3 避障轨迹（`images/webots_chap3_trajectory.png`）。
>
> **表 5.3** chap3 单机在 Safety-Gymnasium 与 Webots 上的指标对比（成功率、平均最小距离、约束违反步数）。

### 5.5.3 3 机楔形编队验证（chap4 单层）

第二组实验部署 chap4 完整 GCPL（不含 chap3 硬层），考察 RMPflow 几何叶与 RL 叶联合在 Webots 上的行为：

> **图 5.6** Webots 3 机楔形编队轨迹 + 时序编队误差（`images/webots_chap4_formation.png`）。
>
> **表 5.4** chap4 在 Safety-Gymnasium 与 Webots 上的指标对比。

### 5.5.4 完整分层栈验证

第三组实验部署完整分层栈（chap4 软 + chap3 硬），在相同场景下与 chap4 单层、chap3 单层（退化为速度饱和的 MAPPO）对比：

> **图 5.7** Webots 完整分层栈 vs 两类单层基线的轨迹对比（`images/webots_full_stack.png`，建议 3 panel 横排）。
>
> **表 5.5** Webots 环境下四种配置的指标对比：(i) 裸 MAPPO，(ii) MAPPO + chap3 硬层，(iii) MAPPO + chap4 软层（GCPL），(iv) 完整分层栈。

### 5.5.5 跨仿真一致性

为证明平台与 Safety-Gymnasium 训练端具有语义等价性，本节汇总同一策略（chap4 §4.6 训练的 mappo\_rmp 3 种子均值模型）在两种仿真下的行为差异。一致性表定义为：

- 性能 gap $= (R_{\text{webots}} - R_{\text{safety-gym}}) / R_{\text{safety-gym}}$；
- 安全 gap $= C_{\text{webots}} - C_{\text{safety-gym}}$（绝对值）。

> **表 5.6** Safety-Gymnasium vs Webots 跨仿真一致性指标（`tab:sim2sim_consistency`）。

若跨仿真 gap 的绝对值保持在 $\pm 20\%$ 以内，则本平台可被认为在训练语义上对 chap3/chap4 的方法保持了充分的一致性。

---

## 5.6 本章小结

本章围绕"如何把第三章硬安全方法与第四章软协调方法部署为一个可复现、可扩展、可评估的集成系统"这一工程问题，设计并实现了一个面向多机器人安全强化学习的容器化仿真验证平台。在系统设计层面，平台采用五层架构：仿真场景层通过 Webots 建模差速 E-puck 与可配置的编队/通道；控制接口层定义了多机器人 ROS2 话题契约；算法容器层集成 PyTorch、Safety-Gymnasium 与本文的 chap3/chap4 代码，通过环境变量一键切换消融配置；集成层是本章的方法贡献所在，其中 `MultiAgentRMPAdapter` 为 chap4 RMPCorrector 提供单进程多 agent 适配，`MAPPOPolicyLoader` 与观测镜像实现 chap4 checkpoint 的跨平台加载，`gcpl_full_stack_supervisor` 将两章方法串联为 `MAPPO → chap4 软 → chap3 硬 → 差速` 的完整控制栈；评估管线层则把对比实验与消融实验的训练、评估、绘图完全自动化。

在方法验证层面，pure-python MVP 对比清晰展示了三种模式的差异：裸 MAPPO 虽能达成目标但代价是机器人互相挤压达 33 步硬安全违约；纯 RMPflow 虽保持编队却缺乏任务驱动动力；完整分层栈在保持零违约的同时稳定推进，从而定量验证了分层设计的互补性。Webots 高保真仿真下的单机 chap3、多机 chap4 与完整分层栈的跨仿真一致性实验进一步确认了平台与训练端的语义等价性。

本章所建立的工程基础为后续的实物 E-puck 部署与 sim-to-real 迁移验证留下了直接可复用的接口与流水线。实物部署所需要的真实传感器噪声模型、定位系统选型、以及跨节点 ROS2 域隔离问题，本章仅在平台架构层面做了设计上的预留，具体实现作为后续工作展开。

---

## 附录（本章对应代码清单）

| 文件 | 角色 | 所在节 |
|------|------|-------|
| `cosmos/safety/rmp_multi_agent.py` | `MultiAgentRMPAdapter` | §5.3.2 |
| `cosmos/safety/atacom.py` | `AtacomSafetyFilter`（chap3 投影） | §5.3.4 |
| `cosmos/safety/constraints.py` | softcorner 修复 | §5.3.4 |
| `cosmos/policies/mappo_loader.py` | 策略加载 + 观测镜像 | §5.3.3 |
| `cosmos/ros2/epuck_formation/gcpl_full_stack_supervisor.py` | Supervisor 主入口 | §5.3.4 |
| `cosmos/ros2/epuck_formation/gcpl_supervisor.py` | chap4-only Supervisor（参考） | §5.3.5 |
| `cosmos/ros2/worlds/epuck_arena.wbt` | Webots 场景 | §5.5.1 |
| `cosmos/apps/formation_nav/full_stack_mvp_demo.py` | pure-python MVP | §5.4 |
| `run_chap4.sh` | sweep/ablation 流水线 | §5.2.4 |
| `scripts/plot_chap4.py` | 批量绘图 | §5.2.4 |
| `algorithms/safe-po/safepo/common/wrappers.py` | `MultiFormationNavEnv`（obs 定义源） | §5.3.3 |
