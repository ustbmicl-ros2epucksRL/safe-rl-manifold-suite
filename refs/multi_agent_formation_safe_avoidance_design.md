# 多智能体编队安全避障：需求与设计方案

> 基于 Safety-Gymnasium MA + RMPflow + ATACOM 的 MuJoCo 多智能体编队
> USTB MICL Lab · 硕士论文研究

---

## 1. 问题定义与研究目标

### 1.1 核心问题

在 MuJoCo 物理仿真环境中，N 个独立 Point 智能体需要同时完成以下耦合任务：

1. **编队控制（Formation Control）**：智能体之间维持期望的几何构型（如三角形、正方形、V 形等）
2. **安全避障（Safe Obstacle Avoidance）**：智能体在运动过程中避免与静态障碍物和其他智能体发生碰撞
3. **目标导航（Goal Navigation）**：编队整体向目标区域移动，或各智能体到达各自目标位置
4. **约束满足保证（Constraint Satisfaction Guarantee）**：上述安全约束在训练和部署过程中均需严格满足（ATACOM 硬约束）

### 1.2 技术路线

**MAPPO（策略学习）+ ATACOM（硬安全投影）+ RMPflow（几何基础）**，基于 Safety-Gymnasium 多智能体框架扩展实现，直接在 MuJoCo 中完成全部测试。

### 1.3 实现基础：Safety-Gymnasium MA

本方案不从零构建，而是基于 safe-po 中已跑通的多智能体链路做最小扩展：

| 现有模块 | 已有能力 | 本方案扩展 |
|----------|---------|-----------|
| `SafetyPointMultiGoal{0,1,2}-v0` | 2 个独立 Point agent + MuJoCo 障碍物 | agent 数量 2 → N |
| `MultiGoalEnv` (wrappers.py:68) | 多智能体接口封装、obs/action 空间 | → `FormationMultiGoalEnv`：加编队观测 |
| `MultiNavAtacom` (mult_cm.py:7) | ATACOM 碰撞约束 + 障碍物约束 | → `FormationNavAtacom`：加编队距离约束 |
| `mappo_cm.py` | MAPPO + ATACOM 训练循环 | → `mappo_cm_formation.py`：加编队奖励 |
| `ShareDummyVecEnv` | 向量化并行 | 直接复用 |
| `multi-robot-rmpflow` | RMPflow 树结构、叶策略 | Jacobian 几何框架参考 |

**改动量**：3 个继承扩展文件 + 配置文件，不修改任何现有代码。

---

## 2. 需求分析

### 2.1 功能需求

#### FR-1: 编队构型定义

- **FR-1.1**：支持多种编队拓扑（完全图、链式、星形、自定义邻接矩阵）
- **FR-1.2**：支持预定义编队形状（三角形、正方形、V 形、菱形、圆形）
- **FR-1.3**：编队参数可配置（期望间距 `d_ij`、编队半径、旋转角度）
- **FR-1.4**：支持编队切换（运行时从 V 形切换到一字形等）

#### FR-2: 安全约束

- **FR-2.1**：智能体间碰撞避免（`||q_i - q_j|| >= r_safe`，∀i≠j）
- **FR-2.2**：静态障碍物避免（`||q_i - o_k|| >= r_obs`，∀i,∀k）
- **FR-2.3**：边界约束（`q_i ∈ [x_min, x_max] × [y_min, y_max]`）
- **FR-2.4**：速度约束（`||v_i|| <= v_max`）
- **FR-2.5**：约束在整个训练和执行过程中严格满足（硬约束，非惩罚项）

#### FR-3: 任务目标

- **FR-3.1**：编队整体导航到目标区域
- **FR-3.2**：编队在移动过程中保持构型（允许一定弹性变形）
- **FR-3.3**：到达目标时编队误差在容许范围内
- **FR-3.4**：支持领航者-跟随者（Leader-Follower）和分布式共识两种编队模式

#### FR-4: 训练与评估

- **FR-4.1**：基于 MAPPO 的分布式策略学习
- **FR-4.2**：ATACOM 流形投影保证训练全程安全
- **FR-4.3**：支持 TensorBoard / WandB 日志
- **FR-4.4**：提供编队误差、安全违规率、到达率等评估指标

### 2.2 非功能需求

- **NFR-1**：智能体数量可扩展（2~10 个独立 Point agent）
- **NFR-2**：约束求解实时性（单步 < 5ms，支持 1000+ 步 MuJoCo 仿真）
- **NFR-3**：基于 Safety-Gymnasium MA 继承扩展，不修改现有代码
- **NFR-4**：全部实验在 MuJoCo 中完成（与基线共用仿真器）

### 2.3 约束层次

```
约束优先级（从高到低）：
┌─────────────────────────────────┐
│ L1: 碰撞避免（硬约束）            │  ← 绝对不可违反
│   - 智能体间碰撞                 │
│   - 障碍物碰撞                   │
│   - 边界约束                     │
├─────────────────────────────────┤
│ L2: 运动约束（硬约束）            │  ← 物理限制
│   - 速度上限                     │
│   - 加速度上限                   │
├─────────────────────────────────┤
│ L3: 编队保持（软约束→硬约束）      │  ← 可通过 ATACOM 强制
│   - 智能体间距维持               │
│   - 构型形状维持                 │
├─────────────────────────────────┤
│ L4: 目标到达（优化目标）          │  ← 通过奖励驱动
│   - 编队中心向目标移动            │
│   - 最终到达目标                 │
└─────────────────────────────────┘
```

---

## 3. 系统架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop (MAPPO-CM-Formation)       │
│                                                                 │
│  ┌───────────┐    ┌──────────────────┐    ┌──────────────────┐ │
│  │  Policy    │    │  ATACOM           │    │  Environment     │ │
│  │  Network   │───▶│  Formation       │───▶│  (Safety-Gym)    │ │
│  │  (Actor)   │    │  Wrapper          │    │                  │ │
│  │            │◀───│                   │◀───│                  │ │
│  └───────────┘    └──────────────────┘    └──────────────────┘ │
│       │                    │                       │            │
│       ▼                    ▼                       ▼            │
│  ┌───────────┐    ┌──────────────────┐    ┌──────────────────┐ │
│  │  Critic   │    │  Constraint       │    │  Reward          │ │
│  │  Network  │    │  Manager          │    │  Computation     │ │
│  │ (R + C)   │    │  (Formation +     │    │  (Formation +    │ │
│  │           │    │   Safety)         │    │   Navigation)    │ │
│  └───────────┘    └──────────────────┘    └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 模块分层（基于 Safety-Gymnasium MA 继承扩展）

```
safepo/
├── common/
│   ├── constrained_manifold/
│   │   ├── constraints.py          # [已有] StateConstraint, ConstraintsSet
│   │   ├── manifold.py             # [已有] AtacomEnvWrapper
│   │   ├── formation.py            # [新增] FormationTopology, FormationShape,
│   │   │                           #        FormationDistanceConstraint
│   │   └── __init__.py
│   ├── wrappers.py                 # [已有] MultiGoalEnv, ShareEnv
│   ├── mult_cm.py                  # [已有] MultiNavAtacom (碰撞+障碍物约束)
│   ├── formation_cm.py             # [新增] FormationNavAtacom(MultiNavAtacom)
│   │                               #        继承, 新增 _make_formation_f()
│   ├── formation_wrappers.py       # [新增] FormationMultiGoalEnv(MultiGoalEnv)
│   │                               #        继承, N agent + 编队观测
│   └── ...
├── multi_agent/
│   ├── mappo_cm.py                 # [已有] MAPPO-CM 训练循环
│   ├── mappo_cm_formation.py       # [新增] 基于 mappo_cm.py, 加编队奖励
│   └── marl_cfg/
│       └── formation/              # [新增] 编队实验配置
│           ├── triangle_3agents.yaml
│           ├── square_4agents.yaml
│           └── ...

不修改任何已有文件，全部通过继承扩展。
```

### 3.3 数据流

```
每步执行流程：

  Agent_i 策略网络             ATACOM 投影             环境
  ┌─────────────┐          ┌─────────────────┐      ┌──────┐
  │ obs_i → π_i │──α_i──▶ │ 1. 计算约束 Jc   │──a_i─▶│      │
  │   (Gaussian) │         │ 2. 零空间投影 Nc  │      │ step │
  │              │         │ 3. 误差修正 dq_err│      │      │
  │              │         │ 4. 动作截断       │      │      │
  └─────────────┘         └─────────────────┘      └──┬───┘
        ▲                         ▲                    │
        │                         │                    ▼
        │                  ┌──────┴────────┐     ┌──────────┐
        └──────────────────│ 约束管理器     │     │ obs, r,  │
                           │               │     │ cost,    │
                           │ • 碰撞约束     │◀────│ done     │
                           │ • 编队约束     │     └──────────┘
                           │ • 边界约束     │
                           └───────────────┘
```

---

## 4. 核心模块详细设计

### 4.1 编队约束定义（`formation.py`）

#### 4.1.1 编队拓扑（FormationTopology）

```python
class FormationTopology:
    """定义智能体之间的编队连接关系"""

    def __init__(self, num_agents, topology_type='complete'):
        """
        Args:
            num_agents: 智能体数量 N
            topology_type: 拓扑类型
                - 'complete': 完全图，所有对之间有约束
                - 'chain': 链式，相邻对之间有约束
                - 'star': 星形，leader 与所有 follower 之间有约束
                - 'custom': 自定义邻接矩阵
        """
        self.num_agents = num_agents
        self.adjacency = self._build_adjacency(topology_type)
        self.edges = self._extract_edges()  # [(i, j), ...] 无序对列表

    @property
    def num_edges(self):
        return len(self.edges)

    def _build_adjacency(self, topology_type):
        """构建邻接矩阵 A ∈ {0,1}^{N×N}"""
        N = self.num_agents
        if topology_type == 'complete':
            A = np.ones((N, N)) - np.eye(N)
        elif topology_type == 'chain':
            A = np.zeros((N, N))
            for i in range(N - 1):
                A[i, i+1] = A[i+1, i] = 1
        elif topology_type == 'star':
            A = np.zeros((N, N))
            for i in range(1, N):
                A[0, i] = A[i, 0] = 1
        return A

    def _extract_edges(self):
        """提取无向边列表 [(i, j)]，i < j"""
        edges = []
        N = self.num_agents
        for i in range(N):
            for j in range(i+1, N):
                if self.adjacency[i, j] > 0:
                    edges.append((i, j))
        return edges
```

#### 4.1.2 编队形状（FormationShape）

```python
class FormationShape:
    """预定义编队几何构型，计算期望间距矩阵"""

    SHAPES = {
        'triangle': lambda N, r: _regular_polygon(3, r),      # N 取 3
        'square':   lambda N, r: _regular_polygon(4, r),       # N 取 4
        'pentagon':  lambda N, r: _regular_polygon(5, r),      # N 取 5
        'line':     lambda N, r: _line_formation(N, r),        # 一字排列
        'v_shape':  lambda N, r: _v_formation(N, r),           # V 形
        'circle':   lambda N, r: _regular_polygon(N, r),       # 正 N 边形
    }

    def __init__(self, shape_type, num_agents, radius=0.5):
        """
        Args:
            shape_type: 编队形状类型
            num_agents: 智能体数量
            radius: 编队半径（正多边形外接圆半径 / 线阵间距）
        """
        self.positions = self.SHAPES[shape_type](num_agents, radius)
        self.desired_distances = self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        """计算期望间距矩阵 D[i,j] = ||p_i - p_j||"""
        N = len(self.positions)
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                D[i, j] = np.linalg.norm(
                    self.positions[i] - self.positions[j]
                )
        return D

    def get_desired_distance(self, i, j):
        """获取智能体 i 和 j 之间的期望距离"""
        return self.desired_distances[i, j]


def _regular_polygon(n, r):
    """正 n 边形，外接圆半径 r"""
    angles = [2 * np.pi * k / n for k in range(n)]
    return np.array([[r * np.cos(a), r * np.sin(a)] for a in angles])

def _line_formation(N, spacing):
    """一字排列，间距 spacing"""
    return np.array([[i * spacing, 0.0] for i in range(N)])

def _v_formation(N, spacing):
    """V 形编队"""
    positions = []
    positions.append([0.0, 0.0])  # leader at origin
    for i in range(1, N):
        side = 1 if i % 2 == 1 else -1
        rank = (i + 1) // 2
        positions.append([
            -rank * spacing * np.cos(np.pi / 6),  # 后方
             side * rank * spacing * np.sin(np.pi / 6)   # 两侧展开
        ])
    return np.array(positions)
```

#### 4.1.3 编队约束（FormationConstraint）

将编队距离约束集成到 ATACOM 的 `StateConstraint` 框架中：

```python
class FormationDistanceConstraint:
    """
    编队距离约束：维持智能体 i 与智能体 j 之间的期望距离 d_ij

    约束形式（双边不等式，用两个单边不等式表示）：
      c_lower(q_i) = -(||q_i - q_j|| - d_ij) + delta  <= 0   (不能太近)
      c_upper(q_i) = (||q_i - q_j|| - d_ij) - delta    <= 0   (不能太远)

    其中 delta 为容许变形量（弹性带宽）

    或者使用单个等式约束 + slack：
      c(q_i) = ||q_i - q_j|| - d_ij = 0  (通过 slack 变量松弛)
    """

    def __init__(self, agent_idx, other_idx, desired_distance,
                 tolerance=0.1, slack_type='softcorner', slack_beta=10.0):
        """
        Args:
            agent_idx: 当前智能体索引
            other_idx: 编队邻居索引
            desired_distance: 期望间距 d_ij
            tolerance: 容许变形量 delta
            slack_type: 松弛类型
            slack_beta: 松弛参数
        """
        self.agent_idx = agent_idx
        self.other_idx = other_idx
        self.d_ij = desired_distance
        self.tolerance = tolerance

    def build_constraints(self, other_positions_accessor):
        """
        构建 StateConstraint 对象列表

        Args:
            other_positions_accessor: 返回其他智能体当前位置的回调函数

        Returns:
            list[StateConstraint]
        """
        d_ij = self.d_ij
        delta = self.tolerance

        def formation_lower(q, x=None):
            """不能比期望距离近太多"""
            q_j = other_positions_accessor(self.other_idx)
            dist = np.linalg.norm(q[:2] - q_j[:2])
            return -(dist - d_ij) + delta  # <= 0 when dist >= d_ij - delta

        def formation_upper(q, x=None):
            """不能比期望距离远太多"""
            q_j = other_positions_accessor(self.other_idx)
            dist = np.linalg.norm(q[:2] - q_j[:2])
            return (dist - d_ij) - delta  # <= 0 when dist <= d_ij + delta

        def jac_lower(q, x=None):
            q_j = other_positions_accessor(self.other_idx)
            diff = q[:2] - q_j[:2]
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                return np.zeros((1, len(q)))
            J = np.zeros((1, len(q)))
            J[0, :2] = diff / dist  # ∂dist/∂q_i = (q_i - q_j) / ||q_i - q_j||
            return J

        def jac_upper(q, x=None):
            return -jac_lower(q, x)  # 方向相反

        constraints = []

        # 下界约束（不能太近）
        constraints.append(StateConstraint(
            dim_q=3,  # [x, y, theta]
            dim_out=1,
            fun=formation_lower,
            jac_q=jac_lower,
            slack_type=self.slack_type,
            slack_beta=self.slack_beta,
        ))

        # 上界约束（不能太远）
        constraints.append(StateConstraint(
            dim_q=3,
            dim_out=1,
            fun=formation_upper,
            jac_q=jac_upper,
            slack_type=self.slack_type,
            slack_beta=self.slack_beta,
        ))

        return constraints
```

### 4.2 编队 ATACOM 包装器（`formation_cm.py`，继承 MultiNavAtacom）

```python
class FormationNavAtacom(MultiNavAtacom):
    """
    多智能体编队导航的 ATACOM 约束流形包装器

    继承 MultiNavAtacom 已有能力：
    - _make_hazard_f(): 障碍物约束
    - _make_inter_agent_f(): 智能体间碰撞约束
    - step_action_function(): ATACOM 零空间投影

    新增编队扩展：
    - _make_formation_f(): 编队距离约束
    - 编队感知的观测空间扩展
    - 编队奖励信号
    """

    def __init__(self, base_env, num_agents, formation_shape,
                 formation_topology, timestep=0.002, K_c=100,
                 collision_radius=0.3, formation_tolerance=0.1,
                 slack_beta_collision=1.0, slack_beta_formation=10.0):
        """
        Args:
            base_env: 基础多智能体环境
            num_agents: 智能体数量
            formation_shape: FormationShape 实例
            formation_topology: FormationTopology 实例
            timestep: 仿真步长
            K_c: 约束误差反馈增益
            collision_radius: 碰撞安全距离
            formation_tolerance: 编队距离容许偏差
            slack_beta_collision: 碰撞约束松弛参数
            slack_beta_formation: 编队约束松弛参数
        """
        self.env = base_env
        self.num_agents = num_agents
        self.formation = formation_shape
        self.topology = formation_topology
        self.K_c = K_c
        self.dt = timestep

        # 构建每个智能体的约束集
        self.per_agent_constraints = self._build_all_constraints()

        # 每个智能体的 ATACOM 投影器
        self.per_agent_atacom = [
            AtacomEnvWrapper(
                base_env=None,  # 不直接包装环境，手动管理
                constraints=cs,
                step_size=timestep,
                K_c=K_c,
            )
            for cs in self.per_agent_constraints
        ]

        # 状态存储
        self.q = np.zeros((num_agents, 3))  # [x, y, theta] per agent
        self._other_positions = [None] * num_agents

    def _build_all_constraints(self):
        """为每个智能体构建完整约束集"""
        constraints_list = []

        for i in range(self.num_agents):
            cs = ConstraintsSet(dim_q=3, dim_x=0)

            # === L1: 碰撞避免约束 ===

            # 1a. 智能体间碰撞
            for j in range(self.num_agents):
                if i == j:
                    continue
                cs.add_constraint(self._make_inter_agent_collision(
                    agent_i=i, agent_j=j,
                    radius=self.collision_radius
                ))

            # 1b. 静态障碍物碰撞
            for obs_pos, obs_radius in self._get_obstacles():
                cs.add_constraint(self._make_obstacle_collision(
                    obstacle_pos=obs_pos,
                    obstacle_radius=obs_radius
                ))

            # 1c. 边界约束
            for wall_constraint in self._make_boundary_constraints():
                cs.add_constraint(wall_constraint)

            # === L3: 编队距离约束 ===
            for (a, b) in self.topology.edges:
                if a == i or b == i:
                    other = b if a == i else a
                    d_ij = self.formation.get_desired_distance(i, other)
                    fc = FormationDistanceConstraint(
                        agent_idx=i,
                        other_idx=other,
                        desired_distance=d_ij,
                        tolerance=self.formation_tolerance,
                    )
                    for sc in fc.build_constraints(
                        lambda idx: self._other_positions[idx]
                    ):
                        cs.add_constraint(sc)

            constraints_list.append(cs)

        return constraints_list

    def _make_inter_agent_collision(self, agent_i, agent_j, radius):
        """智能体间碰撞约束: -||q_i - q_j|| + 2*radius <= 0"""
        def fun(q, x=None):
            q_j = self._other_positions[agent_j]
            return -np.linalg.norm(q[:2] - q_j[:2]) + 2 * radius

        def jac_q(q, x=None):
            q_j = self._other_positions[agent_j]
            diff = q[:2] - q_j[:2]
            dist = np.linalg.norm(diff)
            J = np.zeros((1, 3))
            if dist > 1e-6:
                J[0, :2] = -diff / dist
            return J

        return StateConstraint(
            dim_q=3, dim_out=1, fun=fun, jac_q=jac_q,
            slack_type='softcorner', slack_beta=self.slack_beta_collision,
        )

    def step(self, actions):
        """
        执行一步：对每个智能体的动作进行 ATACOM 投影后执行

        Args:
            actions: list[np.ndarray]，每个智能体的原始策略输出 α_i

        Returns:
            obs, rewards, costs, dones, infos
        """
        # 1. 更新所有智能体位置
        self.q = self._get_all_positions()
        self._update_other_positions()

        # 2. 对每个智能体进行 ATACOM 投影
        safe_actions = []
        for i in range(self.num_agents):
            alpha_i = actions[i]
            a_i = self._atacom_project(alpha_i, self.q[i], i)
            safe_actions.append(a_i)

        # 3. 执行安全动作
        obs, rewards, costs, dones, infos = self.env.step(safe_actions)

        # 4. 计算编队奖励（叠加到 rewards 上）
        formation_reward = self._compute_formation_reward()
        for i in range(self.num_agents):
            rewards[i] += formation_reward[i]

        # 5. 扩展观测（加入编队信息）
        obs = self._augment_observations(obs)

        return obs, rewards, costs, dones, infos

    def _atacom_project(self, alpha, q_i, agent_idx):
        """
        ATACOM 动作投影：将策略输出投影到约束流形的零空间

        核心公式：
            Jc = [J_q, J_s]                     # 约束 Jacobian
            Jc_inv = Jc^T (Jc Jc^T)^{-1}        # 伪逆
            Nc = (I - Jc_inv Jc)[:, :dim_null]   # 零空间基
            dq_null = Nc @ alpha                 # 零空间动作
            dq_err = -K_c * Jc_inv @ c(q)        # 约束误差修正
            dq = dq_null + dq_err                # 最终安全动作
        """
        cs = self.per_agent_constraints[agent_idx]

        # 约束值与 Jacobian
        c_val = cs.c(q_i)
        J_q, J_x, J_s = cs.get_jacobians(q_i)

        # 构建完整 Jacobian
        Jc = np.hstack([J_q, J_s])

        # 伪逆
        Jc_inv = Jc.T @ np.linalg.inv(Jc @ Jc.T + 1e-8 * np.eye(Jc.shape[0]))

        # 零空间投影
        dim_null = cs.dim_null
        Nc = (np.eye(Jc.shape[1]) - Jc_inv @ Jc)[:, :dim_null]

        # 安全动作
        dq_null = Nc @ (alpha * self.dq_max)
        dq_err = -self.K_c * (Jc_inv @ c_val)
        dq = dq_null + dq_err

        # 截断到动作限制
        scale = 1.0 / max(np.max(np.abs(dq[:3]) / self.action_limit), 1.0)
        dq = dq[:3] * scale

        return dq

    def _compute_formation_reward(self):
        """
        编队保持奖励

        r_formation_i = -w_f * Σ_{j∈N(i)} (||q_i - q_j|| - d_ij)²

        对每个智能体，惩罚与其编队邻居之间的距离偏差
        """
        rewards = np.zeros(self.num_agents)
        w_f = 0.1  # 编队奖励权重

        for i in range(self.num_agents):
            for (a, b) in self.topology.edges:
                if a == i or b == i:
                    j = b if a == i else a
                    dist = np.linalg.norm(self.q[i, :2] - self.q[j, :2])
                    d_ij = self.formation.get_desired_distance(i, j)
                    rewards[i] -= w_f * (dist - d_ij) ** 2

        return rewards

    def _augment_observations(self, obs):
        """
        扩展观测空间，加入编队相关信息

        原始 obs_i 拼接：
        - 到编队邻居的相对位置 (Δx_ij, Δy_ij) for j ∈ N(i)
        - 到编队邻居的距离误差 (||q_i - q_j|| - d_ij) for j ∈ N(i)
        - 编队中心位置（相对于智能体 i）
        """
        augmented = []
        formation_center = np.mean(self.q[:, :2], axis=0)

        for i in range(self.num_agents):
            extra = []

            # 编队邻居相对信息
            for (a, b) in self.topology.edges:
                if a == i or b == i:
                    j = b if a == i else a
                    rel_pos = self.q[j, :2] - self.q[i, :2]
                    d_ij = self.formation.get_desired_distance(i, j)
                    dist_err = np.linalg.norm(rel_pos) - d_ij
                    extra.extend([rel_pos[0], rel_pos[1], dist_err])

            # 编队中心（相对）
            rel_center = formation_center - self.q[i, :2]
            extra.extend([rel_center[0], rel_center[1]])

            augmented.append(np.concatenate([obs[i], np.array(extra)]))

        return augmented
```

### 4.3 奖励函数设计

```python
class FormationRewardComputer:
    """
    多目标奖励函数设计

    总奖励 = w_nav * r_navigation + w_form * r_formation + w_smooth * r_smoothness
    """

    def __init__(self, w_nav=1.0, w_form=0.1, w_smooth=0.01):
        self.w_nav = w_nav
        self.w_form = w_form
        self.w_smooth = w_smooth

    def compute(self, q, q_prev, goal, formation_shape, topology):
        """
        Args:
            q: (N, 3) 当前所有智能体状态
            q_prev: (N, 3) 上一步状态
            goal: (2,) 编队中心目标位置
            formation_shape: FormationShape 实例
            topology: FormationTopology 实例
        """
        rewards = np.zeros(len(q))

        for i in range(len(q)):
            # --- 导航奖励 ---
            # 编队中心到目标的距离变化（鼓励编队整体靠近目标）
            center = np.mean(q[:, :2], axis=0)
            center_prev = np.mean(q_prev[:, :2], axis=0)
            dist_to_goal = np.linalg.norm(center - goal)
            dist_to_goal_prev = np.linalg.norm(center_prev - goal)
            r_nav = dist_to_goal_prev - dist_to_goal  # 正值表示靠近

            # --- 编队保持奖励 ---
            r_form = 0.0
            for (a, b) in topology.edges:
                if a == i or b == i:
                    j = b if a == i else a
                    dist = np.linalg.norm(q[i, :2] - q[j, :2])
                    d_ij = formation_shape.get_desired_distance(i, j)
                    r_form -= (dist - d_ij) ** 2

            # --- 平滑性奖励 ---
            # 惩罚大的加速度变化（鼓励平滑运动）
            velocity = q[i, :2] - q_prev[i, :2]
            r_smooth = -np.linalg.norm(velocity) ** 2

            rewards[i] = (self.w_nav * r_nav
                         + self.w_form * r_form
                         + self.w_smooth * r_smooth)

        return rewards
```

### 4.4 训练脚本设计（`mappo_cm_formation.py`，基于 mappo_cm.py）

```python
"""
MAPPO-CM-Formation: 基于 Safety-Gymnasium MA + ATACOM 的多智能体编队训练

基于 mappo_cm.py 最小改动，仅替换环境创建和追加编队奖励。

用法:
    python mappo_cm_formation.py \
        --task SafetyPointMultiGoal1-v0 \
        --num-agents 4 \
        --formation-shape square \
        --formation-radius 0.5 \
        --topology complete \
        --seed 0

与 mappo_cm.py 的 3 处差异：
1. 环境: FormationMultiGoalEnv (继承 MultiGoalEnv) + FormationNavAtacom (继承 MultiNavAtacom)
2. 奖励: rollout 中追加 compute_formation_reward()
3. 日志: 追加编队误差、编队保持率等指标
"""

def main():
    args = parse_args()

    # 1. 创建基础环境
    base_env = make_multi_agent_env(args.task, args.num_envs)

    # 2. 定义编队
    formation = FormationShape(
        shape_type=args.formation_shape,
        num_agents=args.num_agents,
        radius=args.formation_radius,
    )
    topology = FormationTopology(
        num_agents=args.num_agents,
        topology_type=args.topology,
    )

    # 3. 用 ATACOM 编队包装器包装环境
    env = FormationNavAtacom(
        base_env=base_env,
        num_agents=args.num_agents,
        formation_shape=formation,
        formation_topology=topology,
        timestep=args.timestep,
        K_c=args.K_c,
        collision_radius=args.collision_radius,
        formation_tolerance=args.formation_tolerance,
    )

    # 4. 策略网络（扩展的观测空间）
    obs_dim = env.observation_space.shape[0]  # 包含编队信息
    act_dim = env.action_space.shape[0]
    policy = MultiAgentActorVCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=[128, 128],
        num_agents=args.num_agents,
    )

    # 5. 训练循环（MAPPO with CTDE）
    for epoch in range(args.num_epochs):
        # 收集轨迹
        trajectories = collect_rollouts(env, policy, args.rollout_length)

        # 计算优势估计（GAE）
        compute_advantages(trajectories, gamma=0.99, lam=0.95)

        # PPO 更新（每个智能体独立策略，共享评价者）
        for agent_idx in range(args.num_agents):
            ppo_update(
                policy.actors[agent_idx],
                policy.critic,  # centralized critic
                trajectories[agent_idx],
                clip_ratio=0.2,
                target_kl=0.01,
            )

        # 日志
        log_metrics(epoch, trajectories, formation, topology)
```

---

## 5. 约束耦合与优先级处理

### 5.1 约束冲突解决机制

当碰撞避免约束和编队保持约束冲突时（例如编队要求靠近但碰撞要求远离），需要优先级处理：

```
约束聚合策略：

方案 A：加权 Jacobian 合并（当前采用）
    ┌────────────────────────────────────────┐
    │ Jc = [ w_coll * J_collision ]          │
    │      [ w_form * J_formation ]          │
    │      [ w_wall * J_boundary  ]          │
    │                                        │
    │ 权重设置：                              │
    │   w_coll = 10.0  （最高优先）           │
    │   w_wall = 10.0                        │
    │   w_form = 1.0   （较低优先）           │
    └────────────────────────────────────────┘

    零空间投影自然处理：碰撞约束占据的维度优先保证，
    编队约束在剩余自由度中尽量满足。

方案 B：级联投影（备选，更严格的优先级）
    ┌────────────────────────────────────────┐
    │ Step 1: 先用碰撞约束做零空间投影        │
    │   N_coll = null(J_collision)           │
    │   alpha_1 = N_coll @ alpha             │
    │                                        │
    │ Step 2: 在碰撞安全子空间中做编队投影     │
    │   J_form_proj = J_form @ N_coll        │
    │   N_form = null(J_form_proj)           │
    │   alpha_2 = N_form @ alpha_1           │
    │                                        │
    │ 保证碰撞约束始终满足，编队约束尽力而为   │
    └────────────────────────────────────────┘
```

### 5.2 Slack 变量参数选择

```
约束类型          | slack_type    | slack_beta | threshold | 原因
-----------------|---------------|------------|-----------|-------------------
碰撞避免（智能体）| softcorner    | 30.0       | 0.01      | 硬约束，快速响应
碰撞避免（障碍物）| softcorner    | 30.0       | 0.01      | 硬约束，快速响应
边界约束          | softcorner    | 30.0       | 0.01      | 硬约束
编队距离（下界）  | softplus      | 10.0       | 0.05      | 允许一定弹性
编队距离（上界）  | softplus      | 10.0       | 0.05      | 允许一定弹性
速度约束          | square        | 5.0        | 0.1       | 较软，渐进式
```

---

## 6. 观测空间与动作空间设计

### 6.1 观测空间（每个智能体）

```
obs_i = [
    # === 自身状态 (dim: 5) ===
    x_i, y_i,              # 位置
    vx_i, vy_i,            # 速度
    theta_i,               # 朝向

    # === 目标信息 (dim: 3) ===
    goal_x - x_i,          # 到目标的相对位置
    goal_y - y_i,
    dist_to_goal,          # 到目标的距离

    # === 编队邻居信息 (dim: 3 * |N(i)|) ===
    # 对每个编队邻居 j ∈ N(i)：
    x_j - x_i,             # 相对位置 x
    y_j - y_i,             # 相对位置 y
    ||q_j - q_i|| - d_ij,  # 距离误差

    # === 编队全局信息 (dim: 2) ===
    center_x - x_i,        # 编队中心相对位置 x
    center_y - y_i,        # 编队中心相对位置 y

    # === 障碍物信息 (dim: 3 * K) ===
    # 对每个障碍物 k (最近 K 个)：
    obs_x_k - x_i,         # 障碍物相对位置 x
    obs_y_k - y_i,         # 障碍物相对位置 y
    dist_to_obs_k,          # 到障碍物距离

    # === 其他智能体信息 (dim: 3 * (N-1)) ===
    # 对每个非邻居智能体 m：
    x_m - x_i,
    y_m - y_i,
    dist_to_m,
]

总维度: 5 + 3 + 3*|N(i)| + 2 + 3*K + 3*(N-1-|N(i)|)
示例 (N=4, 完全图, K=3): 5 + 3 + 9 + 2 + 9 + 0 = 28
```

### 6.2 动作空间

```
alpha_i ∈ ℝ^{dim_null}    # ATACOM 零空间维度

dim_null = dim_q + dim_slack - dim_constraints
         = 3 + n_slack - n_constraints

实际输入环境的动作经过 ATACOM 投影：
a_i = ATACOM_project(alpha_i) ∈ ℝ^{dim_q}  # [dx, dy, dtheta]
```

### 6.3 集中式评价者观测（CTDE）

```
share_obs = [
    # 所有智能体状态
    x_1, y_1, vx_1, vy_1, theta_1,
    ...,
    x_N, y_N, vx_N, vy_N, theta_N,

    # 编队全局信息
    center_x, center_y,
    formation_error,  # 编队总误差

    # 目标信息
    goal_x, goal_y,

    # 障碍物信息（全局）
    obs_1_x, obs_1_y, obs_1_r,
    ...,
]
```

---

## 7. 评估指标体系

### 7.1 安全指标

| 指标 | 定义 | 目标 |
|------|------|------|
| 碰撞率 | 任一时刻发生碰撞的 episode 比例 | 0% |
| 约束违反次数 | 整个 episode 中约束值 > 0 的总步数 | 0 |
| 最小安全距离 | min_{i≠j, t} ||q_i(t) - q_j(t)|| | > r_safe |
| 边界违反 | 任一智能体越界的次数 | 0 |

### 7.2 编队指标

| 指标 | 定义 | 目标 |
|------|------|------|
| 平均编队误差 | E_t[Σ_{(i,j)∈E} \|  \|\|q_i - q_j\|\| - d_{ij} \| / \|E\|] | < tolerance |
| 最大编队误差 | max_{(i,j)∈E, t} \|  \|\|q_i - q_j\|\| - d_{ij} \| | < 2 * tolerance |
| 编队收敛时间 | 编队误差首次 < threshold 的时间步 | 越小越好 |
| 编队保持率 | 编队误差 < threshold 的时间步比例 | > 95% |

### 7.3 任务指标

| 指标 | 定义 | 目标 |
|------|------|------|
| 到达率 | 编队中心到达目标区域的 episode 比例 | > 90% |
| 到达时间 | 首次到达目标的平均时间步 | 越小越好 |
| 路径效率 | 最短路径长度 / 实际路径长度 | > 0.8 |
| 累积奖励 | episode 总奖励 | 最大化 |

### 7.4 基线对比（全部在 MuJoCo / Safety-Gymnasium MA 中运行）

```
需要与以下方法对比（共用 SafetyPointMultiGoal 环境）：

1. MAPPO (无安全约束)         — 上界参考（任务性能）    [已有: mappo.py]
2. MAPPO-Lag (拉格朗日法)     — 基于惩罚的安全方法      [已有: mappo_lag.py]
3. MACPO (信任域约束优化)     — 基于约束优化的安全方法   [已有: macpo.py]
4. MAPPO-CM (无编队约束)      — 仅碰撞避免的 ATACOM    [已有: mappo_cm.py]
5. RMPflow (手工策略)         — 纯几何方法，无 RL       [参考: multi-robot-rmpflow]
6. MAPPO-CM-Formation (本方案) — 完整方案              [新增: mappo_cm_formation.py]

优势：基线 1-4 在 safe-po 中已实现，可直接运行对比。
```

---

## 8. 实验场景设计（全部在 MuJoCo 中）

> 所有场景基于 SafetyPointMultiGoal 环境扩展，使用 MuJoCo 物理仿真。

### 8.1 场景一：基础编队导航（Baseline）

```
环境：MuJoCo 空旷区域（无障碍物, SafetyPointMultiGoal0 扩展）
智能体：3 个独立 Point agent（MuJoCo sphere body + slide joints）
编队：等边三角形，间距 0.5
任务：从左侧出发，保持编队移动到右侧目标
目的：验证编队约束的基本效果

     目标区域
        ★
       / \
      /   \
     ●─────●     ←──  编队出发
      \   /
       \ /
        ●
```

### 8.2 场景二：障碍物穿越

```
环境：MuJoCo 走廊 + 障碍物（SafetyPointMultiGoal1 扩展）
智能体：4 个独立 Point agent
编队：正方形，间距 0.4
任务：编队穿越障碍物区域
目的：验证编队与避障的协调（编队需要变形穿过窄道后恢复）

  ████████████████████
  ██                ██
  ██  ●──●    ⬤    ██
  ██  |  |     ⬤   ██
  ██  ●──●  ⬤   ★  ██
  ██                ██
  ████████████████████
  ⬤ = 障碍物    ★ = 目标
```

### 8.3 场景三：多编队交汇

```
环境：MuJoCo 开放区域（SafetyPointMultiGoal0 扩展至 6 agent）
智能体：6 个独立 Point agent (两组三角形编队)
编队：两个独立三角形，组间需避碰
任务：两组编队交叉穿越对方区域
目的：验证多编队间的安全避障

    ▲          ▽
   / \   →   / \   ←
  /   \     /   \
 ●─────●   ●─────●

 组 A →              ← 组 B
        交叉穿越
```

### 8.4 场景四：动态编队切换

```
环境：MuJoCo 包含窄道的区域（自定义 XML 墙壁）
智能体：4 个独立 Point agent
任务：
  Phase 1: 正方形编队通过开阔区域
  Phase 2: 切换为一字编队穿过窄道
  Phase 3: 恢复正方形编队到达目标

  ████████████████████████████
  ██          ██           ██
  ██  ●──●   ██   ●●●●   ██
  ██  |  |   窄道          ★
  ██  ●──●   ██   ●●●●   ██
  ██          ██           ██
  ████████████████████████████
```

---

## 9. 与 RMPflow 的对应关系

```
RMPflow 概念                    本方案对应
──────────────────────────────────────────────────────
RMPRoot (联合配置空间)     ←→    全局状态 q = (q_1, ..., q_N)
RMPNode (子任务空间)       ←→    per_agent_constraints (每智能体约束集)
RMPLeaf (叶策略)           ←→    RL 策略 π_i + ATACOM 投影
GoalAttractorUni           ←→    目标导航奖励 r_nav
CollisionAvoidance         ←→    碰撞 StateConstraint (ATACOM 硬约束)
FormationDecentralized     ←→    编队 FormationDistanceConstraint (ATACOM)
Damper                     ←→    平滑性奖励 r_smooth
Pushforward (J @ x_dot)   ←→    约束 Jacobian J_q 计算
Pullback (J^T @ f)         ←→    ATACOM 零空间投影 (I - J^+ J)
Resolve (M^+ f)            ←→    ATACOM 动作合成 dq = dq_null + dq_err
```

**关键区别**：
- RMPflow：所有叶策略都是手工设计的解析函数，确定性
- 本方案：目标导航策略由 RL 学习（π_θ），安全约束由 ATACOM 确定性保证
- 本质上是把 RMPflow 的"手工力场设计"替换为"RL 学习 + 流形安全投影"

---

## 10. 实现路线图（基于 Safety-Gymnasium MA 扩展）

### Phase 1：编队约束模块 + 环境扩展（Week 1-2）

```
目标：实现编队约束核心模块，在 SafetyPointMultiGoal 上验证

任务：
☐ 实现 FormationTopology 类（拓扑定义）
☐ 实现 FormationShape 类（形状定义、距离矩阵）
☐ 实现 FormationDistanceConstraint 类
☐ 实现 FormationMultiGoalEnv（继承 MultiGoalEnv）
    - 扩展 MuJoCo XML 支持 N 个 agent
    - override _get_obs() 加编队误差
☐ 单元测试：约束值和 Jacobian 数值验证
☐ 验证：用 SafetyPointMultiGoal 环境 + 额外 agent body

产出文件：
  safepo/common/constrained_manifold/formation.py
  safepo/common/formation_wrappers.py
  tests/test_formation_constraints.py
```

### Phase 2：FormationNavAtacom + RMPflow 集成（Week 3-4）

```
目标：将编队约束集成到 ATACOM，接入 RMPflow Jacobian

任务：
☐ 实现 FormationNavAtacom（继承 MultiNavAtacom）
    - 新增 _make_formation_f()
    - 复用 _make_hazard_f() 和 _make_inter_agent_f()
    - override _get_q() 支持 N 个 agent
☐ 接入 RMPflow Jacobian（可选，先用解析 Jacobian 验证）
☐ 手工策略测试：用简单控制器验证 ATACOM 编队投影
☐ MuJoCo 渲染验证：观察多智能体编队运动

产出文件：
  safepo/common/formation_cm.py
```

### Phase 3：训练流程集成（Week 5-6）

```
目标：MAPPO-CM-Formation 在 MuJoCo 中完整训练

任务：
☐ 实现 mappo_cm_formation.py（基于 mappo_cm.py）
    - 环境创建改用 FormationMultiGoalEnv + FormationNavAtacom
    - rollout 中追加 compute_formation_reward()
    - 日志中追加编队指标
☐ 配置文件：triangle_3agents.yaml, square_4agents.yaml
☐ 基础场景训练（场景一：空旷区域 3 agent 三角形编队）
☐ 调参：K_c, slack_beta, formation_tolerance, w_formation

产出文件：
  safepo/multi_agent/mappo_cm_formation.py
  safepo/multi_agent/marl_cfg/formation/*.yaml
```

### Phase 4：实验与对比（Week 7-8）

```
目标：完成全部实验场景，与基线对比

任务：
☐ 场景二：障碍物穿越（4 agent, 6 障碍物）
☐ 场景三：多编队交汇（6 agent, 两组）
☐ 场景四：动态编队切换（4 agent, 窄道）
☐ 基线对比（全部在 MuJoCo 中）：
    MAPPO / MAPPO-Lag / MACPO / MAPPO-CM / RMPflow / 本方案
☐ 消融实验：编队约束权重、容许变形量、slack 参数
☐ 可扩展性：2/3/4/6/8/10 智能体
☐ MuJoCo 渲染录制 + 论文图表
```

---

## 11. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 编队约束与碰撞约束冲突导致无可行解 | ATACOM 投影失败 | 编队约束用更大 tolerance 松弛；采用级联投影（碰撞优先） |
| 约束数量增加导致零空间维度过低 | RL 策略自由度不足 | 减少编队拓扑边数（用 chain/star 替代 complete）；增大 tolerance |
| Jacobian 矩阵奇异/病态 | 数值不稳定 | 正则化 (Jc Jc^T + εI)^{-1}；添加阻尼项 |
| 编队松弛后无法恢复 | 编队误差持续偏大 | 增大 K_c 增益；在奖励中加入编队恢复奖励 |
| 训练收敛慢 | 实验效率低 | 课程学习（先简单后复杂场景）；预训练（先导航后编队） |
| SafetyPointMultiGoal XML 扩展 N>2 agent 不兼容 | 环境创建失败 | 直接在 MuJoCo XML 中插入 body；或继承 Safety-Gymnasium Builder 类 |
| MultiNavAtacom._get_q() 硬编码 2 agent | qpos 索引错误 | FormationNavAtacom override _get_q() 支持 N 个 freejoint |

---

## 12. RMPflow 论文谱系与综述

### 12.1 论文发展脉络

```
时间线：
2018 ──── 2019 ──── 2020 ──── 2021 ──── 2024
  │         │         │         │         │
  ▼         ▼         ▼         ▼         ▼

[P1] RMP 原始理论           ─┐
  Ratliff et al. 2018        │
                             ├── 理论奠基
[P2] RMPflow 计算图框架      │
  Cheng et al. 2018/2021   ─┘
        │
        ├──▶ [P3] 多机器人 RMPflow        ← 编队导航的直接参考
        │     Li, Mukadam et al. 2019
        │
        ├──▶ [P4] 可学习 Lyapunov 权重    ← 度量权重自动学习
        │     Mukadam et al. 2020
        │
        ├──▶ [P5] 从演示学习 RMP          ← 叶策略学习
        │     Li, Rana et al. 2020
        │
        ├──▶ [P6] 欠驱动系统扩展          ← 扩展到非全驱动系统
        │     Wingo et al. 2020
        │
        ├──▶ NVIDIA Isaac Sim 产品化      ← 工业落地
        │
        └──(几何思想启发)──▶ [P7] ATACOM   ← 安全 RL 约束流形
                             Liu et al. 2021
                               │
                               └──▶ [P8] ATACOM 理论扩展
                                     Liu et al. 2024
```

### 12.2 各论文详细信息

#### P1: Riemannian Motion Policies（RMP 原始理论）

- **作者**: Nathan Ratliff, Jan Issac, Daniel Kappler, Stan Birchfield, Dieter Fox
- **年份**: 2018
- **链接**: [arXiv:1801.02854](https://arxiv.org/abs/1801.02854)
- **出处**: arXiv preprint

**核心贡献**：

提出 RMP（Riemannian Motion Policy）这一数学对象——一个二阶动力系统（加速度场）配上一个 Riemannian 度量：

```
RMP = (a(x, ẋ), M(x, ẋ))
  a : 加速度策略（在任务空间中想怎么动）
  M : Riemannian 度量（哪个方向更重要）
```

提出 Pullback 操作，可以把不同任务空间的 RMP 通过 Jacobian 转换合成到统一的配置空间。

**用途**：单个机械臂的避障 + 目标到达。

**局限**：理论性较强，还没有形成工程化的计算图框架。

---

#### P2: RMPflow — A Computational Graph for Automatic Motion Policy Generation

- **作者**: Ching-An Cheng, Mustafa Mukadam, Jan Issac, Stan Birchfield, Dieter Fox, Byron Boots, Nathan Ratliff
- **年份**: 2018 (arXiv), 2020 (ISRR proceedings), 2021 (IEEE T-ASE)
- **链接**: [arXiv:1811.07049](https://arxiv.org/abs/1811.07049), [IEEE T-ASE](https://ieeexplore.ieee.org/document/9388869/)
- **出处**: ISRR 2019 / IEEE Transactions on Automation Science and Engineering

**核心贡献**：

将 P1 的理论工程化为**计算图框架**（树结构），定义了三类节点和两个核心操作：

```
节点类型：
  RMPRoot  — 根节点（联合配置空间）
  RMPNode  — 中间节点（子任务空间）
  RMPLeaf  — 叶节点（具体策略，输出 f 和 M）

核心操作：
  Pushforward（前推）：状态 (x, ẋ) 从根经 Jacobian 传到叶
    x_child = ψ(x_parent)
    ẋ_child = J(x_parent) · ẋ_parent

  Pullback（回拉）：力 f 和度量 M 从叶经 J^T 聚合回根
    f_parent += J^T · (f_child - M_child · J̇ · ẋ_parent)
    M_parent += J^T · M_child · J

几何一致性定理：
  若每个叶节点的 M_k 半正定，则聚合后的 M_root 半正定
  解 a = M_root⁺ · f_root 存在且唯一
```

**用途**：机器人操作（抓取、放置）中同时避障 + 到达目标。

**与本项目的关系**：`multi-robot-rmpflow/rmp.py` 中 `RMPRoot`/`RMPNode`/`RMPLeaf` 的直接理论来源。

---

#### P3: Multi-Objective Policy Generation for Multi-Robot Systems Using Riemannian Motion Policies

- **作者**: Anqi Li, Mustafa Mukadam, Magnus Egerstedt, Byron Boots
- **年份**: 2019 (arXiv), 2022 (ISRR 2019 proceedings)
- **链接**: [arXiv:1902.05177](https://arxiv.org/abs/1902.05177), [Springer](https://link.springer.com/chapter/10.1007/978-3-030-95459-8_16)
- **出处**: International Symposium on Robotics Research (ISRR) 2019

**核心贡献**：

将 RMPflow 从单机器人扩展到**多机器人系统**，设计了 7 种面向多机器人的 RMP 叶策略：

```
叶策略                              用途
─────────────────────────────────────────────────
GoalAttractorUni                    目标吸引（自适应增益）
CollisionAvoidance                  单障碍物避障
CollisionAvoidanceDecentralized     分布式智能体间避碰
CollisionAvoidanceCentralized       集中式智能体间避碰
FormationDecentralized              分布式编队距离保持
FormationCentralized                集中式编队距离保持
Damper                              速度阻尼（稳定性）
```

提出了**集中式 vs 分布式**两种多机器人 RMPflow 架构，并在 Georgia Tech **Robotarium 平台**上用 9 个真实机器人验证了编队保持与避碰的协调。

**用途**：多机器人编队控制、碰撞避免、目标导航。

**与本项目的关系**：`multi-robot-rmpflow/` 代码库的直接来源论文，编队叶策略（FormationDecentralized/Centralized）是本方案中编队约束设计的参考蓝本。

---

#### P4: Riemannian Motion Policy Fusion through Learnable Lyapunov Function Reshaping

- **作者**: Mustafa Mukadam, Ching-An Cheng, Dieter Fox, Byron Boots, Nathan Ratliff
- **年份**: 2020 (CoRL 2019)
- **链接**: [arXiv:1910.02646](https://arxiv.org/abs/1910.02646), [PMLR](https://proceedings.mlr.press/v100/mukadam20a.html)
- **出处**: Conference on Robot Learning (CoRL) 2019

**核心贡献**：

解决 RMPflow 中**各叶策略的度量权重需要手工调参**的问题：

```
问题：M_goal vs M_collision 的相对权重影响行为质量
      手工调参耗时且对场景敏感

方法：引入可学习的权重函数重塑 Lyapunov 函数
  w(x) : 神经网络参数化的权重函数
  M_new = w(x) · M_old
  通过反向传播从数据中学习 w(x)

理论保证：
  在温和条件下（w(x) > 0），加权后的合成策略仍全局 Lyapunov 稳定
```

**用途**：让子任务融合权重从数据中自动学习，替代手工调参。

**与本项目的关系**：启发了 ATACOM 中约束优先级的权重设计——可以用类似思路学习碰撞约束 vs 编队约束的相对权重。

---

#### P5: Learning Reactive Motion Policies in Multiple Subtask Spaces

- **作者**: Anqi Li, Ching-An Cheng, M. Asif Rana, Man Xie, Karl Van Wyk, Nathan Ratliff, Byron Boots
- **年份**: 2020 (CoRL 2019)
- **链接**: [PDF](https://homes.cs.washington.edu/~bboots/files/CORL2019_Learning_RMPs.pdf), [PMLR](http://proceedings.mlr.press/v100/rana20a/rana20a.pdf)
- **出处**: Conference on Robot Learning (CoRL) 2019

**核心贡献**：

从人类演示中**学习** RMP 叶策略，而不是手工设计：

```
问题：RMPflow 的每个叶策略 (f, M) 都是手工设计的解析函数
      对复杂任务难以设计出好的策略

方法：
  1. 将人类演示视为非欧子任务空间中的运动
  2. 在每个子任务空间中学习 (f_θ, M_θ)（神经网络拟合）
  3. 用 RMPflow 的 Pushforward/Pullback 合成全局策略

关键观察：
  演示 → 分解到各子任务空间 → 拟合 RMP → RMPflow 合成
```

**用途**：从演示学习复杂的机器人操作策略。

**与本项目的关系**：验证了"用学习方法替代手工 RMP 叶策略"的可行性。本方案进一步用 RL（而非模仿学习）来学习策略。

---

#### P6: Extending Riemannian Motion Policies to Underactuated Systems

- **作者**: Jacob Wingo, Ching-An Cheng, Mustafa Mukadam, Byron Boots, Nathan Ratliff
- **年份**: 2020 (ICRA 2020)
- **链接**: [PDF](https://www.chinganc.com/docs/wingo2020extending.pdf), [IEEE](https://ieeexplore.ieee.org/abstract/document/9196866)
- **出处**: IEEE International Conference on Robotics and Automation (ICRA) 2020

**核心贡献**：

将 RMPflow 从全驱动系统扩展到**欠驱动系统**：

```
问题：RMPflow 假设所有自由度直接可控（全驱动）
      轮式机器人、倒立摆等欠驱动系统不满足这一假设

方法：
  将动力学分解为：
    全驱动子系统（直接可控的自由度）
    + 残差动力学（被动自由度的耦合）
  在全驱动子系统上应用 RMPflow
  残差部分通过耦合项补偿
```

**用途**：轮式倒立摆机器人的多任务控制。

**与本项目的关系**：如果编队中的智能体是非全驱动的（如差速驱动小车），需要参考此扩展。

---

#### P7: Robot Reinforcement Learning on the Constraint Manifold（ATACOM）

- **作者**: Puze Liu (刘普泽), Kuo Zhang, Davide Tateo, Snehal Jauhri, Jan Peters, Georgia Chalvatzaki
- **年份**: 2021 (CoRL 2021)
- **链接**: [PMLR](https://proceedings.mlr.press/v164/liu22c/liu22c.pdf), [项目页](https://puzeliu.github.io/corl-ATACOM)
- **出处**: Conference on Robot Learning (CoRL) 2021

**核心贡献**：

提出 ATACOM（Acting on the Tangent space of the Constraint Manifold），首次将**约束流形的切空间投影**与 RL 结合：

```
核心思想：
  约束 c(q) ≤ 0 定义一个流形 M = {q : c(q) = 0}
  智能体在流形的切空间 T_q M 中探索
  ATACOM 投影保证动作始终在切空间内

关键公式（对应代码 manifold.py）：
  dq = (I - J⁺J) · α           ← 零空间投影（切空间内运动）
     + (-K_c · J⁺ · c(q))      ← 误差修正（拉回流形表面）

优势：
  1. 同时处理等式和不等式约束
  2. 约束在整个学习过程中维持在容许范围内
  3. 不需要初始可行策略——可以从零开始学习
  4. 将约束 RL 问题转化为标准无约束 RL 问题
```

**用途**：机器人操作中的安全 RL（单智能体）。

**与本项目的关系**：`constrained_manifold/manifold.py` 和 `constraints.py` 的直接理论来源。本方案将其扩展到多智能体编队场景。

---

#### P8: Safe Reinforcement Learning on the Constraint Manifold: Theory and Applications

- **作者**: Puze Liu, Kuo Zhang, Davide Tateo, Snehal Jauhri, Jan Peters, Georgia Chalvatzaki
- **年份**: 2024 (IEEE T-RO)
- **链接**: [arXiv:2404.09080](https://arxiv.org/abs/2404.09080), [项目页](https://puzeliu.github.io/TRO-ATACOM)
- **出处**: IEEE Transactions on Robotics

**核心贡献**：

ATACOM 的完整理论分析 + 真实机器人实验：

```
新增理论：
  1. 证明了 ATACOM 的约束满足保证（理论上界）
  2. 分析了不同 slack 类型（softcorner/softplus/exp/square）的收敛性质
  3. 证明了 slack 变量初始化的可行性条件

新增实验：
  1. 空气曲棍球（高速动态任务）— 真实机器人
  2. 与 Lagrangian/CBF/Safety Layer 的系统对比
  3. 不同 slack 类型的消融实验
```

**用途**：ATACOM 的理论基础论文。

**与本项目的关系**：提供了 slack 参数选择的理论依据，指导编队约束的 slack_type 和 slack_beta 配置。

---

#### 工业应用：NVIDIA Isaac Sim 中的 RMPflow

- **链接**: [Isaac Sim 文档](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/manipulators/concepts/rmpflow.html)

RMPflow 已被 NVIDIA 产品化，集成到 Isaac Sim 机器人仿真平台中，用于工业机械臂的实时运动规划。说明该理论框架已经足够成熟，具备实际部署能力。

---

### 12.3 论文引用关系

```
引用关系图：

[P1] RMP ────────────────────────────┐
  │                                  │
  ▼                                  ▼
[P2] RMPflow ──────────► [P4] Learnable Lyapunov (权重学习)
  │
  ├──────────────────────► [P5] Learning RMPs (策略学习)
  │
  ├──────────────────────► [P6] Underactuated (欠驱动扩展)
  │
  ├──────────────────────► [P3] Multi-Robot RMPflow (多机器人)
  │                          │
  │                          └──► 本代码库 multi-robot-rmpflow/
  │
  └──(几何启发)────────────► [P7] ATACOM (约束RL)
                               │     │
                               │     └──► 本代码库 constrained_manifold/
                               │
                               └──► [P8] ATACOM 理论 (T-RO)

本方案 = [P3] 的编队/避碰任务 + [P7] 的安全保证 + MAPPO 的策略学习
```

---

## 13. RMPflow 在编队导航任务中的具体使用方式

### 13.1 RMPflow 原始做法（纯几何，无 RL）

在 P3 (Multi-Robot RMPflow) 中，编队导航的实现方式是**完全手工设计**的：

#### 步骤一：构建 RMPflow 树

以 4 个智能体的三角形编队导航为例：

```
RMPRoot (8维: [x1,y1, x2,y2, x3,y3, x4,y4])
│
├── Agent_0 节点 (ψ: 提取 [x1,y1])
│   ├── GoalAttractor_0 (叶: 吸引向目标)
│   │   ψ(q) = q - goal_0
│   │   f = -gain·tanh(α·||x||)·x/||x|| - η·w·ẋ
│   │   M = w·I    (w 随距离自适应)
│   │
│   ├── CollisionAvoidance_01 (叶: 对 agent_1 避碰)
│   │   ψ(q) = ||q - q_1||/R - 1
│   │   f = -∂Φ/∂x - ξ - η·g·ẋ    (斥力场)
│   │   M = g(x) = w/x⁴             (距离越近度量越大)
│   │
│   ├── CollisionAvoidance_02 (叶: 对 agent_2 避碰)
│   ├── CollisionAvoidance_03 (叶: 对 agent_3 避碰)
│   │
│   ├── FormationDecentralized_01 (叶: 与 agent_1 保持距离 d_01)
│   │   ψ(q) = ||q - q_1|| - d_01
│   │   f = -gain·x·w - η·w·ẋ       (PD控制器)
│   │   M = w·I
│   │
│   ├── FormationDecentralized_02 (叶: 与 agent_2 保持距离 d_02)
│   ├── FormationDecentralized_03 (叶: 与 agent_3 保持距离 d_03)
│   │
│   └── Damper_0 (叶: 速度阻尼)
│       f = -η·w·ẋ
│       M = w·I
│
├── Agent_1 节点 (类似结构)
├── Agent_2 节点
└── Agent_3 节点

叶节点总数 = 4 × (1 + 3 + 3 + 1) = 32
```

#### 步骤二：每步执行 Pushforward → Pullback → Resolve

```python
# 伪代码：RMPflow 编队导航每步执行

for t in range(T):
    # 1. 设置根节点状态
    root.set_root_state(x_all, x_dot_all)

    # 2. Pushforward: 状态从根传播到所有叶节点
    #    每个叶节点通过 ψ 和 J 得到自己任务空间的状态
    root.pushforward()

    # 3. 更新动态参数（分布式避碰/编队需要知道其他智能体的最新位置）
    for leaf in collision_leaves:
        leaf.update_params()     # c ← other_agent.x
    for leaf in formation_leaves:
        leaf.update_params()     # c ← other_agent.x

    # 4. Pullback: 每个叶节点计算 (f, M)，沿 J^T 聚合回根
    #    f_root = Σ J_k^T (f_k - M_k J̇_k ẋ)
    #    M_root = Σ J_k^T M_k J_k
    root.pullback()

    # 5. Resolve: 在根节点求解加速度
    x_ddot = root.resolve()     # a = M_root⁺ · f_root

    # 6. 积分更新状态
    x_dot_all += x_ddot * dt
    x_all += x_dot_all * dt
```

#### 步骤三：RMPflow 中各叶策略的具体力场设计

**GoalAttractorUni（目标吸引）**：

```
数学形式：
  x = q - goal          任务空间：到目标的误差向量
  β = exp(-||x||² / 2σ²)   高斯衰减
  w = (w_u - w_l)·β + w_l  自适应权重：远处高，近处低
  s = tanh(α·||x||)        饱和函数

  grad_Φ = s/||x|| · w · x · gain    势场梯度力
  B·ẋ = η · w · ẋ                    阻尼力

  f = -grad_Φ - B·ẋ - ξ    (ξ 为 Coriolis 修正项)
  M = w · I

行为：
  远离目标 → 强吸引力，快速靠近
  接近目标 → 权重降低，平滑停止
  速度阻尼防止振荡
```

**CollisionAvoidanceDecentralized（分布式避碰）**：

```
数学形式：
  x = ||q_i - q_j|| / R - 1    任务空间：归一化距离（<0 表示碰撞）

  w = 1/x⁴   (x≥0)            权重函数：距离越近越大
      1e10    (x<0)            碰撞时极大

  u = ε + min(0, ẋ)·ẋ          速度调制（只在靠近时激活）
  g = w · u                    综合增益

  grad_Φ = α · w · (-4/x⁵)    势场梯度（排斥力）
  ξ = 0.5 · ẋ · ẋ_rel · u · grad_w    相对速度 Coriolis 项
  B·ẋ = η · g · ẋ              阻尼

  f = -grad_Φ - ξ - B·ẋ
  M = g + 0.5·ẋ·w·grad_u

行为：
  远处 (x >> 0): w ≈ 0，几乎无排斥
  近处 (x → 0): w → ∞，强排斥
  关键：ẋ_rel 使用相对速度，考虑两个移动智能体的对向接近
```

**FormationDecentralized（分布式编队）**：

```
数学形式：
  x = ||q_i - q_j|| - d_ij    任务空间：实际距离与期望距离的偏差

  f = -gain · x · w - η · w · ẋ    PD控制器
  M = w · I

行为：
  x > 0（太远）: f < 0，向邻居拉近
  x < 0（太近）: f > 0，推开邻居
  线性比例控制 + 速度阻尼
```

### 13.2 RMPflow 原始做法的优势与局限

```
优势：
  ✓ 实时性好（解析计算，无需前向推理）
  ✓ 几何一致性保证（Pullback 保持度量正定）
  ✓ 可扩展（增加新子任务 = 加新叶节点）
  ✓ 物理直觉好（每个叶策略有明确的力学含义）
  ✓ 稳定性可证明（Lyapunov 意义下全局稳定）

局限：
  ✗ 所有叶策略都是手工设计，对复杂场景适应性差
  ✗ 超参数（α, η, gain, w_u, w_l, σ, R）需要大量手调
  ✗ 避障策略基于简单势场，无法处理复杂障碍物形状
  ✗ 编队策略是线性 PD 控制器，对大扰动恢复慢
  ✗ 无法从经验中学习和改进
  ✗ 多目标权衡完全依赖 M 的手工设计
```

### 13.3 本方案的改进：用 RL 替代手工叶策略

核心思想：**保留 RMPflow 的几何框架，但把手工力场替换为 RL 学习的策略 + ATACOM 安全投影**。

```
RMPflow 原始做法                    本方案 (MAPPO-CM-Formation)
─────────────────────────────────────────────────────────────────

树结构:                             约束结构:
  RMPRoot                            全局状态 q = (q_1,...,q_N)
  ├── Agent_i                        ├── Agent_i 约束集
  │   ├── GoalAttractor (手工 f,M)   │   ├── π_i(α|o_i)  ← RL 策略
  │   ├── CollisionAvoid (手工 f,M)  │   ├── c_coll: ATACOM 硬约束
  │   ├── Formation (手工 f,M)       │   ├── c_form: ATACOM 编队约束
  │   └── Damper (手工 f,M)          │   └── c_wall: ATACOM 边界约束

动作生成:                           动作生成:
  a = M_root⁺ · f_root              a = ATACOM(α)
    = 度量加权的力合成                  = 零空间投影 + 误差修正

子任务权衡:                         子任务权衡:
  通过 M_k 的大小控制                 碰撞/边界: ATACOM 硬约束（绝对满足）
  手工设定                            编队: ATACOM 约束 + 奖励双重机制
  无优先级保证                        导航: 奖励驱动（RL 学习最优策略）
                                     优先级: L1 碰撞 > L2 编队 > L3 导航
```

### 13.4 几何对应关系的数学推导

为什么 ATACOM 的零空间投影与 RMPflow 的 Pullback 在数学上是相关的：

```
=== RMPflow Pullback ===

给定叶节点 k 的 RMP (f_k, M_k)，映射 ψ_k 和 Jacobian J_k：

  f_root = Σ_k J_k^T · M_k · (M_k⁻¹ f_k - J̇_k ẋ)
  M_root = Σ_k J_k^T · M_k · J_k

  解: a = M_root⁺ · f_root

含义：每个叶节点的力通过 J_k^T 映射回配置空间，
     按各自的度量 M_k 加权合成。


=== ATACOM 零空间投影 ===

给定约束 c(q) = 0，Jacobian J = ∂c/∂q：

  J⁺ = J^T (J J^T)⁻¹           伪逆
  N = I - J⁺ J                  零空间投影矩阵
  a = N · α - K_c · J⁺ · c(q)  安全动作

含义：策略输出 α 投影到约束 Jacobian 的零空间，
     只保留不影响约束的分量。


=== 数学联系 ===

关键观察：当 RMPflow 的 M_collision → ∞ 时（碰撞约束最高优先）

  M_root ≈ J_coll^T · M_coll · J_coll    (碰撞度量主导)

  目标力在碰撞度量下的投影：
  a_goal = (I - J_coll^T (J_coll J_coll^T)⁻¹ J_coll) · a_goal_desired

  这正好等于 ATACOM 的零空间投影！
  N_coll = I - J_coll⁺ J_coll

因此：
  ATACOM 的零空间投影 ≡ RMPflow 中碰撞度量趋向无穷时的极限情况
  ATACOM 把 RMPflow 的"软优先级"（通过 M 大小控制）
  变成了"硬优先级"（约束绝对满足，不是通过大权重近似）
```

---

## 14. MARL + 黎曼流形约束的完整集成方案

### 14.1 三层架构的协同机制

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  第一层：MAPPO 策略学习（"在哪个方向上用力"）                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │   Agent_i 观测:                                               │  │
│  │   o_i = [自身状态, 邻居相对位置, 编队误差, 目标方向, 障碍物]      │  │
│  │                          ↓                                    │  │
│  │   Actor π_θ_i : o_i → N(μ_i, σ_i)  → 采样 α_i               │  │
│  │   (神经网络, 128×128 MLP)                                      │  │
│  │                                                               │  │
│  │   Critic V_φ : s_global → V(s)     (集中式, 训练时使用)         │  │
│  │                                                               │  │
│  │   α_i 的含义: "我想在零空间中沿这个方向运动"                      │  │
│  │   α_i 不直接控制关节/速度，而是流形上的坐标                       │  │
│  │                                                               │  │
│  └───────────────────────────────┬───────────────────────────────┘  │
│                                  │ α_i (可能不安全的动作意图)        │
│                                  ▼                                  │
│  第二层：ATACOM 黎曼流形约束投影（"怎么保证安全"）                    │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │   约束集 C_i (每个智能体独立维护):                               │  │
│  │                                                               │  │
│  │   c_coll_ij = -||q_i - q_j|| + 2r ≤ 0    ∀j≠i  (碰撞避免)    │  │
│  │   c_obs_ik  = -||q_i - o_k|| + r_o ≤ 0   ∀k    (障碍物)      │  │
│  │   c_wall    = ±q_i ± bound ≤ 0                  (边界)       │  │
│  │   c_form_ij = |  ||q_i-q_j||-d_ij  | - δ ≤ 0   (编队距离)   │  │
│  │                                                               │  │
│  │   总 Jacobian:                                                │  │
│  │   Jc = ∂[c_coll; c_obs; c_wall; c_form] / ∂[q; s]            │  │
│  │       ∈ ℝ^{m × (n+m_slack)}                                   │  │
│  │                                                               │  │
│  │   零空间投影:                                                  │  │
│  │   N = (I - Jc⁺ Jc)[:, :dim_null]    "约束之外的自由方向"       │  │
│  │                                                               │  │
│  │   安全动作合成:                                                │  │
│  │   a_i = N · α_i                      零空间中的运动            │  │
│  │       + (-K_c · Jc⁺ · c(q))          约束误差修正              │  │
│  │                                                               │  │
│  └───────────────────────────────┬───────────────────────────────┘  │
│                                  │ a_i (确定性安全的动作)           │
│                                  ▼                                  │
│  第三层：RMPflow 几何基础（"为什么数学上正确"）                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │   ATACOM 的零空间投影 N = I - J⁺J                              │  │
│  │                                                               │  │
│  │   在 RMPflow 理论中等价于:                                     │  │
│  │     当约束叶节点的度量 M_constraint → ∞ 时                      │  │
│  │     目标策略力被投影到约束法方向的正交补空间                      │  │
│  │                                                               │  │
│  │   几何一致性保证:                                              │  │
│  │     ✓ 多约束的 Jacobian 拼接 = 多叶节点的 Pullback 聚合         │  │
│  │     ✓ 投影后的度量 M_projected 仍半正定                         │  │
│  │     ✓ 约束优先级 = 度量权重的极限排序                            │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.2 一个完整时间步的详细执行流程

以 4 个智能体三角形编队穿越障碍物为例，展示每个模块的具体计算：

```
时刻 t，智能体 i=0 的完整计算流程：

═══════════════════════════════════════════════════════════════
 Step 1: 观测构建
═══════════════════════════════════════════════════════════════

  环境状态:
    q_0 = [1.2, 0.8, 0.3]     (位置 x,y 和朝向 θ)
    q_1 = [1.5, 1.1, 0.1]
    q_2 = [0.9, 1.2, -0.2]
    q_3 = [1.8, 0.5, 0.5]
    goal = [3.0, 2.0]
    obstacle = [2.0, 1.0], r=0.3

  构建 o_0:
    自身状态:    [1.2, 0.8, v_x, v_y, 0.3]                    5维
    目标:        [1.8, 1.2, 2.16]                              3维
    邻居(完全图): [0.3,0.3,err_01], [-0.3,0.4,err_02], [0.6,-0.3,err_03]  9维
    编队中心:    [0.15, 0.1]                                   2维
    障碍物:      [0.8, 0.2, 0.72]                              3维
    总计:        22维

═══════════════════════════════════════════════════════════════
 Step 2: MAPPO 策略输出
═══════════════════════════════════════════════════════════════

  Actor 网络 π_θ_0:
    o_0 → MLP(128,128) → μ_0 = [0.7, -0.3]    (零空间坐标)
                        → σ_0 = [0.2, 0.15]

  采样: α_0 ~ N(μ_0, σ_0) = [0.65, -0.25]

  α_0 的含义: "我想在安全动作空间中往 [0.65, -0.25] 方向运动"
  （这不是 [vx, vy]，而是流形切空间中的坐标）

═══════════════════════════════════════════════════════════════
 Step 3: 约束评估
═══════════════════════════════════════════════════════════════

  碰撞约束（3个）:
    c_01 = -||q_0-q_1|| + 0.6 = -√(0.09+0.09) + 0.6 = -0.42 + 0.6 = 0.18?
    不对, 重新算: ||q_0-q_1|| = √(0.3²+0.3²) = 0.424
    c_01 = -0.424 + 0.6 = 0.176  ← 危险！快要碰了！

    c_02 = -||q_0-q_2|| + 0.6 = -√(0.09+0.16) + 0.6 = -0.5 + 0.6 = 0.1
    c_03 = -||q_0-q_3|| + 0.6 = -√(0.36+0.09) + 0.6 = -0.67 + 0.6 = -0.07 ✓

  障碍物约束:
    c_obs = -||q_0-obs|| + 0.3+0.15 = -√(0.64+0.04) + 0.45 = -0.825 + 0.45 = -0.375 ✓

  编队约束（假设 d_01=0.5, δ=0.1）:
    dist_01 = 0.424
    c_form_01_lower = -(0.424 - 0.5) + 0.1 = 0.176  ← 在容许范围边缘
    c_form_01_upper = (0.424 - 0.5) - 0.1 = -0.176 ✓

  边界约束:
    c_wall_x_max = 1.2 - 5.0 = -3.8  ✓
    ...

═══════════════════════════════════════════════════════════════
 Step 4: ATACOM 零空间投影
═══════════════════════════════════════════════════════════════

  约束向量（含 slack 变量后变为等式约束）:
    c̃ = [c̃_01, c̃_02, c̃_03, c̃_obs, c̃_wall..., c̃_form_01_lower, ...]

  Jacobian（约束对 [q_x, q_y, q_θ, s_1, s_2, ...] 的偏导）:
    Jc ∈ ℝ^{m × (3 + m_slack)}

  计算伪逆:
    Jc⁺ = Jc^T (Jc Jc^T + 1e-8·I)⁻¹

  零空间基:
    dim_null = (3 + m_slack) - m    (总变量数 - 约束数)
    N = (I - Jc⁺ Jc)[:, :dim_null]

  安全动作合成:
    dq_null = N · α_0 · dq_max        零空间运动 (RL想做的)
    dq_err  = -100 · Jc⁺ · c̃(q,s)    约束修正 (把c_01拉回安全)
    dq = dq_null + dq_err

  关键效果:
    c_01 = 0.176 > 0 (快要碰 agent_1)
    → dq_err 产生一个"推离 agent_1"的修正力
    → α_0 中"靠近 agent_1"的分量被零空间投影去除
    → 最终 a_0 不会让智能体 0 更靠近智能体 1

  截断:
    scale = 1 / max(|dq[:3]|/a_max, 1)
    a_0 = dq[:3] · scale

═══════════════════════════════════════════════════════════════
 Step 5: 环境执行 + 奖励计算
═══════════════════════════════════════════════════════════════

  环境执行 a_0（对所有智能体并行）:
    s', r_env, cost, done = env.step([a_0, a_1, a_2, a_3])

  叠加编队奖励:
    r_form_0 = -0.1 · Σ_j (||q_0-q_j|| - d_0j)²
    r_total_0 = r_env + r_form_0

═══════════════════════════════════════════════════════════════
 Step 6: 存入 Buffer
═══════════════════════════════════════════════════════════════

  buffer.store(o_0, α_0, r_total_0, cost_0, V_r(s), V_c(s), log_π(α_0))

  注意: 存的是 α_0（零空间坐标），不是 a_0（环境动作）
  因为 MAPPO 学习的是在零空间中的策略，ATACOM 投影是确定性的
```

### 14.3 训练循环中各模块的更新

```
每个 Epoch 的完整训练流程:

╔══════════════════════════════════════════════════════════════╗
║ Phase A: 数据收集 (Rollout)                                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  for step in range(rollout_length):                          ║
║      for agent_i in range(N):                                ║
║          α_i = π_θ_i(o_i)           ← MAPPO Actor            ║
║          a_i = ATACOM(α_i, q_i)     ← 流形投影（确定性）      ║
║      obs, r, c, done = env.step(a)  ← 环境执行                ║
║      r += formation_reward(q)       ← 编队奖励                ║
║      buffer.store(...)                                       ║
║                                                              ║
║  ATACOM 在此阶段只做前向计算（无梯度）                         ║
║  RL 策略和约束投影完全解耦                                    ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║ Phase B: 优势估计 (GAE)                                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  for agent_i in range(N):                                    ║
║      δ_t = r_t + γ·V(s_{t+1}) - V(s_t)                      ║
║      A_t = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}     ← GAE        ║
║      A_t = (A_t - mean) / std                  ← 标准化      ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║ Phase C: 策略更新 (PPO)                                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  for agent_i in range(N):          ← 分布式 Actor 更新       ║
║      for epoch_k in range(K):                                ║
║          r_t = π_θ_new(α|o) / π_θ_old(α|o)                  ║
║          L = -E[min(r_t·A_t, clip(r_t,1±ε)·A_t)]            ║
║          θ_i ← θ_i - lr · ∇L                                ║
║                                                              ║
║  Critic 更新:                       ← 集中式 Critic 更新     ║
║      L_V = E[(V_φ(s) - V_target)²]                          ║
║      φ ← φ - lr · ∇L_V                                      ║
║                                                              ║
║  注意:                                                        ║
║    ✓ MAPPO 只更新 Actor 和 Critic 网络参数                    ║
║    ✗ ATACOM 无参数需要更新（纯解析计算）                       ║
║    ✗ RMPflow 无参数需要更新（提供理论保证）                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### 14.4 MARL 中各智能体的约束耦合

多智能体场景下，各智能体的约束不是独立的——agent_i 的碰撞约束依赖 agent_j 的位置，反之亦然。这种耦合的处理方式：

```
分布式约束管理（对应 RMPflow 的 Decentralized 架构）：

  每个 Agent_i 独立维护自己的约束集 C_i
  但 C_i 中的碰撞/编队约束引用了其他智能体的位置

  Agent_0 的约束集:
    c_coll(q_0, q_1): "我到 agent_1 的距离"  ← 需要知道 q_1
    c_coll(q_0, q_2): "我到 agent_2 的距离"  ← 需要知道 q_2
    c_form(q_0, q_1): "我和 agent_1 的编队距离" ← 需要知道 q_1

  每步开始时:
    1. 读取所有智能体的当前位置 q_all
    2. 更新每个智能体的 _other_positions
    3. 各智能体独立计算自己的 ATACOM 投影
    4. 所有安全动作一起送入环境

  对应 RMPflow 中的 update_params():
    CollisionAvoidanceDecentralized.update_params()
    FormationDecentralized.update_params()
    每步更新邻居位置 c ← other_agent.x

优势:
  ✓ 分布式计算（每个智能体独立求解自己的约束投影）
  ✓ 可扩展（新增智能体只需新增一组约束）
  ✓ 信息需求小（只需知道邻居位置，不需要邻居的策略/动作）

理论保证:
  虽然每个智能体独立投影，但碰撞约束是双向的：
    Agent_0 的 c_01 ≤ 0  AND  Agent_1 的 c_10 ≤ 0
    两个约束函数值相同（||q_0-q_1|| = ||q_1-q_0||）
    所以双方都会推离，安全性有冗余保证
```

### 14.5 RMPflow 几何在 MARL 训练中的三个关键作用

#### 作用一：约束 Jacobian 提供安全梯度信息

```
RMPflow 的 Pushforward 机制告诉 ATACOM：
  约束 c(q) 对状态 q 的敏感度是什么？

例子：碰撞约束
  c = -||q_i - q_j|| + r_safe
  J = ∂c/∂q_i = -(q_i - q_j) / ||q_i - q_j||

  J 的几何含义（RMPflow 视角）：
    J 是从配置空间到"碰撞距离"任务空间的 Pushforward 映射
    它告诉我们：q_i 沿哪个方向运动会最快改变碰撞距离

  J 在 ATACOM 中的作用：
    J⁺ · c(q) 给出了"修正约束违反"的最高效方向
    (I - J⁺J) 给出了"完全不影响约束"的运动方向
```

#### 作用二：零空间投影保证多任务不冲突

```
RMPflow 的 Pullback 机制告诉我们：
  多个子任务的力可以通过 J^T 聚合

ATACOM 的零空间投影在数学上等价于：
  "碰撞约束的度量 M_coll → ∞ 时的 RMPflow Resolve"

多约束场景的零空间分析：
  约束数 m = 3(碰撞) + 2(编队) + 4(边界) = 9
  状态维度 n = 3 (x, y, θ)
  Slack 维度 m_slack = 9
  总变量 = n + m_slack = 12
  零空间维度 = 12 - 9 = 3

  RL 策略在 3 维零空间中自由探索
  这 3 个维度正好是"满足所有约束后的剩余自由度"
```

#### 作用三：度量加权提供约束优先级

```
RMPflow 中度量 M 的大小决定了子任务的优先级：
  M_collision >> M_formation >> M_goal
  → 避碰最重要，编队次之，导航最灵活

ATACOM 中的对应机制：
  碰撞约束: slack_beta = 30, threshold = 0.01  (硬约束)
  编队约束: slack_beta = 10, threshold = 0.05  (较软)

  slack_beta 越大 → slack 函数越陡峭 → 约束越"硬"
  → 对应 RMPflow 中 M 越大

  当碰撞和编队约束冲突时：
    碰撞约束的 Jacobian 贡献更大的度量权重
    零空间投影自然优先满足碰撞约束
    编队约束在碰撞安全的前提下尽量满足
```

### 14.6 与 RMPflow 对比的本方案完整优势

```
                        RMPflow 纯几何        本方案 (MARL + 流形约束)
                        ──────────────        ───────────────────────
策略来源               手工力场设计            RL 学习（MAPPO）
安全保证               无硬保证（力可能被覆盖） 确定性硬保证（ATACOM 投影）
编队控制               线性 PD 控制器          RL 策略（可处理非线性场景）
避障行为               势场法（局部最优）       RL 学习（全局更优路径）
多目标权衡             手工 M 设计（需调参）    约束优先级 + RL 自动权衡
环境适应性             固定策略，不适应新场景    从经验中学习和改进
训练时安全             N/A（无训练过程）        ATACOM 保证训练全程安全
理论基础               RMPflow 几何一致性      RMPflow + 约束流形理论
可扩展性               加叶节点                加约束 + 加智能体

关键创新点：
  1. RL 策略替代手工力场 → 更强的适应性和泛化能力
  2. ATACOM 替代度量加权 → 从软优先级升级为硬约束保证
  3. 保留 RMPflow 几何框架 → 多约束合成有理论保证
  4. 编队约束集成到 ATACOM → 编队保持也有硬保证（RMPflow 做不到）
```

---

## 15. 基于 Safety-Gymnasium MA 的编队扩展方案

> **实现基础**：直接基于 Safety-Gymnasium 多智能体框架扩展，不从零构建环境。
> 复用现有的 `MultiGoalEnv` + `MultiNavAtacom` + `mappo_cm.py` 训练循环。

### 15.1 现有基础设施（已可用）

```
Safety-Gymnasium MA 已有的完整链路：

  SafetyPointMultiGoal{0,1,2}-v0           ← 2 个独立 Point agent + 障碍物
        ↓
  MultiGoalEnv (wrappers.py:68-157)        ← 多智能体接口封装
  │  ├── num_agents = 2                    │
  │  ├── _get_obs() → 含 agent_id one-hot  │
  │  └── step(actions) → (obs, share_obs, rewards, costs, dones, infos)
        ↓
  MultiNavAtacom (mult_cm.py:7-228)        ← ATACOM 约束投影（已有）
  │  ├── per_agent_constraints             │  ← 每个 agent 独立约束集
  │  ├── _make_hazard_f()                  │  ← 障碍物约束
  │  ├── _make_inter_agent_f(agent_idx)    │  ← 智能体间碰撞约束
  │  └── step_action_function(alpha, q, i) │  ← 零空间投影
        ↓
  MAPPO-CM (mappo_cm.py)                   ← 训练循环（已有）
  │  ├── MultiAgentActor / Critic          │
  │  ├── SeparatedReplayBuffer             │
  │  └── PPO 更新                          │
        ↓
  ShareDummyVecEnv / ShareSubprocVecEnv    ← 向量化并行（已有）
```

**关键发现：这套链路已经跑通了 2 个独立 Point agent 的安全导航。**

### 15.2 需要扩展的三个点

```
                现有能力                          编队扩展
              ──────────                        ──────────
智能体数量     固定 2 个                         → N 个 (2~10)
约束类型       碰撞避免 + 障碍物                  → + 编队距离约束
几何基础       无                                → + RMPflow Jacobian

具体改动：

  ┌──────────────────────────────────────────────────────────────┐
  │ 改动 1: MultiGoalEnv → FormationMultiGoalEnv                 │
  │   - num_agents: 2 → N (参数化)                                │
  │   - 扩展 MuJoCo XML: 动态添加 N 个 Point body                │
  │   - obs 中增加编队误差信息                                     │
  │                                                                │
  │ 改动 2: MultiNavAtacom → FormationNavAtacom                    │
  │   - 新增 _make_formation_f(agent_idx): 编队距离约束函数         │
  │   - 在 per_agent_constraints 中追加编队约束                     │
  │   - 集成 RMPflow 叶节点的 Jacobian                             │
  │                                                                │
  │ 改动 3: mappo_cm.py → mappo_cm_formation.py                   │
  │   - 奖励函数追加编队项                                          │
  │   - 配置文件新增编队参数                                         │
  └──────────────────────────────────────────────────────────────┘
```

### 15.3 改动 1: 环境扩展（MultiGoalEnv → FormationMultiGoalEnv）

```python
# 文件: safepo/common/wrappers.py (扩展 MultiGoalEnv)

class FormationMultiGoalEnv(MultiGoalEnv):
    """
    在 MultiGoalEnv 基础上扩展：
    1. 支持 N 个智能体（原来固定 2 个）
    2. 观测中增加编队误差
    3. 奖励中增加编队保持项

    现有代码改动最小化：
    - 继承 MultiGoalEnv 的 reset/step/render
    - 仅 override _get_obs() 和 _compute_rewards()
    """

    def __init__(self, task, num_agents, formation_config):
        # 基于 SafetyPointMultiGoal 的 MuJoCo XML 扩展
        # 动态添加额外的 Point agent body
        self.num_agents = num_agents
        self.formation_config = formation_config

        # 计算编队期望距离矩阵
        self.desired_distances = formation_distances(
            formation_config['shape'],
            num_agents,
            formation_config['radius']
        )

        # 调用父类初始化（加载 MuJoCo 环境）
        super().__init__(task=task)

        # 覆盖 num_agents（父类硬编码为 2）
        self.num_agents = num_agents

    def _extend_mujoco_xml(self, base_xml, num_agents):
        """
        在 SafetyPointMultiGoal 的 XML 基础上追加额外 agent body

        SafetyPointMultiGoal 已有:
          - agent_0 (Point robot): freejoint, sphere geom
          - agent_1 (Point robot): freejoint, sphere geom
          - hazards: cylinder geom
          - goal: cylinder geom

        追加:
          - agent_2 ~ agent_{N-1}: 同样的 Point body
        """
        for i in range(2, num_agents):
            angle = 2 * np.pi * i / num_agents
            x, y = 1.5 * np.cos(angle), 1.5 * np.sin(angle)
            # 插入新 body 到 worldbody
            agent_xml = f'''
            <body name="agent_{i}" pos="{x} {y} 0.1">
                <freejoint name="agent_{i}_joint"/>
                <geom name="agent_{i}_geom" type="sphere"
                      size="0.15" rgba="{0.2+0.08*i} 0.4 {0.8-0.08*i} 1"/>
            </body>'''
            # ... 插入到 XML 中

    def _get_obs(self):
        """
        扩展观测: 基础观测 + 编队误差

        原 MultiGoalEnv._get_obs():
          obs_i = [自身状态, agent_id_onehot, ...]

        扩展后:
          obs_i = [自身状态, agent_id_onehot,
                   邻居相对位置(2*(N-1)),
                   编队距离误差(N-1),            ← 新增
                   编队中心偏差(2)]               ← 新增
        """
        base_obs = super()._get_obs()
        positions = self._get_all_positions()

        extended_obs = []
        for i in range(self.num_agents):
            form_errors = []
            for j in range(self.num_agents):
                if j != i:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    form_errors.append(dist - self.desired_distances[i, j])

            center = positions.mean(axis=0)
            center_offset = center - positions[i]

            extended_obs.append(np.concatenate([
                base_obs[i],
                np.array(form_errors, dtype=np.float32),
                center_offset.astype(np.float32),
            ]))

        return extended_obs

    def _get_all_positions(self):
        """从 Safety-Gymnasium 环境读取所有 agent 位置"""
        # Safety-Gymnasium 的 Point agent 位置存在 data.qpos 中
        # 具体索引取决于 agent 的 freejoint 在 XML 中的顺序
        positions = np.zeros((self.num_agents, 2))
        for i in range(self.num_agents):
            # freejoint: qpos = [x, y, z, qw, qx, qy, qz], 取 x, y
            offset = i * 7  # 每个 freejoint 占 7 个 qpos
            positions[i] = self.env.data.qpos[offset:offset+2]
        return positions
```

### 15.4 改动 2: 约束扩展（MultiNavAtacom → FormationNavAtacom）

```python
# 文件: safepo/common/mult_cm.py (扩展 MultiNavAtacom)

class FormationNavAtacom(MultiNavAtacom):
    """
    在 MultiNavAtacom 基础上新增编队距离约束

    现有约束（继承）:
      - _make_hazard_f(): 障碍物约束    c_obs = -||q_i - o_k|| + r
      - _make_inter_agent_f(): 碰撞约束  c_coll = -||q_i - q_j|| + 2r

    新增约束:
      - _make_formation_f(): 编队约束
          c_form_lower = -(||q_i - q_j|| - d_ij) - δ ≤ 0  (不能太近)
          c_form_upper =  (||q_i - q_j|| - d_ij) - δ ≤ 0  (不能太远)
    """

    def __init__(self, base_env, formation_config, **kwargs):
        self.formation_config = formation_config
        self.desired_distances = formation_config['desired_distances']
        self.formation_tolerance = formation_config.get('tolerance', 0.1)

        super().__init__(base_env, **kwargs)

        # 为每个 agent 追加编队约束
        for i in range(self.num_agents):
            formation_constraints = self._make_formation_f(i)
            for fc in formation_constraints:
                self.per_agent_constraints[i].add_constraint(fc)

    def _make_formation_f(self, agent_idx):
        """
        为 agent_idx 创建编队距离约束

        对邻接表中的每个邻居 j:
          c_lower(q) = -(||q_i - q_j|| - d_ij) - δ ≤ 0
          c_upper(q) = (||q_i - q_j|| - d_ij) - δ ≤ 0

        这保证 |  ||q_i - q_j|| - d_ij  | ≤ δ

        Jacobian (RMPflow FormationDecentralized 等价):
          ∂c/∂q_i = ±(q_i - q_j) / ||q_i - q_j||
        """
        constraints = []
        topology = self.formation_config.get('topology', 'complete')

        for j in range(self.num_agents):
            if j == agent_idx:
                continue

            # 根据拓扑决定是否有此约束边
            if topology == 'complete' or self._has_edge(agent_idx, j, topology):
                d_ij = self.desired_distances[agent_idx, j]
                delta = self.formation_tolerance

                # 下界约束: 不能比期望距离近太多
                def f_lower(q, _j=j, _d=d_ij, _delta=delta):
                    q_i = q[:2]
                    q_j = self._other_positions[_j]
                    dist = np.linalg.norm(q_i - q_j)
                    return -(dist - _d) - _delta

                # 上界约束: 不能比期望距离远太多
                def f_upper(q, _j=j, _d=d_ij, _delta=delta):
                    q_i = q[:2]
                    q_j = self._other_positions[_j]
                    dist = np.linalg.norm(q_i - q_j)
                    return (dist - _d) - _delta

                constraints.append(StateConstraint(
                    f=f_lower, type='ineq',
                    slack_type='softcorner', slack_beta=10.0
                ))
                constraints.append(StateConstraint(
                    f=f_upper, type='ineq',
                    slack_type='softcorner', slack_beta=10.0
                ))

        return constraints

    def _integrate_rmpflow_jacobian(self, agent_idx):
        """
        可选：从 RMPflow 叶节点获取 Jacobian 替代手工计算

        RMPflow 的 FormationDecentralized 叶节点计算的 Jacobian
        与 _make_formation_f 中解析计算的 Jacobian 数学上等价：
          J = ∂c/∂q = ±(q_i - q_j) / ||q_i - q_j||

        使用 RMPflow 的优势：
          1. 保持与 RMPflow 理论框架的一致性
          2. 可获取度量 M 用于约束优先级加权
          3. 未来扩展到更复杂约束时更方便
        """
        if hasattr(self.base_env, 'rmp_trees'):
            jacobians, metrics = self.base_env.get_rmpflow_jacobians(agent_idx)
            return jacobians, metrics
        return None, None
```

### 15.5 改动 3: 训练循环扩展

```python
# 文件: safepo/multi_agent/mappo_cm_formation.py
# 基于 mappo_cm.py 最小改动

# === 改动点 1: 环境创建 ===
# mappo_cm.py 原来调用 make_cm_multi_goal_env()
# 改为调用新函数:

def make_formation_env(task, num_agents, formation_config, seed, cfg_train):
    """
    创建编队导航环境

    复用 env.py 中的环境创建流程，替换 wrapper:
      MultiGoalEnv → FormationMultiGoalEnv
      MultiNavAtacom → FormationNavAtacom
    """
    env = FormationMultiGoalEnv(
        task=task,
        num_agents=num_agents,
        formation_config=formation_config,
    )
    env = FormationNavAtacom(
        base_env=env,
        formation_config=formation_config,
        K_c=100,
        collision_radius=0.3,
    )
    return env

# === 改动点 2: 奖励计算 ===
# 在 rollout 中追加编队奖励

def compute_formation_reward(positions, desired_distances, w_formation=0.1):
    """
    编队保持奖励（追加到环境原始奖励上）

    r_form_i = -w · Σ_j (||q_i - q_j|| - d_ij)²
    """
    N = len(positions)
    rewards = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if j != i:
                dist = np.linalg.norm(positions[i] - positions[j])
                rewards[i] -= w_formation * (dist - desired_distances[i, j]) ** 2
    return rewards

# === 改动点 3: 配置文件 ===
# 在 marl_cfg/ 中新增编队配置

formation_config = {
    "shape": "triangle",       # 编队形状
    "num_agents": 3,           # 智能体数量
    "radius": 0.5,             # 编队半径
    "topology": "complete",    # 拓扑类型
    "tolerance": 0.1,          # 距离容许偏差
    "w_formation": 0.1,        # 编队奖励权重
}
```

### 15.6 完整对接流程

```python
# 基于 Safety-Gymnasium MA 的完整链路

# 1. 创建编队环境（扩展自 SafetyPointMultiGoal）
env = FormationMultiGoalEnv(
    task="SafetyPointMultiGoal1-v0",
    num_agents=3,
    formation_config=formation_config,
)
# 内部: 在 SafetyPointMultiGoal 的 MuJoCo XML 上追加 agent body

# 2. ATACOM 编队约束包装（扩展自 MultiNavAtacom）
env = FormationNavAtacom(
    base_env=env,
    formation_config=formation_config,
    K_c=100,
    collision_radius=0.3,
)
# 内部: 继承碰撞/障碍物约束 + 新增编队距离约束

# 3. 向量化（复用现有）
vec_env = ShareDummyVecEnv([lambda: env], device="cpu")

# 4. MAPPO 训练（复用 mappo_cm.py，仅追加编队奖励）
runner = MAPPOCMRunner(config, vec_env, policy)
runner.run()

# 对接关系图:
#
#   SafetyPointMultiGoal1-v0  (Safety-Gymnasium, 已有)
#          │
#          ▼
#   FormationMultiGoalEnv     (继承 MultiGoalEnv, 改动 1)
#          │  + N 个 agent, + 编队观测
#          ▼
#   FormationNavAtacom        (继承 MultiNavAtacom, 改动 2)
#          │  + 编队约束, + RMPflow Jacobian
#          ▼
#   mappo_cm_formation.py     (基于 mappo_cm.py, 改动 3)
#          │  + 编队奖励
#          ▼
#   训练输出
```

### 15.7 关键文件对照表

```
现有文件 (safe-po)                      扩展文件
────────────────                       ──────────
safepo/common/wrappers.py              → 新增 FormationMultiGoalEnv 类
  └── MultiGoalEnv (line 68)              继承, override _get_obs()

safepo/common/mult_cm.py               → 新增 FormationNavAtacom 类
  └── MultiNavAtacom (line 7)             继承, 新增 _make_formation_f()

safepo/multi_agent/mappo_cm.py         → 新增 mappo_cm_formation.py
  └── MAPPOCMRunner                       复制, 追加编队奖励计算

safepo/utils/config.py                 → 新增编队环境 ID 注册
  └── multi_agent_goal_tasks              追加 FormationMultiGoal 系列

safepo/multi_agent/marl_cfg/           → 新增编队配置文件
                                          formation_3agents.yaml 等

总改动量: 3 个新文件（继承扩展）+ 1 个配置文件
不修改任何现有文件
```

### 15.8 实施注意事项

```
1. MuJoCo XML 扩展方式
   SafetyPointMultiGoal 的 XML 由 Safety-Gymnasium 动态生成
   扩展方式: 在生成后的 XML string 中插入额外 agent body
   或: 继承 Safety-Gymnasium 的 Builder 类, override 生成逻辑

2. qpos/qvel 索引
   SafetyPointMultiGoal 的 Point agent 使用 freejoint (7 DOF qpos)
   N 个 agent: qpos 长度 = N * 7 (x,y,z,qw,qx,qy,qz per agent)
   位置提取: positions[i] = data.qpos[7*i : 7*i+2]

3. MultiNavAtacom._get_q() 兼容
   现有 _get_q() 针对 2 agent 硬编码
   FormationNavAtacom 需要 override 支持 N 个 agent

4. 与 RMPflow 的集成是可选的
   Phase 1: 先用手工解析 Jacobian（与现有 MultiNavAtacom 一致）
   Phase 2: 接入 RMPflow 叶节点 Jacobian（验证几何一致性）
   两种方式的约束投影结果应完全相同

5. 渲染和可视化
   Safety-Gymnasium 自带 MuJoCo 渲染
   额外 agent 会自动出现在渲染中
   可通过 rgba 区分不同 agent
```
