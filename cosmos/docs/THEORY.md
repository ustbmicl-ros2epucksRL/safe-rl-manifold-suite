# 理论基础：ATACOM + RMPflow + COSMOS

本文档详细总结项目所依赖的三大理论支柱：
1. **ATACOM**：约束流形上的安全强化学习（单智能体基础理论）
2. **RMPflow**：黎曼运动策略的计算图框架
3. **COSMOS**：COordinated Safety On Manifold for multi-agent Systems（本项目核心贡献）

---

## 0. 理论关系架构图

### 0.1 MARL + 流形约束 + 黎曼流形 整体关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COSMOS 理论架构                                       │
│         (MARL + 约束流形 + 黎曼流形 的融合)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      MARL 层 (MAPPO)                                  │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  多智能体策略学习: π_θ(α|o)                                      │  │  │
│  │  │  • 集中训练分散执行 (CTDE)                                       │  │  │
│  │  │  • 共享参数的 Actor-Critic                                       │  │  │
│  │  │  • 输出原始动作 α ∈ ℝ^d (未经安全约束)                           │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │ α (原始动作)                              │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    约束流形层 (ATACOM/COSMOS)                          │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  约束流形: M = {q ∈ ℝ^n : c(q) ≤ 0}                             │  │  │
│  │  │                                                                 │  │  │
│  │  │  核心投影公式:                                                   │  │  │
│  │  │  ┌─────────────────────────────────────────────────────────┐    │  │  │
│  │  │  │  dq = N_c · α + (-K_c · J_c⁺ · c(q))                    │    │  │  │
│  │  │  │       ├─────┘         └──────────────┘                  │    │  │  │
│  │  │  │   切空间运动            误差修正                         │    │  │  │
│  │  │  │  (零空间投影)         (拉回流形边界)                     │    │  │  │
│  │  │  └─────────────────────────────────────────────────────────┘    │  │  │
│  │  │                                                                 │  │  │
│  │  │  约束类型:                                                       │  │  │
│  │  │  • 智能体间碰撞: c_ij = r_safe - ||p_i - p_j||                  │  │  │
│  │  │  • 障碍物碰撞:   c_ik = r_obs - ||p_i - o_k||                   │  │  │
│  │  │  • 边界约束:     c_b = ||p_i|| - arena_bound                    │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │ dq_safe (约束满足的动作)                  │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    黎曼流形层 (RMPflow)                                │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  黎曼运动策略: RMP = (a(x,ẋ), M(x,ẋ))                           │  │  │
│  │  │  • a: 加速度策略 (任务空间中的期望运动)                          │  │  │
│  │  │  • M: 黎曼度量 (半正定，编码方向重要性)                          │  │  │
│  │  │                                                                 │  │  │
│  │  │  树形聚合:                                                       │  │  │
│  │  │  ┌─────────────────────────────────────────────────────────┐    │  │  │
│  │  │  │  Pushforward: x_child = ψ(x_parent), ẋ = J·ẋ_parent     │    │  │  │
│  │  │  │  Pullback:    f += J^T·f_child, M += J^T·M_child·J      │    │  │  │
│  │  │  │  Resolve:     a = M_root⁺ · f_root                      │    │  │  │
│  │  │  └─────────────────────────────────────────────────────────┘    │  │  │
│  │  │                                                                 │  │  │
│  │  │  叶策略 (几何先验):                                              │  │  │
│  │  │  • GoalAttractor: 目标吸引力                                    │  │  │
│  │  │  • CollisionAvoidance: 障碍斥力                                 │  │  │
│  │  │  • FormationControl: 编队形状保持                               │  │  │
│  │  │  • Damper: 速度阻尼                                             │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │ f_rmp (几何引导力)                        │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        动作融合层                                      │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  u_final = dq_safe + β · f_rmp                                  │  │  │
│  │  │            ├──────┘       └─────┘                               │  │  │
│  │  │         安全保证       几何引导                                  │  │  │
│  │  │        (硬约束)       (软优化)                                   │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │ u_final (最终安全动作)                    │
│                                  ▼                                          │
│                          多智能体环境执行                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 0.2 三层架构的职责划分

| 层级 | 核心方法 | 数学空间 | 职责 | 保证 |
|------|---------|---------|------|------|
| **MARL 层** | MAPPO | 动作空间 ℝ^d | 学习最优策略 | 奖励最大化 |
| **约束流形层** | ATACOM/COSMOS | 约束流形 M | 安全投影 | **硬约束满足** |
| **黎曼流形层** | RMPflow | 黎曼流形 (M, g) | 几何引导 | 平滑运动 |

### 0.3 信息流与梯度流

```
                    前向传播 (动作生成)
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │   观测 o ──▶ MAPPO ──▶ α ──▶ ATACOM ──▶ dq_safe  │
    │              π_θ           N_c·α + K_c·J_c⁺·c   │
    │                                    │            │
    │                                    ▼            │
    │                              RMPflow ──▶ f_rmp  │
    │                                    │            │
    │                                    ▼            │
    │                              u = dq_safe + βf   │
    │                                                  │
    └──────────────────────────────────────────────────┘

                    反向传播 (策略学习)
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │   ∇_θ J = E[∇_θ log π_θ(α|o) · A(s,α)]          │
    │           ↑                                      │
    │   注意: RL 学习的是 α 到奖励的映射               │
    │         ATACOM 层对 α 连续可微，梯度可传播       │
    │         安全投影对策略学习是"透明"的             │
    │                                                  │
    └──────────────────────────────────────────────────┘
```

### 0.4 为什么需要三层？

| 单独使用 | 问题 | COSMOS 解决方案 |
|---------|------|----------------|
| **只有 MARL** | 无安全保证，碰撞频繁 | + 约束流形层提供硬安全 |
| **只有 ATACOM** | 无学习能力，策略固定 | + MARL 层学习最优行为 |
| **只有 RMPflow** | 软约束，参数敏感 | + 约束流形层兜底 |
| **MARL + ATACOM** | 缺乏几何先验，学习慢 | + RMPflow 提供编队引导 |

---

## 1. ATACOM 原理详解

### 1.1 核心思想

**ATACOM** (Acting on the Tangent space of the Constraint Manifold) 是一种将安全约束"硬编码"到动作空间的方法，确保智能体在**整个学习过程中**（包括探索阶段）都满足安全约束。

```
核心洞察：
  约束 c(q) ≤ 0 定义一个约束流形 M = {q : c(q) = 0}
  智能体在流形的切空间 T_q M 中探索
  ATACOM 投影保证动作始终在切空间内
```

### 1.2 数学公式

#### 基本投影公式

对于状态 $q$ 和约束 $c(q) \leq 0$：

```
J_c = ∂c/∂q                              (约束 Jacobian)
N_c = I - J_c^+ J_c                      (零空间投影矩阵)
dq = N_c · α + (-K_c · J_c^+ · c(q))
     ├────────┘   └─────────────────┘
     切空间运动     误差修正（拉回流形）
```

其中：
- $\alpha \in \mathbb{R}^{d_\alpha}$ 是 RL 策略的原始输出
- $J_c^+ = J_c^T (J_c J_c^T + \epsilon I)^{-1}$ 是阻尼伪逆
- $K_c > 0$ 是约束修正增益
- $N_c$ 投影到约束的零空间（切空间）

#### Slack 变量处理不等式约束

对于不等式约束 $c(q) \leq 0$，引入 slack 变量 $s$：

```
c(q) + \phi(s) = 0     (转换为等式约束)

其中 \phi(s) 是 slack 惩罚函数：
  softcorner: φ(s) = -log(-expm1(βs)) / β   (锐利，推荐)
  softplus:   φ(s) = log(1 + exp(βs)) / β   (平滑)
  square:     φ(s) = β · s²                  (简单)
```

增广 Jacobian 变为：
```
J_aug = [J_c | diag(∂φ/∂s)]   (dim_out × (dim_q + dim_s))
```

### 1.3 理论保证

根据 Liu et al. 2024 (IEEE T-RO) 的分析：

| 性质 | 保证 |
|------|------|
| **约束满足** | 若初始 $c(q_0) < 0$，则 $\forall t > 0: c(q_t) < 0$ |
| **连续可微** | $\text{ATACOM}(q, \alpha)$ 关于 $\alpha$ 连续可微，可用于策略梯度 |
| **零空间探索** | RL 策略仅在不违反约束的方向上探索 |
| **Slack 收敛** | 在温和条件下，slack 变量收敛到使 $c + \phi(s) = 0$ |

### 1.4 与其他安全 RL 方法对比

| 方法 | 安全保证 | 学习效率 | 适用约束类型 |
|------|---------|---------|-------------|
| **ATACOM** | 硬保证（零空间投影） | 高（无约束 RL） | 状态约束 |
| Lagrangian | 软约束（期望） | 中 | 期望约束 |
| CBF | 硬保证（李雅普诺夫） | 需设计 | 安全集 |
| Safety Layer | 硬保证（投影） | 中 | 线性约束 |
| CPO/TRPO-Lag | 软约束（KL 惩罚） | 低 | 期望约束 |

---

## 2. RMPflow 原理详解

### 2.1 核心概念：RMP

**RMP** (Riemannian Motion Policy) 是一个二阶动力系统与黎曼度量的组合：

```
RMP = (a(x, ẋ), M(x, ẋ))

其中：
  a : 加速度策略（在任务空间中想怎么动）
  M : 黎曼度量（描述哪个方向更重要，半正定矩阵）
```

**关键创新**：通过黎曼度量 $M$ 编码每个子任务的重要性和方向偏好，避免简单力叠加导致的干扰和振荡。

### 2.2 计算图结构

RMPflow 将 RMP 工程化为树形计算图：

```
节点类型：
┌─────────────────────────────────────────────────────────┐
│  RMPRoot  — 根节点（联合配置空间，如 [x1,y1,...,xn,yn]） │
│  RMPNode  — 中间节点（子任务空间映射）                   │
│  RMPLeaf  — 叶节点（具体策略，输出 f 和 M）              │
└─────────────────────────────────────────────────────────┘
```

### 2.3 两个核心操作

#### Pushforward（前推）：状态从根向叶传播

```
x_child = ψ(x_parent)             # 任务空间映射
ẋ_child = J · ẋ_parent            # 速度通过 Jacobian 转换

其中 J = ∂ψ/∂x_parent
```

#### Pullback（回拉）：力和度量从叶向根聚合

```
f_parent += J^T · (f_child - M_child · J̇ · ẋ_parent)
M_parent += J^T · M_child · J
```

**几何一致性定理**：若每个叶节点的 $M_k$ 半正定，则聚合后的 $M_{\text{root}}$ 也半正定，解 $a = M_{\text{root}}^+ \cdot f_{\text{root}}$ 存在且唯一。

### 2.4 典型叶策略

| 叶策略 | 任务空间映射 $\psi(q)$ | 力场 $f$ | 用途 |
|--------|----------------------|---------|------|
| **GoalAttractorUni** | $q - \text{goal}$ | $-\gamma \tanh(\alpha \|x\|) \frac{x}{\|x\|} - \eta w \dot{x}$ | 目标吸引 |
| **CollisionAvoidance** | $\|q - \text{obs}\|/R - 1$ | $-\frac{\partial \Phi}{\partial x} - \eta g \dot{x}$ | 斥力避障 |
| **FormationDecentralized** | $\|q - q_j\| - d_{ij}$ | $-\gamma x w - \eta w \dot{x}$ | 编队距离保持 |
| **Damper** | $q$ | $-\eta w \dot{x}$ | 速度阻尼 |

### 2.5 编队导航树结构示例

以 4 智能体正方形编队为例：

```
RMPRoot (8维: [x1,y1, x2,y2, x3,y3, x4,y4])
│
├── Agent_0 节点 (ψ: 提取 [x1,y1])
│   ├── GoalAttractor_0           ← 吸引向目标点
│   ├── CollisionAvoidance_01,02,03  ← 对其他 3 个智能体避碰
│   ├── FormationDecentralized_01,02,03  ← 与其他 3 个智能体保持期望距离
│   └── Damper_0                  ← 速度阻尼
│
├── Agent_1 节点 (类似结构)
├── Agent_2 节点
└── Agent_3 节点

总叶节点数 = 4 × (1 + 3 + 3 + 1) = 32
```

### 2.6 每步执行流程

```python
for t in range(T):
    # 1. 设置根节点状态
    root.set_root_state(x_all, x_dot_all)

    # 2. Pushforward: 状态从根传播到所有叶节点
    root.pushforward()

    # 3. 更新动态参数（分布式策略需要最新位置）
    for leaf in collision_leaves + formation_leaves:
        leaf.update_params()

    # 4. Pullback: 叶节点计算 (f, M)，沿 J^T 聚合回根
    root.pullback()

    # 5. Resolve: 求解加速度
    a = M_root⁺ · f_root

    # 6. 积分更新
    v += a * dt
    x += v * dt
```

### 2.7 RMPflow 的局限性

| 问题 | 说明 | 本项目解决方案 |
|------|------|---------------|
| **软约束** | 斥力无法保证 100% 不碰撞 | ATACOM 硬约束 |
| **参数敏感** | 权重需手工调参 | RL 自动学习 |
| **无学习能力** | 纯几何，策略固定 | MAPPO 策略学习 |
| **复杂约束** | 难以处理动态优先级 | COSMOS 优先级系统 |

### 2.8 RMPflow 在 COSMOS 中的作用

#### 2.8.1 核心作用：几何先验引导

RMPflow 在 COSMOS 框架中扮演**几何先验提供者**的角色，而非安全保证者：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RMPflow 在 COSMOS 中的定位                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   问题: MARL + ATACOM 可以保证安全，但学习效率低                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  原因:                                                          │   │
│   │  • RL 策略从零开始探索，缺乏几何直觉                             │   │
│   │  • 编队任务的几何结构（距离、角度）需要大量样本学习              │   │
│   │  • ATACOM 只保证安全，不提供任务引导                            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   解决: RMPflow 提供几何先验                                            │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  作用:                                                          │   │
│   │  • 编队形状保持 → FormationDecentralized 叶策略                 │   │
│   │  • 目标导航 → GoalAttractor 叶策略                              │   │
│   │  • 运动平滑 → Damper 叶策略                                     │   │
│   │  • 黎曼度量 M 自动平衡多目标冲突                                │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 2.8.2 与 ATACOM 的协作关系

```
               ATACOM (硬约束)              RMPflow (软引导)
                    │                            │
                    ▼                            ▼
            ┌──────────────┐            ┌──────────────┐
            │  安全边界    │            │  期望轨迹    │
            │  "不能做什么" │            │  "应该做什么" │
            └──────┬───────┘            └──────┬───────┘
                   │                           │
                   │    ┌──────────────┐      │
                   └───▶│   COSMOS     │◀─────┘
                        │   动作融合    │
                        └──────┬───────┘
                               │
                               ▼
                        最终安全动作
```

**职责分工**：

| 组件 | 职责 | 性质 | 失败后果 |
|------|------|------|---------|
| **ATACOM** | 约束满足 | 硬约束 | 碰撞 (不可接受) |
| **RMPflow** | 轨迹优化 | 软约束 | 编队偏差 (可接受) |

#### 2.8.3 RMPflow 提供的具体引导

**1. 编队形状保持**

```python
# FormationDecentralized 叶策略
# 每对智能体 (i,j) 保持期望距离 d_ij

ψ(q) = ||q_i - q_j|| - d_ij    # 任务空间：距离误差

f = -γ · ψ · w(ψ) - η · w(ψ) · ψ̇   # 弹簧+阻尼力
M = w(ψ) · I                        # 黎曼度量

# w(ψ) 为权重函数：距离误差越大，权重越高
```

**2. 目标吸引**

```python
# GoalAttractor 叶策略
# 智能体被吸引向目标点

ψ(q) = q - q_goal                   # 任务空间：到目标位移

f = -γ · tanh(α||ψ||) · ψ/||ψ|| - η · w · ψ̇   # 有界吸引力
M = w(ψ) · I

# tanh 确保吸引力有界，避免远距离时力过大
```

**3. 运动平滑 (阻尼)**

```python
# Damper 叶策略
# 抑制速度振荡

f = -η · w · ẋ    # 阻尼力
M = w · I

# 防止智能体在约束边界附近振荡
```

#### 2.8.4 黎曼度量的作用

黎曼度量 $M$ 是 RMPflow 的核心创新，解决多目标冲突：

```
场景：智能体同时受到目标吸引和编队约束

传统方法 (简单力叠加):
  f_total = f_goal + f_formation
  问题: 权重如何设置？不同距离下是否应该变化？

RMPflow 方法 (黎曼几何):
  (f_total, M_total) = Pullback(
      (f_goal, M_goal),
      (f_formation, M_formation)
  )
  a = M_total⁺ · f_total

  优势:
  • M 自动编码每个力的"重要性方向"
  • 近距离时避障 M 大，主导运动
  • 远距离时目标 M 大，主导运动
  • 无需手工设计距离-权重函数
```

#### 2.8.5 RMPflow 如何加速 RL 学习

```
┌───────────────────────────────────────────────────────────────────────┐
│                  学习效率对比                                          │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   无 RMPflow (纯 MARL + ATACOM):                                      │
│   ┌─────────────────────────────────────────────────────────────────┐ │
│   │  • RL 需要学习: 编队形状 + 目标导航 + 避碰 + 速度控制           │ │
│   │  • 探索空间: 整个安全动作空间                                    │ │
│   │  • 收敛: ~1000 episodes                                         │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│   有 RMPflow (COSMOS 完整框架):                                       │
│   ┌─────────────────────────────────────────────────────────────────┐ │
│   │  • RL 只需学习: 在 RMPflow 引导上的微调                         │ │
│   │  • 探索空间: RMPflow 基线附近的残差                             │ │
│   │  • 收敛: ~200 episodes                                          │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│   公式:                                                               │
│     α_rl = α_rmpflow + Δα_learned                                    │
│     RL 学习的是残差 Δα，而非完整策略                                  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

#### 2.8.6 实现中的融合方式

```python
def cosmos_step(obs, positions, velocities):
    # 1. MAPPO 输出原始动作
    alpha_rl = mappo.get_action(obs)

    # 2. RMPflow 计算几何引导力
    rmp_tree.set_root_state(positions, velocities)
    rmp_tree.pushforward()
    rmp_tree.pullback()
    f_rmp = rmp_tree.resolve()  # 黎曼几何计算的加速度

    # 3. 融合 RL 动作和 RMPflow 引导
    alpha_combined = alpha_rl + beta * f_rmp  # β 为混合系数

    # 4. ATACOM 安全投影 (硬约束)
    dq_safe = atacom.project(alpha_combined, positions)

    return dq_safe
```

#### 2.8.7 总结：RMPflow 的三重作用

| 作用 | 描述 | 对应叶策略 |
|------|------|-----------|
| **几何先验** | 提供编队形状、目标导航的几何知识 | FormationDecentralized, GoalAttractor |
| **多目标平衡** | 黎曼度量自动解决力冲突 | 所有叶策略的 M 聚合 |
| **学习加速** | 将 RL 探索空间从全策略缩小到残差 | 作为 baseline 被 RL 修正 |

**关键洞察**：RMPflow 不提供安全保证（这是 ATACOM 的职责），但它使得安全约束下的学习更加高效。

---

## 3. COSMOS：多智能体流形协调安全

**COSMOS** (COordinated Safety On Manifold for multi-agent Systems) 是本项目的核心贡献，将约束流形方法扩展到多智能体系统。

### 3.1 多智能体安全的独特挑战

| 挑战 | 描述 | 复杂度 |
|------|------|--------|
| **耦合动力学** | Agent i 的安全依赖 Agent j 的动作 | $O(N^2)$ 约束 |
| **信息结构** | 集中式 vs 分布式知识 | 通信拓扑 |
| **可扩展性** | $N$ 智能体有 $N(N-1)/2$ 对碰撞约束 | 组合爆炸 |
| **死锁** | 相互阻塞导致无法移动 | 需协调机制 |
| **活锁** | 振荡行为，竞争目标 | 优先级调度 |
| **公平性** | 约束满足在智能体间的均衡 | 权重分配 |

### 3.2 COSMOS 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      COSMOS 系统架构                         │
│      (COordinated Safety On Manifold for multi-agent Systems) │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌───────────────┐      ┌───────────────┐                  │
│   │  MAPPO 策略   │──α──▶│  优先级计算   │                  │
│   │  (RL 探索)    │      │  (危险度评估)  │                  │
│   └───────────────┘      └───────┬───────┘                  │
│                                  │                          │
│   ┌──────────────────────────────┼──────────────────────┐   │
│   │                              ▼                       │   │
│   │  ┌────────────────────────────────────────────────┐ │   │
│   │  │           约束系统 (C-COSMOS / D-COSMOS)        │ │   │
│   │  ├────────────────────────────────────────────────┤ │   │
│   │  │  • 智能体间碰撞: c_ij = -||qi-qj|| + r_safe    │ │   │
│   │  │  • 障碍物碰撞:   c_ik = -||qi-ok|| + r_obs     │ │   │
│   │  │  • 边界约束:     c_b  = |qi| - arena_bound     │ │   │
│   │  │  • 耦合约束:     c_形状, c_连通性              │ │   │
│   │  └────────────────────────────────────────────────┘ │   │
│   │                              │                       │   │
│   │                              ▼                       │   │
│   │  ┌────────────────────────────────────────────────┐ │   │
│   │  │              零空间投影                         │ │   │
│   │  │  dq = N_c · α_优先级加权 + (-K_c · J_c⁺ · c)  │ │   │
│   │  └────────────────────────────────────────────────┘ │   │
│   │                                                      │   │
│   └──────────────────────────────┼──────────────────────┘   │
│                                  │                          │
│   ┌──────────────────────────────┼──────────────────────┐   │
│   │                              ▼                       │   │
│   │  ┌────────────────────────────────────────────────┐ │   │
│   │  │              CBF 安全层                         │ │   │
│   │  │  ∇h · q̇ ≥ -α_cbf · h   (前向不变性保证)       │ │   │
│   │  └────────────────────────────────────────────────┘ │   │
│   │                              │                       │   │
│   │                              ▼                       │   │
│   │  ┌────────────────────────────────────────────────┐ │   │
│   │  │            RMPflow 编队力混合                   │ │   │
│   │  │  dq_final = dq_safe + β · f_formation          │ │   │
│   │  └────────────────────────────────────────────────┘ │   │
│   │                                                      │   │
│   └──────────────────────────────┼──────────────────────┘   │
│                                  │                          │
│                                  ▼                          │
│                          安全动作输出                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 集中式 vs 分布式 COSMOS

#### 集中式 COSMOS (C-COSMOS)

```
联合配置空间: q_joint = [q_1, q_2, ..., q_N] ∈ R^{2N}

优点:
  • 全局最优解
  • 考虑所有智能体耦合
  • 避免分布式冲突

缺点:
  • O(N²) 约束规模
  • 需要全局通信
  • 计算复杂度高
```

#### 分布式 COSMOS (D-COSMOS)

```
每个智能体独立求解，使用邻居信息:

Agent i: dq_i = N_c^i · α_i + error_correction_i

优点:
  • 可扩展性好
  • 仅需局部通信
  • 并行计算

缺点:
  • 可能不一致
  • 需要处理冲突
  • 保守估计
```

### 3.4 耦合约束

#### 编队形状约束

保持多智能体围成的多边形面积不低于阈值：

```python
# 三角形面积 (Shoelace 公式)
Area = 0.5 * |det([p1-p0, p2-p0])|

约束: Area_desired - Area ≤ 0  (面积不能太小)
```

#### 连通性约束

确保编队不会过于分散：

```python
# 最大智能体间距离
max_dist = max(||p_i - p_j||, ∀i,j)

约束: max_dist - connectivity_bound ≤ 0
```

### 3.5 优先级机制

动态优先级基于危险程度：

```python
def compute_priority(agent_i):
    danger = 0.0

    # 智能体间危险
    for j in other_agents:
        d = distance(agent_i, agent_j)
        if d < 2 * safety_radius:
            danger += (2 * safety_radius - d) / safety_radius

    # 障碍物危险
    for obs in obstacles:
        d = distance(agent_i, obs)
        if d < 2 * safety_radius:
            danger += (2 * safety_radius - d) / safety_radius

    # 边界危险
    margin = boundary - |position|
    if margin < safety_radius:
        danger += (safety_radius - margin) / safety_radius

    return 1.0 + danger  # 危险越高，优先级越高
```

高优先级智能体获得更小的零空间（更受约束），低优先级智能体有更多移动自由度。

### 3.6 死锁检测与解决

#### 检测

```python
def detect_deadlock():
    # 过去 N 步的平均移动量
    avg_movement = mean(||p_t - p_{t-1}||, t in last N steps)

    # 约束是否活跃（智能体在尝试移动但被阻止）
    constraints_active = num_active_constraints > threshold

    return avg_movement < epsilon AND constraints_active
```

#### 解决

```python
def resolve_deadlock(alphas):
    priorities = compute_priorities()

    # 低优先级智能体添加小扰动
    perturbation = random_normal() * (1 / priorities)

    return alphas + perturbation
```

### 3.7 安全证书

实时计算安全边界：

```python
certificate = {
    "min_inter_agent_margin": min(||p_i - p_j|| - r_safe),
    "min_obstacle_margin": min(||p_i - o_k|| - r_obs),
    "min_boundary_margin": min(arena_bound - |p_i|),
    "formation_area": shoelace_area(positions),
    "is_safe": all margins > 0
}
```

---

## 4. 三者融合：COSMOS + RMPflow + MAPPO

### 4.1 架构图

```
        RL 策略 (MAPPO)              RMPflow 叶策略
              │                           │
              ▼                           ▼
         原始动作 α               编队/避碰几何力 f_rmp
              │                           │
              └───────────┬───────────────┘
                          ▼
                   COSMOS 安全滤波器
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
     零空间投影      CBF 安全校正    RMPflow 混合
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                     安全动作输出
                          │
                          ▼
                    多智能体环境
```

### 4.2 各组件职责

| 组件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **MAPPO** | 学习高层决策和探索 | 观测 | 原始动作 α |
| **RMPflow** | 提供编队几何先验 | 位置、速度 | 编队力 f |
| **COSMOS** | 硬编码安全约束 | α, 位置, 速度 | 安全速度 dq |
| **CBF** | 额外安全层 | dq, 状态 | 修正后 dq |

### 4.3 训练流程

```python
for episode in range(num_episodes):
    obs = env.reset()
    cosmos.reset(positions)

    for step in range(max_steps):
        # 1. RL 策略输出原始动作
        alphas, log_probs = mappo.get_actions(obs)

        # 2. COSMOS 安全投影
        safe_actions = cosmos.project(
            alphas, positions, velocities, dt
        )

        # 3. 环境步进
        obs, rewards, costs, dones, infos = env.step(safe_actions)

        # 4. 存储经验（使用原始 α，不是 safe_actions）
        buffer.insert(obs, alphas, log_probs, rewards, ...)

    # 5. PPO 更新
    mappo.update(buffer)
```

**关键点**：RL 策略学习的是原始动作 α 到奖励的映射，COSMOS 是透明的安全层。

---

## 5. 论文引用

### ATACOM 系列

```bibtex
@inproceedings{liu2021atacom,
  title={Robot Reinforcement Learning on the Constraint Manifold},
  author={Liu, Puze and Zhang, Kuo and Tateo, Davide and Jauhri, Snehal and Peters, Jan and Chalvatzaki, Georgia},
  booktitle={CoRL},
  year={2021}
}

@article{liu2024atacom,
  title={Safe Reinforcement Learning on the Constraint Manifold: Theory and Applications},
  author={Liu, Puze and Zhang, Kuo and Tateo, Davide and Jauhri, Snehal and Peters, Jan and Chalvatzaki, Georgia},
  journal={IEEE T-RO},
  year={2024}
}
```

### RMPflow 系列

```bibtex
@article{ratliff2018rmp,
  title={Riemannian Motion Policies},
  author={Ratliff, Nathan and Issac, Jan and Kappler, Daniel and Birchfield, Stan and Fox, Dieter},
  journal={arXiv:1801.02854},
  year={2018}
}

@article{cheng2021rmpflow,
  title={RMPflow: A Computational Graph for Automatic Motion Policy Generation},
  author={Cheng, Ching-An and Mukadam, Mustafa and Issac, Jan and Birchfield, Stan and Fox, Dieter and Boots, Byron and Ratliff, Nathan},
  journal={IEEE T-ASE},
  year={2021}
}

@inproceedings{li2019multirobot,
  title={Multi-Objective Policy Generation for Multi-Robot Systems Using Riemannian Motion Policies},
  author={Li, Anqi and Mukadam, Mustafa and Egerstedt, Magnus and Boots, Byron},
  booktitle={ISRR},
  year={2019}
}
```

### 多智能体强化学习

```bibtex
@inproceedings{yu2022mappo,
  title={The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games},
  author={Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan and Wang, Yu and Bayen, Alexandre and Wu, Yi},
  booktitle={NeurIPS},
  year={2022}
}
```

---

## 6. 实现文件索引

| 文件 | 内容 |
|------|------|
| `safety/cosmos.py` | **COSMOS 完整实现** (COordinated Safety On Manifold for multi-agent Systems) |
| `safety/atacom.py` | 基础 ATACOM 实现（向后兼容） |
| `safety/constraints.py` | 约束和 Slack 变量 |
| `safety/rmp_policies.py` | RMPflow 编队策略 |
| `algo/mappo.py` | MAPPO 算法 |
| `env/formation_env.py` | 编队导航环境 |
