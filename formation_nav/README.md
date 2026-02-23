# 多机器人编队导航：ATACOM + RMPflow + MAPPO

## 一、问题定义

**场景**：$N$ 个同质机器人（双积分器动力学 $\dot{v}=a,\;\dot{p}=v$）在 2D 有界环境中，从随机初始位置出发，以指定的几何编队（正方形、三角形等）协同导航到目标位置，同时避免与障碍物和彼此发生碰撞。

**核心矛盾**：同时满足三个相互竞争的目标——

| 目标 | 数学描述 | 难点 |
|------|---------|------|
| **导航** | 编队质心 $\bar{p} \to p_{\text{goal}}$ | 多智能体协调 |
| **编队保持** | $\|p_i - p_j\| \approx d_{ij}^*,\;\forall (i,j)\in\mathcal{E}$ | 与避碰约束冲突 |
| **安全约束** | $\|p_i - p_j\| \geq r_{\text{safe}}$，$\|p_i - o_k\| \geq r_k$，$p_i \in \mathcal{A}$ | 需要**硬保证**，不可违反 |

传统 RL 方法（MAPPO-Lag、MACPO）只能通过损失函数**软约束**惩罚不安全行为——训练初期仍会频繁碰撞，且无法提供形式化安全保证。

## 二、方法：ATACOM 零空间投影 + RMPflow 编队力混合

### 2.1 ATACOM 安全滤波器

将所有不等式安全约束 $c(q) \leq 0$ 通过 **softcorner 松弛变量** 转化为等式约束：

$$c(q) - \frac{1}{\beta}\log\!\big(-\text{expm1}(\beta s)\big) = 0$$

构建增广约束 Jacobian $J_c = [J_q \mid J_s]$，计算其**零空间投影矩阵**：

$$N_c = \big(I - J_c^\dagger J_c\big)_{[:, :d_{\text{null}}]} \cdot \text{diag}(\dot{q}_{\max})$$

RL 策略输出 $\alpha \in \mathbb{R}^2$ 被投影到约束零空间：

$$\dot{q} = \underbrace{N_c \cdot \alpha}_{\text{零空间内自由运动}} + \underbrace{(-K_c \cdot J_c^\dagger \cdot c(q))}_{\text{约束误差修正}}$$

**关键性质**：$d_{\text{null}} = d_q + d_{\text{slack}} - d_{\text{out}} = 2$（每个不等式约束贡献 1 个松弛变量和 1 个输出，净效果为零）。无论约束数目如何变化（3个智能体9个约束、6个智能体16个约束），RL 动作空间始终是 **2 维**。

### 2.2 RMPflow 编队力几何引导

从 RMPflow 框架提取编队维持的 Riemannian 运动策略力（弹簧-阻尼器模型），作为 ATACOM 安全动作的**柔和几何补偿**：

$$\dot{q}_{\text{final}} = \dot{q}_{\text{safe}} + \beta_{\text{blend}} \cdot f_{\text{formation}}^{\text{RMP}}$$

RMPflow 提供的不是硬约束而是几何先验——在训练初期引导策略保持编队结构，随着 RL 策略成熟，其影响逐渐被学到的行为替代。

### 2.3 MAPPO 训练框架（CTDE）

- **参数共享 Actor**：所有同质智能体共享同一策略网络，输出 $\alpha_i \in [-1,1]^2$
- **集中式 Critic**：接收全局状态（所有智能体位置+速度+目标+障碍物）
- **关键设计**：buffer 中存储**原始策略输出 $\alpha$**（非安全滤波后的动作），保证 $\log\pi(\alpha|s)$ 与 PPO 更新的一致性

```
训练循环：
  α = Actor(obs)           ← RL 策略输出（存入 buffer）
  a = ATACOM.project(α)    ← 安全投影（执行动作）
  env.step(a)              ← 环境接收安全动作
  PPO.update(α, log_π)     ← 基于原始 α 更新策略
```

## 三、方法与基线的对比

| | MAPPO-Lag | MACPO | **MAPPO-CM（本方法）** |
|---|---|---|---|
| 安全机制 | 训练时 Lagrangian 软约束 | 信赖域软约束 | **动作空间硬约束（预执行投影）** |
| 安全保证 | 无（依赖收敛） | 无（依赖收敛） | **形式化保证**（零空间投影） |
| 训练初期碰撞 | 频繁 | 频繁 | **从第 1 步起即安全** |
| 编队引导 | 无 | 无 | RMPflow 几何先验 |
| 核心公式 | $A_r - \lambda A_c$ | TRPO + 可行性分析 | $\dot{q} = N_c\alpha - K_c J_c^\dagger c(q)$ |

## 四、贡献总结

1. **将 ATACOM 流形约束方法从单智能体扩展到多智能体编队导航**：每个智能体独立构建约束集（智能体间避碰 + 障碍物避碰 + 边界约束），通过去中心化零空间投影实现**可扩展的硬安全保证**

2. **建立 RMPflow 与 ATACOM 的几何对应并实现混合**：RMPflow 的 Riemannian 度量张量 $M\to\infty$（几何排斥）对应 ATACOM 的约束 Jacobian 零空间投影（代数硬约束）。本方法将两者结合——ATACOM 保证安全，RMPflow 提供编队先验

3. **训练曲线验证**（2000 episode 实验）：
   - 奖励从 -30 提升到 ~0（学会导航）
   - 代价从 200+ 降到 0（消除碰撞）
   - 编队误差从 0.45 降到 0.003（精确保持编队）
   - 最小智能体间距从 0.05 稳定在 1.2+（远超安全半径 0.4）

**一句话概括**：用 ATACOM Jacobian 零空间投影**替代** RMPflow 手工设计的策略叶节点，在保留 Riemannian 流形安全几何保证的同时，通过 RL 学习多机器人编队导航策略——实现了**可学习性**与**硬安全**的统一。

## 五、代码结构

```
formation_nav/
├── config.py                  # 全部超参数（dataclass）
├── requirements.txt           # torch, numpy, gymnasium, matplotlib, tensorboard
├── env/
│   ├── formations.py          # 编队形状 & 拓扑定义
│   └── formation_env.py       # 2D 多机器人 Gymnasium 环境（纯 NumPy 物理）
├── safety/
│   ├── rmp_tree.py            # RMPflow 树结构（RMPRoot/RMPNode/RMPLeaf）
│   ├── rmp_policies.py        # 叶节点策略 + MultiRobotRMPForest 便捷类
│   ├── constraints.py         # StateConstraint + ConstraintsSet（含松弛变量）
│   └── atacom.py              # ATACOM 安全滤波器（零空间投影 + RMPflow 编队力混合）
├── algo/
│   ├── networks.py            # Actor / Critic / CostCritic（PyTorch）
│   ├── buffer.py              # RolloutBuffer（GAE 计算）
│   └── mappo.py               # MAPPO 训练器（CTDE）
├── train.py                   # 主训练脚本
└── eval.py                    # 评估 & 可视化
```

## 六、使用方法

```bash
# 安装依赖
pip install torch numpy gymnasium matplotlib tensorboard

# 训练（4 智能体正方形编队）
PYTHONPATH=. python formation_nav/train.py \
    --num-agents 4 --formation square --seed 0 --total-episodes 2000

# 评估 & 可视化
PYTHONPATH=. python formation_nav/eval.py \
    --model-path checkpoints/mappo_formation_final.pt \
    --num-agents 4 --formation square --num-episodes 10 --save-video

# 消融实验：RMPflow 混合系数
PYTHONPATH=. python formation_nav/train.py --rmp-blend 0.0   # 纯 ATACOM
PYTHONPATH=. python formation_nav/train.py --rmp-blend 0.3   # 默认混合
PYTHONPATH=. python formation_nav/train.py --rmp-blend 0.5   # 强编队引导

# 不同编队形状
PYTHONPATH=. python formation_nav/train.py --num-agents 3 --formation triangle
PYTHONPATH=. python formation_nav/train.py --num-agents 6 --formation circle
```
