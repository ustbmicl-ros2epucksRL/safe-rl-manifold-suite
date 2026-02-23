# ATACOM博士论文学习笔记

## 论文信息

| 项目 | 内容 |
|------|------|
| **标题** | Safe Reinforcement Learning for Robotics: From Exploration to Policy Learning |
| **作者** | Puze Liu |
| **机构** | TU Darmstadt, Intelligent Autonomous Systems Lab |
| **导师** | Prof. Jan Peters (主), Prof. Marco Pavone (副) |
| **提交时间** | 2024年10月，答辩2024年12月6日 |
| **类型** | 博士论文 (Dr. rer. nat.)，约190页含附录 |

---

## 0. 这篇论文解决什么问题？

### 0.1 总问题：RL在真实机器人上不安全

强化学习(RL)在仿真中表现优秀，但部署到真实机器人时面临一个根本矛盾：

> **RL需要通过"试错"来学习，但真实机器人试错的代价是不可接受的** —— 关节过力矩会烧毁电机，碰撞会损坏机器人和环境中的人。

现有Safe RL方法（如PPO-Lagrangian、CPO、WCSAC）通过**在损失函数中加入约束惩罚项**来间接约束策略，但这只能保证**训练收敛后**策略大致安全，无法保证**训练过程中每一步**都安全。这对真实机器人是不够的。

### 0.2 核心研究问题

> **"需要多少领域知识（动力学模型、约束函数），才能达到什么级别的安全保障？"**

这个问题将Safe RL方法按"领域知识需求—安全保障级别"的权衡关系分成一个谱系。

### 0.3 三个子问题（对应三章）

#### 子问题1（Ch2）：如果我们完全知道机器人动力学和约束函数，能否保证训练中每一步都安全？

**现状的不足**：
- CBF(Control Barrier Function)方法需要在每一步求解QP，多约束时QP可能不可行
- CBF将RL动作做后处理投影，改变了RL的动作分布，破坏了策略梯度的正确性
- 没有一个统一框架能同时处理不等式约束、等式约束、高阶动力学

**解决**：ATACOM —— 在约束流形的切空间中定义RL动作空间，结构性地保证安全

#### 子问题2（Ch3）：如果约束函数未知或太复杂无法手工设计怎么办？

**现状的不足**：
- 真实场景中障碍物形状复杂（铰接人体、不规则家具），手工距离函数不现实
- 即使能设计当前步的距离约束，也不保证**长期安全**（当前安全但几步后被逼入死角）
- 约束的不确定性（动态障碍物运动预测误差）被完全忽略

**解决**：
- ReDSDF：学习光滑的距离场作为约束函数，替代手工设计
- D-ATACOM：学习长期安全约束（FVF），用分布式RL捕获不确定性（CVaR）

#### 子问题3（Ch4）：如果连动力学模型都没有怎么办？

**现状的不足**：
- 基于值函数的安全评价 $V_c^\pi(s) \leq \eta$ 有三个问题：
  1. 多约束不同尺度难以平衡（碰撞代价 vs 速度超限代价）
  2. 安全阈值 $\eta$ 语义不清，不同任务需要不同调参
  3. 可行域是真实安全集的**超集**，允许一定违反
- 没有捕获安全预测的**不确定性**（未访问过的状态被乐观/悲观地错误评估）

**解决**：SPF/DSPF —— 用无量纲的安全概率替代代价值函数，用Beta分布建模不确定性

### 0.4 问题之间的关系图

```
                        领域知识多                    领域知识少
                           │                           │
安全保障强 ←──── Ch2 ATACOM ────→ Ch3 ReDSDF/D-ATACOM ────→ Ch4 SPF/DSPF ────→ 安全保障弱
                 │                    │                       │
              已知动力学            已知动力学               未知动力学
              已知约束              学习约束                 学习安全评估
              每步安全(SafeExp)     每步+长期安全            期望安全(SPL)
              理论保证              半理论保证               统计保证
```

**核心洞见**：领域知识越多，安全保障越强，但适用范围越窄。论文系统探索了这一权衡的整个谱系。

---

## 1. Chapter 2: ATACOM —— 约束流形上的安全探索

### 1.1 解决的具体问题

**问题**：给定已知的机器人动力学 $\dot{s} = f(s) + g(s)u$ 和约束 $k(s) \leq 0$，如何设计一个动作变换 $W: \mathcal{U} \to \mathcal{A}$，使得**无论RL策略输出什么**，执行后都不违反约束？

**与CBF的区别**：CBF是"RL先输出动作，然后QP投影修正"；ATACOM是"RL直接在安全的切空间中输出，从结构上不可能违反约束"。

### 1.2 关键思想

将约束 $k(s) \leq 0$ 通过松弛变量 $\mu$ 转化为等式约束 $c(s, \mu) = k(s) + \alpha(\mu) = 0$，定义一个约束流形 $\mathcal{M}$，然后**只在流形的切空间中运动**。

### 1.3 数学框架

**约束流形**：
$$\mathcal{M} = \{(s, \mu) \in \mathbb{R}^{S+K} : k(s) + \alpha(\mu) = 0\}$$

这是 $\mathbb{R}^{S+K}$ 中的 $S$ 维子流形（$K$ 个约束"消耗"了 $K$ 个自由度）。

**增广动力学**：
$$\begin{bmatrix} \dot{s} \\ \dot{\mu} \end{bmatrix} = \begin{bmatrix} f(s) \\ 0 \end{bmatrix} + \begin{bmatrix} g(s) & 0 \\ 0 & I_K \end{bmatrix} \begin{bmatrix} u_s \\ u_\mu \end{bmatrix}$$

**安全控制器（核心公式 Eq 2.12）**：

$$\begin{bmatrix} u_s \\ u_\mu \end{bmatrix} = \underbrace{-J_u^\dagger \psi}_{\text{(1)漂移补偿}} \underbrace{- \lambda J_u^\dagger c}_{\text{(2)收缩项}} + \underbrace{B_u \cdot u}_{\text{(3)RL动作}}$$

三项的含义：
1. **漂移补偿** $-J_u^\dagger \psi$：自然动力学 $f(s)$ 会推动系统，可能推向约束边界，这一项精确抵消
2. **收缩项** $-\lambda J_u^\dagger c$：如果因数值误差偏离了约束流形，以速率 $\lambda$ 拉回
3. **RL动作** $B_u u$：$B_u$ 是 $J_u$ 零空间的正交基，RL策略 $\pi$ 输出 $u \in \mathbb{R}^{S}$ 在此空间中自由移动

其中：
- $J_u = J_c \tilde{g}$：约束函数对控制输入的雅可比
- $\psi = J_c \tilde{f}$：漂移对约束变化率的影响
- $J_u^\dagger = J_u^T(J_u J_u^T)^{-1}$：Moore-Penrose伪逆

### 1.4 为什么安全？（Theorem 3）

定义 $V = \frac{1}{2}c^Tc$（约束偏离的"能量"）。对 $V$ 求导：

$$\dot{V} = c^T \dot{c} = c^T J_u(-J_u^\dagger \psi - \lambda J_u^\dagger c + B_u u)$$

由于 $J_u B_u = 0$（$B_u$ 在 $J_u$ 的零空间中），第三项消失：

$$\dot{V} = -\lambda c^T (J_u J_u^\dagger) c \leq 0$$

由 LaSalle 不变性原理 → $c \to 0$ → 系统渐近收敛到约束流形。

**ISS鲁棒性（Theorem 5）**：存在有界扰动 $\|\eta\| \leq \eta_c$ 时，约束违反有界 $\|c(t)\| \leq \eta_c / \lambda$。

### 1.5 实用技术细节

| 技术 | 解决的问题 | 方法 |
|------|----------|------|
| **指数松弛** $\alpha(\mu) = e^{\beta\mu}-1$ | 线性松弛在约束边界和内部的动作空间大小一样 | 远离约束时大动作空间，接近时小动作空间 |
| **光滑切空间基** (Alg 3) | SVD分解的基在不同状态可能跳变 | 正交Procrustes问题：$R^* = VU^T$ |
| **漂移裁剪** $\hat{\psi}_i = \max(\psi_i, 0)$ | 补偿所有漂移过于保守 | 只补偿推向约束边界方向的分量 |
| **可分离状态空间** | 约束涉及不可直接控制的状态（如速度） | 分离DCS/DUS，通过DCS间接控制 |
| **二阶动力学** | 位置约束在加速度控制下需要考虑速度 | 类CBF约束 $k^*(s,\dot{s}) = \zeta(k(s)) + J_k\dot{s} \leq 0$ |
| **等式约束** | 编队保持等需要精确等式约束 | 无需松弛变量，直接流形投影 |

### 1.6 ATACOM vs CBF

| 方面 | ATACOM | CBF-QP |
|------|--------|--------|
| 机制 | 动作空间变换（结构安全） | 动作后处理投影 |
| 多约束 | 自然处理（增广维度） | QP约束增多可能不可行 |
| 等式约束 | 直接支持 | 需额外处理 |
| RL兼容性 | 策略在切空间中输出，分布不被修改 | 投影改变动作分布，影响策略梯度 |
| 计算 | 伪逆 + SVD | QP求解 |
| 局限 | 需要高控制频率、需要动力学模型 | 需要Lipschitz连续性 |

### 1.7 实验

| 环境 | 测试内容 | 结果 |
|------|---------|------|
| 2D-StaticEnv | 松弛变量类型对比 | 指数松弛 >> 线性松弛 |
| 2D-DynamicEnv | 动态障碍物 | 速度观测显著帮助漂移补偿 |
| QuadrotorEnv | ATACOM vs CBF-QP | ATACOM更灵活，CBF在高维约束中困难 |
| AirHockeySim | 动力学不匹配下的安全性 | ISS保证 → 约束违反有界 |
| Real AirHockey | 真实机器人在线微调 | 全程安全，成功学到击球策略 |

---

## 2. Chapter 3: 学习约束与长期安全

### 2.1 ReDSDF：解决"约束函数太复杂无法手工设计"的问题

#### 2.1.1 具体问题

**场景**：机器人与人在共享工作空间中交互（HRI）。需要避免碰撞，但——
- 人体是铰接体，形状随姿态变化
- 用球体近似人体太粗糙（造成不必要的保守或漏洞）
- 即使知道网格模型，计算精确距离场也很困难（不光滑、梯度不连续）

**需要**：一个**可微的、光滑的距离函数** $d(q, x)$，给定关节配置 $q$ 和查询点 $x$，输出带符号距离。

#### 2.1.2 解决方案

ReDSDF = Regularized Deep Signed Distance Field

- 输入：关节配置 $q$ + 查询点 $x$
- 输出：带符号距离 $d$（正=外部，负=内部）
- 归纳偏置：网络结构保证距离场随远离表面而增大
- 训练损失：$\mathcal{L} = \mathcal{L}_{SDF} + \lambda_1 \mathcal{L}_{grad} + \lambda_2 \mathcal{L}_{Eikonal}$
  - $\mathcal{L}_{Eikonal}$：$\|\nabla d\| = 1$（真正距离场的性质）

与ATACOM集成：$k(s) = d_{threshold} - d_{ReDSDF}(q, x)$，雅可比通过自动微分。

#### 2.1.3 实验结果

| 实验 | 关键数据 |
|------|---------|
| WBC双臂（TIAGo++） | ReDSDF cautious：0碰撞/1000次；球体法：49碰撞/1000次 |
| 人机共享工作空间 | ReDSDF：95碰撞/1000次 vs 球体法：171碰撞/1000次 |
| 真实机器人HRI（送水杯） | 96%成功率，0碰撞，最小距离0.185m |
| 计算效率 | ReDSDF计算量线性增长 vs 球体法二次增长（PoI数量）|

### 2.2 D-ATACOM：解决"当前步安全 ≠ 长期安全"的问题

#### 2.2.1 具体问题

**场景**：TIAGo机器人导航避开移动的Fetch机器人。

ATACOM只保证**当前步**满足距离约束 $d(robot, obstacle) \geq d_{min}$，但：
- 当前安全不意味着下一步安全（机器人可能高速冲向障碍物，当前还没撞但马上要撞）
- 需要**预见未来**：当前状态是否在"前向不变集"中？
- 约束是确定性的，但动态障碍物的运动有**不确定性**

#### 2.2.2 解决方案

**Feasibility Value Function (FVF)**：

$$V_F^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t \max(k(s_t), 0) \middle| s_0 = s\right]$$

$V_F^\pi(s) = 0$ → 从 $s$ 出发未来永远不违反约束（Feasible Set $\mathcal{S}_F$）。

**分布式FVF**：$V_F^\pi(s) \sim \mathcal{N}(\mu_\phi^F(s), \sigma_\phi^F(s))$

**CVaR风险约束**：
$$\text{CVaR}_\alpha^F(s) = \mu^F(s) + \frac{\varphi(\Phi^{-1}(\alpha))}{1-\alpha} \sigma^F(s) \leq \delta$$

$\alpha$ 越高 → 越关注尾部风险 → 越保守。

**自适应阈值 $\delta$**：根据实际代价 vs 预算自动调节。

**解决非稳态MDP问题**：约束在训练中变化 → 动作映射 $W$ 变化。解决：Q函数在原始动作空间学习 $Q_\omega(s, a)$，而非切空间。

#### 2.2.3 实验结果

| 环境 | D-ATACOM vs 基线 |
|------|-----------------|
| Cartpole | 学习最快，约束违反最小 |
| Navigation | **唯一**展现主动碰撞避免行为的方法（Lagrangian方法陷入局部最优） |
| 3DoF AirHockey | 接近已知前向不变约束的性能，同时约束违反更低 |

#### 2.2.4 局限

- 需要已知动力学模型
- 需要探索不安全状态来训练FVF
- 无法处理等式约束
- 不探索已知约束+学习约束的结合

---

## 3. Chapter 4: SPF/DSPF —— 无模型的安全概率函数

### 3.1 解决的具体问题

#### 3.1.1 问题1：值函数安全评价的缺陷

现有方法用值函数 $V_c^\pi(s) = \mathbb{E}[\sum \gamma^t c(s_t)]$ 估计未来累积代价，要求 $V_c^\pi(s) \leq \eta$。

三个问题：
1. **多约束尺度不一**：碰撞代价可能是0/1，速度超限代价是连续值，直接相加无意义
2. **阈值 $\eta$ 无直觉语义**：$\eta = 5$ 是安全还是不安全？不同任务完全不同
3. **可行域是安全集的超集**：$\{s: V_c^\pi(s) \leq \eta\}$ 包含了实际不安全但"平均代价低"的状态

#### 3.1.2 问题2：安全预测的不确定性被忽略

训练初期，安全评价网络对未访问状态的估计不可靠：
- 乐观估计 → 进入不安全区域
- 悲观估计 → 过度保守，性能差

没有区分"确信这里安全"和"不确定这里是否安全"。

### 3.2 SPF：安全概率函数

#### 3.2.1 核心思想

不估计"未来代价是多少"，而是估计**"从当前状态出发，整条轨迹一直安全的概率是多少"**。

**SPF定义**：$\Psi^\pi(s) := p(c = 1 | s)$，其中 $c = 1$ 表示从 $s$ 出发整条轨迹一直安全。

**Safe Probability Bellman Equation (SPBE)**：

$$\Psi^\pi(s) = (1 - \gamma)p_b(s) + \gamma p_b(s)\mathbb{E}_{s'}[\Psi^\pi(s')]$$

直觉：
- $(1-\gamma)p_b(s)$：当前安全且轨迹结束（被吸收）
- $\gamma p_b(s)\mathbb{E}[\Psi^\pi(s')]$：当前安全且未结束，后续也一直安全

#### 3.2.2 SPF如何解决问题1

| 值函数方法 | SPF |
|----------|-----|
| $V_c^\pi \in [0, \frac{1}{1-\gamma}]$，尺度随任务变化 | $\Psi^\pi \in [0, 1]$，无量纲概率 |
| 多约束需要权衡代价尺度 | 所有约束统一为"安全/不安全"二分类 |
| 可行域 $\{V_c \leq \eta\} \supseteq \mathcal{S}_S$ | 可行域 $\{\Psi \geq \eta\} \subseteq \mathcal{S}_S$ |
| $\eta$ 含义模糊 | $\eta = 0.9$ 就是"90%概率安全" |

**关键性质（Theorem 7）**：SPBE算子是 $\gamma$-收缩映射 → TD学习保证收敛。

### 3.3 DSPF：分布式安全概率函数

#### 3.3.1 解决问题2：捕获不确定性

将SPF从**点估计**扩展为**分布估计**：

- 低阶：$C_x \sim \text{Bernoulli}(c; \psi)$
- 高阶：$\psi \sim \text{Beta}(\psi; \alpha_1, \alpha_2)$

通过**Prior Network**区分三种不确定性：
- **Data Uncertainty**（随机不确定性）：安全边界处的固有模糊
- **Knowledge Uncertainty**（认知不确定性）：因缺乏数据造成的不确定

$$U(x) = \underbrace{\mathcal{H}[\mathbb{E}[p(c|\psi)]]}_{\text{Total Uncertainty}} - \underbrace{\mathbb{E}[\mathcal{H}[p(c|\psi)]]}_{\text{Data Uncertainty}}$$

#### 3.3.2 SUKB：安全上界知识边界

$$\mathcal{K}_\zeta(x) = p(c=1|x; \mathcal{D}) + \zeta U(x)$$

SUKB = 安全概率 + 知识不确定性。

行为：
- 确信安全 → 高SUKB → 允许探索
- 确信不安全 → 低SUKB → 避免
- **不确定** → 高不确定性 → 高SUKB → **鼓励探索**（乐观原则）

有界探索保证（Proposition 5）：不安全状态最多被访问 $N$ 次后SUKB下降到阈值以下。

### 3.4 算法

**DSPF-QLearning（离散动作空间）**：
$$\pi(a|s) = \begin{cases} \epsilon\text{-greedy in } \mathcal{A}_S & \text{if } \mathcal{A}_S \neq \emptyset \\ \arg\max_{a} \mathcal{K}_\zeta(s, a) & \text{otherwise} \end{cases}$$

**DSPF-AC（连续动作空间）**：构造 Safe-Q 分布
$$p_\mathcal{K}^\pi(a,s) = \frac{1}{Z_k}\exp\left(-\beta_k[\mathcal{K}_\zeta^\pi(s,a) - \eta]_-^2\right)$$

策略优化：$\min_\pi \text{KL}[\pi(a|s) \| \frac{1}{Z}\exp(\beta_q Q^\pi) p_\mathcal{K}^\pi]$

### 3.5 实验

**GridWorld（离散）**：
- DSPF-Sample在AllUnsafe环境失败（目标分布偏向不安全样本）
- DSPF-SafeProb（Classifier-based）解决偏差问题，所有环境都表现良好

**Safety Gymnasium（连续）**：HopperVelocity, HalfCheetahVelocity, AntVelocity, PointCircle
- SPF-AC/DSPF-AC：安全概率阈值 $\eta = 0.9$ 对所有任务通用（vs Lag-SAC/WCSAC需要逐任务调参）
- 收敛更慢但最终性能可比，安全性更高
- 意外发现：连续任务中SPF-AC和DSPF-AC差异不大（原因：软约束优化+NN平均化）

---

## 4. Chapter 5: 结论

### 4.1 核心回答

> 对于严格安全要求的任务，**领域知识是必要的**。动力学模型+约束函数 → ATACOM可以在真实机器人上安全训练。在缺乏领域知识时，model-free方法经过充分训练也能学到安全策略，**但实际部署依赖于精心调校的仿真器**。

### 4.2 贡献总结

| 章节 | 方法 | 需要什么 | 安全级别 | 关键创新 |
|------|------|---------|---------|---------|
| Ch2 | ATACOM | 动力学 + 约束 | 每步安全（渐近） | 约束流形切空间投影 |
| Ch3a | ReDSDF | 动力学 + 网格数据 | 碰撞避免 | 正则化深度SDF |
| Ch3b | D-ATACOM | 动力学 + 训练学习 | 每步 + 长期安全 | FVF + CVaR + 自适应阈值 |
| Ch4 | SPF/DSPF | 无 | 期望安全 | 安全概率Bellman + SUKB |

### 4.3 开放问题

| 方向 | 问题 | 挑战 |
|------|------|------|
| 执行器限制 | 安全动作可能不存在 | 如何将执行器限制纳入约束设计 |
| 高维输入 | 图像/点云如何构造约束 | PointNet/YOLO等预训练模型集成 |
| 语义约束 | "不要撞人"如何变成可微约束 | LLM → 约束函数的自动转化 |
| 离散动力学 | 接触等非连续动力学 | 时间离散化ATACOM |
| 学习动力学 | 模型与策略同时学习 | 稳定性问题 |
| 混合方法 | 结合model-based与model-free | 如何利用不完美的先验知识 |

---

## 5. 与硕士课题的关系

### 5.1 直接技术基础

本论文是硕士课题 `safeRL_manifold` 项目的直接理论基础：

| 论文内容 | 代码对应 |
|---------|---------|
| Ch2 ATACOM算法 | `constrained_manifold/manifold.py` → `AtacomEnvWrapper` |
| 约束流形定义 | `constrained_manifold/constraints.py` → `ConstraintsSet`, `StateConstraint` |
| 单智能体集成 | `single_cm.py` |
| 多智能体集成 | `mult_cm.py` |

### 5.2 论文是单智能体的，硕士课题扩展到多智能体

论文全文聚焦**单智能体**。硕士课题的核心创新是多智能体扩展：

| 单智能体（论文） | 多智能体（硕士课题） |
|-----------------|-------------------|
| 单约束流形 $\mathcal{M}$ | 联合/分布式约束流形 |
| 单系统动力学 | 耦合/解耦多体动力学 |
| 单策略切空间 | 分布式切空间分解 |
| FVF长期安全 | 多智能体FVF（考虑他者策略） |
| SPF安全概率 | 多智能体交互下的安全概率 |

### 5.3 RMPflow联系

RMPflow与ATACOM共享黎曼几何基础：
- RMPflow：树状pushforward/pullback传播 $x, \dot{x}$ 和力+度量
- ATACOM：约束流形切空间投影

硕士课题可以：
1. 用RMPflow层次结构作为多智能体ATACOM的几何骨架
2. 将RMPflow手工叶节点替换为RL策略（论文核心思路）
3. 在multi-robot-rmpflow去中心化框架上叠加ATACOM安全层

### 5.4 五个可行研究方向

1. **分布式ATACOM**：每个智能体维护局部约束流形，通信交换约束信息
2. **层次化安全**：RMPflow层次 + ATACOM安全层 = 可扩展多机器人安全
3. **多智能体D-ATACOM**：联合FVF考虑其他智能体行为不确定性
4. **多智能体SPF**：安全概率函数处理多智能体不确定性交互
5. **编队+ATACOM**：编队保持（等式约束Ch2.7.3）+ 碰撞避免（不等式约束）

---

## 6. 关键公式速查

| 名称 | 公式 |
|------|------|
| **ATACOM控制器** | $[u_s; u_\mu] = -J_u^\dagger\psi - \lambda J_u^\dagger c + B_u u$ |
| **约束流形** | $\mathcal{M} = \{(s,\mu): k(s) + \alpha(\mu) = 0\}$ |
| **安全性证明** | $V = \frac{1}{2}c^Tc$, $\dot{V} = -\lambda c^T J_u J_u^\dagger c \leq 0$ |
| **ISS界** | $\|c(t)\| \leq \eta_c / \lambda$ |
| **FVF** | $V_F^\pi(s) = \mathbb{E}[\sum \gamma^t \max(k(s_t), 0) \| s_0 = s]$ |
| **CVaR约束** | $\mu^F + \frac{\varphi(\Phi^{-1}(\alpha))}{1-\alpha}\sigma^F \leq \delta$ |
| **SPBE** | $\Psi^\pi(s) = (1-\gamma)p_b(s) + \gamma p_b(s)\mathbb{E}_{s'}[\Psi^\pi(s')]$ |
| **SUKB** | $\mathcal{K}_\zeta(x) = p(c=1|x;\mathcal{D}) + \zeta U(x)$ |
| **Safe-Q分布** | $p_\mathcal{K}^\pi = \frac{1}{Z}\exp(-\beta_k[\mathcal{K}_\zeta - \eta]_-^2)$ |

---

## 7. 对应出版物

| 章节 | 论文 | 会议/期刊 |
|------|------|---------|
| Ch2 | "Robot RL on the Constraint Manifold" | CoRL 2022 |
| Ch2扩展 | "Safe RL on the Constraint Manifold: Theory and Applications" | T-RO 2025 (条件录用) |
| Ch3 ReDSDF | "Regularized Deep SDF for Reactive Motion Generation" | IROS 2022 |
| Ch3 D-ATACOM | "Safe RL of Dynamic High-Dim Robotic Tasks" | ICRA 2023 |
| Ch3 D-ATACOM2 | "Handling Long-Term Safety and Uncertainty in Safe RL" | CoRL 2024 |
| Ch4 SPF/DSPF | Günster, Liu, Peters, Tateo | CoRL 2024 |
| 相关 | ROSCOM: Robust Safe RL on Stochastic Constraint Manifolds | T-ASE 2024 |
