# 多智能体编队：强化学习 / RMP / 奖励参数与图结构说明

本文档汇总当前代码里**多智能体编队任务**（以 `SafetyPointMultiFormationGoal0-v0` 为例）的全部可调参数、默认数值，以及编队图 **边 (e)**、**期望方向 (r\*)** 的构造方式与奖励公式。实现分散在 **Safe-Policy-Optimization** 与 **Safety-Gymnasium**（底层任务与奖励）中。

---

## 1. 算法入口与分工

| 入口 | 说明 |
|------|------|
| `safepo.multi_agent.mappo` | MAPPO，**无** RMP 动作修正 |
| `safepo.multi_agent.mappo_rmp` | MAPPO + **RMPCorrector**：在策略输出动作上叠加 RMP 加速度并加权融合 |

编队环境构造：`safepo/common/env.py` 中 `make_formation_nav_env` → `MultiFormationNavEnv` → `safety_gymnasium.make(task, num_agents=..., formation_*)`。

---

## 2. 强化学习（MAPPO）参数

### 2.1 配置文件（编队任务走顶层，不用 `mamujoco` 段）

路径（相对已安装包或源码树 `safepo/multi_agent/`）：

- `marl_cfg/mappo/config.yaml`
- `marl_cfg/mappo_rmp/config.yaml`

编队任务**不合并** YAML 里的 `mamujoco:` 小节，以下**顶层键**对编队生效（与仓库当前文件一致）：

| 键 | 数值 | 含义 |
|----|------|------|
| `num_env_steps` | `100000000` | 总环境步数 |
| `episode_length` | `8` | 每轮收集步数（与 rollout 相乘得每轮总步数） |
| `n_rollout_threads` | `80` | 并行环境数（可用 `--num-envs` 覆盖） |
| `n_eval_rollout_threads` | `1` | 评估并行环境数 |
| `hidden_size` | `512` | Actor/Critic MLP 隐层宽度 |
| `layer_N` | `2` | MLP 层数 |
| `gamma` | `0.96` | 折扣因子 |
| `gae_lambda` | `0.95` | GAE λ |
| `clip_param` | `0.2` | PPO clip |
| `learning_iters` | `5` | 每批数据上 PPO 更新轮数 |
| `num_mini_batch` | `1` | mini-batch 数 |
| `actor_lr` | `9e-5` | 策略学习率 |
| `critic_lr` | `5e-3` | 价值网络学习率 |
| `entropy_coef` | `0.0` | 策略熵系数 |
| `max_grad_norm` | `10` | 梯度裁剪 |
| `target_kl` | `0.016` | KL 相关阈值（实现中与 PPO 流程配合） |
| `use_gae` | `True` | 使用 GAE |
| `use_popart` | `True` | PopArt 价值归一化 |
| `use_feature_normalization` | `True` | 特征归一化 |
| `use_ReLU` | `True` | 激活函数 |
| `eval_interval` / `log_interval` | `25` | 日志间隔（episode 索引） |
| `cost_limit` | 默认由命令行 `--cost-limit` 写入，默认 `25.0` | 写入 `cfg_train`；是否作为约束视算法 |

命令行常用覆盖：`--total-steps` → `num_env_steps`；`--num-envs` → 同时设置 `n_rollout_threads` 与 `n_eval_rollout_threads`；`--seed`、`--device`、`--device-id`、`--experiment` 等见 `safepo/utils/config.py` 中 `multi_agent_args()`。

---

## 3. 编队与命令行（环境与 RMP 共用一套 cfg）

在 `multi_agent_args()` 中写入 `cfg_train` 的项（默认值如下），并传给 `MultiFormationNavEnv` 与 `RMPCorrector`：

| 参数 | 默认 | 可选 / 说明 |
|------|------|-------------|
| `--num_agents` | `2` | 机器人数量 |
| `--formation-shape` | `mesh` | `mesh` \| `line` \| `wedge` \| `circle` |
| `--formation-line-axis` | `x` | `line` 时：`x`（沿 +x 排开）或 `y`（沿 +y） |
| `--formation-wedge-half-angle-deg` | `35.0` | `wedge` 半张角（度） |
| `--formation-target-distance` | `0.5` | 特征尺度：**邻距 / 弦长**（米），同时用于环境奖励约束与 RMP 边长 |

可选仅通过 YAML 扩展（未单独做 argparse）：`formation_desired_direction`，默认 `[0.0, 1.0]`（世界系下期望参考方向）。

---

## 4. RMP 参数（`safepo/multi_agent/rmp_corrector.py`）

仅在 **`mappo_rmp`** 且成功 `import` 本地包 `multi-robot-rmpflow` 时启用（见文件内 `sys.path` 与 `RMP_AVAILABLE`）。

### 4.1 从 `config`（即 `cfg_train`）读取的项与默认

| 键 | 默认 | 含义 |
|----|------|------|
| `use_rmp` | `True` | 是否启用 RMP |
| `formation_target_distance` | `0.5` | 与命令行一致；构图边长 / 间距 |
| `formation_shape` | `mesh` | 与上表相同 |
| `formation_line_axis` | `x` | 与上表相同 |
| `formation_wedge_half_angle_deg` | `35.0` | 与上表相同 |
| `formation_desired_direction` | `[0.0, 1.0]` | **`mesh`**：统一期望边方向 **r\***；**`circle`**：用于确定多边形朝向（见下节） |
| `formation_orientation_alpha` | `1.0` | 方向叶 `FormationOrientationDecentralized` 的 α\_θ |
| `formation_orientation_eta` | `2.0` | 方向叶 η\_θ |
| `formation_orientation_c_metric` | `10.0` | 方向叶度量 c\_θ |
| `collision_safety_radius` | `0.3` | 智能体–智能体避碰与 **Sigwall** 几何半径尺度 |
| `rmp_weight` | `1` | 策略动作与 RMP 解算加速度融合系数：`a = a_rl + rmp_weight * a_rmp`（实现中对 2D 加速度与动作逐维相加） |

### 4.2 RMP 图：**边 (e)** 与 **(i, j, d, r\*)**

边由 `safepo/multi_agent/formation_spec.py` 的 `build_formation_edges(shape, num_agents, spacing, global_direction, ...)` 生成。每条边为：

`(i, j, d_ij, r_star)`

- **i, j**：机器人索引；hub 挂在 **i** 上，叶节点相对 **j** 的位置构造 **i→j** 的期望。
- **d\_ij**：该边期望**几何距离**（米）。
- **r\_star**：世界系下该边期望**单位方向向量**。

与底层 **奖励** 使用的距离约束同源几何，但奖励侧在 `formation_reward_edges.build_formation_distance_constraints` 中去重为无向 `(i, j, d)`（`i < j`），见第 5 节。

各 **`formation_shape`** 下图的构造要点（`spacing = formation_target_distance`，即 CLI 默认多为 `0.5`）：

#### `mesh`（及别名 `full` / `default` / `complete`）

- **边**：所有无序对 `(i, j), i < j`，共 `C(N,2)` 条。
- **距离**：每条 `d_ij = spacing`。
- **r\***：同一条单位向量 `gdu = normalize(formation_desired_direction)`，所有边共用（legacy 行为）。

#### `line`（及别名 `line_horizontal` / `row`）

- **轴**：`formation_line_axis == "x"` → 期望沿 **+x**，`r_star = [1,0]`；`"y"` → **+y**，`r_star = [0,1]`。
- **边**：相邻 `(i, i+1)`，**双向**各一条有向边（实现里 `_add_pair`），`d = spacing`。

#### `wedge`（及别名 `v` / `vee`）

- **N = 2**：一条无向边等价双向有向边，`d = spacing`，`r_star = [1,0]`。
- **N ≥ 3**：顶点为 agent `0`，半角 `ha = wedge_half_angle_rad`（默认 35°）。在几何模板里先放各机器人**标称位置**，再对 `(0,k)` 与相邻翼上 `(k, k+1)` 连边；**每条边的 `d` 为标称点之间的欧氏距离**（不一定都等于 `spacing`）。角度在 `[-ha, +ha]` 上均分。

#### `circle`（及别名 `ring` / `polygon`）

- **弦长**：相邻顶点弦长 = `spacing`。
- **外接圆半径**：`R = spacing / (2 sin(π/N))`。
- **顶点位置**：从 `global_direction` 对应方位角开始，每隔 `2π/N` 取正 N 边形顶点。
- **边**：环上 `(i, i+1)`（含 `N-1→0`），双向，`d` 为相邻顶点距离（等于弦长 `spacing`）。

### 4.3 RMP 叶节点（代码中写死的超参）

| 模块 | 参数 | 数值 |
|------|------|------|
| `CollisionAvoidanceDecentralized`（两两机器人） | `R` | `collision_safety_radius`（默认 `0.3`） |
| | `eta` | `1.0` |
| `FormationDecentralized`（每条编队边） | `gain` | `1.0` |
| | `eta` | `2.0` |
| | `w` | `10.0` |
| `FormationOrientationDecentralized` | `alpha_theta` / `eta_theta` / `c_theta` | 见上表 `formation_orientation_*` |
| `CollisionAvoidance`（Sigwall，静态圆） | 墙心 | **代码写死** `[-0.8, 0]`、`[0.8, 0]`（与部分任务 XML 中墙位置可能略有偏差，以 RMP 为准） |
| | `R` | 与 `collision_safety_radius` 相同 |
| | `epsilon` | `1e-3` |
| | `alpha` | `1e-5` |
| | `eta` | `1.0` |

---

## 5. 强化学习奖励（Safety-Gymnasium 任务层）

实现文件（安装源码时路径类似）：

- `safety_gymnasium/tasks/safe_multi_agent/tasks/multi_formation/multi_formation_level0.py` — `MultiFormationGoalLevel0`
- `safety_gymnasium/tasks/safe_multi_agent/tasks/multi_formation/formation_reward_edges.py` — `build_formation_distance_constraints`

### 5.1 目标与边界相关常数（`MultiFormationGoalLevel0`）

| 项 | 数值 | 说明 |
|----|------|------|
| `goal.reward_distance` | `2.0` | 稠密：**靠近目标**的势差奖励系数 |
| `goal.reward_goal` | `2.5` | 稀疏：进入目标半径内一次性奖励 |
| 目标位置 | `(0, 2.0)` | 通道末端 |
| `dist_delta` 裁剪 | `[-0.8, 0.8]` | 每步距离差分项裁剪 |
| `boundary_penalty_scale` | `0.2` | 越界 cost 时对 reward 的额外减分 |
| `_reward_clip` | `10.0` | 每 agent 每步 reward 截断到 `[-10, 10]` |

若构造任务时 **未** 传入 `formation_target_distance`，任务内默认 **`0.4`**；经 SafePO 的 `make_formation_nav_env` 通常会传入 `cfg_train` 中的值（CLI 默认 **`0.5`**），此时以传入值为准。

### 5.2 编队距离约束边（奖励用）

`build_formation_distance_constraints` 与 RMP 的 `build_formation_edges` **同一套形状规则**，但输出为**无向**唯一边 `(i, j, d_target)`（`i < j`），用于计算误差。**mesh** 时另返回 `mesh_r_star` 用于对齐项。

### 5.3 每步奖励分解（每个 agent 相同的全局编队项 + 各自目标项）

对位置 `positions`：

1. **平均边距误差**  
   `distance_error = mean_{(i,j) in edges} | ||p_i - p_j|| - d_target(i,j) |`  
   （边集为去重后的无向约束。）

2. **编队形状奖励（共享标量，加到每个 agent）**  
   `formation_reward = formation_reward_scale * exp(-distance_error / formation_tolerance)`  

   | 字段 | 默认 |
   |------|------|
   | `formation_reward_scale` | `0.04` |
   | `formation_tolerance` | `0.1` |

3. **形状辅助项 `shape_aux`（标量，加到每个 agent）**  
   - **mesh**：`mesh_alignment_reward_scale * mean( cos 夹角 )`，其中夹角为实际边方向与 `mesh_r_star` 的点积（实现为边方向单位向量与 `mesh_r_star` 的点积平均）。默认 `mesh_alignment_reward_scale = 0.025`。  
   - **line**：惩罚垂直于轴线方向的分散：`- line_width_reward_scale * Var(坐标)`，默认 `line_width_reward_scale = 0.02`。  
   - **wedge / circle**：仅由距离约束体现，此项为 `0`。

4. **每 agent 目标项**  
   - `dist_delta * goal.reward_distance`（较上一步更接近目标为正）  
   - 若到达目标：`+ goal.reward_goal`  
   - 若越界：` - boundary_penalty_scale`  

最后对每 agent：`clip(reward, -10, 10)`。

---

## 6. 一致性说明（便于调参）

1. **`formation_target_distance`**：建议命令行显式设定，使 **环境奖励边长**、**RMP 边长**一致；默认 CLI 为 `0.5`，与 RMPCorrector 默认一致。  
2. **墙的位置**：任务里 `Sigwalls` 可在 `multi_formation_level0` 中设为约 `±1.0`；RMP 里 Sigwall 避障圆心为 **`±0.8`**。若需严格一致，需分别改任务 XML/逻辑与 `rmp_corrector.py` 中的 `sigwall_centers`。  
3. **仅 MAPPO**：无 RMP 时仍使用上节 **Safety-Gymnasium** 奖励；**mappo_rmp** 在此基础上对动作做 RMP 修正。

---

## 7. 相关文件索引（Safe-Policy-Optimization）

| 文件 | 内容 |
|------|------|
| `safepo/utils/config.py` | `multi_agent_args`、编队 CLI 写入 `cfg_train` |
| `safepo/common/env.py` | `make_formation_nav_env` |
| `safepo/common/wrappers.py` | `MultiFormationNavEnv` |
| `safepo/multi_agent/formation_spec.py` | RMP 边列表 `build_formation_edges` |
| `safepo/multi_agent/rmp_corrector.py` | `RMPCorrector`、叶节点与 Sigwall |
| `safepo/multi_agent/marl_cfg/mappo/config.yaml` | MAPPO 默认超参 |
| `safepo/multi_agent/marl_cfg/mappo_rmp/config.yaml` | MAPPO+RMP 默认超参 |

更简的运行示例见同目录 [MULTI_AGENT_FORMATION_RUN.md](MULTI_AGENT_FORMATION_RUN.md)。
