# 多机器人编队导航实验设置说明

本文档详细说明当前多机器人编队导航实验的配置，包括奖励函数、代价函数、观测空间、动作空间、RMP设置等。

## 1. 环境设置

### 1.1 任务名称
- **任务ID**: `SafetyPointMultiFormationGoal0-v0`
- **任务类型**: 多智能体编队导航任务
- **机器人类型**: Point（点机器人）

### 1.2 环境布局
- **通道设置**: 
  - 两个固定墙（Sigwalls）形成狭长通道
  - 左墙位置: `(-0.8, 0)`
  - 右墙位置: `(0.8, 0)`
  - 墙厚度: `0.05` 米
  - 安全通道宽度: 约 `1.6` 米（从 `-0.8` 到 `0.8`）

- **目标位置**: 
  - 固定目标位置: `(0, 2.0)`（在通道尾部）
  - 目标大小: 由 `Goal.keepout=0.305` 决定

- **初始位置**:
  - 机器人初始生成区域: `[(-0.7, -0.8, 0.7, 0.8)]`
  - 确保机器人在安全区域内生成，避免与墙碰撞

### 1.3 机器人数量
- 默认: `2` 个机器人（`agent_0` 和 `agent_1`）
- 可通过命令行参数 `--num_agents` 动态指定（支持更多机器人）

---

## 2. 奖励函数 (Reward)

奖励函数为每个智能体返回一个字典，格式为 `{'agent_0': float, 'agent_1': float}`。

### 2.1 距离奖励 (Distance Reward)
- **公式**: `(last_dist_goal - current_dist_goal) * goal.reward_distance`
- **说明**: 
  - 当机器人接近目标时获得正奖励
  - 远离目标时获得负奖励
  - `goal.reward_distance = 1.0`（默认值）
  - `last_dist_goal` 记录上一时刻到目标的距离

### 2.2 到达目标奖励 (Goal Achievement Reward)
- **公式**: `goal.reward_goal`（当机器人到达目标时）
- **条件**: `dist_goal(agent_id) <= goal.size`
- **默认值**: `goal.reward_goal = 1.0`

### 2.3 编队保持奖励 (Formation Reward)
- **公式**: `formation_reward_scale * exp(-distance_error / formation_tolerance)`
- **参数**:
  - `formation_target_distance = 0.4` 米（目标编队距离）
  - `formation_tolerance = 0.1` 米（允许的误差范围）
  - `formation_reward_scale = 0.5`（奖励缩放因子）
- **计算方式**:
  ```python
  inter_agent_distance = ||agent_0_pos - agent_1_pos||
  distance_error = |inter_agent_distance - formation_target_distance|
  formation_reward = 0.5 * exp(-distance_error / 0.1)
  ```
- **特点**: 
  - 当两个机器人距离接近 `0.4` 米时，奖励最大
  - 距离误差越小，奖励越大（指数衰减）
  - 两个机器人都获得相同的编队奖励

### 2.4 碰撞墙惩罚 (Wall Collision Penalty)
- **公式**: `-0.5`（当机器人碰撞墙时）
- **触发条件**: `cost[agent_name].get('cost_out_of_boundary', 0) > 0`
- **说明**: 在奖励中额外惩罚，与代价函数配合使用

### 2.5 总奖励
每个智能体的总奖励为：
```
reward[agent_i] = distance_reward + goal_reward + formation_reward - wall_penalty
```

---

## 3. 代价函数 (Cost)

代价函数返回每个智能体的代价字典，格式为 `{'agent_0': dict, 'agent_1': dict}`。

### 3.1 边界越界代价 (Out of Boundary Cost)
- **键名**: `cost_out_of_boundary`
- **值**: `1.0`（当机器人越界时）
- **判断逻辑**:
  ```python
  # 安全区域边界（考虑机器人半径和容差）
  safe_left = -0.8 + 0.05/2 + 0.05 - 0.05 = -0.775
  safe_right = 0.8 - 0.05/2 - 0.05 + 0.05 = 0.775
  
  # 如果机器人 x 坐标超出 [safe_left, safe_right]，则触发代价
  if agent_x < safe_left or agent_x > safe_right:
      cost['cost_out_of_boundary'] = 1.0
  ```
- **说明**: 
  - 墙厚度: `0.05` 米
  - 机器人半径: `0.05` 米
  - 容差: `0.05` 米（避免浮点误差）

### 3.2 智能体间碰撞代价 (Inter-Agent Collision Cost)
- **键名**: `cost_contact_other`
- **值**: 由 `contact_other_cost` 配置决定
- **触发条件**: 两个机器人发生物理接触

### 3.3 总代价 (Cost Sum)
- **键名**: `cost_sum`
- **公式**: 所有代价项的总和
  ```python
  cost_sum = sum(v for k, v in agent_cost.items() if k.startswith('cost_'))
  ```

---

## 4. 观测空间 (Observation Space)

### 4.1 底层观测 (Base Observation)
来自 Safety-Gymnasium 的原始观测，包括：

#### 4.1.1 传感器观测
- **accelerometer**: 加速度计（3维）
- **velocimeter**: 速度计（3维）
- **gyro**: 陀螺仪（3维）
- **magnetometer**: 磁力计（3维）
- **accelerometer1**: 第二个机器人的加速度计（3维）
- **velocimeter1**: 第二个机器人的速度计（3维）
- **gyro1**: 第二个机器人的陀螺仪（3维）
- **magnetometer1**: 第二个机器人的磁力计（3维）

#### 4.1.2 关节观测（如果适用）
- **hinge_pos**: 铰链关节位置
- **hinge_vel**: 铰链关节速度
- **ballquat**: 球关节四元数
- **ballangvel**: 球关节角速度

#### 4.1.3 自由关节观测
- **freejoint_pos**: 自由关节位置（z坐标）
- **freejoint_qvel**: 自由关节速度（3维）

#### 4.1.4 障碍物观测（如果启用）
- **goal_lidar**: 目标激光雷达观测（`num_bins` 维）
- **goal_comp**: 目标罗盘观测（2维）
- **sigwalls_lidar**: 墙激光雷达观测（如果启用）

### 4.2 包装器增强观测 (Wrapper Enhanced Observation)

在 `MultiFormationNavEnv` 中，观测被进一步处理：

#### 4.2.1 局部观测 (Local Observation)
- **组成**: `[base_state, agent_id_one_hot]`
- **agent_id_one_hot**: 
  - 对于 `agent_0`: `[1.0, 0.0, ...]`
  - 对于 `agent_1`: `[0.0, 1.0, ...]`
  - 维度: `num_agents` 维
- **标准化**: 
  ```python
  obs_i = (obs_i - mean(obs_i)) / (std(obs_i) + 1e-8)
  ```

#### 4.2.2 共享观测 (Shared Observation)
- **组成**: 所有机器人共享相同的全局状态
- **标准化**: 
  ```python
  share_obs = (state - mean(state)) / (std(state) + 1e-8)
  ```

### 4.3 观测空间定义
```python
observation_space = Box(low=-10, high=10, shape=(obs_size,))
share_observation_space = Box(low=-10, high=10, shape=(share_obs_size,))
```

---

## 5. 动作空间 (Action Space)

### 5.1 动作类型
- **机器人类型**: Point（点机器人）
- **动作维度**: `2` 维（v, omiga）
- **动作空间**: `Box(low=-1, high=1, shape=(2,))`
- **说明**: 
  - 动作值范围: `[-1, 1]`


---

## 6. RMP 设置 (Riemannian Motion Policy)

RMP 用于实时动作修正，包括碰撞避免和编队保持。

### 6.1 RMP 配置参数
- **formation_target_distance**: `0.5` 米（RMP 编队目标距离）
- **collision_safety_radius**: `0.3` 米（碰撞安全半径）
- **rmp_weight**: `0.1`（RMP 修正权重）

### 6.2 RMP 树结构
为每个并行环境创建独立的 RMP 树：

#### 6.2.1 机器人节点
- 每个机器人对应一个 `RMPNode`
- 映射函数 `phi`: 提取机器人位置 `[x_i, y_i]`
- 雅可比矩阵 `J`: 从全局状态到机器人状态的映射

#### 6.2.2 碰撞避免节点
- **类型**: `CollisionAvoidanceDecentralized`
- **数量**: 每对机器人之间创建一个节点
- **参数**: 
  - `R = 0.3`（安全半径）
  - `eta = 1.0`（阻尼系数）

#### 6.2.3 编队控制节点
- **类型**: `FormationDecentralized`
- **数量**: 每对机器人之间创建一个节点（双向）
- **参数**:
  - `d = 0.5`（目标距离）
  - `gain = 1.0`（增益）
  - `eta = 2.0`（阻尼系数）
  - `w = 10.0`（权重）

### 6.3 RMP 动作修正流程
1. **获取状态**: 从环境获取所有机器人的位置和速度
2. **更新 RMP 树**: 
   - `set_root_state(x, x_dot)`
   - `pushforward()`
   - 更新所有叶子节点
   - `pullback()`
3. **解析加速度**: `acceleration_correction = r.resolve()`
4. **修正动作**: 
   ```python
   corrected_action = original_action + rmp_weight * acceleration_correction
   ```
5. **裁剪动作**: 确保动作在有效范围内

---

## 7. 训练设置

### 7.1 Leader-Follower 架构
- **Leader**: `agent_0`（使用 PPO 训练）
- **Followers**: `agent_1, agent_2, ...`（RMP 编队跟随）

### 7.2 Follower 策略
- **RMP编队跟随**: 通过 RMP 确保碰撞避免和编队保持

### 7.3 训练配置
- **算法**: PPO (Proximal Policy Optimization)
- **只训练 Leader**: 只有 `agent_0` 的策略网络和值网络被训练
- **RMP 实时修正**: 在 `collect()` 和 `eval()` 中应用 RMP 修正

---

## 8. 关键参数总结

### 8.1 环境参数
| 参数 | 值 | 说明 |
|------|-----|------|
| 通道宽度 | 1.6 米 | 从 -0.8 到 0.8 |
| 目标位置 | (0, 2.0) | 通道尾部 |
| 初始区域 | (-0.7, -0.8, 0.7, 0.8) | 机器人生成区域 |

### 8.2 奖励参数
| 参数 | 值 | 说明 |
|------|-----|------|
| reward_distance | 1.0 | 距离奖励系数 |
| reward_goal | 1.0 | 到达目标奖励 |
| formation_target_distance | 0.4 米 | 编队目标距离 |
| formation_tolerance | 0.1 米 | 编队误差容忍度 |
| formation_reward_scale | 0.5 | 编队奖励缩放 |
| wall_penalty | -0.5 | 碰撞墙惩罚 |

### 8.3 RMP 参数
| 参数 | 值 | 说明 |
|------|-----|------|
| formation_target_distance | 0.5 米 | RMP 编队目标距离 |
| collision_safety_radius | 0.3 米 | 碰撞安全半径 |
| rmp_weight | 0.1 | RMP 修正权重 |

### 8.4 代价参数
| 参数 | 值 | 说明 |
|------|-----|------|
| cost_out_of_boundary | 1.0 | 边界越界代价 |
| contact_other_cost | 由配置决定 | 智能体间碰撞代价 |

---

## 9. 运行命令示例

```bash
# 基本训练（2个机器人，启用 RMP）
python ppo_leader.py \
    --task SafetyPointMultiFormationGoal0-v0 \
    --seed 0 \
    --num-envs 1 \
    --num_agents 2 \
    --render-mode human \
    --use_rmp True

# 更多机器人（5个）
python ppo_leader.py \
    --task SafetyPointMultiFormationGoal0-v0 \
    --seed 0 \
    --num-envs 4 \
    --num_agents 5 \
    --use_rmp True
```

---

## 10. 注意事项

1. **编队距离不一致**: 
   - 奖励函数中的编队目标距离是 `0.4` 米
   - RMP 中的编队目标距离是 `0.5` 米
   - 这可能导致奖励和 RMP 修正之间的轻微不一致

2. **目标位置**: 
   - 代码中目标位置设置为 `(0, 2.0)`，但注释中提到 `(0.8, 0.0)`
   - 需要确认实际使用的目标位置

3. **观测空间**: 
   - 观测空间大小取决于启用的传感器和障碍物观测
   - 实际维度可能因配置而异

4. **RMP 可用性**: 
   - 如果 RMP 模块不可用，系统会自动禁用 RMP 修正
   - 训练仍可正常进行，但不会有实时动作修正

