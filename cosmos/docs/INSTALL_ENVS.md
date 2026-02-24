# 环境安装指南

## 核心环境（内置，无需安装）

### 1. Formation Navigation
```bash
# 已内置，直接使用
python -m cosmos.train env=formation_nav algo=mappo safety=cosmos
```

### 2. E-puck Simulation
```bash
# 已内置，直接使用
python -m cosmos.train env=epuck_sim algo=mappo safety=cbf
```

---

## Safe RL Benchmark（推荐安装）

### 3. Safety-Gymnasium
```bash
# 安装
pip install safety-gymnasium

# 使用
python -m cosmos.train env=safety_gym algo=mappo safety=cbf

# 可用环境
# - SafetyPointGoal1-v0: 点机器人导航
# - SafetyCarGoal1-v0: 小车导航
# - SafetyPointButton1-v0: 按钮任务
# - SafetyPointPush1-v0: 推箱子任务
```

### 4. VMAS（GPU加速多智能体）
```bash
# 安装
pip install vmas

# 使用（支持GPU加速）
python -m cosmos.train env=vmas algo=mappo

# 可用场景
# - navigation: 多智能体导航
# - formation_control: 编队控制
# - transport: 协作搬运
```

---

## 可选环境

### 5. MuJoCo（高精度物理）
```bash
# 安装
pip install mujoco gymnasium[mujoco]

# 使用
python -m cosmos.train env=mujoco algo=maddpg
```

### 6. Multi-Agent MuJoCo
```bash
# 安装
pip install multiagent-mujoco mujoco

# 使用
python -m cosmos.train env=ma_mujoco algo=mappo
```

### 7. Webots E-puck（Sim2Real）
```bash
# 1. 下载安装 Webots
#    https://cyberbotics.com/

# 2. 设置环境变量
export WEBOTS_HOME=/Applications/Webots.app  # macOS
# export WEBOTS_HOME=/usr/local/webots      # Linux

# 3. 使用
python -m cosmos.train env=webots_epuck algo=mappo
```

---

## 快速安装（论文实验推荐）

```bash
# 安装论文实验所需的所有环境
pip install safety-gymnasium vmas

# 验证安装
python -c "
import safety_gymnasium
import vmas
print('All environments installed successfully!')
"
```

---

## 环境对比

| 环境 | 多智能体 | 安全约束 | GPU | Sim2Real | 论文用途 |
|------|---------|---------|-----|----------|---------|
| formation_nav | ✅ 4+ | ✅ | ❌ | ❌ | 主实验 |
| epuck_sim | ✅ 4+ | ✅ | ❌ | ✅ | 真实验证 |
| safety_gym | ❌ 1 | ✅ | ❌ | ❌ | Baseline对比 |
| vmas | ✅ 100+ | ❌ | ✅ | ❌ | 可扩展性 |

---

## 论文实验配置示例

```bash
# 实验1: COSMOS主实验（formation_nav）
python -m cosmos.train env=formation_nav algo=mappo safety=cosmos \
    experiment.num_episodes=1000

# 实验2: 安全性对比（Safety-Gym）
python -m cosmos.train env=safety_gym algo=mappo safety=cbf \
    env.env_id=SafetyPointGoal1-v0

# 实验3: 算法对比
python -m cosmos.train env=formation_nav algo=qmix safety=cosmos
python -m cosmos.train env=formation_nav algo=maddpg safety=cosmos

# 实验4: 可扩展性（VMAS, 大规模）
python -m cosmos.train env=vmas algo=mappo \
    env.num_agents=16 env.num_envs=64 env.device=cuda
```
