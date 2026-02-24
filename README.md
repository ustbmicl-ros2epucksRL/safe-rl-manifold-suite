# COSMOS: Safe Multi-Agent Reinforcement Learning Framework

**COSMOS** (COordinated Safety On Manifold for multi-agent Systems) - 基于约束流形的多智能体安全强化学习框架

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ustbmicl-ros2epucksRL/safe-rl-manifold-suite/blob/master/cosmos/examples/Epuck_Colab_Demo.ipynb)

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COSMOS 系统架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         应用层 (Applications)                        │  │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │
│   │  │ 编队导航      │  │ E-puck 仿真  │  │ ROS2 部署    │               │  │
│   │  │ formation_nav│  │ examples/    │  │ ros2_ws/     │               │  │
│   │  └──────────────┘  └──────────────┘  └──────────────┘               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         核心框架 (cosmos/)                           │  │
│   │                                                                     │  │
│   │   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   │  │
│   │   │  环境层   │   │  算法层   │   │  安全层   │   │  运行层   │   │  │
│   │   │  envs/    │   │  algos/   │   │  safety/  │   │  runners/ │   │  │
│   │   │           │   │           │   │           │   │  buffers/ │   │  │
│   │   │ •Formation│   │ •MAPPO    │   │ •CBF      │   │           │   │  │
│   │   │ •Epuck    │   │ •QMIX     │   │ •COSMOS   │   │ •Episode  │   │  │
│   │   │ •SafetyGym│   │ •MADDPG   │   │ •ATACOM   │   │ •Parallel │   │  │
│   │   │ •MuJoCo   │   │           │   │           │   │           │   │  │
│   │   │ •VMAS     │   │           │   │           │   │           │   │  │
│   │   └───────────┘   └───────────┘   └───────────┘   └───────────┘   │  │
│   │                                                                     │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │                    基础设施层                                │   │  │
│   │   │  Registry (组件注册)  │  Hydra Config (配置管理)  │  WandB   │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
safe-rl-manifold-suite/
│
├── cosmos/                      # 🎯 统一框架 (所有代码整合于此)
│   ├── train.py                 # 训练入口: python -m cosmos.train
│   ├── trainer.py               # 统一训练器
│   ├── registry.py              # 组件注册器
│   │
│   ├── configs/                 # Hydra 配置
│   ├── envs/                    # 环境层 (formation_nav, epuck, safety_gym, ...)
│   ├── algos/                   # 算法层 (mappo, qmix, maddpg)
│   ├── safety/                  # 安全层 (cbf, atacom, rmpflow)
│   ├── buffers/                 # 缓冲区 (rollout, replay)
│   ├── runners/                 # 运行器
│   ├── utils/                   # 工具函数
│   │
│   ├── apps/                    # 应用层
│   │   └── formation_nav/       # 编队导航应用 (demo, benchmark)
│   │
│   ├── tests/                   # ✅ 测试套件
│   ├── examples/                # 📚 示例 (Jupyter Notebook)
│   ├── scripts/                 # 🔧 工具脚本
│   ├── docs/                    # 📖 文档
│   └── ros2/                    # 🤖 ROS2 E-puck 部署
│
├── refs/                        # 📑 参考文献 (PDF, 笔记)
├── paper/                       # 📄 论文资料
│
├── algorithms/                  # Git 子模块 (外部参考)
├── envs/                        # Git 子模块 (外部参考)
│
├── setup.py                     # pip 安装
├── setup.sh                     # 环境安装
└── README.md
```

---

## 核心组件

### 1. 环境 (Environments)

| 环境 | 描述 | 智能体数 | 安装 |
|------|------|---------|------|
| `formation_nav` | 多机器人编队导航 | 可变 | 内置 |
| `epuck_sim` | E-puck 机器人仿真 | 可变 | 内置 |
| `safety_gym` | Safety-Gymnasium | 1 | `pip install safety-gymnasium` |
| `mujoco` | MuJoCo 物理仿真 | 可变 | `pip install mujoco` |
| `vmas` | 向量化多智能体仿真 | 可变 | `pip install vmas` |

### 2. 算法 (Algorithms)

| 算法 | 类型 | 描述 |
|------|------|------|
| `mappo` | On-Policy | Multi-Agent PPO with CTDE |
| `qmix` | Value-Based | Value Decomposition with Mixing Network |
| `maddpg` | Off-Policy | Multi-Agent DDPG with Centralized Critic |

### 3. 安全滤波器 (Safety Filters)

| 滤波器 | 方法 | 描述 |
|--------|------|------|
| `cbf` | Control Barrier Function | 基于 QP 的安全动作投影 |
| `cosmos` | Manifold Projection | 约束流形 + RMPflow |
| `none` | Pass-through | 无安全约束 (基线) |

---

## 数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              训练数据流                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐         │
│    │   Env   │ obs  │ Policy  │action│ Safety  │ safe │   Env   │         │
│    │  reset  │─────▶│  (RL)   │─────▶│ Filter  │─────▶│  step   │         │
│    └─────────┘      └─────────┘      └─────────┘      └────┬────┘         │
│                                                            │               │
│         ┌──────────────────────────────────────────────────┘               │
│         │ (obs, reward, cost, done)                                        │
│         ▼                                                                  │
│    ┌─────────┐                                                             │
│    │ Buffer  │                                                             │
│    │ (GAE)   │                                                             │
│    └────┬────┘                                                             │
│         │                                                                  │
│         ▼                                                                  │
│    ┌─────────┐                                                             │
│    │ Update  │                                                             │
│    │ Policy  │                                                             │
│    └─────────┘                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/ustbmicl-ros2epucksRL/safe-rl-manifold-suite.git
cd safe-rl-manifold-suite

# 方式1: 自动安装
chmod +x setup.sh && ./setup.sh

# 方式2: 手动安装
pip install -e .
pip install torch numpy scipy matplotlib gymnasium hydra-core omegaconf

# 可选: 安装额外环境
pip install safety-gymnasium mujoco vmas
```

### 验证安装

```bash
python -m cosmos.tests.test_all_envs
```

### 运行训练

```bash
# 使用 COSMOS 框架
python -m cosmos.train env=formation_nav algo=mappo safety=cbf

# 切换环境
python -m cosmos.train env=epuck_sim algo=mappo safety=cbf

# 切换算法
python -m cosmos.train env=formation_nav algo=qmix safety=cbf

# 自定义参数
python -m cosmos.train env=formation_nav algo=mappo safety=cbf \
    env.num_agents=6 \
    experiment.num_episodes=500

# 使用 formation_nav 应用演示
python -m cosmos.apps.formation_nav.demo
```

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ustbmicl-ros2epucksRL/safe-rl-manifold-suite/blob/master/cosmos/examples/Epuck_Colab_Demo.ipynb)

```python
!pip install torch numpy matplotlib gymnasium -q
!git clone https://github.com/ustbmicl-ros2epucksRL/safe-rl-manifold-suite.git
%cd safe-rl-manifold-suite
!pip install -e . -q
!python -m cosmos.tests.test_all_envs
```

---

## 程序说明

### 主程序: COSMOS 框架 (`cosmos/`)

统一的配置驱动训练框架，所有代码整合于此目录。

```bash
# 训练
python -m cosmos.train env=formation_nav algo=mappo safety=cbf

# 编队导航演示
python -m cosmos.apps.formation_nav.demo

# 基准测试
python -m cosmos.apps.formation_nav.benchmark
```

**架构层次:**
```
cosmos/
├── envs/      # 环境层 (formation_nav, epuck_sim, safety_gym, ...)
├── algos/     # 算法层 (mappo, qmix, maddpg)
├── safety/    # 安全层 (cbf, atacom, rmpflow)
├── buffers/   # 缓冲区 (rollout, replay)
├── runners/   # 运行器 (episode, parallel)
└── apps/      # 应用层 (formation_nav demo/benchmark)
```

### 测试套件 (`cosmos/tests/`)

```bash
python -m cosmos.tests.test_all_envs
```

### ROS2 部署 (`cosmos/ros2/`)

```bash
cd cosmos/ros2 && colcon build
ros2 launch epuck_formation epuck_formation.launch.py
```

---

## 配置系统

### Hydra 配置示例

```yaml
# cosmos/configs/config.yaml
defaults:
  - env: formation_nav
  - algo: mappo
  - safety: cosmos

experiment:
  name: cosmos_exp
  seed: 42
  num_episodes: 200
  device: auto

logging:
  use_wandb: false
  output_dir: outputs
```

### 命令行覆盖

```bash
# 修改环境参数
python -m cosmos.train env.num_agents=8

# 修改算法参数
python -m cosmos.train algo.actor_lr=1e-4

# 多配置 sweep
python -m cosmos.train -m algo=mappo,qmix,maddpg
```

---

## 安全滤波器原理

### CBF (Control Barrier Function)

```
min  ||u - u_nom||²           # 最小化与原始动作的偏差
s.t. ḣ(x,u) + αh(x) ≥ 0       # CBF 安全条件

其中:
- h(x) = ||p_i - p_j||² - d_safe²  (碰撞避免)
- α > 0 为 CBF 增益
```

### COSMOS (Manifold Projection)

```
u* = N · u_nom + J⁺ · (-α·c(q))
     ↑              ↑
  零空间分量    约束校正分量

其中:
- c(q) = 0 为约束方程 (编队/连通性)
- J = ∂c/∂q 为约束雅可比
- N = I - J⁺J 为零空间投影矩阵
```

---

## 性能指标

| 环境 | 训练速度 | 碰撞率 |
|------|---------|--------|
| formation_nav | ~10k steps/sec | 0% (with CBF) |
| epuck_sim | ~5k steps/sec | 0% (with CBF) |
| safety_gym | ~1k steps/sec | <1% |

---

## 扩展开发

### 添加新环境

```python
# cosmos/envs/my_env.py
from cosmos.registry import ENV_REGISTRY
from cosmos.envs.base import BaseMultiAgentEnv

@ENV_REGISTRY.register("my_env")
class MyEnv(BaseMultiAgentEnv):
    def reset(self, seed=None):
        return obs, share_obs, info

    def step(self, actions):
        return obs, share_obs, rewards, costs, dones, infos, truncated

    def get_constraint_info(self):
        return {"positions": ..., "velocities": ...}
```

### 添加新算法

```python
# cosmos/algos/my_algo.py
from cosmos.registry import ALGO_REGISTRY
from cosmos.algos.base import BaseMARLAlgo

@ALGO_REGISTRY.register("my_algo")
class MyAlgo(BaseMARLAlgo):
    def get_actions(self, obs, deterministic=False):
        return actions, log_probs

    def update(self, buffer):
        return {"loss": loss}
```

---

## 参考文献

| 方法 | 论文 | 用途 |
|------|------|------|
| ATACOM | Liu et al., CoRL 2021 | 约束流形投影 |
| CBF | Ames et al., 2017 | 控制屏障函数 |
| RMPflow | Cheng et al., WAFR 2018 | 几何运动策略 |
| MAPPO | Yu et al., NeurIPS 2022 | 多智能体 PPO |
| QMIX | Rashid et al., ICML 2018 | 值分解 |
| MADDPG | Lowe et al., NeurIPS 2017 | 多智能体 DDPG |

---

## License

MIT License

---

## 贡献

欢迎提交 Issue 和 Pull Request。

## 联系

- GitHub: [ustbmicl-ros2epucksRL](https://github.com/ustbmicl-ros2epucksRL)
