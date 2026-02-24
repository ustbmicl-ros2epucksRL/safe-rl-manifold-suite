# Safe Multi-Robot Formation Navigation

基于约束流形的多机器人安全编队导航系统

## 研究背景

多机器人编队导航需要同时满足三个相互竞争的目标：

| 目标 | 描述 | 挑战 |
|------|------|------|
| **导航** | 编队整体移动到目标位置 | 多智能体协调 |
| **编队保持** | 维持期望的几何队形 | 与避碰约束冲突 |
| **安全约束** | 避免碰撞（智能体间、障碍物、边界） | 需要硬保证 |

传统强化学习方法只能通过奖励函数"软约束"惩罚碰撞，无法提供形式化安全保证。

## 解决方案：COSMOS

**COSMOS** (COordinated Safety On Manifold for multi-agent Systems) 是本项目提出的多智能体安全框架：

```
┌─────────────────────────────────────────────────────────┐
│                     COSMOS 架构                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   MAPPO 策略 ──→ 原始动作 α ──→ COSMOS 安全滤波 ──→ 安全动作  │
│       ↑                              │                  │
│       │                              ↓                  │
│   奖励/观测 ←───────────────── 多智能体环境             │
│                                                         │
│   关键组件:                                              │
│   • 约束流形投影 (零空间)                                │
│   • CBF 安全校正                                        │
│   • RMPflow 编队力引导                                  │
│   • 死锁检测与解决                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 核心特性

- **形式化安全保证**：从训练第1步起零碰撞
- **集中式/分布式模式**：适应不同通信条件
- **耦合约束处理**：编队形状、连通性约束
- **优先级机制**：基于危险度动态调整

## 项目结构

```
safe-rl-manifold-suite/
├── formation_nav/              # 核心实现：多机器人编队导航
│   ├── safety/
│   │   ├── cosmos.py           # COSMOS 安全滤波器
│   │   ├── atacom.py           # ATACOM 约束流形投影
│   │   ├── rmp_tree.py         # RMPflow 树结构
│   │   └── rmp_policies.py     # RMPflow 编队策略
│   ├── algo/
│   │   ├── mappo.py            # MAPPO 算法
│   │   ├── networks.py         # Actor/Critic 网络
│   │   └── buffer.py           # 经验回放缓冲区
│   ├── env/
│   │   ├── formation_env.py    # 编队导航环境
│   │   └── formations.py       # 编队形状与拓扑
│   ├── train.py                # 训练脚本
│   ├── eval.py                 # 评估脚本
│   ├── benchmark.py            # 基准测试（RMPflow vs MAPPO）
│   ├── COSMOS_Demo.ipynb       # Colab 演示 Notebook
│   └── README.md               # 详细文档
│
├── cosmos/                     # 统一训练框架（配置驱动）
│   ├── registry.py             # 组件注册器
│   ├── trainer.py              # 统一训练器
│   ├── train.py                # Hydra 训练入口
│   ├── configs/                # YAML 配置文件
│   ├── envs/                   # 环境基类与封装
│   ├── algos/                  # 算法基类与封装
│   └── safety/                 # 安全滤波器基类与封装
│
├── refs/                       # 参考文献与阅读笔记
├── paper/                      # 论文资料
├── ARCHITECTURE.md             # 架构设计文档
└── CLAUDE.md                   # Claude AI 开发指南
```

## 快速开始

### 方式一：自动安装（推荐）

```bash
# 1. 运行安装脚本
chmod +x setup.sh
./setup.sh

# 2. 激活环境
conda activate cosmos

# 3. 验证安装
python test_all_envs.py

# 4. 运行实验
./run_experiments.sh quick
```

### 方式二：手动安装

```bash
# 创建环境
conda create -n cosmos python=3.10 -y
conda activate cosmos

# 安装依赖
pip install torch numpy scipy matplotlib gymnasium
pip install hydra-core omegaconf wandb tqdm

# 安装可选环境
pip install safety-gymnasium vmas mujoco

# 安装 COSMOS
pip install -e .
```

### 方式三：Google Colab

```python
!pip install gymnasium torch hydra-core omegaconf -q
!git clone https://github.com/ustbmicl-ros2epucksRL/safe-rl-manifold-suite.git
%cd safe-rl-manifold-suite
!pip install -e . -q
!python test_all_envs.py
```

## 多环境实验

### 可用环境

| 环境 | 命令 | 安装 |
|------|------|------|
| 编队导航 | `env=formation_nav` | 内置 |
| E-puck模拟 | `env=epuck_sim` | 内置 |
| Safety-Gym | `env=safety_gym` | `pip install safety-gymnasium` |
| VMAS | `env=vmas` | `pip install vmas` |

### 运行训练

```bash
# 编队导航 + MAPPO + COSMOS
python -m cosmos.train env=formation_nav algo=mappo safety=cosmos

# E-puck + QMIX + CBF
python -m cosmos.train env=epuck_sim algo=qmix safety=cbf

# Safety-Gym 基准测试
python -m cosmos.train env=safety_gym algo=mappo safety=cbf \
    env.env_id=SafetyPointGoal1-v0

# 自定义参数
python -m cosmos.train \
    env=formation_nav \
    algo=mappo \
    safety=cosmos \
    env.num_agents=6 \
    experiment.num_episodes=1000 \
    logging.use_wandb=true
```

### 批量实验

```bash
# 快速测试
./run_experiments.sh quick

# 编队实验
./run_experiments.sh formation

# 安全性对比
./run_experiments.sh safety

# 消融实验
./run_experiments.sh ablation

# 全部实验
./run_experiments.sh all
```

### 结果分析

```bash
# 生成图表和表格
python scripts/analyze_results.py experiments/TIMESTAMP/

# 输出:
# - learning_curves.png/pdf
# - safety_comparison.png/pdf
# - results_table.tex
```

## 演示结果

训练 200 轮后的典型结果：

| 指标 | 结果 |
|------|------|
| 碰撞次数 | 0 (100% 安全) |
| 编队误差 | < 0.02 |
| 训练速度 | ~2000 FPS |

## 理论基础

| 方法 | 来源 | 作用 |
|------|------|------|
| **ATACOM** | Liu et al. 2021, 2024 | 约束流形投影 |
| **RMPflow** | Cheng et al. 2018 | 几何运动策略 |
| **MAPPO** | Yu et al. 2022 | 多智能体强化学习 |

详细理论请参考 [`formation_nav/docs/THEORY.md`](formation_nav/docs/THEORY.md)

## 参考文献

1. Liu et al., "Robot Reinforcement Learning on the Constraint Manifold", CoRL 2021
2. Liu et al., "Safe RL on the Constraint Manifold: Theory and Applications", IEEE T-RO 2024
3. Cheng et al., "RMPflow: A Computational Graph for Automatic Motion Policy Generation", WAFR 2018
4. Li et al., "Multi-Robot RMPflow", ISRR 2019
5. Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022

## License

MIT License
