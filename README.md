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
├── formation_nav/           # 主要代码
│   ├── safety/
│   │   ├── cosmos.py        # COSMOS 核心实现
│   │   ├── atacom.py        # 基础 ATACOM
│   │   └── rmp_policies.py  # RMPflow 编队策略
│   ├── algo/
│   │   └── mappo.py         # MAPPO 算法
│   ├── env/
│   │   └── formation_env.py # 编队导航环境
│   ├── demo.py              # 演示脚本
│   ├── COSMOS_Demo.ipynb    # Colab Notebook
│   └── docs/THEORY.md       # 理论文档
├── refs/                    # 参考文献笔记
└── paper/                   # 论文资料
```

## 快速开始

### 本地运行

```bash
# 安装依赖
pip install torch numpy gymnasium matplotlib scipy pillow

# 运行演示
PYTHONPATH=. python formation_nav/demo.py --episodes 200
```

### Google Colab

```python
!pip install gymnasium -q
!git clone https://github.com/ustbmicl-ros2epucksRL/safe-rl-manifold-suite.git
%cd safe-rl-manifold-suite
!python formation_nav/demo.py --episodes 200
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
