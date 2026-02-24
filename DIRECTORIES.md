# 目录结构说明

## cosmos/ - 主框架 (全部代码)

所有代码整合于此目录，包含完整的 Safe MARL 框架。

```
cosmos/
├── train.py           # 训练入口
├── trainer.py         # 训练器
├── registry.py        # 组件注册
│
├── configs/           # Hydra 配置文件
├── envs/              # 环境层 (formation_nav, epuck, safety_gym, ...)
├── algos/             # 算法层 (mappo, qmix, maddpg)
├── safety/            # 安全层 (cbf, atacom, rmpflow)
├── buffers/           # 经验缓冲区
├── runners/           # 训练运行器
├── utils/             # 工具函数
│
├── apps/              # 应用层
│   └── formation_nav/ # 编队导航应用 (demo, benchmark)
│
├── tests/             # 测试套件
├── examples/          # 示例与演示 (Jupyter Notebook)
├── scripts/           # 工具脚本 (结果分析)
├── docs/              # 文档
└── ros2/              # ROS2 E-puck 部署包
```

**使用方式:**
```bash
# 训练
python -m cosmos.train env=formation_nav algo=mappo safety=cbf

# 测试
python -m cosmos.tests.test_all_envs

# 演示
python -m cosmos.apps.formation_nav.demo
```

---

## cosmos/ 子目录详情

### algos/ - 算法层
MARL 算法实现。

| 文件 | 说明 |
|------|------|
| `mappo.py` | Multi-Agent PPO |
| `qmix.py` | QMIX 值分解 |
| `maddpg.py` | MADDPG |
| `base.py` | 算法基类 |
| `networks.py` | 神经网络 |

### envs/ - 环境层
多智能体环境封装。

| 文件 | 说明 |
|------|------|
| `formation_nav.py` | 编队导航环境 |
| `webots_wrapper.py` | E-puck Webots 仿真 |
| `safety_gym_wrapper.py` | Safety-Gymnasium 封装 |
| `vmas_wrapper.py` | VMAS 矢量环境 |
| `mujoco_wrapper.py` | MuJoCo 环境 |
| `formations.py` | 编队几何形状 |
| `base.py` | 环境基类 |

### safety/ - 安全层
安全滤波器实现。

| 文件 | 说明 |
|------|------|
| `cbf.py` | Control Barrier Function |
| `atacom.py` | ATACOM 约束流形投影 |
| `rmp_tree.py` | RMPflow 树结构 |
| `rmp_policies.py` | RMP 行为策略 |
| `constraints.py` | 约束定义 |
| `base.py` | 安全滤波器基类 |

### apps/ - 应用层
特定应用的实现。

```
apps/
└── formation_nav/
    ├── config.py      # 应用配置
    ├── demo.py        # 交互演示
    ├── demo_visualization.py  # 可视化
    └── benchmark.py   # 性能基准测试
```

### tests/ - 测试套件
验证所有组件正常工作。

```bash
python -m cosmos.tests.test_all_envs
```

### examples/ - 示例
Jupyter Notebook 演示。

| 文件 | 说明 |
|------|------|
| `Epuck_Colab_Demo.ipynb` | Google Colab E-puck 可视化演示 |

### scripts/ - 工具脚本
数据分析和实验工具。

| 文件 | 说明 |
|------|------|
| `analyze_results.py` | 实验结果分析与绘图 |

### docs/ - 文档
安装指南和使用说明。

| 文件 | 说明 |
|------|------|
| `ROS2_WEBOTS_SETUP.md` | ROS2 + Webots 环境配置 |

### ros2/ - ROS2 部署
E-puck 机器人 ROS2 部署包，用于实物机器人。

```
ros2/
├── package.xml        # ROS2 包配置
├── CMakeLists.txt     # CMake 配置
├── config/            # ROS2 参数配置
├── launch/            # 启动文件
├── scripts/           # 节点脚本
└── worlds/            # Webots 仿真世界
```

**使用方式 (Ubuntu):**
```bash
cd cosmos/ros2 && colcon build
ros2 launch epuck_formation epuck_formation.launch.py
```

---

## 根目录其他文件夹

### refs/ - 参考文献
论文 PDF 和阅读笔记。

| 文件 | 内容 |
|------|------|
| `ATACOM-TRO-Liu2024.pdf` | ATACOM 约束流形论文 |
| `MultiRobotRMP-Li2019.pdf` | 多机器人 RMPflow |
| `MADDPG-Lowe2017.pdf` | MADDPG 算法 |
| `reading-notes.md` | 阅读笔记 |

### paper/ - 论文资料
毕业论文和课题相关材料。

```
paper/
└── 3_Control_SafeMARL/    # 课题「控制与 SafeMARL」
    ├── 任务1.../          # IROS 论文
    ├── 毕业论文/          # 本科毕设
    └── 硕士毕业论文/      # 硕士论文
```

---

## Git 子模块 (外部参考)

这些目录是 Git 子模块，引用外部仓库作为参考实现：

| 目录 | 仓库 | 说明 |
|------|------|------|
| `algorithms/multi-robot-rmpflow` | chengzizhuo/multi-robot-rmpflow | RMPflow 参考实现 |
| `algorithms/safe-po` | chengzizhuo/safe-po | 安全策略优化参考 |
| `envs/safety-gymnasium` | chengzizhuo/safe-gym | Safety-Gym 参考 |

**注意:** 这些是外部参考代码，不是本项目的核心代码。核心实现都在 `cosmos/` 目录中。

**初始化子模块 (可选):**
```bash
git submodule update --init --recursive
```

---

## 生成数据 (自动忽略)

这些目录由训练/测试生成，已在 `.gitignore` 中忽略：

| 目录 | 内容 |
|------|------|
| `checkpoints/` | 训练模型检查点 (.pt) |
| `demo_output/` | 演示输出 (图片、GIF、模型) |
| `outputs/` | Hydra 训练输出 |
| `results/` | 实验结果 (metrics.json) |

---

## 配置文件

| 文件 | 说明 |
|------|------|
| `setup.py` | pip 包安装配置 |
| `setup.sh` | Conda 环境安装脚本 |
| `run_experiments.sh` | 批量实验脚本 |
| `requirements.txt` | Python 依赖 |
| `.gitmodules` | Git 子模块配置 |
| `ARCHITECTURE.md` | 详细架构文档 |
| `CLAUDE.md` | Claude AI 开发指南 |
| `README.md` | 项目说明 |
| `DIRECTORIES.md` | 本文件 |
| `INSTALL_ENVS.md` | 环境安装指南 |
