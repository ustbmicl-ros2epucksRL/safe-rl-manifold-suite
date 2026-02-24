# 目录结构说明

## 核心目录

### cosmos/ - 主框架 (核心代码)
所有代码整合于此目录，包含完整的 Safe MARL 框架。

```
cosmos/
├── train.py           # 训练入口
├── trainer.py         # 训练器
├── registry.py        # 组件注册
├── configs/           # Hydra 配置文件
├── envs/              # 环境层 (formation_nav, epuck, safety_gym, ...)
├── algos/             # 算法层 (mappo, qmix, maddpg)
├── safety/            # 安全层 (cbf, atacom, rmpflow)
├── buffers/           # 经验缓冲区
├── runners/           # 训练运行器
└── apps/              # 应用层
    └── formation_nav/ # 编队导航应用 (demo, benchmark)
```

**使用方式:**
```bash
python -m cosmos.train env=formation_nav algo=mappo safety=cbf
```

---

### tests/ - 测试套件
验证所有组件正常工作。

```bash
python tests/test_all_envs.py
```

---

### examples/ - 示例与演示
包含 Jupyter Notebook 演示。

- `Epuck_Colab_Demo.ipynb` - Google Colab E-puck 可视化演示

---

### ros2_ws/ - ROS2 部署
E-puck 机器人 ROS2 部署包，用于实物机器人。

```
ros2_ws/src/epuck_formation/
├── package.xml        # ROS2 包配置
├── launch/            # 启动文件
├── scripts/           # 节点脚本
└── worlds/            # Webots 仿真世界
```

**使用方式 (Ubuntu):**
```bash
cd ros2_ws && colcon build
ros2 launch epuck_formation epuck_formation.launch.py
```

---

### docs/ - 文档
安装指南和使用说明。

- `ROS2_WEBOTS_SETUP.md` - ROS2 + Webots 环境配置

---

### scripts/ - 工具脚本
数据分析和实验工具。

- `analyze_results.py` - 实验结果分析与绘图

---

### refs/ - 参考文献
论文 PDF 和阅读笔记。

| 文件 | 内容 |
|------|------|
| `ATACOM-TRO-Liu2024.pdf` | ATACOM 约束流形论文 |
| `MultiRobotRMP-Li2019.pdf` | 多机器人 RMPflow |
| `MADDPG-Lowe2017.pdf` | MADDPG 算法 |
| `reading-notes.md` | 阅读笔记 |

---

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

## Git 子模块 (外部引用)

这些目录是 Git 子模块占位符，引用外部仓库：

| 目录 | 仓库 | 说明 |
|------|------|------|
| `algorithms/multi-robot-rmpflow` | chengzizhuo/multi-robot-rmpflow | RMPflow 参考实现 |
| `algorithms/safe-po` | chengzizhuo/safe-po | 安全策略优化参考 |
| `envs/safety-gymnasium` | chengzizhuo/safe-gym | Safety-Gym 参考 |

**初始化子模块:**
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

## 遗留目录

### formation_nav/ - 旧代码 (可删除)
原独立的编队导航实现，现已整合到 `cosmos/apps/formation_nav/`。

**状态:** 保留用于参考，后续可删除。

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
