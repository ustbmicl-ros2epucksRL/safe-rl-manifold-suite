# COSMOS 详细架构文档

## 一、代码统计

| 指标 | 值 |
|------|-----|
| 总 Python 文件 | 48 |
| 总代码行数 | ~9,000+ |
| 抽象基类 | 4 (Env, Algo, Safety, Buffer) |
| 已实现环境 | 7 |
| 已实现算法 | 3 |
| 已实现安全滤波器 | 3 |
| 配置文件 (YAML) | 14 |

---

## 二、程序清单

| 程序 | 入口 | 功能 |
|------|------|------|
| **COSMOS 训练** | `python -m cosmos.train` | 统一训练框架 |
| **编队导航演示** | `python -m cosmos.apps.formation_nav.demo` | 可视化演示 |
| **基准测试** | `python -m cosmos.apps.formation_nav.benchmark` | 性能对比 |
| **测试套件** | `python -m cosmos.tests.test_all_envs` | 组件测试 |
| **ROS2 部署** | `ros2 launch epuck_formation ...` | 机器人部署 |

---

## 三、目录结构详解

```
safe-rl-manifold-suite/
│
├── cosmos/                              # 统一框架 (~9,000+ 行代码)
│   ├── __init__.py
│   ├── train.py                         # 📌 Hydra 训练入口 (115行)
│   ├── trainer.py                       # 统一训练器 (377行)
│   ├── registry.py                      # 组件注册器 (192行)
│   │
│   ├── configs/                         # Hydra 配置文件
│   │   ├── config.yaml                  # 主配置
│   │   ├── env/                         # 环境配置 (formation_nav, vmas, ...)
│   │   ├── algo/                        # 算法配置 (mappo, qmix, maddpg)
│   │   └── safety/                      # 安全滤波配置 (cosmos, cbf, none)
│   │
│   ├── envs/                            # 环境层 (3,122行, 7个环境)
│   │   ├── __init__.py
│   │   ├── base.py                      # BaseMultiAgentEnv 抽象基类
│   │   ├── formation_nav.py             # 编队导航环境 (426行)
│   │   ├── formations.py                # 编队形状与拓扑
│   │   ├── env_wrapper.py               # 外部环境包装器
│   │   ├── vmas_wrapper.py              # VMAS 向量化环境 (223行)
│   │   ├── safety_gym_wrapper.py        # Safety-Gymnasium (368行)
│   │   ├── mujoco_wrapper.py            # MuJoCo 环境 (338行)
│   │   ├── webots_wrapper.py            # Webots E-puck (755行)
│   │   └── epuck_visualizer.py          # E-puck 可视化
│   │
│   ├── algos/                           # 算法层 (1,527行, 3个算法)
│   │   ├── __init__.py
│   │   ├── base.py                      # BaseMARLAlgo, OnPolicyAlgo, OffPolicyAlgo
│   │   ├── mappo.py                     # Multi-Agent PPO (331行)
│   │   ├── qmix.py                      # QMIX 值分解 (500行)
│   │   └── maddpg.py                    # Multi-Agent DDPG (428行)
│   │
│   ├── safety/                          # 安全层 (1,795行)
│   │   ├── __init__.py
│   │   ├── base.py                      # BaseSafetyFilter 抽象基类
│   │   ├── cosmos_filter.py             # COSMOS + CBF 滤波器 (478行)
│   │   ├── atacom.py                    # ATACOM 流形投影 (411行)
│   │   ├── constraints.py               # StateConstraint, ConstraintsSet (155行)
│   │   ├── rmp_tree.py                  # RMPflow 树结构 (132行)
│   │   └── rmp_policies.py              # RMP 叶策略 (414行)
│   │
│   ├── buffers/                         # 缓冲区 (283行)
│   │   ├── __init__.py
│   │   ├── rollout_buffer.py            # On-policy PPO 缓冲区
│   │   └── replay_buffer.py             # Off-policy 回放缓冲区
│   │
│   ├── runners/                         # 运行器 (~250行)
│   │   ├── __init__.py
│   │   ├── episode_runner.py            # 回合收集器
│   │   └── parallel_runner.py           # 并行环境运行器
│   │
│   ├── utils/                           # 工具函数
│   │   └── checkpoint.py                # 检查点管理
│   │
│   ├── apps/                            # 应用层
│   │   └── formation_nav/               # 编队导航应用
│   │       ├── config.py                # 应用配置 (dataclass)
│   │       ├── demo.py                  # 📌 训练 + 可视化演示
│   │       ├── demo_visualization.py    # 视频生成
│   │       └── benchmark.py             # 📌 基准测试
│   │
│   ├── tests/                           # 测试套件
│   │   └── test_all_envs.py             # 环境集成测试
│   │
│   ├── scripts/                         # 分析脚本
│   │   └── analyze_results.py           # 结果分析与绘图
│   │
│   ├── examples/                        # 示例
│   │   └── Epuck_Colab_Demo.ipynb       # Colab 演示
│   │
│   ├── ros2/                            # ROS2 部署
│   │   └── epuck_formation/
│   │       ├── launch/                  # ROS2 launch 文件
│   │       ├── scripts/                 # 控制节点
│   │       ├── worlds/                  # Webots 世界文件
│   │       └── config/                  # 参数配置
│   │
│   └── docs/                            # 设计文档
│       ├── THEORY.md                    # 理论基础
│       ├── ARCHITECTURE.md              # 本文件
│       ├── DIRECTORIES.md               # 目录说明
│       ├── INSTALL_ENVS.md              # 环境安装
│       └── ROS2_WEBOTS_SETUP.md         # ROS2 部署
│
├── refs/                                # 参考文献与学习笔记
│   ├── *.pdf                            # 参考论文
│   └── *.md                             # 学习笔记
│
├── artifacts/                           # 生成数据 (gitignored)
│   ├── checkpoints/                     # 模型检查点
│   ├── demo_output/                     # 演示输出
│   ├── outputs/                         # Hydra 输出
│   └── results/                         # 实验结果
│
├── algorithms/                          # Git 子模块 (外部参考)
├── envs/                                # Git 子模块 (外部参考)
├── paper/                               # 论文资料
│
├── README.md                            # 项目说明
└── CLAUDE.md                            # Claude 开发指南
```

---

## 四、注册表系统

### 4.1 三大注册表

```python
# cosmos/registry.py

ENV_REGISTRY      # 环境注册表
ALGO_REGISTRY     # 算法注册表
SAFETY_REGISTRY   # 安全滤波器注册表
BUFFER_REGISTRY   # 缓冲区注册表
```

### 4.2 已注册组件

| 注册表 | 名称 | 别名 | 类 |
|--------|------|------|-----|
| **ENV** | formation_nav | formation, nav | FormationNavEnv |
| | vmas | - | VMASWrapper |
| | safety_gym | - | SafetyGymWrapper |
| | mujoco | - | MuJoCoWrapper |
| | ma_mujoco | - | MultiAgentMuJoCoWrapper |
| | webots_epuck | - | WebotsEpuckEnv |
| | epuck_sim | - | EpuckSimEnv |
| **ALGO** | mappo | ppo, ippo | MAPPO |
| | qmix | - | QMIX |
| | maddpg | - | MADDPG |
| **SAFETY** | cosmos | atacom, manifold | COSMOSFilter |
| | cbf | - | CBFFilter |
| | none | passthrough | NoSafetyFilter |
| **BUFFER** | rollout | on_policy, ppo_buffer | RolloutBuffer |
| | replay | off_policy | ReplayBuffer |
| | episode_replay | qmix_buffer | EpisodeReplayBuffer |

### 4.3 注册表使用

```python
# 注册组件
@ENV_REGISTRY.register("my_env", aliases=["alias1", "alias2"])
class MyEnv(BaseMultiAgentEnv):
    pass

# 构建组件
env = ENV_REGISTRY.build("formation_nav", cfg=env_cfg)
algo = ALGO_REGISTRY.build("mappo", obs_dim, share_obs_dim, act_dim, num_agents)
safety = SAFETY_REGISTRY.build("cosmos", env_cfg, safety_cfg, constraint_info)
```

---

## 五、核心类接口

### 5.1 环境接口 (BaseMultiAgentEnv)

```python
class BaseMultiAgentEnv(gym.Env, ABC):
    # 必须实现的属性
    @property
    def num_agents(self) -> int: ...
    @property
    def observation_space(self) -> spaces.Space: ...
    @property
    def action_space(self) -> spaces.Space: ...
    @property
    def share_observation_space(self) -> spaces.Space: ...

    # 必须实现的方法
    def reset(self, seed=None) -> Tuple[obs, share_obs, info]: ...
    def step(self, actions) -> Tuple[obs, share_obs, rewards, costs, dones, infos, truncated]: ...
    def get_constraint_info(self) -> Dict[str, Any]: ...
```

**constraint_info 结构**:
```python
{
    "positions": np.ndarray,       # (num_agents, 2)
    "velocities": np.ndarray,      # (num_agents, 2)
    "desired_distances": np.ndarray,  # (num_agents, num_agents)
    "topology_edges": List[Tuple], # [(0,1), (1,2), ...]
    "obstacles": np.ndarray        # (num_obstacles, 3): x, y, radius
}
```

### 5.2 算法接口 (BaseMARLAlgo)

```python
class BaseMARLAlgo(ABC):
    def get_actions(self, obs, deterministic=False) -> Tuple[actions, log_probs]: ...
    def get_values(self, share_obs) -> values: ...
    def update(self, buffer) -> Dict[str, float]: ...
    def save(self, path): ...
    def load(self, path): ...
```

### 5.3 安全滤波器接口 (BaseSafetyFilter)

```python
class BaseSafetyFilter(ABC):
    def reset(self, constraint_info: Dict): ...
    def project(self, actions, constraint_info, dt=0.05) -> safe_actions: ...
    def update(self, constraint_info: Dict): ...

    # 可选方法
    def get_safety_margin(self, constraint_info) -> float: ...
    def is_safe(self, constraint_info) -> bool: ...
```

---

## 六、程序架构图

### 6.1 整体框架

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COSMOS Framework                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     train.py (Hydra 入口)                            │   │
│   │                              │                                       │   │
│   │                      ┌───────┴───────┐                              │   │
│   │                      │    Hydra      │                              │   │
│   │                      │   配置加载     │                              │   │
│   │                      └───────┬───────┘                              │   │
│   │                              │                                       │   │
│   │                      ┌───────┴───────┐                              │   │
│   │                      │   Trainer     │                              │   │
│   │                      │   (377行)     │                              │   │
│   │                      └───────┬───────┘                              │   │
│   └──────────────────────────────┼──────────────────────────────────────┘   │
│                                  │                                          │
│       ┌──────────────────────────┼──────────────────────────┐               │
│       │                          │                          │               │
│       ▼                          ▼                          ▼               │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐          │
│   │ENV_REGISTRY │         │ALGO_REGISTRY│         │SAFETY_REGIS │          │
│   │             │         │             │         │             │          │
│   │ formation   │         │   mappo     │         │   cosmos    │          │
│   │ epuck_sim   │         │   qmix      │         │    cbf      │          │
│   │ safety_gym  │         │   maddpg    │         │    none     │          │
│   │ mujoco      │         │             │         │             │          │
│   │ vmas        │         │             │         │             │          │
│   │ webots      │         │             │         │             │          │
│   └──────┬──────┘         └──────┬──────┘         └──────┬──────┘          │
│          │                       │                       │                  │
│          └───────────────────────┼───────────────────────┘                  │
│                                  │                                          │
│                    ┌─────────────┴─────────────┐                            │
│                    │                           │                            │
│                    ▼                           ▼                            │
│             ┌─────────────┐             ┌─────────────┐                     │
│             │   Runner    │             │   Buffer    │                     │
│             │  (episode/  │             │ (rollout/   │                     │
│             │  parallel)  │             │  replay)    │                     │
│             └─────────────┘             └─────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 训练数据流

```
python -m cosmos.train env=formation_nav algo=mappo safety=cosmos
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Trainer.__init__(cfg)                                                       │
│      ├── _build_env()   → FormationNavEnv                                   │
│      ├── _build_algo()  → MAPPO (Actor + Critic)                            │
│      ├── _build_safety() → COSMOSFilter (ATACOM + RMPflow)                  │
│      └── _build_buffer() → RolloutBuffer                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  trainer.train()                                                             │
│      │                                                                       │
│      └── for episode in range(num_episodes):                                │
│              │                                                               │
│              ├── obs, share_obs = env.reset()                               │
│              ├── safety.reset(constraint_info)                              │
│              │                                                               │
│              └── for step in range(max_steps):                              │
│                      │                                                       │
│                      ├── actions, log_probs = algo.get_actions(obs)        │
│                      ├── values = algo.get_values(share_obs)                │
│                      ├── constraint_info = env.get_constraint_info()        │
│                      │                                                       │
│                      ├── safe_actions = safety.project(actions,             │
│                      │                                 constraint_info, dt) │
│                      │                                                       │
│                      ├── next_obs, rewards, costs, dones = env.step(        │
│                      │                                       safe_actions)  │
│                      │                                                       │
│                      └── buffer.insert(obs, actions, rewards, ...)          │
│                                                                              │
│              ├── buffer.compute_returns_and_advantages(last_values)         │
│              └── algo.update(buffer)                                        │
│                                                                              │
│      └── save_checkpoint()                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 安全投影流程 (COSMOS)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COSMOSFilter.project()                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: actions (N, act_dim), constraint_info                                │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  1. 约束评估                                                          │  │
│  │     c_ij = r_safe - ||p_i - p_j||        (智能体间碰撞)               │  │
│  │     c_ik = r_obs - ||p_i - o_k||         (障碍物碰撞)                 │  │
│  │     c_b  = ||p_i|| - arena_bound         (边界约束)                   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  2. ATACOM 零空间投影                                                 │  │
│  │     J_c = ∂c/∂q                           (约束 Jacobian)             │  │
│  │     J_c⁺ = J_c^T @ (J_c @ J_c^T + ε*I)⁻¹  (阻尼伪逆)                  │  │
│  │     N_c = I - J_c⁺ @ J_c                  (零空间投影矩阵)            │  │
│  │     dq = N_c @ α + (-K_c * J_c⁺ @ c)      (投影 + 修正)               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  3. RMPflow 几何引导 (可选)                                           │  │
│  │     rmp_tree.set_root_state(positions, velocities)                    │  │
│  │     rmp_tree.pushforward()                                            │  │
│  │     rmp_tree.pullback()                                               │  │
│  │     f_rmp = rmp_tree.resolve()            (几何引导力)                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  4. 动作融合                                                          │  │
│  │     safe_actions = dq_safe + β * f_rmp                                │  │
│  │     safe_actions = clip(safe_actions, -dq_max, dq_max)                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  输出: safe_actions (N, act_dim)                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 七、类继承关系

### 7.1 环境类

```
gym.Env (ABC)
    │
    └── BaseMultiAgentEnv (cosmos/envs/base.py)
            │
            ├── FormationNavEnv        # 编队导航 (426行)
            ├── EpuckSimEnv            # E-puck 仿真 (481行)
            ├── WebotsEpuckEnv         # Webots E-puck (755行)
            ├── SafetyGymWrapper       # Safety-Gymnasium (368行)
            ├── MuJoCoWrapper          # MuJoCo 单智能体 (338行)
            ├── MultiAgentMuJoCoWrapper # MuJoCo 多智能体
            └── VMASWrapper            # VMAS 向量化 (223行)
```

### 7.2 算法类

```
BaseMARLAlgo (ABC)
    │
    ├── OnPolicyAlgo (ABC)
    │   └── MAPPO (331行)
    │       ├── Actor (MLP + Gaussian)
    │       ├── Critic (MLP)
    │       └── CostCritic (MLP, 可选)
    │
    └── OffPolicyAlgo (ABC)
        ├── QMIX (500行)
        │   ├── RNNAgent (GRU)
        │   └── QMixer (Hypernetwork)
        │
        └── MADDPG (428行)
            ├── Actor (Deterministic MLP)
            └── CentralizedCritic
```

### 7.3 安全滤波类

```
BaseSafetyFilter (ABC)
    │
    ├── COSMOSFilter (478行)          # 约束流形 + RMPflow
    │   ├── ConstraintsSet
    │   │   └── StateConstraint (多个)
    │   └── MultiRobotRMPForest
    │       └── RMPRoot → RMPNode → RMPLeaf (多个)
    │
    ├── CBFFilter                     # Control Barrier Function
    │   └── _solve_cbf_qp()
    │
    └── NoSafetyFilter                # 直通 (基线)
```

### 7.4 RMPflow 结构

```
MultiRobotRMPForest
    │
    └── RMPRoot (联合配置空间 [x1,y1,...,xn,yn])
            │
            ├── Agent_0 (RMPNode)
            │   ├── GoalAttractorUni (RMPLeaf)      # 目标吸引
            │   ├── CollisionAvoidance (RMPLeaf)    # 避障斥力 (×N-1)
            │   ├── FormationDecentralized (RMPLeaf) # 编队保持 (×N-1)
            │   └── Damper (RMPLeaf)                # 速度阻尼
            │
            ├── Agent_1 (RMPNode, 类似结构)
            ├── Agent_2 (RMPNode)
            └── ...
```

---

## 八、配置系统

### 8.1 Hydra 配置结构

```yaml
# cosmos/configs/config.yaml
defaults:
  - env: formation_nav
  - algo: mappo
  - safety: cosmos

experiment:
  name: cosmos_exp
  seed: 42
  num_episodes: 5000
  eval_interval: 100
  save_interval: 500

wandb:
  enabled: false
  project: cosmos
```

### 8.2 环境配置示例

```yaml
# cosmos/configs/env/formation_nav.yaml
name: formation_nav
num_agents: 4
num_obstacles: 4
arena_size: 10.0
formation_shape: square
formation_radius: 1.0
dt: 0.05
max_steps: 500

reward:
  w_nav: 1.0
  w_formation: 0.1
  w_smooth: 0.01
  goal_bonus: 10.0
```

### 8.3 算法配置示例

```yaml
# cosmos/configs/algo/mappo.yaml
name: mappo
actor_lr: 3e-4
critic_lr: 3e-4
clip_param: 0.2
ppo_epochs: 10
num_mini_batch: 4
entropy_coef: 0.01
max_grad_norm: 0.5
gamma: 0.99
gae_lambda: 0.95
```

### 8.4 安全配置示例

```yaml
# cosmos/configs/safety/cosmos.yaml
name: cosmos
safety_radius: 0.5
K_c: 50.0
dq_max: 0.8
eps_damping: 1e-4
slack_type: softcorner
slack_beta: 30.0
use_rmpflow: true
rmp_formation_blend: 0.3
```

### 8.5 命令行覆盖

```bash
# 基本训练
python -m cosmos.train env=formation_nav algo=mappo safety=cosmos

# 参数覆盖
python -m cosmos.train env.num_agents=6 algo.actor_lr=1e-4

# 多运行扫描
python -m cosmos.train -m algo=mappo,qmix env.num_agents=4,6,8

# 使用不同环境
python -m cosmos.train env=vmas algo=maddpg safety=cbf
```

---

## 九、扩展指南

### 9.1 添加新环境

```python
# cosmos/envs/my_env.py
from cosmos.envs.base import BaseMultiAgentEnv
from cosmos.registry import ENV_REGISTRY

@ENV_REGISTRY.register("my_env", aliases=["myenv"])
class MyEnv(BaseMultiAgentEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        # 初始化

    @property
    def num_agents(self) -> int:
        return self.cfg.num_agents

    @property
    def observation_space(self):
        return spaces.Box(...)

    @property
    def action_space(self):
        return spaces.Box(...)

    @property
    def share_observation_space(self):
        return spaces.Box(...)

    def reset(self, seed=None):
        # 返回 (obs, share_obs, info)
        pass

    def step(self, actions):
        # 返回 (obs, share_obs, rewards, costs, dones, infos, truncated)
        pass

    def get_constraint_info(self):
        return {
            "positions": ...,
            "velocities": ...,
            ...
        }
```

```yaml
# cosmos/configs/env/my_env.yaml
name: my_env
num_agents: 4
param1: value1
```

### 9.2 添加新算法

```python
# cosmos/algos/my_algo.py
from cosmos.algos.base import BaseMARLAlgo, OnPolicyAlgo
from cosmos.registry import ALGO_REGISTRY

@ALGO_REGISTRY.register("my_algo")
class MyAlgo(OnPolicyAlgo):
    def __init__(self, obs_dim, share_obs_dim, act_dim, num_agents, cfg):
        # 初始化网络
        pass

    def get_actions(self, obs, deterministic=False):
        # 返回 (actions, log_probs)
        pass

    def get_values(self, share_obs):
        # 返回 values
        pass

    def update(self, buffer):
        # 返回 {"loss": ..., ...}
        pass
```

### 9.3 添加新安全滤波器

```python
# cosmos/safety/my_filter.py
from cosmos.safety.base import BaseSafetyFilter
from cosmos.registry import SAFETY_REGISTRY

@SAFETY_REGISTRY.register("my_filter")
class MyFilter(BaseSafetyFilter):
    def __init__(self, env_cfg, safety_cfg, constraint_info):
        pass

    def reset(self, constraint_info):
        pass

    def project(self, actions, constraint_info, dt=0.05):
        # 返回 safe_actions
        pass
```

---

## 十、性能参考

| 组件 | 指标 | 参考值 |
|------|------|--------|
| formation_nav 训练 | 速度 | ~10,000 steps/sec |
| epuck_sim 训练 | 速度 | ~5,000 steps/sec |
| safety_gym 训练 | 速度 | ~1,000 steps/sec |
| COSMOS 滤波 | 延迟 | < 1 ms |
| CBF 滤波 | 延迟 | < 0.5 ms |
| MAPPO 推理 | 延迟 | < 0.5 ms |
| 碰撞率 (with COSMOS) | 安全性 | 0% |

---

## 十一、参考文献

1. Liu et al., "Robot Reinforcement Learning on the Constraint Manifold", CoRL 2021
2. Liu et al., "Safe Reinforcement Learning on the Constraint Manifold: Theory and Applications", IEEE T-RO 2024
3. Cheng et al., "RMPflow: A Computational Graph for Automatic Motion Policy Generation", WAFR 2018
4. Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022
5. Rashid et al., "QMIX: Monotonic Value Function Factorisation", ICML 2018
6. Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", NeurIPS 2017
7. Ames et al., "Control Barrier Functions: Theory and Applications", ECC 2017
