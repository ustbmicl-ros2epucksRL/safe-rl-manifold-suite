# 第 5 章 · refRoboticAcademy 修改方案

> 创建日期: 2026-04-19
> 目标: 在上游 RoboticsAcademy (Webots 版) 基础上构造 `multi_robot_formation` 练习,
>       将 chap3+chap4 分层控制栈接入, 作为 chap5 实验平台
> 范围: **只改 Webots 相关路径** (不动 Gazebo 部分); 不做实物部署
> 参考章节: `chap5.md` §5.2 (平台模块扩展), §5.3 (分层栈集成)

---

## 0. 前提: 上游 follow_line_webots 已有的结构

上游开源版 follow_line_webots 是**单 E-puck 循线**教学练习, 展示了"Webots + ROS2 + HAL/GUI + 浏览器"的完整链路。我们把它作**架构模板**,新增一个 `multi_robot_formation` 练习。

```
exercises/static/exercises/
├── follow_line_webots/         ← 上游参考模板
│   ├── python_template/ros2_humble/
│   │   ├── HAL.py              # 单机 HAL: getGroundSensors / setV / setW
│   │   ├── GUI.py              # 图像推送 (cv_bridge + /gui_image topic)
│   │   ├── WebGUI.py           # 自定义 WebGUI 视图
│   │   └── aerial_view_controller.py   # 俯视相机 extern controller
│   ├── webots_projects/
│   │   ├── worlds/follow_line_indoor.wbt
│   │   └── controllers/        # (空的; controller 由 extern 提供)
│   ├── react-components/       # 前端自定义 React
│   ├── resources/              # 练习页面描述/图片
│   └── scripts/
└── multi_robot_formation/      ← 本文要新增的目录 (5 个子目录复刻结构)
```

上游的 Webots 容器镜像 `jderobot/robotics-webots:humble`
通过 `docker-compose-webots.yaml` 与主 `robotics-academy` 容器组网, 两者通过 **ROS2 DDS (ROS_DOMAIN_ID=0)** 跨容器通信。

---

## 1. 修改总览 (按 chap5.md 五层架构对应)

| 层 | 改什么 | 新增 / 改动文件 | 工作量 |
|---|---|---|---|
| L1 场景层 | 新增 3 机 E-puck + sigwalls + goal 的 .wbt 文件 | `worlds/epuck_formation.wbt` | ~100 行 .wbt |
| L2 接口层 | 多机 ROS2 话题: `/agent_i/odom`, `/cmd_vel`, `/goal` | 修 `HAL.py` 为 multi-agent 版 | ~150 行 Python |
| L3 容器层 | 给 Webots 容器装 torch/safepo/cosmos, 共享 `runs/` | 改 `Dockerfile.webots` + `docker-compose-webots.yaml` | ~30 行 |
| L4 集成层 | Webots supervisor 运行 MAPPO + chap4 + chap3 | 复用 `gcpl_full_stack_supervisor.py` + 对接 | ~80 行 wrapper |
| L5 评估层 | 容器内批量实验 + 产出 chap5 图表 | `scripts/run_chap5_exp.sh` | ~60 行 shell |
| 前端(选) | React 组件展示编队误差时序 | `react-components/FormationViz.jsx` | 跳过也行 |

**总代码量 ~400-500 行**, 绝大部分是模板化改造。

---

## 2. Layer 1 · 新增 Webots 场景

### 2.1 新增文件

`exercises/static/exercises/multi_robot_formation/webots_projects/worlds/epuck_formation.wbt`

### 2.2 场景内容 (与 chap4 §4.6.1 对齐)

```
WorldInfo { basicTimeStep 32 }
# Arena: 3m × 3m 平面
# Sigwalls: (-1.0, 0) 和 (1.0, 0) 位置两个静态墙
# Goal: (0, 2.0) 位置放置一个可见的 marker (红色圆盘)

DEF EPUCK_0 E-puck {
  translation  0.0  -0.75  0
  rotation     0 0 1  1.5708
  name         "epuck_0"
  controller   "<extern>"   # 由 Supervisor 集中调度
  supervisor   TRUE
}
DEF EPUCK_1 E-puck { ... translation -0.35 -0.95 0 ... }
DEF EPUCK_2 E-puck { ... translation  0.35 -0.95 0 ... }

# Sigwalls (实心方块, 用作障碍)
DEF SIGWALL_L Solid { translation -1.0 0 0.1 ... }
DEF SIGWALL_R Solid { translation  1.0 0 0.1 ... }

# Goal marker (视觉参考, 不是物理障碍)
DEF GOAL Solid { translation 0 2.0 0.01 ... }

# 全局 supervisor 节点, 用 extern 控制器
DEF GCPL_SUPERVISOR Robot {
  name "gcpl_supervisor"
  controller "<extern>"
  supervisor TRUE
  children [
    Emitter { name "cmd_emitter"  channel 1 }
  ]
}
```

### 2.3 配套 slave 控制器

每个 E-puck 也挂一个 `<extern>` controller, 读 supervisor 广播的 cmd, 设置轮速:

```python
# webots_projects/controllers/epuck_slave/epuck_slave.py
# 接 receiver, 拿 "i v_L v_R" 字符串, 设 motors
```

或者更简单的路线: 让 supervisor 直接通过 Supervisor API 写入 E-puck 的 `setVelocity()`, 完全省掉 receiver 通信。缺点是 slave 机器人们必须由 supervisor 用 DEF 引用, 灵活性差一点。

**推荐**: Supervisor 用 ROS2 话题 `/agent_i/cmd_vel` 下发, slave 订阅 ROS2 话题转 motor, 与上游 `follow_line_webots` 的模式一致。这样 L2 接口层可以复用 ROS2 机制。

---

## 3. Layer 2 · ROS2 多机接口改造

### 3.1 上游 follow_line_webots 的 HAL 是什么样

```python
# 上游 HAL.py (摘)
class HAL_Node(Node):
    def __init__(self):
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub_gs = self.create_subscription(Range, '/gs0', self.gs0_cb, 10)
        # 单机假设: 只有一个 /cmd_vel, 只有一个 /gs*

def setV(v): hal.pub_cmd_vel(Twist(linear=Vector3(x=v)))
def setW(w): hal.pub_cmd_vel(Twist(angular=Vector3(z=w)))
```

### 3.2 多机版本设计

新建 `exercises/static/exercises/multi_robot_formation/python_template/ros2_humble/HAL.py`:

```python
class MultiAgentHAL:
    """多机 HAL. 话题命名 /agent_0/cmd_vel, /agent_0/odom, ..."""

    def __init__(self, num_agents=3):
        self.N = num_agents
        rclpy.init()
        self.node = Node("multi_agent_hal")
        self.cmd_pubs = [
            self.node.create_publisher(Twist, f'/agent_{i}/cmd_vel', 10)
            for i in range(num_agents)
        ]
        self.odom_subs = [...]
        self.agent_poses = np.zeros((num_agents, 3))   # x, y, yaw
        self.agent_vels  = np.zeros((num_agents, 2))   # vx, vy

    def setCmdVel(self, i, v, w):
        """下发差速指令给 agent i."""
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.cmd_pubs[i].publish(msg)

    def getAllPoses(self) -> np.ndarray:
        """返回 (N, 3) 的 pose 数组."""
        rclpy.spin_once(self.node, timeout_sec=0.0)
        return self.agent_poses.copy()

    def getAllVels(self) -> np.ndarray:
        return self.agent_vels.copy()

    def getGoal(self) -> np.ndarray:
        return np.array([0.0, 2.0])   # 硬编码场景 goal
```

用户代码示例 (对标 follow_line 的教学模式):

```python
from HAL import MultiAgentHAL

hal = MultiAgentHAL(num_agents=3)
while True:
    poses = hal.getAllPoses()     # (3, 3)
    vels = hal.getAllVels()       # (3, 2)
    # 自定义控制逻辑 ...
    for i in range(3):
        hal.setCmdVel(i, v=0.1, w=0.0)
```

### 3.3 额外: Supervisor 模式

对 chap5 主实验, 我们**不让用户写控制逻辑**, 而是让上面的 `gcpl_full_stack_supervisor.py` 作为 extern controller 直接跑, 绕过 HAL:

```
webots-container 内:
  gcpl_full_stack_supervisor.py 作为 Webots extern controller
    ↓
  读取 DEF EPUCK_0/1/2 的 Supervisor 状态
  ↓
  ROS2 发布 /agent_i/cmd_vel
    ↓
  E-puck slave controller 订阅并转 motor
```

HAL 层在这里主要是**教学/调试用**, chap5 主实验不依赖。

---

## 4. Layer 3 · Docker 容器依赖扩展

### 4.1 当前 `scripts/RADI/webots/Dockerfile.webots`

上游镜像 `jderobot/robotics-webots:humble` 预装:
- Webots R2023b
- ROS2 Humble
- Python 3.10 + rclpy

**缺失**: PyTorch, safety-gymnasium, safepo, cosmos (本文代码)

### 4.2 扩展方案

**方案 A (推荐, 最小改动)**: 用 `pip install` 在 entrypoint 里按需安装

在 `docker-compose-webots.yaml` 的 `webots:` 服务里加:

```yaml
webots:
  image: jderobot/robotics-webots:humble
  volumes:
    # ...已有...
    # 本文新增: 挂载 safe-rl-manifold-suite 源码进容器
    - ../safe-rl-manifold-suite:/workspace/safe-rl-manifold-suite:ro
    # 挂载训练产出, 让 supervisor 能读 MAPPO checkpoint
    - ../runs:/workspace/runs:ro
  environment:
    - PYTHONPATH=/workspace/safe-rl-manifold-suite:/workspace/safe-rl-manifold-suite/algorithms/multi-robot-rmpflow
    - MULTI_ROBOT_RMPFLOW_PATH=/workspace/safe-rl-manifold-suite/algorithms/multi-robot-rmpflow
```

在 `start_webots_vnc.sh` (或同级 entrypoint) 启动前加:

```bash
pip install --quiet torch>=1.10 --index-url https://download.pytorch.org/whl/cpu
pip install --quiet -e /workspace/safe-rl-manifold-suite/envs/safety-gymnasium
pip install --quiet -e /workspace/safe-rl-manifold-suite/algorithms/safe-po --no-deps
```

这样**不需要重新 build 镜像**, 每次启动时自动把依赖装好, 对容器缓存友好。

**方案 B (更规范)**: 构建新镜像 `ustbmicl/robotics-webots-chap5:latest`

把上面的 pip install 步骤直接写进 `Dockerfile.webots` 里 `FROM jderobot/robotics-webots:humble` 之后, 构建一次就缓存了, 启动更快。适合发论文后做 Docker Hub 发布。

### 4.3 volume 挂载: 让 Supervisor 读 checkpoint

```yaml
volumes:
  - ../runs:/workspace/runs:ro   # chap4 训练产物 (read-only)
  - ../safe-rl-manifold-suite:/workspace/safe-rl-manifold-suite:ro
```

然后 Supervisor 代码里:

```python
from cosmos.policies.mappo_loader import MAPPOPolicyLoader
loader = MAPPOPolicyLoader.latest(
    "/workspace/runs/Base/SafetyPointMultiFormationGoal0-v0/mappo_rmp"
)
```

---

## 5. Layer 4 · 分层栈 Supervisor 集成

### 5.1 Webots extern controller 注册

在 `.wbt` 中 supervisor 节点的 `controller` 字段设为 `"<extern>"`:

```
DEF GCPL_SUPERVISOR Robot {
  name "gcpl_supervisor"
  controller "<extern>"
  supervisor TRUE
}
```

然后在容器内用 `webots-controller` CLI 启动 supervisor:

```bash
WEBOTS_CONTROLLER_URL=ipc://webots-container:1234/gcpl_supervisor \
  python3 /workspace/safe-rl-manifold-suite/cosmos/ros2/epuck_formation/gcpl_full_stack_supervisor.py
```

这里 `ipc://` URL 由 Webots 容器内部分配。

### 5.2 启动流程脚本

新增 `exercises/static/exercises/multi_robot_formation/scripts/launch_chap5.sh`:

```bash
#!/bin/bash
# 1. 启动 Webots 场景
webots --stdout --stderr --batch --mode=fast \
  /workspace/safe-rl-manifold-suite/cosmos/ros2/worlds/epuck_formation.wbt &
WEBOTS_PID=$!
sleep 5

# 2. 启动 ROS2 → Webots 桥接 (若需要)
# webots_ros2_driver 已由上游镜像启动; 这里只起 supervisor

# 3. 启动 GCPL full-stack supervisor
WEBOTS_CONTROLLER_URL=ipc://webots-container:1234/gcpl_supervisor \
  python3 /workspace/safe-rl-manifold-suite/cosmos/ros2/epuck_formation/gcpl_full_stack_supervisor.py

wait $WEBOTS_PID
```

### 5.3 参数传递 (env var 全部沿用 chap4)

```bash
# 在 docker-compose-webots.yaml 或 launch_chap5.sh 里:
GCPL_NUM_AGENTS=3
GCPL_SHAPE=wedge
GCPL_DIST=0.5
GCPL_SAFETY_R=0.3
GCPL_HARD_SAFETY_R=0.25
GCPL_RL_WEIGHT=10.0
GCPL_ENABLE_ATACOM=1
GCPL_MAPPO_CKPT_DIR=/workspace/runs/Base/SafetyPointMultiFormationGoal0-v0/mappo_rmp
```

---

## 6. Layer 5 · 批量实验脚本

### 6.1 容器内运行 chap5 实验

`scripts/run_chap5_exp.sh`:

```bash
#!/bin/bash
# 本脚本在 webots-container 内运行, 跑 4 组对比实验
set -e

SUPERVISOR_PY=/workspace/safe-rl-manifold-suite/cosmos/ros2/epuck_formation/gcpl_full_stack_supervisor.py
OUT_DIR=/workspace/runs/chap5_webots

for mode in chap3_only chap4_only gcpl_full mappo_only; do
    for seed in 0 1 2; do
        case $mode in
            chap3_only)
                GCPL_RL_WEIGHT=0 GCPL_ENABLE_ATACOM=1 ;;
            chap4_only)
                GCPL_RL_WEIGHT=10 GCPL_ENABLE_ATACOM=0 ;;
            gcpl_full)
                GCPL_RL_WEIGHT=10 GCPL_ENABLE_ATACOM=1 ;;
            mappo_only)
                GCPL_RL_WEIGHT=10 GCPL_ENABLE_ATACOM=0
                GCPL_USE_FORMATION_LEAF=false
                GCPL_USE_ORIENTATION_LEAF=false
                GCPL_USE_COLLISION_LEAF=false ;;
        esac
        # webots --batch 模式跑一次, 结果写 $OUT_DIR/$mode/seed_$seed.json
        python3 $SUPERVISOR_PY > $OUT_DIR/$mode/seed_$seed.log
    done
done

# 聚合出 §5.5 的 tab 和 fig
python3 /workspace/safe-rl-manifold-suite/scripts/plot_chap5.py \
    --runs $OUT_DIR --out /workspace/runs/chap5_webots/figs
```

### 6.2 新增的 `scripts/plot_chap5.py`

与 `plot_chap4.py` 同结构, 但:
- 读的是 Webots 生成的 json (而非 safepo 的 progress.csv)
- 产出: `fig:webots_full_stack.png`、`tab:sim2sim_consistency`

---

## 7. 前端 (可选, 跳过也能写论文)

### 7.1 上游 noVNC 已能展示 Webots 原生窗口

这对 chap5 实验录屏/截图已够用。

### 7.2 若要加自定义 WebGUI

模仿 follow_line_webots 的 `react-components/`, 写一个:
- `FormationErrorPlot.jsx` — 实时画 $\bar e(t)$
- `TrajectoryOverlay.jsx` — 叠加 3 机轨迹

但这只影响教学演示效果, chap5 实验**可以不做**这一步。

---

## 8. 改动规模汇总

| 新增/改动文件 | 路径 | 行数估计 | 性质 |
|---|---|---|---|
| `.wbt` 场景 | `exercises/static/exercises/multi_robot_formation/webots_projects/worlds/epuck_formation.wbt` | 100 | 新增 |
| slave controller | `webots_projects/controllers/epuck_slave/epuck_slave.py` | 60 | 新增 |
| 多机 HAL | `python_template/ros2_humble/HAL.py` | 150 | 新增 (拷贝 follow_line 改造) |
| launch 脚本 | `scripts/launch_chap5.sh` | 30 | 新增 |
| 实验脚本 | `scripts/run_chap5_exp.sh` | 60 | 新增 |
| Dockerfile 扩展 (方案 A) | `docker-compose-webots.yaml` volume/env | 20 | 改动 |
| Dockerfile 扩展 (方案 B) | `scripts/RADI/webots/Dockerfile.webots` | 20 | 改动 |
| plot_chap5.py | `scripts/plot_chap5.py` (参考 plot_chap4.py) | 200 | 新增 |
| **总计** | | **~640 行** | |

**纯配置/模板化代码居多, 不需要重写框架**。

---

## 9. 相对上游的改动记录 (chap5 论文里需要明示的清单)

以下所有路径**相对 `refRoboticAcademy/` 根目录**:

### 9.1 全新增加 (属于本文贡献)

- `exercises/static/exercises/multi_robot_formation/` (整个目录是新增的)
  - `webots_projects/worlds/epuck_formation.wbt` ← chap4 §4.6.1 实验场景的 Webots 实现
  - `webots_projects/controllers/epuck_slave/` ← 轮速订阅器
  - `python_template/ros2_humble/HAL.py` ← 多机 HAL
  - `scripts/launch_chap5.sh` ← 启动入口
  - `scripts/run_chap5_exp.sh` ← 批量实验
  - `scripts/plot_chap5.py` ← 出图脚本
  - `resources/description.md` ← 练习描述 (教学用)
- `cosmos/ros2/epuck_formation/gcpl_supervisor.py` (本文已写)
- `cosmos/ros2/epuck_formation/gcpl_full_stack_supervisor.py` (本文已写)
- `cosmos/safety/rmp_multi_agent.py` (本文已写)
- `cosmos/policies/mappo_loader.py` (本文已写)

### 9.2 修改上游文件 (记录差异即可, 不是重写)

- `docker-compose-webots.yaml`
  - 加 volume: `../safe-rl-manifold-suite:/workspace/safe-rl-manifold-suite:ro`
  - 加 volume: `../runs:/workspace/runs:ro`
  - 加 env: `MULTI_ROBOT_RMPFLOW_PATH`, `PYTHONPATH`
- `scripts/RADI/webots/Dockerfile.webots` (可选, 若走方案 B)
  - 追加 `pip install torch ...`
- `scripts/RADI/webots/start_webots_vnc.sh` (可选, 若走方案 A)
  - 启动前加 pip install

### 9.3 **不动**的上游文件

- Django 后端 (academy/)
- React 前端 (react_frontend/)
- PostgreSQL schema
- Gazebo 相关 (Dockerfile.gazebo, follow_line_gazebo)
- RoboticsApplicationManager (src/RoboticsApplicationManager/)

这些都是上游既有能力, 我们直接复用。

---

## 10. 风险与替代方案

| 风险 | 缓解 |
|------|------|
| 多 extern controller 并行在 Webots 里偶发阻塞 (上游已知, 见 `modified/` 下的文档) | 优先让 supervisor 用 Supervisor API 直接设 E-puck motor, 避免多 extern 竞争 |
| Webots 容器的 Python 版本可能与宿主训练环境不一致 | 启动前 `pip install` 兼容 3.10+; 若冲突则走方案 B 构建新镜像 |
| ROS2 跨容器 DDS 偶尔不同步 | `ROS_DOMAIN_ID` 统一, 且加 retry 机制 |
| `webots --mode=fast` 下无图像输出 | 对 chap5 批量实验这是好事 (无头快跑); 要截图时改 `--mode=realtime` |
| volume 挂载在某些 Docker 版本需要 absolute path | 启动脚本里用 `$(realpath ...)` |

---

## 11. 渐进式实施顺序

### 11.1 Day 1 (~半天, 跑通单 agent)
- [ ] 复制 `follow_line_webots` 结构到 `multi_robot_formation`
- [ ] 写 `epuck_formation.wbt` (先放 1 个 E-puck, 验证容器启动正常)
- [ ] 改 docker-compose 的 volume + env, 确保 `import cosmos` 能在容器内成功

### 11.2 Day 2 (~一天, 跑通 3 机)
- [ ] 在 .wbt 加到 3 个 E-puck + sigwalls + goal marker
- [ ] 写 slave controller, 验证 ROS2 `/agent_i/cmd_vel` 能驱动每机轮速
- [ ] 写 launch_chap5.sh, 启动 supervisor + 看 ROS2 `ros2 topic echo /agent_0/cmd_vel` 有输出

### 11.3 Day 3 (~一天, chap5 主实验)
- [ ] 跑 4 档对比 (chap3_only / chap4_only / gcpl_full / mappo_only)
- [ ] 写 run_chap5_exp.sh 批量执行
- [ ] 用 plot_chap5.py 出 §5.5 的表格和图

### 11.4 Day 4 (可选, 前端)
- [ ] 写 React 组件展示编队误差时序

### 11.5 Day 5 (文档)
- [ ] chap5.tex 写作, 引用上面所有图表
- [ ] chap1 / chap6 同步去 sim-to-real 字样

---

## 12. 对接 chap5.md 的正文

本文件产生的代码改动和配置, 对应 `chap5.md` 的:

| chap5.md 节 | 本方案对应 |
|---|---|
| §5.2.1 多机器人 Webots 场景构建 | §2 本文件 L1 |
| §5.2.2 ROS2 多机通信接口设计 | §3 本文件 L2 |
| §5.2.3 安全 RL 算法容器化 | §4 本文件 L3 |
| §5.2.4 批量评估与可视化管线 | §6 本文件 L5 |
| §5.3 分层控制栈的平台集成 | §5 本文件 L4 (已有代码) |
| §5.5 Webots 高保真验证 | §11 本文件 Day 3 实验 |

---

## 13. 下一步

按实施顺序开始 Day 1:

1. `mkdir -p refRoboticAcademy/exercises/static/exercises/multi_robot_formation/{webots_projects/{worlds,controllers/epuck_slave},python_template/ros2_humble,scripts,resources}`
2. 参考 `follow_line_webots` 复制必要的骨架 (但去掉循线专属代码)
3. 先写 `epuck_formation.wbt` 单 E-puck 版, 验证容器启动
4. 调 `docker-compose-webots.yaml` 加 volume + env, 验证 `import cosmos`
5. 迭代到 3 机器人

要我现在就开始 Day 1 的代码实施吗 (新建目录 + 写 .wbt + docker compose patch)？
