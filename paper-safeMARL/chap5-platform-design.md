# 第 5 章设计方案（v2 · 仿真融合, 不含实物）

> 更新日期: 2026-04-19
> 核心判断: chap5 = **为多机器人安全 RL 设计扩展的容器化仿真验证平台, 并在其上集成 chap3/chap4 的分层控制栈**
> 范围: 纯仿真, 不做 sim-to-real / 不做 E-puck 实物
> 基础: JdeRobot RoboticsAcademy 本地 fork (`refRoboticAcademy/`)

---

## 1. 章标题建议

**不要**在标题里点 RoboticsAcademy 的名字（那是工具，不是方法）。

推荐候选（按抽象度排序）：

1. **多机器人安全强化学习的容器化仿真验证平台**（推荐）
2. 支持分层安全控制的可复现仿真验证平台设计与实现
3. 面向安全多智能体强化学习的仿真集成与验证框架
4. 分层安全强化学习的容器化仿真验证平台

**选 1 的理由**：直接点出"多机器人 + 安全 RL + 容器化 + 仿真验证"四要素，不提具体工具，方法和平台双重贡献都能承载。

---

## 2. 重新定位: 平台修改是设计贡献, 不是"借了工具"

之前的 v1 稿把 chap5 定位为"扩展现有平台"，这个方向对，但**矫枉过正** — 把 chap5 写得像工程搬运工。

**关键纠正**：

> refRoboticAcademy 是**面向单机教学**的平台，它原本**不支持** `多 robot + safety-RL + 可复现 sweep`。我们对它的**改动本身就是系统级设计贡献**，应当详细陈述。

所以 chap5 要给足以下内容的篇幅:

| 层次 | chap5 内容 | 是否贡献 |
|------|------|--------|
| JdeRobot 原有 Web + Docker + noVNC 框架 | 1-2 段，简介并引用 | ❌ |
| **本文对框架的扩展** | **2-3 节，详细陈述** | ✅ |
| **chap3+chap4 分层控制栈在该框架内的集成** | **3-4 节，核心贡献** | ✅ |
| 功能融合验证（MVP + Webots） | 2-3 节 | ✅ |

---

## 3. 本文对基础平台的具体修改（chap5 的工程创新点）

按模块整理，每项都可在 chap5 §5.2 展开：

### 3.1 多机器人场景层
- **上游缺失**：仅有单机/单无人机练习（follow_line / drone_corridor 等），没有多 robot 场景
- **本文新增**：
  - `cosmos/ros2/worlds/epuck_arena.wbt`：支持 N 个差速 E-puck 的 Webots 场景
  - 静态障碍（Sigwalls）+ 可配置初始位置 + 可配置编队形状（line/wedge/circle/mesh）
  - 跨 exercise 复用的通用场景模板（对应 chap4 实验设定）

### 3.2 控制接口层（ROS2 话题契约）
- **上游**：HAL/GUI 的 Python API 针对单机，API 假设"一个机器人一个 process"
- **本文新增**：多机器人 ROS2 话题命名约定（`/agent_i/odom`, `/agent_i/cmd_vel`, `/formation/state`），并实现 Supervisor 节点做集中调度

### 3.3 安全 RL 算法容器层
- **上游**：Docker 镜像预装 Gazebo/Webots/ROS2, 但**不含** torch / safety-gymnasium / MAPPO 训练环境
- **本文新增**：把 safepo + safety-gymnasium + chap3/chap4 代码整合进容器的 Python 环境，用 `docker-compose` 的 volume 挂载把训练产物 (`runs/`) 暴露给 Supervisor 读取
- 新增的 container env vars: `GCPL_*_OVERRIDE` 系列用于 sweep 模式下的消融

### 3.4 分层控制栈集成层（最核心）
- **上游**：平台没有"策略+安全过滤器+差速映射"的分层概念
- **本文新增**：
  - `MultiAgentRMPAdapter`（cosmos/safety/rmp_multi_agent.py）：把 chap4 RMPCorrector 封装为单进程多 agent 接口
  - `MAPPOPolicyLoader` + `safetygym_obs_mirror`（cosmos/policies/mappo_loader.py）：从 checkpoint 复原 MAPPO 策略，并通过 obs 镜像保证跨平台 obs 一致
  - `gcpl_full_stack_supervisor`（cosmos/ros2/epuck_formation/）：Webots supervisor 节点，按 `MAPPO → chap4 软 → chap3 硬 → 差速` 管道执行

### 3.5 评估与可复现性层
- **上游**：提供单次 exercise 结果展示，无批量统计
- **本文新增**：
  - `run_chap4.sh` 的 `sweep` / `ablation_sweep` / `plot` 模式
  - `scripts/plot_chap4.py` 聚合 seed 出 mean ± std 图
  - Pure-python MVP demo（`cosmos/apps/formation_nav/full_stack_mvp_demo.py`）作为免依赖对比验证

### 3.6 数值稳定性工程补丁（细节）
- `cosmos/safety/constraints.py` 的 softcorner slack NaN 修复（β·s 钳制到 [−20, −1e-6]）
- `algorithms/safe-po/safepo/evaluate.py` CUDA fork `spawn` 修复
- `algorithms/safe-po/safepo/multi_agent/mappo*.py` 多处 cuda/cpu 设备对齐
- 这些属于工程细节，可在 §5.3.4 或附录提及

---

## 4. 建议 chap5 新大纲（12-15 页）

```
第5章 多机器人安全强化学习的容器化仿真验证平台
├── 引言 (0.5 页)
│   安全多机 RL 从单机训练到多机集成验证的工程缺口
│
├── 5.1 平台需求与总体设计 (1.5 页)
│   ├── 5.1.1 对基础平台的功能需求 (多 robot / safety-RL / 可复现 sweep)
│   ├── 5.1.2 分层结构设计: 场景层 / 控制接口层 / 算法容器层 / 集成层 / 评估层
│   └── 5.1.3 本平台与上游教学平台的关系
│
├── 5.2 平台模块扩展 (3-4 页)  ← 工程贡献
│   ├── 5.2.1 多机器人 Webots 场景构建 (E-puck 差速建模 + 可配置编队/通道)
│   ├── 5.2.2 ROS2 多机通信接口设计 (话题命名 / 频率 / 消息类型)
│   ├── 5.2.3 安全 RL 算法容器化 (torch/safepo/safety-gymnasium 集成)
│   └── 5.2.4 批量评估与可视化管线
│
├── 5.3 分层控制栈的平台集成 (3-4 页)  ← 方法贡献
│   ├── 5.3.1 融合架构: MAPPO → chap4 软协调 → chap3 硬投影 → 差速映射
│   ├── 5.3.2 MultiAgentRMPAdapter 设计 (chap4 软层)
│   ├── 5.3.3 MAPPOPolicyLoader + 观测镜像 (RL 策略加载)
│   ├── 5.3.4 Supervisor 实现与数值稳定性处理
│   └── 5.3.5 三个关键融合点 (动作空间 / 约束软硬划分 / 状态共享)
│
├── 5.4 功能验证 · pure-python MVP (2-3 页)
│   ├── 5.4.1 场景与评估指标
│   ├── 5.4.2 三模式对比: mappo_only / rmp_only / fusion
│   │   (fig:chap5_mvp 3-panel 对比图)
│   └── 5.4.3 消融: chap3 硬层对最小安全距离的影响
│
├── 5.5 高保真验证 · Webots (3-4 页)
│   ├── 5.5.1 E-puck 场景配置
│   ├── 5.5.2 单机避障 (chap3 端到端在 Webots 中的行为)
│   ├── 5.5.3 3 机楔形穿通道 (chap4 端到端)
│   ├── 5.5.4 完整分层栈 (fusion) 的安全/效率对比
│   └── 5.5.5 与 Safety-Gymnasium 基准的跨仿真一致性 (tab:sim2sim_consistency)
│
└── 5.6 本章小结 (0.5 页)
    └─ 平台贡献 + 方法贡献总结; 实物部署作为 future work 明示
```

---

## 5. 关键图表清单

### 5.1 §5.1 - §5.3 方法图

| ID | 类型 | 内容 | 状态 |
|----|------|------|------|
| `fig:platform_architecture` | 平台架构图 | 场景/接口/算法/集成/评估 5 层, 标出"本文新增" vs "上游已有" | ⚠ 待绘 (需替换现 `fig:platform_framework`) |
| `fig:deploy_pipeline` | 分层栈管道 | MAPPO→chap4→chap3→差速 3 栏示意 | ⚠ 待绘 (chap5-fusion-design.md §1 已备) |
| `tab:chap4_hard_soft` | 硬软层对照 | 已在 chap4 §4.2 有类似表 | 复用 |
| `tab:interface_alignment` | 动作/约束接口对照 | 说明两层交界的契约 | ⚠ 待写 |

### 5.2 §5.4 MVP 验证

| ID | 类型 | 内容 | 状态 |
|----|------|------|------|
| `fig:chap5_mvp` | 3 模式对比 | mappo_only / rmp_only / fusion 三栏轨迹图 | ✅ 已生成 (`/tmp/chap5_real_policy.png`, 用真实 MAPPO checkpoint) |
| `tab:chap5_mvp_stats` | 多 seed 统计 | 3 模式 × 3 seed × 指标表 (reached/coll/min_dist) | ⚠ 待跑 (脚本可跑 5 seed 出 mean±std) |

### 5.3 §5.5 Webots 验证

| ID | 类型 | 内容 | 状态 |
|----|------|------|------|
| `fig:webots_env` | 场景截图 | E-puck × 3 + Sigwalls + goal | ⚠ 待 Webots 导出 |
| `fig:webots_chap3_single` | 单机 chap3 轨迹 | 约束流形避障效果 | ⚠ 待跑 |
| `fig:webots_chap4_multi` | 3 机 chap4 轨迹 | GCPL 编队穿通道 | ⚠ 待跑 |
| `fig:webots_full_stack` | 完整栈对比 | mappo / rmpflow / gcpl / full-stack 4 档 | ⚠ 待跑 |
| `tab:sim2sim_consistency` | 跨仿真一致性 | Safety-Gym vs Webots 的 4 指标对比 | ⚠ 待汇总 |

---

## 6. 需要补跑/补建的具体工作（按工期排序）

### 🔴 必做 (~3-4 天)

- [ ] 绘 `fig:platform_architecture` 和 `fig:deploy_pipeline` (2h, draw.io / tikz)
- [ ] Pure-python MVP 跑 5 seed 取统计, 出 `tab:chap5_mvp_stats` (~1h 运行)
- [ ] Webots 场景 `.wbt` 文件准备 (sigwalls + 3 epuck + goal; 1 个 session)
- [ ] 在 Webots 里跑 §5.5.2-5.5.5 的 4 组场景 (~1-2 天)
- [ ] 整理 Webots 轨迹图 + 对比表 (~半天)

### 🟠 应做

- [ ] 写 chap5.tex 新正文 (12-15 页, ~1-2 天)
- [ ] 同步更新 chap1 / chap6 (去掉 sim-to-real 关键词)

### 🟢 可做可不做

- [ ] RoboticsAcademy exercise 框架集成 (让 chap5 部署能从 Web 浏览器启动), 可以放附录或 future work

---

## 7. 对陈述的诚实度要求

即便不做实物, chap5 也要诚实。以下原则：

1. **明确承认上游**：第一次提 Docker / noVNC / Web 前端时加一条参考 JdeRobot RoboticsAcademy 的引用
2. **突出本文改动**：每一节都能回答 "这和上游相比多了什么？"
3. **实物明示 future work**：§5.6 本章小结最后一句："本章专注于仿真域的分层控制栈集成与验证；实物 E-puck 部署与 sim-to-real 迁移作为后续工作。" （防止评审误以为做了实物）
4. **方法 vs 工程分隔**：凡是 chap3/chap4 已讲过的方法，chap5 只简述并标引；chap5 重点在"如何让它们在同一平台上协同"

---

## 8. 与 chap3 / chap4 / chap6 的对齐

- **chap3 末尾**: 补 "本章方法在第 5.5.2 节于 Webots 高保真仿真中进一步验证"
- **chap4 末尾**: 补 "第 5.5.3 节给出 Webots 环境下的多机协调效果; 第 5.5.4 节与 chap3 硬安全层集成评估"
- **chap5 引言**: 回顾 chap3+chap4 各自解决的子问题, 引出"需要一个平台把两层拼起来"的动机
- **chap6 总结**: 三章合璧 (chap3 硬安全方法 + chap4 软协调方法 + chap5 平台集成与仿真验证), 实物迁移列在 future work

- **论文题目**: 若原题含 "sim-to-real" 或 "实物验证" 字样, 去掉
- **chap1 研究目标**: 若列了 "实机部署", 改为 "建立可复现的安全多机 RL 仿真验证平台"

---

## 9. v1 草案的什么段落需要删

`chap5-platform-design.md` v1 里下列段落现在作废：

- §3.2 新大纲的 §5.5 E-puck 实物部署子节 → 全删
- §5.2 实物实验矩阵 (RE-1 ~ RE-4) → 全删
- §7 "两级保底方案" 中的 C 理想方案 → 简化
- §8 "对接当前工作" 中关于 chap5 sim-to-real 的段落 → 改为仿真融合
- `chap5-revision-plan.md` 里的 §5.3 E-puck 实物部署 → 全删或标为 future work
- `chap5-fusion-design.md` 里关于实物分布式 agent_node 的段落 → 保留作 future work 附录, 不进主线

---

## 10. 现成资产（已经可以引用进 chap5 的代码 + 图）

| 资产 | 路径 | 用在 |
|------|------|------|
| 平台扩展代码 | `cosmos/safety/rmp_multi_agent.py` | §5.3.2 |
| 平台扩展代码 | `cosmos/policies/mappo_loader.py` | §5.3.3 |
| 平台扩展代码 | `cosmos/ros2/epuck_formation/gcpl_full_stack_supervisor.py` | §5.3.4 |
| 平台扩展代码 | `run_chap4.sh` sweep 模式 | §5.2.4 |
| 平台扩展代码 | `scripts/plot_chap4.py` | §5.2.4 |
| MVP 验证图 | `/tmp/chap5_real_policy.png` (真实 MAPPO) | §5.4.2 |
| MVP 动画 | `/tmp/chap5_fusion.gif` | §5.4 or 答辩 |
| Webots 场景 | `cosmos/ros2/worlds/epuck_arena.wbt` | §5.5.1 |
| 设计文档 | `chap5-fusion-design.md` | 作 §5.3 的完整蓝图 |

---

## 11. 下一步建议 (按优先级)

1. **现在** (几分钟) — 敲定标题: "多机器人安全强化学习的容器化仿真验证平台" 是否 OK？
2. **明天** S1 出结果后, 跑 Webots 5.5 的场景 (~1 天)
3. **后天** 按新大纲重写 chap5.tex + 同步 chap1/chap6

要我现在就按这个新方案把 chap5.tex 的**骨架**（章节 + 段落头 + 占位图表）写出来吗？正文 S1+Webots 数据出来后再填。
