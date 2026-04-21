# 第 4 章 研究问题 (RQ) 与交付物清单

> 最后更新: 2026-04-19
> 对应论文: `contents/chap4.tex` (GCPL = MAPPO + RMPflow + RL 叶节点)
> 依赖文档: `chap4-structure-revision.md`, `chap4-sweep-guide.md`, `chap4-progress.md`

---

## 0. 定位回顾

第 4 章在第 3 章**硬安全层**基础上构建**多机软协调层**。论证分两层：
- 方法层：几何叶节点 (距离/方向) + RL 叶节点 在 RMPflow 计算图中统一合成
- 实验层：对比 + 消融回答 RQ

本章实验**不**验证硬安全（那是 chap5 的职责），只验证**软协调**在编队导航场景下的效果。

---

## 1. 研究问题 (RQ)

### 🎯 RQ1 · 对比：GCPL 能否在编队导航任务上同时兼顾性能与软安全？

> 与 MAPPO（纯 RL）、MAPPO_Lag（拉格朗日安全 RL）、RMPflow（纯几何）四类基线对比累计回报、安全代价、编队完成度、任务成功率。

**期望结论**：GCPL 在 Reward、Cost、Formation Completion、Success Rate **四个指标上同时占优**，Pareto 前沿显著优于基线。

### 🎯 RQ2 · 消融：GCPL 的各叶节点各自贡献多少？

> 拆开 GCPL，评估：
> - 基础 MAPPO
> - MAPPO + 距离叶（仅编队尺度约束）
> - MAPPO + 距离叶 + 方向叶（距离 + 方向刚性）
> - MAPPO + 距离叶 + 方向叶 + 避碰叶（完整 GCPL）

**期望结论**：方向叶的引入能显著减少编队旋转/剪切（反映在编队完成度 FC↑），避碰叶显著降低 cost。这里的"方向叶"正是本文在 Multi-Robot RMPflow 之上的原创贡献。

### 🎯 RQ3 · 融合机制：RL-as-leaf (fusion_mode=leaf) 是否优于线性加权 (additive)？

> 在同一 MAPPO + 几何叶设置下，比较：
> - `fusion_mode=leaf` (路径 A, 本章核心)
> - `fusion_mode=additive` (回退: a_final = a_RL + rmp_weight × a_RMP)

**期望结论**：leaf 模式让 RL 先验在共同黎曼度量下被加权，策略意图与几何先验能**协商**而非单向覆盖；additive 模式在几何力较大时会被覆盖，Pareto 前沿更差。

### 🟡 RQ4 · 泛化：GCPL 能否迁移到不同编队形状？

> 在 line / wedge / circle / mesh 四种形状上分别运行 GCPL，并与 MAPPO 对比。代码已支持，论文目前降级为直线代表。

**期望结论**：GCPL 在所有四种形状上都优于 MAPPO；相对优势在"形状刚性更强"的 wedge / circle 上更大。

### 🟡 RQ5 · (可选) 扩展：GCPL 对机器人数量 N 是否稳健？

> 在 N=3/5/7 上比较 GCPL vs MAPPO 的 Pareto。

**期望结论**：GCPL 的相对优势随 N 增大而扩大（多机任务的协调需求更强）。

---

## 2. 交付物总表

### 图表与 RQ 的对应关系

| 图/表 ID | 类型 | 支撑 RQ | 当前状态 | 数据来源 |
|---------|------|-------|---------|---------|
| `fig:RMPflow framework` | 算法框架图 | 方法介绍 | ✅ 已有 | 手工绘制 |
| `fig:RL-RMPflow树` | 任务树示意 | 方法介绍 | ✅ 已有 | 手工绘制 |
| `fig:hard_soft_interface` | chap3–chap4 对照图 | §4.2 引入 | ⚠ 待绘 | 手工（或用 `tab:chap4-hard-soft` 代替） |
| `fig:rmp_env` | 实验场景图 | 所有 RQ | ✅ 已有 | 手工/env 截图 |
| `fig:train_result` | 训练曲线 (reward+cost) | **RQ1** | ⚠ 待跑 | sweep 产物: `images/chap4_sweep/training_reward.png`, `training_cost.png` |
| `fig:eval_bar` | 4 算法 × 4 指标 | **RQ1** | ⚠ 待跑 | `images/chap4_sweep/eval_bar.png` |
| `fig:cost_reward_pareto` (**建议新增**) | Pareto 散点 | **RQ1** | ⚠ 待跑 | `images/chap4_sweep/cost_reward_pareto.png` |
| `fig:rmp_result` | 消融训练曲线 | **RQ2** | ⚠ 待跑 | ablation sweep 产物 |
| `fig:ablation_bar` (**建议新增**) | 4 消融档 × 4 指标 | **RQ2** | ⚠ 待跑 | ablation sweep 产物 |
| `fig:shape_bar` (**建议新增**) | 4 形状 × 4 算法 | **RQ4** | 可选 | 扩展 sweep |
| `fig:fusion_pareto` (**建议新增**) | leaf vs additive | **RQ3** | 可选 | 小规模 fusion sweep |
| `fig:trajectory` (**锦上添花**) | 轨迹叠加图 | **RQ1/RQ2** | ⚠ 需扩测 rollout | 额外 eval rollout |
| `tab:chap4-hard-soft` | chap3–chap4 对照表 | §4.2 | ✅ 已写 | `contents/chap4.tex` |
| `tab:GCPL_params` | 训练 + GCPL 超参表 | 复现性 | ✅ 已写 | 手工 |
| `tab:line_formation_result` | 4 算法 4 指标表 | **RQ1** | ⚠ 待填真实数字 | `runs/Base/eval_result.txt` |
| `tab:ablation_result` (**建议新增**) | 消融 4 档 4 指标 | **RQ2** | ⚠ 待跑 | ablation eval |
| `tab:shape_result` (**可选**) | 4 形状 × GCPL/MAPPO | **RQ4** | 可选 | 扩展 sweep |

图例说明：✅ 已完成；⚠ 待跑或待绘；**建议新增** = 论文尚未占位但会显著加分；**锦上添花** = 非必须。

---

## 3. 每项交付的具体 Spec

### 3.1 RQ1 对比实验交付

#### `fig:train_result` — 训练曲线
- 布局：一行两子图 (左 reward, 右 cost)
- 每子图 4 条线 × 阴影带（多 seed std）
- 横轴：训练步数（0 → 500k for medium）
- 图例：MAPPO / MAPPO_Lag / RMPflow / GCPL (ours)
- 生成：`scripts/plot_chap4.py` 自动产出 `training_reward.png` + `training_cost.png`

#### `fig:eval_bar` — 评估柱状图
- 布局：一行 3 子图 (reward / cost / success_rate)
- 每子图 4 柱 × 误差棒 (seed std)
- 生成：`scripts/plot_chap4.py` 自动产出 `eval_bar.png`

#### `fig:cost_reward_pareto` — Pareto 散点（**建议新增**，答辩加分）
- 布局：单图，横轴 cost，纵轴 reward
- 每点一个 (算法, seed) 组合，共 4×3=12 点
- 配色：GCPL 蓝，RMPflow 绿，MAPPO 红，MAPPO_Lag 橙
- 视觉结论："左上角 = 最优"，若 GCPL 簇位于 Pareto 前沿
- 生成：`scripts/plot_chap4.py` 自动产出 `cost_reward_pareto.png`

#### `tab:line_formation_result` — 对比实验表
列：Algorithm / Reward / Cost / Formation_Completion (%) / Success_Rate (%)
行：4 算法（GCPL 粗体）
数据格式：`mean ± std`，从 `runs/Base/eval_result.txt` 的 `SUMMARY` 行聚合

---

### 3.2 RQ2 消融实验交付

#### Ablation 配置（需要额外 sweep）

| 档位 | 几何叶 | 说明 |
|-----|-------|------|
| A | 全关 | 裸 MAPPO |
| B | 距离叶 | MAPPO + 距离保持 |
| C | 距离 + 方向 | MAPPO + 距离 + 方向（本文新增的方向叶） |
| D | 距离 + 方向 + 避碰 | 完整 GCPL |

代码侧实现：通过 `rmp_corrector` 的 `use_rmp` + 按叶节点开关（需扩展 `config` 里加 `use_formation_leaf`/`use_orientation_leaf`/`use_collision_leaf` 三个字段，或写 4 个 config 变体）。

#### `fig:rmp_result` — 消融训练曲线
- 一行两子图 (reward + cost)
- 4 条线 × 阴影带

#### `fig:ablation_bar`（**建议新增**）
- 4 档 × 4 指标柱状图

#### `tab:ablation_result`（**建议新增**）
- 4 档 × 4 指标表

---

### 3.3 RQ3 融合机制（leaf vs additive）

- 固定 GCPL + line formation + 3 seed
- 两个 config：`fusion_mode=leaf` vs `fusion_mode=additive`
- 交付 1 张 Pareto 散点（`fig:fusion_pareto`）
- 交付 1 小段正文说明
- **可并入** `fig:cost_reward_pareto` 作为额外两簇点，不必单独成图

---

### 3.4 RQ4 形状泛化（可选）

- 4 形状 × {MAPPO, GCPL} × 3 seed = 24 runs
- 交付：柱状图 `fig:shape_bar`（横轴 4 形状分组、两柱 MAPPO vs GCPL），一张表 `tab:shape_result`
- 若资源紧：只跑 wedge + circle，写进附录

---

### 3.5 RQ5 scale（可选，如有精力）

- 3 种 N × {MAPPO, GCPL} × 3 seed = 18 runs
- 交付：一张双轴柱图 + 简短段落

---

## 4. 实验矩阵（跑什么、顺序、成本）

### Must-do 最小闭环（覆盖 RQ1 + RQ2）

| Sweep | 内容 | config | runs 数 | 单 run | 总时长 |
|------|------|--------|--------|--------|-------|
| S1: 对比 | 4 算法 × 3 seed × line | medium | 12 | ~30-60 min | **6-12 h** |
| S2: 消融 | 4 档 × 3 seed × line | medium | 12 | ~30-60 min | **6-12 h** |
| **合计** | | | 24 | | **12-24 h CPU+GPU** |

单卡 4090L 预计 12-16 h 可以跑完全部 24 个 run（不并行）。

### Nice-to-have（RQ3-5）

| Sweep | 内容 | 额外 runs | 额外时长 |
|------|------|---------|---------|
| S3: fusion 消融 | 2 fusion × 3 seed | 6 | ~3-6 h |
| S4: 形状泛化 | 4 形状 × 2 算法 × 3 seed | 24 | ~12-24 h |
| S5: N 扩展 | 3 N × 2 算法 × 3 seed | 18 | ~9-18 h |

### 数据落盘

- 训练产物：`runs/Base/SafetyPointMultiFormationGoal0-v0/<algo>/seed-*/`
- 评估汇总：`runs/Base/eval_result.txt`
- 图表输出：`paper-safeMARL/images/chap4_sweep/`

---

## 5. 论文中位置映射

chap4.tex 4.6 实验节的结构已经占位好，现在要做的是**填数字 + 替换图**：

```
4.6 实验
├── 4.6.1 任务与环境            → fig:rmp_env (已有)
├── 4.6.2 评价指标              → (无图)
├── 4.6.3 参数设置              → tab:GCPL_params (已有, 可能需更新)
└── 4.6.4 实验结果
    ├── 4.6.4.1 对比实验
    │   ├── fig:train_result    ← S1 产出
    │   ├── fig:eval_bar        ← S1 产出 (新增)
    │   ├── fig:cost_reward_pareto ← S1 产出 (建议新增)
    │   └── tab:line_formation_result ← S1 产出
    └── 4.6.4.2 消融实验
        ├── fig:rmp_result      ← S2 产出
        ├── fig:ablation_bar    ← S2 产出 (建议新增)
        └── tab:ablation_result ← S2 产出 (建议新增)
```

---

## 6. 当前阻塞项 & 近期 TODO

### 🔴 阻塞 RQ1 的事

- [ ] 跑 S1 (medium × 4 算法 × 3 seed) - 12 runs - 约 6-12 h
- [ ] 跑完后用 `scripts/plot_chap4.py` 出图, 替换 `fig:train_result`
- [ ] 用 `runs/Base/eval_result.txt` 的数字填 `tab:line_formation_result`

### 🔴 阻塞 RQ2 的事

- [x] 在 `rmp_corrector` 里加 3 个叶节点独立开关 (`use_formation_leaf` / `use_orientation_leaf` / `use_collision_leaf`) — **完成 2026-04-19**
- [x] 扩展 `run_chap4.sh` 加 `ablation_sweep` 模式，定义 A/B/C/D 4 档 config — **完成 2026-04-19**
- [x] `evaluate.py` / `plot_chap4.py` 识别 `abl_*` 目录 — **完成 2026-04-19**
- [ ] 跑 S2 (12 runs, ~6-12 h)
- [ ] 消融结果填 `tab:ablation_result`、替换 `fig:rmp_result`

S2 命令（等 S1 完成后）:
```bash
CHAP4_MODE=ablation_sweep CHAP4_SCALE=medium CHAP4_DEVICE=cuda \
  CHAP4_SKIP_INSTALL=1 CHAP4_SKIP_SANITY=1 \
  CHAP4_SEEDS="0 1 2" \
  CHAP4_PLOT_OUT=images/chap4_ablation \
  bash run_chap4.sh
```

4 档 config 映射:
| 档位 | 几何叶 | 入口 | 产出目录 |
|-----|------|------|---------|
| A | 无 | `mappo.py` | `runs/.../abl_A/` |
| B | 距离叶 | `mappo_rmp.py` + `CHAP4_USE_ORIENTATION_LEAF=false` + `CHAP4_USE_COLLISION_LEAF=false` | `runs/.../abl_B/` |
| C | 距离 + 方向叶 | `mappo_rmp.py` + `CHAP4_USE_COLLISION_LEAF=false` | `runs/.../abl_C/` |
| D | 全开（完整 GCPL） | `mappo_rmp.py` 默认 | `runs/.../abl_D/` |

### 🟠 提升工作（可选）

- [ ] RQ3: fusion_mode 消融 (S3, ~3-6 h)
- [ ] RQ4: 形状泛化 (S4, ~12-24 h)
- [ ] 轨迹可视化 (`fig:trajectory`): 改 eval 保存 (p_x, p_y, t) 时序
- [ ] chap3-chap4 接口示意图 (`fig:hard_soft_interface`)

### 🟢 收尾

- [ ] 论文正文对齐新数字 (去掉当前 table 里的占位数字)
- [ ] 消融小节补写（目前 §4.6.4.2 是简化版）
- [ ] 若保留 wedge/circle 的论证：正文 §4.6.1 改"以直线编队为代表" 为"覆盖四种形状"

---

## 7. 风险 & 备选方案

| 风险 | 备选 |
|------|------|
| S1 (对比) 12h 跑不完 | 缩 seed 数到 2，或缩 total_steps 到 300k |
| RMPflow 基线仍是 `rl_leaf_weight=0.01` 近似，不够严格 | 在 `mappo_rmp.py` 加 `--rmp-only` 分支跳过 PPO，严格 pure-geometric。工作量 ~30 min |
| 消融叶节点开关尚未实现 | 先用 `fusion_mode=additive + rmp_weight=0` 做"纯 MAPPO 对照"，或直接用 mappo 基线 |
| 训练曲线 `fig:rmp_result` 和对比 `fig:train_result` 共用 progress.csv 列名差异 | 已在 `plot_chap4.py` 中做 default key 容错；首次 sweep 后 grep 列名兜底 |

---

## 附：快速命令清单

```bash
# RQ1 对比 (medium × 4 × 3)
CHAP4_MODE=sweep CHAP4_SCALE=medium CHAP4_SKIP_INSTALL=1 \
  CHAP4_SEEDS="0 1 2" bash run_chap4.sh

# 仅重绘 (数据在位)
CHAP4_MODE=plot bash run_chap4.sh

# 小规模冒烟 (15 min 内完成全流程)
CHAP4_MODE=sweep CHAP4_SCALE=small CHAP4_SEEDS="0" bash run_chap4.sh
```

图表产物最终路径：`paper-safeMARL/images/chap4_sweep/*.png`。
