# 第 4 章 · 严格 RMPflow 基线与目录重复说明

> 创建日期: 2026-04-19
> 问题来源: 对 `rl_leaf_weight=0.01` 近似 RMPflow 基线的严格性质疑

---

## 1. 问题：chap4 论文里的 "RMPflow 基线" 不够严格

chap4 对比实验列的第 3 个基线是 **RMPflow**（纯几何控制，无学习）。当前实现：

```bash
# run_chap4.sh 里对 rmpflow 的处理 (run_one 函数, algo=rmpflow 分支):
CHAP4_FUSION_MODE_OVERRIDE=leaf
CHAP4_RL_LEAF_WEIGHT_OVERRIDE=0.01    # 把 RL 叶权重压到很小
python -m safepo.multi_agent.mappo_rmp ...
```

即：**仍然在训练 MAPPO 策略**（RL 叶存在，只是权重极小），几何叶节点在融合中主导。物理效果近似纯 RMPflow，但严格意义上：
- MAPPO 策略仍在更新（虽然 RL 叶的贡献被压低）
- 答辩时可被问："你这不是 RMPflow，是 weak-RL + RMPflow"

这个弱点已在 `chap4-sweep-guide.md §6` 明示为 "known limitation"。

---

## 2. 现有相关代码盘点

### 2.1 cosmos 里已有的 pure RMPflow 实现（但用了不同 env）

| 路径 | 内容 | 能否直接用于 chap4? |
|------|------|--------------------|
| `cosmos/apps/formation_nav/benchmark.py` | `compute_rmp_action()` 手写 RMP 解析式 + `run_rmp_episode()` | **不能** — 用的是 `cosmos.envs.formation_nav.FormationNavEnv`，不是 chap4 的 safety-gymnasium `SafetyPointMultiFormationGoal0-v0` |
| `cosmos/apps/formation_nav/demo_visualization.py` | 同上，`--mode rmp` 只做可视化 | **不能** — 同样是 cosmos 自己的 env |

**关键差异**：cosmos 里的 RMP 是**简化手写公式**（goal 吸引 + 编队力 + 障碍排斥 + 阻尼的线性组合），**不是**论文里 GCPL 所用的 `RMPCorrector` + `rmp_leaf.py` 那套 node-pushforward-pullback-resolve 框架。即便迁移 env，也不是同一套 RMPflow 方法，对比意义不同。

### 2.2 safepo 里没有 `--rmp-only` 入口

- `safepo.multi_agent.mappo_rmp` — GCPL 训练入口（含 RL 叶）
- `safepo.multi_agent.mappo` / `mappolag` / `happo` / ... — 各类 RL 算法
- ❌ 没有 `rmpflow_only.py` 或类似的 "纯几何无学习" 入口
- ❌ 没有 `--rmp-only` CLI 标志

---

## 3. 要修改的源文件（确认只改 suite 版本）

**规范路径：所有 chap4 代码改动应在 `safe-rl-manifold-suite/` 下进行**（见 §4 目录重复说明）。

### 3.1 方案 A（推荐）· 给 `mappo_rmp.py` 加 `--rmp-only` 分支

**主改文件**：
1. `safe-rl-manifold-suite/algorithms/safe-po/safepo/multi_agent/mappo_rmp.py`
   - `Runner.__init__`: 保存 `self.rmp_only = config.get('rmp_only', False)`
   - `Runner.run()`: 若 `rmp_only=True`：
     - 跳过 `self.collect(step)`（或令其返回零动作 tensor）
     - 跳过 `self.insert(...)` / `self.compute()` / `self.train()` （因为不需要 PPO 更新）
     - 仍然调用 `self.rmp_corrector.apply_correction()` 让 RMP 合成 a_soft（RL 叶输入为 0）
     - 仍然调用 `env.step(a_soft)` + 奖励/代价记录
   - `Runner.eval()`: 同样逻辑

2. `safe-rl-manifold-suite/algorithms/safe-po/safepo/utils/config.py`
   - 新增 CLI 参数 `--rmp-only`（默认 False），通过 `cfg_train["rmp_only"]` 透传

### 3.2 方案 A 辅助改动

3. `safe-rl-manifold-suite/algorithms/safe-po/safepo/multi_agent/rmp_corrector.py`
   - 无需代码改动。RL 叶 `_a_rl` 默认就是零向量；只要 `mappo_rmp.py` 不调 `set_action`，RL 叶就恒为 0，几何叶主导
   - （可选）打印增加 `rmp_only=True` 的状态显示

4. `safe-rl-manifold-suite/run_chap4.sh`
   - 在 `run_one` 的 `rmpflow` 分支去掉 `CHAP4_RL_LEAF_WEIGHT_OVERRIDE=0.01` 近似，改为 `--rmp-only` 参数
   - 更新 doc 注释

### 3.3 方案 A 工作量估计

- 代码：~30-40 行 Python (mappo_rmp.py) + 3-4 行 (config.py) + ~5 行 (run_chap4.sh)
- 预计实施 + 自测：30-45 分钟
- 对现有 S1 / S2 sweep 结果**不破坏**（可选启用；默认关闭）

### 3.4 方案 B · 不改，保留 `rl_leaf_weight=0.01` 近似

- 工作量 0
- 在 chap4.tex 正文 / 答辩 PPT 明确写："RMPflow baseline 通过将 RL 叶权重降至 0.01 近似实现"
- 风险：评审专家可能要求严格基线

---

## 4. 目录重复问题

### 4.1 两套平行目录

仓库根 `czz-safe-manifold/` 下存在**两套结构**：

| 路径 | 状态 | 是 chap4 工作区吗 |
|------|------|------------------|
| `czz-safe-manifold/algorithms/safe-po/` | 🗑 **空目录**（Feb 27 创建, 从未填充） | ❌ |
| `czz-safe-manifold/algorithms/multi-robot-rmpflow/` | 🗑 **空目录** | ❌ |
| `czz-safe-manifold/cosmos/` | 📦 **有内容**, md5 与 suite 一致 | ⚠ **重复** |
| `czz-safe-manifold/envs/safety-gymnasium/` | 📦 **部分内容**, 比 suite 少 docs/Makefile 等 | ⚠ 不完整副本 |
| `czz-safe-manifold/safe-rl-manifold-suite/algorithms/safe-po/` | ✅ **活文件**, 含所有 chap4 补丁 | ✅ |
| `czz-safe-manifold/safe-rl-manifold-suite/algorithms/multi-robot-rmpflow/` | ✅ **活文件**, 含 RLLeaf 类 | ✅ |
| `czz-safe-manifold/safe-rl-manifold-suite/cosmos/` | ✅ | ✅ |
| `czz-safe-manifold/safe-rl-manifold-suite/envs/safety-gymnasium/` | ✅ **完整**, editable install 指向这里 | ✅ |

### 4.2 Python 实际用的是哪个？

从 `pip show` / `direct_url.json` 推断：

| 包 | 实际 import 路径 |
|-----|-----------------|
| `safety_gymnasium` | `safe-rl-manifold-suite/envs/safety-gymnasium/safety_gymnasium/__init__.py` |
| `safepo` | `safe-rl-manifold-suite/algorithms/safe-po/safepo/__init__.py` |
| `rmp_leaf` (via `MULTI_ROBOT_RMPFLOW_PATH`) | `safe-rl-manifold-suite/algorithms/multi-robot-rmpflow/rmp_leaf.py` |

**结论：所有运行时导入都指向 suite 版本。**

### 4.3 顶层副本的来历 / 用途

- `algorithms/safe-po/`、`algorithms/multi-robot-rmpflow/` 空目录：推测是早期 submodule 初始化失败或清理留下的空壳
- `cosmos/`：似乎是 suite 的同步副本（时间戳与 md5 一致），用途不清
- `envs/safety-gymnasium/`：部分 editable 化时的残余？

### 4.4 规范

**所有 chap4 相关代码修改只在 `safe-rl-manifold-suite/` 下进行**：

```
# ✅ 正确 — 会生效
safe-rl-manifold-suite/algorithms/safe-po/safepo/multi_agent/mappo_rmp.py
safe-rl-manifold-suite/algorithms/multi-robot-rmpflow/rmp_leaf.py
safe-rl-manifold-suite/envs/safety-gymnasium/safety_gymnasium/...

# ❌ 错误 — 改了不会被 Python 加载
czz-safe-manifold/cosmos/...          # stale 副本
czz-safe-manifold/envs/safety-gymnasium/...   # 不完整副本
czz-safe-manifold/algorithms/safe-po/         # 空目录
```

### 4.5 是否清理这些顶层副本？

| 选项 | 风险 | 收益 |
|------|------|------|
| **保留**（现状） | 无 | 无 |
| 删除顶层 `cosmos/`、`envs/`、`algorithms/` 空目录 | 若有脚本硬编码引用会断（需 grep 确认） | 仓库更干净, 消除混淆 |

**推荐：暂不清理**。chap4 工作完成后再统一整理，避免动到 S1/S2 跑期间可能引用的路径。

---

## 5. 行动清单

### 🔴 若决定做方案 A（严格 RMPflow 基线）

- [ ] 在 `safe-rl-manifold-suite/algorithms/safe-po/safepo/multi_agent/mappo_rmp.py` 加 `rmp_only` 分支
- [ ] 在 `safe-rl-manifold-suite/algorithms/safe-po/safepo/utils/config.py` 加 `--rmp-only` CLI 参数
- [ ] 更新 `safe-rl-manifold-suite/run_chap4.sh` 的 `rmpflow` 分支使用 `--rmp-only`
- [ ] 跑 small smoke 验证
- [ ] 等 S1 medium sweep 结束后，只重跑 rmpflow 那一组 3 seed（其余不动）
- [ ] 更新 `chap4-sweep-guide.md` 去掉 "已知限制" 里的 rmpflow 近似段

### 🟡 若决定不做（保留方案 B 近似）

- [ ] 在 `chap4.tex` §4.6.4.1 对比实验正文里明示近似：一句话说明 "RMPflow baseline is instantiated with `rl_leaf_weight=0.01`, which makes the geometric leaves dominate fusion; a pure geometry-only baseline is left as future work."
- [ ] 答辩 PPT 标注

### 🟢 目录清理（收尾）

- [ ] chap4 工作完成后, 考虑删除顶层 stale 副本（`czz-safe-manifold/{cosmos, envs, algorithms/safe-po, algorithms/multi-robot-rmpflow}`）

---

## 6. 当前 S1/S2 的影响

S1 正在跑的 `rmpflow` 3 个 seed 将产出**近似版** RMPflow 数据。方案 A 若实施：

- 不影响 S1 的 mappo_rmp / mappo / mappolag 三组（共 9 runs）
- 只需要重新跑 rmpflow 组（3 runs，~3 h）
- 总增量时间: ~3 h

因此无需现在打断 S1。可以**等 S1 跑完后**再决定方案，重跑成本低。

---

## 附：快速导航

- 本文件：`paper-safeMARL/chap4-rmpflow-strict-baseline.md`
- 相关文档：`chap4-rq-deliverables.md`（RQ 清单）、`chap4-sweep-guide.md`（sweep 使用）、`chap4-progress.md`（进度）
- 代码入口：`safe-rl-manifold-suite/algorithms/safe-po/safepo/multi_agent/mappo_rmp.py`
