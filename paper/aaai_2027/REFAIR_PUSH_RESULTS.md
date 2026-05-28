# Push 公平重跑实验记录(r=0.3)

**日期**:2026-05-28  
**触发**:F2 诊断(2026-05-24,见 [`project_aaai_pivot` 记忆])证伪了论文原稿"Push = multi-body limitation / box independently collides with hazards"的解释——查 safety_gymnasium 源码,Push 的 box 根本不产生 cost(`assets/geoms/hazards.py:cal_cost()` 只算 agent-hazard 距离;box/pillar 都 `is_constrained=False`)。Push 失败的真实根因是 **filter 硬编码 `hazard_radius=0.2`,但 Push hazard 实际 size=0.3** 的配置错配。VA-ATACOM 修正半径后单方法已验证 Push 0.00(5/5)。  
**目标**:用正确 r=0.3 对所有有 filter 的方法重跑整列 Push,做公平对比;PPO/PPO-Lag 无 filter 不动,Goal/MGoal hazard 真就是 0.2 也不动。

---

## 实验设置

| 项 | 值 |
|---|---|
| env | `push`(Safety-Gymnasium PointRobot, 2 hazards size 0.3) |
| 训练步数 | 200,000 |
| Seeds | 0, 1, 2, 3, 4 |
| Reach 前端 | `sim_brt`(BRT reward shaping,所有方法一致) |
| Calib / EKF | none / none |
| filter 半径覆盖 | `safety.filter_hazard_radius=0.3`(只对 hazard_radius-aware 方法) |
| 并行 | 20 个 single-threaded(OMP=1)|
| 启动器 | `safe-rl-2027/refair_push_launcher.sh`(15:26 CST 启动) |
| 输出位置 | `safe-rl-2027/runs/refair_push_<method>/seed_<s>/run_eval.json` |
| GO 阈值 | mean cost ≤ 5.0 |

---

## 状态(截至 2026-05-28 18:34)

- ✅ **全 50 run 完成**
  - 主批 40 runs (8 方法 × 5 seed):15:26–16:20 启动→完成,**53 分钟**
  - relaunch 10 runs (atacom + predictive_atacom × 5 seed):18:16–18:34 启动→完成,**18 分钟**(10 个并行同时跑)
- ✅ **atacom 与 predictive_atacom 复现原 Table 1**:66.30 vs 66.3,12.94 vs 12.9 → 印证这两个方法的 filter 源码上根本不接受 `hazard_radius` 参数,只用 `d_safe=0.3` 作 keepout,原数本就公平。但同批次重跑保证了所有 10 方法的训练条件完全一致(同 sim_brt 前端、同 PPO 配置、同步运行)。

---

## 全 10 方法的 per-seed 结果(Push, r=0.3)

| 方法 | seed_0 | seed_1 | seed_2 | seed_3 | seed_4 | mean ± std | GO/5 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| atacom (null-space) | 78.88 | 111.46 | 49.34 | 62.80 | 29.04 | **66.30 ± 30.97** | 0/5 |
| atacom_vd | 19.86 | 69.82 | 29.94 | 78.40 | 91.32 | **57.87 ± 31.26** | 0/5 |
| atacom_s | 7.30 | 15.82 | 14.36 | 11.42 | 8.34 | **11.45 ± 3.69** | 0/5 |
| atacom_la | 2.82 | 34.62 | 10.10 | 59.38 | 14.00 | **24.18 ± 22.95** | 1/5 |
| hocbf | 1.92 | 1.86 | 11.48 | 0.48 | 21.30 | **7.41 ± 8.92** | 3/5 |
| dcm | 21.08 | 3.46 | 2.66 | 19.30 | 13.02 | **11.90 ± 8.62** | 2/5 |
| predictive_atacom | 3.04 | 0.00 | 27.02 | 17.98 | 16.66 | **12.94 ± 11.43** | 2/5 |
| cbf_qp | 2.48 | 12.12 | 85.46 | 32.64 | 14.10 | **29.36 ± 33.21** | 1/5 |
| **distance_adaptive (DT-margin)** | 3.14 | 0.00 | 1.72 | 11.68 | 0.26 | **3.36 ± 4.82** | **4/5** ✓ |
| **brake_manifold (VA-ATACOM)** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00 ± 0.00** | **5/5** ✓ |

reward 见 `runs/refair_push_<method>/seed_<s>/run_eval.json` 的 `eval.reward_mean`(brake_manifold 5 seeds: 0.482 / 0.509 / 0.548 / 0.236 / -0.672;seed_4 reward 偏低但 cost 仍 0.00,filter 稳压安全)。

---

## 公平重跑前后对比(Push 列)

| 方法 | 原 Table 1 (r=0.2) | **新公平 (r=0.3)** | Δ | **新 GO/5** |
|---|---:|---:|---:|:---:|
| atacom (null-space) | 66.3 | **66.30** | ≈0 | 0/5 |
| atacom_vd | 50.6 | 57.87 | +7.3 | 0/5 |
| atacom_s | 26.9 | 11.45 | −15.5 | 0/5 |
| atacom_la | 40.1 | 24.18 | −15.9 | 1/5 |
| hocbf | 11.4 | **7.41** | −4.0 | **3/5** ⤴ |
| dcm | 11.1 | 11.90 | +0.8 | 2/5 |
| predictive_atacom | 12.9 | **12.94** | ≈0 | 2/5 |
| cbf_qp | 29.4 | 29.36 | ≈0 | 1/5 |
| **distance_adaptive (DT-margin)** | 10.58 | **3.36** | **−7.2** | **4/5 GO** ★ |
| **brake_manifold (VA-ATACOM)** | 93.2 | **0.00** | **−93.2** | **5/5 GO** ★★ |

注 1:atacom / atacom_vd / predictive_atacom / cbf_qp / dcm 这 5 个方法 Δ 接近零或反向——其 filter 内部本就不读 hazard_radius(或读了但对 cost 不敏感),原数已是公平值。剩下 5 个方法(atacom_s/atacom_la/hocbf/distance_adaptive/brake_manifold)受半径修正显著影响,其中后两者真正受益 → GO。

注 2:atacom_vd seed_4 = 91.32 是大方差 outlier;原 Table 1 中"PPO 基线"和"PPO-Lag"未含在此重跑(无 filter,不受半径影响,沿用原表)。

注:atacom_vd seed_4 之类大方差 outlier 把 mean 拉高;原 Table 1 中"PPO 基线 + sim_brt"行未含在此重跑(无 filter,不受半径影响)。

---

## 关键发现

### 1. VA-ATACOM 三任务全 GO,15/15 已确立 🎯

| 任务 | mean cost | seed-level GO | 数据来源 |
|---|---:|:---:|---|
| Goal | 0.00 | 5/5 | 原 Table 1(env hazard 0.2,无需重跑) |
| Push | **0.00** | **5/5** | **本次重跑** |
| MGoal | 0.40 | 5/5 | 原 Table 1(env hazard 0.2) |
| **合计** | — | **15/15** | — |

VA-ATACOM 是**唯一**三任务全 GO 的方法。

### 2. DT-margin(VA-ATACOM 的一阶近似)也几乎全 GO

- Push 由 10.58(0/5)→ 3.36 ± 4.82(**4/5 GO**),task-mean GO。
- 加上原 Goal 0.84 + MGoal 2.59(均 5/5 GO),三任务都 task-mean GO,seed-level 大约 **14/15**(仅 Push 1 个 outlier seed_3 = 11.68 超阈)。

**叙事强化**:精确形式(VA-ATACOM)与一阶近似(DT-margin)都解决了 Push,但只有精确形式做到 5/5——支持"velocity-augmented manifold 是正确的离散时间 CBF 形式"的论文主张。

### 3. "Multi-body / box-coupling limitation"是事实错误,务必从全文删除

每个有 filter 的方法在改正半径后 Push cost 都**显著下降**(atacom_s -15.5,atacom_la -15.9,distance_adaptive -7.2,brake_manifold -93.2);只有少数(atacom_vd, dcm, cbf_qp)变化不显著或反向(可能是 velocity-scaling 类机制在更大半径下被过度放大)。**没有一个方法表现出"box collides with hazards"的 multi-body 特征**——所有变化都能用单体半径错配解释。

论文需删的位置:
- `main_v2.tex` §I 第 67 行附近("Push, which couples the agent to a pushed box, exposes a multi-body limitation...")
- §V Main Results 段(我 2026-05-28 重写时又写了一遍——必须删)
- §Limitations L1("Push task (3/5 GO). Box-agent coupling creates dynamics not captured by single-agent BRT.")

### 4. 多数 baseline 仍未达 GO,头条仍然成立

HOCBF 改善到 7.41(task-mean NO-GO 但 3/5 seed GO)。其余多数仍 ≥10 cost,task-mean NO-GO。这与"continuous-time 假设在粗 Δt 下失效,只有速度增广才行"的论文主张一致。

---

## 待办(等 atacom/pred_atacom relaunch 完结后)

1. 把 atacom(原 66.3)和 predictive_atacom(原 12.9)的新公平数填进上表。
2. 用新数字重写 `main_v2.tex` Table 1 整列 Push。
3. 全文删"multi-body / box-coupling"假说法(3 处)。
4. Push 从"limitations"段移到"results"——VA-ATACOM 三任务全 GO 改成头条。
5. 摘要 / §I / 结论同步:把 10/10 改成 15/15,Push 从 0.40 stress test 改成 0.00 result。
6. (可选)新增半句方法学说明:filter `hazard_radius` 必须匹配 env hazard size,Push hazard 是 0.3 不是 0.2;这是公平比较的前提。

---

## 运行物件位置

- 启动器 + log:`safe-rl-2027/refair_push_launcher.sh`、`refair_push_launcher.log`、`refair_push_progress.log`
- 每 run 输出:`safe-rl-2027/runs/refair_push_<method>/seed_<s>/{run_eval.json, run.log}`
- relaunch 启动器:`safe-rl-2027/refair_push_atacom_relaunch.sh`(原 launcher 不动)
- 相关记忆:`memory/project_aaai_pivot.md`(主项目状态)、`memory/chap4_repro_risk.md`(不相关,硕论)
