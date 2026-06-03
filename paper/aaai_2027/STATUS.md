# AAAI-27 VA-ATACOM 投稿进度 (STATUS)

**Last update**: 2026-06-03
**Deadline**: 2026-07-28 (UTC-12),还有 **~8 周**
**Target**: AAAI-27,第 41 届,2027 年 2 月 16-23 日 Montréal
**当前中稿估计**: **65-78%**(AAAI baseline 24-28% 的 ~2.5-3.5×;详见 `PAPER_AUDIT.md`)

**优先级原则**(see [[feedback_aaai_priority]]):
1. ⭐ **仿真主线** — Safety-Gym + Webots 是核心证据,这条线打磨到 reviewer-proof
2. **论文 polish** — body 7 页内 framing / typesetting / 数字一致性 / supplementary
3. **硬件 (Path C)** — 补充,**等硬件到手再做**,不影响投稿是否成立

---

## ✅ 已完成(P0+P1+P2 + 各续 全 12 task)

### A. 仿真核心证据(数据全跑完 + Wilcoxon/McNemar 显著)

- **Safety-Gym Point Goal/Push/MultiGoal**: 12 method × 3 task × 5 seeds × 200K
  - **VA-ATACOM 三任务全 GO 15/15**(0.00 / 0.00 / 0.40 mean cost)
  - **paired Wilcoxon p < 0.001 vs every baseline**(10/11);DT-margin p=0.003
  - Friedman omnibus χ²=97.05, p=6.8e-16
  - Best baseline = HOCBF **4/15** (audit 后 from 5/15)
  - DT-margin **13/15**(VA-ATACOM 一阶 Taylor 近似)
  - Push 列 r=0.3 fair 重跑完成
  - 每 cell × 12 method = 36 cells 100% 数字 audit 通过(见 `TABLE1_AUDIT.md`)
- **Webots E-puck 3-world × 2-controller × 2-mode = 240 trials**
  - corridor (5-cyl S-curve) + dense (6-cyl) + lshape (5-cyl 154mm 最紧 corner)
  - VA-ATACOM: **1/60 deep (P-ctrl, 60/60 goal), 0/60 deep (PPO, 27/60 goal)**
  - 无 filter: 14/60 / 10/60 deep
  - **paired McNemar p=1.2e-7**;filter saves 23/120, **hurts 0/120**
  - 详见 `WEBOTS_INFERENCE_RESULTS.md`
- **Car Goal generalisation 验证(诚实 negative)**
  - SafetyCarGoal1-v0 4 method × 5 seed × 200K
  - **VA-ATACOM 23.92 (1/5) ← 输给 ATACOM 9.20 (1/5)**(根因:diff-drive action form 与 bicycle 不兼容)
  - 写入 supp **§Generalisation Beyond Point-Robot Dynamics** + main **§VI.Limitations L4**
  - frame:"scope clarification, not refutation of Prop 4 itself"

### B. 论文 source(submission-ready)

- **`aaai27/main_v2_aaai27.pdf`** — 10 pages,body 严格 1-7,**0 errors/warnings/overfull**
  - §I Introduction + Contributions 4 条(含 Cor 1/2 + Webots + Wilcoxon/McNemar)
  - §II Related Work(含 "Position vs HOCBF/DCBF" 段 preempt minor incremental 批评)
  - §III Preliminaries
  - §IV VA-ATACOM Algorithm(§A failure + §B manifold + §C Prop 4 + Cor 1/2 + §D constructive hyperparam)
  - §V Experiments(setup / main / design / coarse / wall-time × 5 sub)
  - §VI Discussion + Limitations **L1-L4** + Sim-to-Real (i-iv Webots)
  - §VII Conclusion(含 specific 数字 + p-value + "closing two-orders-of-magnitude gap" punch)
- **`aaai27/supplementary_dtmargin.pdf`** — 3 pages,4 sections
  - §1 DT-margin Heuristic Baseline(α/h/Δt sweep)
  - §2 Statistical Significance(Wilcoxon table + Friedman + McNemar)
  - §3 Generalisation Beyond Point-Robot Dynamics(Car negative honest)
  - §4 Anticipated Questions(5 reviewer FAQ preempt)
  - §5 Computing Infrastructure
- **`aaai27/repro_checklist.tex`** — 24 答案,**Computational 13/13 yes 全 partial 清掉**
- 用官方 `aaai2027.sty` (2027/05/04, 从 AAAI 官方 author kit 取)

### C. Reviewer reproduce 通路

- **`safe-rl-2027/README.md`** — reviewer-facing,6 命令复现一切
- **`reproduce_table1.sh`** — Table 1 全 180 runs(`--smoke` 2 min;full ~10 h)
- **`reproduce_webots.sh`** — Webots 240 trials(~10 min)
- **`aggregate_table1.py`** — 一键算 mean/GO/Wilcoxon(audited dirs 或 fresh repro)
- **`car_goal_launcher.sh`** — Car Goal 4 method × 5 seed
- **`refair_push_launcher.sh`** — Push fair r=0.3 重跑
- **`audit_missing_cells_launcher.sh`** — Table 1 dir 缺失 3 cell 重跑

### D. Supplementary materials

- **2 个 demo videos**:
  - `supplementary_demo.mp4` 155 s / 1.6 MB(完整 6-trial concat)
  - `supplementary_demo_30s.mp4` 31 s / 573 KB(5× 加速 overview)
- 4 张 overlay PNG(corridor/dense/lshape × P-ctrl/PPO)

### E. 实机(只到 code-ready,等硬件)

- ✅ `safe-rl-2027/experiments/hardware/HARDWARE_COMMS.md`(4-path 通信对比)
- ✅ `epuck_hello.py`(e-puck2 advsercom 20-byte binary + USB ASCII fallback)
- ✅ `dwm1001_reader.py`(DWM1001 PANS-2 shell-mode `lec` 流)
- ✅ `PATH_C_PROTOCOL.md`(9-step 实机协议,~3 day timeline)

---

## ✅ 12 task complete log

| Layer | Task | Commit |
|---|---|---|
| P0-1 | Supplementary 重生 | `62e57c2` |
| P0-2 | Table 1 数字 audit (36 cells × 12 method) | `1999d89` + `f127d10` |
| P0-3 | references.bib 整理(33→29 entries,5 type fix) | `cd4a7f0` |
| P0-4 | abstract polish(270→201 words,hook 前置) | `6fd43ca` |
| P0-5 | Figure 2 dense → lshape(154mm visual stronger) | `05a3635` |
| P1-1 | Wilcoxon + McNemar + Friedman 全显著 | `705e92c` |
| P1-2 | Computing infrastructure 段 | `828ae54` |
| P1-3 | README + reproduce_table1.sh + aggregate.py | `3faa6fe` + `5f75123` |
| P2-1 | Webots demo video 155s + 31s | `9072ce6` + `4eec72d` |
| P0-续 | Conclusion 加 specific 数字 + Limitations L3 + §IV-A Fig 1 ref | `9db5392` + `4a0ca84` |
| P1-续-4 | Car Goal 4 method × 5 seed(诚实 negative + L4) | `9b993a1` + `a507227` |
| P2-续-6 | Reviewer FAQ supp section(5 anticipated questions) | `8228d16` |

---

## 🔄 后续(剩 1 task)

### P3 — 等硬件到位再做(Path C 实机,7 步骤)

| # | 任务 | 预估 |
|:---:|---|---|
| 13 | Phase 3.1 e-puck2 通信打通(`epuck_hello.py` Wi-Fi spin) | 2-4 h |
| 14 | Phase 3.2 a_max 物理标定 | 30 min |
| 15 | Phase 3.3 UWB 部署 | 半天 |
| 16 | Phase 3.4 arena 建 | 1 h |
| 17 | Phase 3.5 host controller 集成 | 半天 |
| 18 | Phase 3.6 20 trials × 2 modes | 1-2 h |
| 19 | Phase 3.7 §VI(v) 加段 + 守 7 页 | 1 h |

**关键决策**:如果到 deadline 前 1 周硬件还没跑通,**就不放实机段** — 仿真证据 sufficient(Webots 240-trial transfer + Wilcoxon + McNemar),论文不依赖这段。

---

## 📊 中稿评估(详见 `PAPER_AUDIT.md`)

| Layer | 累计估升 | 累计估 |
|---|:---:|:---:|
| paper 初版(P0+P1+P2 完成) | — | 55-65% |
| P0-续 3 项(Conclusion 数字 + L3 + Fig 1 ref) | +5-8% | 60-72% |
| P1-续-4(Car 诚实 honest scope) | +2-3% | 62-75% |
| P2-续-6(Reviewer FAQ 5 questions) | +2-3% | **65-78%** |

**对比 AAAI 历年 baseline 24-28%** → 当前 ~**2.5-3.5× 高于平均录用率**。

**主要剩余 risk**:
- Reviewer lottery(1 个 negative reviewer 可能造成 reject,2-3 positive 可救)
- AAAI-27 主席 area routing(ML area chair 更 favor theoretical novelty)
- Concurrent submissions 竞争激烈度

---

## 📂 关键 path 入口

| 资产 | path |
|---|---|
| 主 tex / pdf | `paper/aaai_2027/aaai27/main_v2_aaai27.{tex,pdf}` |
| Supplementary tex / pdf | `paper/aaai_2027/aaai27/supplementary_dtmargin.{tex,pdf}` |
| Reproducibility Checklist | `paper/aaai_2027/aaai27/repro_checklist.tex` |
| 主图 1(velocity augmentation) | `paper/aaai_2027/figures/fig_v_a_trajectory/` |
| 主图 2(lshape 154mm corner) | `paper/aaai_2027/figures/fig_webots_va_atacom/lshape_all_trials_overlay.png` |
| Audit docs | `paper/aaai_2027/{PAPER_AUDIT,TABLE1_AUDIT,REFAIR_PUSH_RESULTS,WEBOTS_INFERENCE_RESULTS}.md` |
| Filter 实现 | `safe-rl-2027/safe_rl/filters/brake_manifold.py`(与 Safety-Gym + Webots + Car 同一份代码) |
| Webots controller | `safe-rl-2027/experiments/webots/controllers/va_atacom_nav/` |
| Webots worlds | `safe-rl-2027/experiments/webots/worlds/epuck_{corridor,dense,lshape}_va_atacom.wbt` |
| 实机 protocol | `safe-rl-2027/experiments/hardware/PATH_C_PROTOCOL.md` |
| Reviewer reproduce | `safe-rl-2027/{README.md,reproduce_table1.sh,reproduce_webots.sh,aggregate_table1.py}` |
| Raw experiment data | `safe-rl-2027/runs/`(gitignored,launcher script 提供复现) |

---

## 🎯 投稿日 critical path(更新)

```
2026-06-03  →  ...    →  2026-07-15  →  2026-07-28
   现在                   freeze          submit deadline
   ↓                      ↓                ↓
[paper ready]            [final           [final PDF +
[12 task done]            proof-read]      supplementary +
[supp ready]                              demo video]
[FAQ ready]
[reproduce ready]
```

**当前状态**:论文 polishing 已 saturate。剩余唯一可做 = **Path C 实机**(等硬件),否则 idle 到 deadline。

每次 progress 改这一个 STATUS.md 即可。
