# AAAI-27 VA-ATACOM 投稿进度 (STATUS)

**Last update**: 2026-05-31
**Deadline**: 2026-07-28 (UTC-12),还有 **~2 个月**
**Target**: AAAI-27,第 41 届,2027 年 2 月 16-23 日 Montréal

**优先级原则**(see [[feedback_aaai_priority]]):
1. ⭐ **仿真主线** — Safety-Gym + Webots 是核心证据,这条线打磨到 reviewer-proof
2. **论文 polish** — body 7 页内 framing / typesetting / 数字一致性 / supplementary
3. **硬件 (Path C)** — 补充,**等硬件到手再做**,不影响投稿是否成立

---

## ✅ 已完成 (累计 5/15 + 24 - 30 May)

### A. 仿真核心证据(数据全跑完)

- **Safety-Gym Point Goal/Push/MultiGoal**: 5 seeds × 200K steps × 10 methods,Push 公平重跑 r=0.3
  - **VA-ATACOM 三任务全 GO 15/15**(Goal 0.00 / Push 0.00 / MultiGoal 0.40 mean cost)
  - 详见 [REFAIR_PUSH_RESULTS.md](REFAIR_PUSH_RESULTS.md)
- **Webots E-puck 3-world × 2-controller × 2-mode = 240 trials**
  - corridor (5-cyl S-curve) + dense (6-cyl scattered) + lshape (5-cyl L-barrier, 154 mm corner)
  - VA-ATACOM: **1/60 deep (P-ctrl, 60/60 goal), 0/60 deep (PPO, 27/60 goal)**
  - 无 filter: 14/60 deep (P-ctrl, 54/60 goal), 10/60 deep (PPO, 27/60 goal)
  - 详见 [WEBOTS_INFERENCE_RESULTS.md](WEBOTS_INFERENCE_RESULTS.md)

### B. 论文 source

- **`aaai27/main_v2_aaai27.tex`** + `aaai27/main_v2_aaai27.pdf`
  - 10 pages total,**body 严格 1-7**(Conclusion + Refs 起均 page 7)
  - 0 errors / 0 warnings / 0 overfull
  - Refs 7末–8顶,Reproducibility Checklist 8–10
  - 用官方 `aaai2027.sty` (2027/05/04),非 aaai2026 — 详见 [[aaai_page_limit]]
- **`aaai27/repro_checklist.tex`** — 24 问全部填完(General 3 yes / Theoretical 8 yes / Dataset relies-no NA / Computational 11 yes + 2 partial)
- **`aaai27/supplementary_dtmargin.tex`**(在 _archived_aaai26 里,需重 port)— DT-margin α/h/Δt 详细 sweep,作 supplementary,不进 body

### C. 理论 + framing

- §IV-C Prop. 4 已重写为速度增广流形离散 CBF 不变性 (`c_{k+1} ≥ (1−α₀Δt)c_k − δ`, `δ=(3/2)Δt²a_max`),三步证明 + Algorithm 1
- §IV-C 加 Corollary 1(Δt→0 退化 ATACOM) + Corollary 2(多障碍 per-step O(m))
- §V 重构:删主文旧三表 → §5.3 Design Study (Table 2,五设计) + Table 4 wall-time
- §VI Sim-to-Real 三段(i)(ii)(iii)+ (iv) Webots E-puck (60 trials,数据已对齐)
- 删了所有事实错误的"box independently collides / multi-body limitation"(F2 已证伪)

### D. 实机(只到 code-ready,等硬件)

- ✅ `safe-rl-2027/experiments/hardware/HARDWARE_COMMS.md` — 4 path 通信对比 + 推荐 Path B Wi-Fi
- ✅ `safe-rl-2027/experiments/hardware/epuck_hello.py` — e-puck2 advsercom 20-byte binary 协议 + USB ASCII fallback
- ✅ `safe-rl-2027/experiments/hardware/dwm1001_reader.py` — PANS-2 shell-mode `lec` 流 reader
- ✅ `safe-rl-2027/experiments/hardware/PATH_C_PROTOCOL.md` — 9 段 step-by-step 协议

---

## 🔄 后续要完成(按 priority 排)

### P0 — 必做(投稿前)

| # | 任务 | 预估 | 备注 |
|:---:|---|---|---|
| 1 | **AAAI-27 supplementary 重新生成** | 1 h | 把 `_archived_aaai26/supplementary_dtmargin.tex` 用 aaai2027.sty 重 compile;含 DT-margin α/h/Δt 3 表 + 2 figure(side-validation) |
| 2 | **论文最终读一遍 + 数字对齐核** | 2 h | spot-check 每个 cited 数字(Table 1 / 五设计 / Webots aggregate)都来源于 commit 的 raw json;catch latent stale claim |
| 3 | **citations 完整性 + bib 整理** | 1 h | 看 `references.bib` 有没 missing entry / typo;某些 cite 是否 outdated arXiv → 改正式 venue |
| 4 | **abstract polish** | 30 min | 让 first sentence 抓眼球;cite 主结果数字 |
| 5 | **figure 替换/优化** | 1 h | Figure 2 当前 dense overlay,考虑换 lshape(154 mm corner 更 visual);Figure 1 trajectory 已 OK |

### P1 — 强烈建议(增竞争力)

| # | 任务 | 预估 | 备注 |
|:---:|---|---|---|
| 6 | **VA-ATACOM 显著性检验** | 2 h | 用 Wilcoxon signed-rank 跑 VA-ATACOM vs 各 baseline 在 15 cells,把 checklist Wilcoxon 从 partial → yes |
| 7 | **算力 infra 段补全** | 30 min | 写明 CPU model / RAM / OS / pytorch/safety-gym version;checklist computing infra partial → yes |
| 8 | **§VI sim-to-real 多 layout 强调** | 30 min | 现在 §VI(iv) 数字是 60-trial aggregate,加一句强调"3 layouts, including a tight 154 mm corner" |
| 9 | **README + 复现脚本** | 2 h | 给 reviewer 看的 top-level `safe-rl-2027/README.md`:env install + 跑 main result + 跑 Webots,1 个 shell script `reproduce_table1.sh` |

### P2 — 锦上添花

| # | 任务 | 预估 | 备注 |
|:---:|---|---|---|
| 10 | **Webots video 投 supplementary** | 1 h | `demo.mp4` 4 个 (per world × policy) 已存;拼一个 30s overview |
| 11 | **Webots 加 1 个 dynamic obstacle world** | 半天 | filter 在动态障碍下行为(stretch goal,**不做也不影响**) |
| 12 | **lshape 加 PPO ablation 不同 BRT horizon** | 半天 | (stretch) |

### P3 — 等硬件到位再做(Path C)

| # | 任务 | 预估 | 备注 |
|:---:|---|---|---|
| 13 | Phase 3.1 e-puck2 通信打通 | 2-4 h | `epuck_hello.py` Wi-Fi spin 验证 |
| 14 | Phase 3.2 a_max 物理标定 | 30 min | 10 次满推急停 |
| 15 | Phase 3.3 UWB 部署 | 半天 | 4 anchor 写坐标 + tag 装 e-puck |
| 16 | Phase 3.4 arena 建 | 1 h | 推荐 lshape layout |
| 17 | Phase 3.5 host controller 集成 | 半天 | `physical_controller.py` 待写 |
| 18 | Phase 3.6 20 trials × 2 modes | 1-2 h | 数据收集 |
| 19 | Phase 3.7 §VI(v) 加段 + 守 7 页 | 1 h | 模板已写在 `PATH_C_PROTOCOL.md` |

**关键决策**:如果到 deadline 前 1 周硬件还没跑通,**就不放实机段** — 仿真证据 sufficient,论文不依赖这段。

---

## 📂 关键 path 入口

| 资产 | path |
|---|---|
| 主 tex / pdf | `paper/aaai_2027/aaai27/main_v2_aaai27.{tex,pdf}` |
| Reproducibility Checklist | `paper/aaai_2027/aaai27/repro_checklist.tex` |
| 主图(velocity augmentation) | `paper/aaai_2027/figures/fig_v_a_trajectory/` |
| Webots 实验数据 audit | `paper/aaai_2027/WEBOTS_INFERENCE_RESULTS.md` |
| Push 公平重跑 audit | `paper/aaai_2027/REFAIR_PUSH_RESULTS.md` |
| Filter 实现 | `safe-rl-2027/safe_rl/filters/brake_manifold.py` |
| Webots controller | `safe-rl-2027/experiments/webots/controllers/va_atacom_nav/` |
| Webots worlds | `safe-rl-2027/experiments/webots/worlds/epuck_{corridor,dense,lshape}_va_atacom.wbt` |
| 实机 protocol | `safe-rl-2027/experiments/hardware/PATH_C_PROTOCOL.md` |
| Raw experiment data | `safe-rl-2027/runs/`(gitignored,需复现) |

---

## 🎯 投稿日 critical path

```
2026-06-01  →  2026-06-15  →  2026-06-30  →  2026-07-15  →  2026-07-28
   现在          P0完成        P1完成         polish freeze    submit deadline
   ↓             ↓             ↓             ↓                ↓
[main_v2.pdf]  [supplementary [Wilcoxon +  [last-pass    [final PDF +
[body 7页]      重生成]        infra 补]    proof-read]   supplementary]
[checklist]    [bib整理]      [README]
[3 worlds 数据]
```

实机 (P3) 不在 critical path 上,**有空再做,做不了不影响投稿**。
