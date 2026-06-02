# AAAI-27 VA-ATACOM 论文 audit + 中稿评估

**Date**: 2026-06-02 (~8 周距 deadline 2026-07-28)
**Status**: P0+P1+P2 全 9/9 完成 ✅,paper submission-ready

---

## 1. 7-page body 布局(逐页 audit)

| Page | 内容 | 密度 |
|:---:|---|---|
| **1** | Title / Abstract (201 w) / §I Intro 起 (Why VA-ATACOM is needed,M1/M2 motivation) | 适中 |
| **2** | §I 续 (Theoretical contribution + Experimental validation 段) + Contributions 4 条 + §II Related Work 起 + §II.A Tangent-Projection / §II.B Set-Based / §II.C Lagrangian | **密** (4 sub) |
| **3** | §II 续 + §III Preliminaries 3 sub (CMDP / ATACOM theorem / Circular geometry) + §IV VA-ATACOM 起 (§IV.A Why Continuous-Time Fail) | **密** (5 sub) |
| **4** | §IV.A 续 (Prop 1-2 M1 chord excursion + M2 drift) + §IV.B Velocity-Augmented Manifold | 适中 |
| **5** | §IV.C **Proposition 4** + §IV.D Constructive Hyperparameter + §IV.E Relation to Prior + §V Experiments 起 + §V.A Setup + §V.B **Main Results** + Table 1 | **密** (6 sub) |
| **6** | Table 1 续 + §V.C **Design Study** Table 2 + §V.D Coarse Rates Table 3 + Figure 1 (V-A trajectory) | 适中 (含 2 tables + 1 figure) |
| **7** | §V.E Wall-time Table 4 + §VI Discussion + Limitations (L1 L2) + Sim-to-Real (i)(ii)(iii)(iv Webots) + §VII Conclusion + References 起 + Figure 2 (lshape overlay) | **密** (多 sub-point, 含 1 fig + 1 table) |

**评估**:
- ✅ Body 严格 1-7,Conclusion + Refs 起均 page 7,**100% 合规** AAAI-27 "7 pages technical content"
- ✅ Refs page 7末-8顶,**不计入 body**(无限 refs)
- ⚠️ **page 2, 5, 7 过密**(每页 4-6 subsection),reviewer 阅读疲劳风险。AAAI 审稿通常 1 reviewer scan 不超过 30 分钟,密 page 易被 skim 漏关键
- ✅ Tables 4 + Figures 2 数量适中,1-7 都有 visual anchor

## 2. 附录 / Supplementary 布局

### A. `supplementary_dtmargin.pdf` (2 pages standalone)

| Section | 内容 | 行数 |
|---|---|---|
| §1 DT-margin Heuristic Baseline | 引言 + Ablation Table + Δt sensitivity Table + α/h sensitivity Table + 1 figure(ablation bar) | ~80 |
| §2 **Statistical Significance** | Wilcoxon table (11 baselines × 15 pairs) + Friedman omnibus + McNemar (120 trials Webots) | ~25 |
| §3 Computing Infrastructure | CPU/RAM/OS + software stack + 200 CPU-hours + seeds 0-4 | ~20 |

### B. `repro_checklist.tex` (3 pages,`\input` 进 main 末尾 / page 9-10 of main PDF)

24 答案:General 3/3 yes / Theoretical 8/8 yes / Dataset relies-no + 6 NA / Computational **13/13 yes**(全 partial 已清)

### C. `supplementary_demo.mp4` + `_30s.mp4` (1.6 MB + 573 KB)

6-trial Webots demo concat,reviewer 可选完整版或 30s overview。

### D. 主仓 audit 文档(不投稿,作 transparency artefact)

- `TABLE1_AUDIT.md` — 36 cells × 12 method 数字对齐 audit
- `REFAIR_PUSH_RESULTS.md` — Push r=0.3 公平重跑 audit
- `WEBOTS_INFERENCE_RESULTS.md` — Webots 240-trial 详细数据

---

## 3. 4 个 Contributions 深度评估

### #1 Velocity-augmented manifold + Prop. 4

| 维度 | 分数 | 注释 |
|---|:---:|---|
| 理论新颖性 | **9/10** | "first discrete-time analogue of ATACOM" 是清晰 novelty claim;Cor. 1 (Δt→0 退回) + Cor. 2 (multi-obstacle) + 一阶 Taylor 近似回收 velocity-margin → 完整数学链 |
| 形式严谨性 | **8/10** | Prop. 4 有 3 步证明 + 显式 constructive 条件 + 计算 defect δ = 1.5Δt²a_max;但 main body 只 sketch,full proof 没在 supp 里(可加) |
| 实用价值 | **9/10** | (Δt, r, v_max) → (d_safe, α_0) 直接 prescribe,无需手 tune |

### #2 Failure-mode diagnosis (Prop. 1-2)

| 维度 | 分数 | 注释 |
|---|:---:|---|
| 解释力 | **9/10** | M1 (chord excursion) + M2 (drift) 各对应一个具体几何机制,实验数据(8 method 全 fail)印证 |
| 与方法 #1 关联 | **8/10** | 把 "why baselines fail" 链到 "why VA-ATACOM works" — 完整 narrative arc |

### #3 Five-design study

| 维度 | 分数 | 注释 |
|---|:---:|---|
| 隔离精度 | **9/10** | 两轴 (velocity-aware × tangent projection) × 5 design → 唯一 (✓, tangent cone) → 0/15 → 5/5 GO,清晰 |
| 防 reviewer attack | **8/10** | "velocity augmentation alone or projection alone 都不够" 直接回应 "VA-ATACOM 只是改进 ATACOM" 这种 minor incremental 批评 |

### #4 Empirical validation across sim and sim-to-real

| 维度 | 分数 | 注释 |
|---|:---:|---|
| Safety-Gym evidence | **10/10** | **15/15 GO, the only method**(audited);Wilcoxon p<0.001 vs every baseline;Friedman p=6.8e-16 |
| Sim-to-real evidence | **9/10** | Webots 240-trial × 3 worlds × 2 controllers;McNemar p=1.2e-7;filter never hurts (0/120 regressions);**lshape 154mm tightest corner** 0/20 deep |
| 缺 physical hardware | **5/10** | Path C 代码 ready 但未跑硬件(等用户);若 reviewer 严苛,会问 "Webots 是 simulator 不是 hardware",这是已知 limitation |

---

## 4. 中稿概率评估

### 4.1 Strength axis(为什么会中)

1. **Theory-empirical fit 紧密** — Prop. 4 不是 standalone theorem,而是与 main result (15/15 GO) 直接挂钩。reviewer 看 abstract 一眼:"theorem 是 ATACOM 离散对应物 + 唯一 15/15 GO" = 完整故事
2. **Audit transparency** — TABLE1_AUDIT.md 把每个 cell mapped to raw json dir,reviewer 复现实验时数字一致 100%。这是 conference 上少见的诚实
3. **Webots transfer 强化 sim-to-real**(P0-5 后 Figure 2 = lshape 154 mm corner,filter 0/20 deep)— reviewer 不能轻易 dismiss "only sim"
4. **Statistical rigor**(P1-1 后)— Wilcoxon p<0.001 / McNemar p=1.2e-7,checklist Computational 13/13 yes,是 AAAI 标的
5. **Reproducibility** — README + `reproduce_table1.sh --smoke`(2 min sanity)+ aggregate.py 一键算 Wilcoxon,reviewer 可直接验证

### 4.2 Risk axis(为什么会被拒)

| 风险 | 严重度 | 缓解措施 |
|---|:---:|---|
| **"只在 simulation,无 physical hardware"** | 中 | 已有 Webots 双 controller transfer + Path C 等硬件(注:实机不在 critical path,per memory);若 reviewer 严苛,Webots 是 rigid-body simulator 还是有说服力 |
| **"Prop. 4 假设可能 restrictive"** | 中 | Prop. 4 需 α₀Δt≤1 + d_safe ≥ (3/2)Δt²a_max + r ≥ d_safe+Δtvmax,3 个条件都 explicit 写出 + 物理量可测;但 reviewer 可能问 "是否在 non-circular obstacles / non-disk agent 仍 work" |
| **"VA-ATACOM 是 ATACOM + 简单 velocity term"** (minor incremental) | 中 | Contribution #3 (5-design study) 显式 isolate velocity augmentation alone OR tangent projection alone 都不够;Cor. 2 multi-obstacle 也是 non-trivial extension |
| **"DT-margin 已是 first-order approx,这只是补完二阶"** | 低 | DT-margin 14/15 → VA-ATACOM 15/15 看似 marginal 1-cell,但 (a) DT-margin 有 α 超参,VA-ATACOM 没;(b) lshape Webots filter on 0/20 vs DT-margin 没 hardware test;(c) statistical sig p=0.003 |
| **"page-2/5/7 密度高,某 sub 容易 skim 漏"** | 低 | 加 figure 引导阅读(已加 Fig 1 trajectory + Fig 2 lshape);abstract 已 hook 强 |
| **"Webots 不是 'real real'"** | 低 | Webots ODE rigid-body + raw noisy GPS,Cyberbotics 是 community 公认 sim-to-real bridge |

### 4.3 概率估计

基于 audit 后的 paper 状态(本次 commit `ee9bf8d`):

**估计录用概率**:

| Scenario | Conditional probability | 整体 |
|---|:---:|:---:|
| Reviewer 群均 fair + 主结果数字打动 | **60-70%** |  |
| Reviewer 偏 theoretical purity (want full Prop 4 proof in main paper) | -10% | |
| Reviewer 严要求 physical hardware | -10-15% | |
| Reviewer 误读 "only sim" 不看 Webots transfer | -5% | |
| **Final** | | **~55-65%** |

对比基线:AAAI 主 track 历年 acceptance rate 24-28%(2024-2026)。本 paper 经过 audit + Wilcoxon + Webots transfer + reproduce script,**显著高于平均录用率**(估 2-3 倍 baseline)。

### 4.4 主要不确定性

1. **Reviewer lottery**:1 个 negative reviewer(尤其偏 robotics hardware school)可能给 reject,2-3 positive 可救
2. **AAAI-27 主席选 area**:VA-ATACOM 跨 ML(safe RL)+ Robotics(CBF)。落 ML area chair 中稿率高一些(theoretical novelty 更重视);落 robotics 中稿率略低(常问 hardware)
3. **Concurrent submissions**:若 AAAI-27 投稿数 + 类似工作多,acceptance rate 下降

---

## 5. 还能做的提升(若想 push 到 70%+)

### P0 后续(若用户决定继续 polish)

1. **Conclusion 段加 numbers**(currently 模糊)— 让最后 1 段 cite 15/15 GO + p=1.2e-7 + 154mm corner specific numbers
2. **§IV-A 加 1 个 figure(M1/M2 visual geometric proof)** — 现在 M1/M2 全文字 + propositions,reviewer 容易漏 intuition
3. **Limitations 段加 "non-circular obstacles" + "non-disk agent" extension proof sketch** — preempt theoretical purity 批评
4. **§II 加 1 句 "Why VA-ATACOM ≠ HOCBF + velocity augmentation"** — preempt minor incremental 批评

### P1 后续(若有时间)

5. **加 "Pendulum / Quadrotor" 1 个非 mobile robot env 验证** — preempt "only mobile robot" 批评(估 ~20 h CPU)
6. **Path C 实机至少 demo 1 个 trial 视频** — 即使 1 个 demo,放进 supp video,把"Physical E-puck validation remains future work" 改成 "preliminary physical demonstration"

### P2 后续(锦上添花)

7. **§II 加 1 句 ACM/ICRA 类似工作比较 table** — "VA-ATACOM vs HOCBF vs DCBF vs ...." 1 行 table(已有,但可加 1 列 "first discrete-time invariance proof" Y/N)
8. **加 reviewer-facing FAQ supplementary section** — 预测 3-5 个常问 + 答(reviewer love when authors 已经回答 anticipated 问题)

---

## 6. 一句话结论

**当前 paper 在 audit-confirmed 数字 + 4-条 contributions + Webots transfer + statistical rigor + reproducibility checklist all-yes 的状态下,中稿概率约 55-65%,显著高于 AAAI baseline 24-28%。** 主要风险是 reviewer 偏好(theoretical vs robotics);最 leverage 的提升是 §IV-A 加 figure + Conclusion 加 numbers + (optional) Pendulum env 验证 + Path C 至少 1 demo trial。
