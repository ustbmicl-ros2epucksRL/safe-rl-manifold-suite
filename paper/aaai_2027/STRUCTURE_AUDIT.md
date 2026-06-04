# Paper Structure Audit — main + supplementary

**Date**: 2026-06-03
**Status**: ✅ architecture sound;1 minor 改善建议(可做可不做)

---

## 1. Main paper structure(7 sections + 16 subsections,7 page body)

```
§I  Introduction                                    page 1-2
    §I.1 Contributions (4 条)
§II Related Work                                    page 2-3
    §II.1 Tangent-Projection Safety Filters
    §II.2 Set-Based and Predictive Safety Methods
    §II.3 Lagrangian Soft Constraints
§III Preliminaries                                  page 3-4
    §III.1 Constrained MDP Setting
    §III.2 ATACOM Continuous-Time Theorem
    §III.3 Circular Obstacle Geometry
§IV VA-ATACOM Algorithm                             page 4-5
    §IV.A Why Continuous-Time Filters Fail (含 Prop 1-2 M1/M2)
    §IV.B VA-ATACOM: The Velocity-Augmented Manifold
    §IV.C Proposition 4: Discrete-Time Forward Invariance + Cor 1/2
    §IV.D Constructive Hyperparameter Selection
    §IV.E Relation to Prior Work
§V Experiments                                      page 5-7
    §V.A Experimental Setup
    §V.B Main Results (Table 1)
    §V.C Design Study: Isolating Velocity Augmentation (Table 2)
    §V.D Robustness to Coarse Control Rates (Table 3)
    §V.E Computational Overhead (Table 4)
§VI Discussion                                       page 7
    §VI.1 Limitations (L1 L2 L3 L4)
    §VI.2 Sim-to-Real Considerations (i ii iii iv-Webots)
§VII Conclusion                                      page 7
```

### 评估

| 维度 | 状态 | 备注 |
|---|:---:|---|
| Standard ML structure | ✅ | 7 sections 完全符合 AAAI 主 track 惯例(Intro / Related / Prelim / Method / Exp / Discuss / Conclusion) |
| Method section 深度(§IV)| ✅ | 5 subsections + Prop 1-4 + Cor 1-2,理论 backbone 全 |
| Experiments section 深度(§V)| ✅ | 5 subsections + 4 tables,fair comparison |
| Discussion 深度(§VI)| ✅ | Limitations L1-L4 全覆盖 + Sim-to-Real(i-iv 4 维度) |
| Conclusion 含具体数字 | ✅ | 15/15 GO + p<0.001 + 1/120 + p=1.2e-7 + "closing the gap" |

### Page density(每 page 多少 sub-section 起始/跨页)

| Page | density | 评 |
|:---:|:---:|---|
| 1 | 2 | normal |
| 2 | 6 | **密**(Contributions + Related Work 3 sub) |
| 3 | 7 | **极密**(Preliminaries 3 + IV 起 + IV-A) |
| 4 | 2 | sparse |
| 5 | 5 | 密(IV-C Prop 4 + IV-D + IV-E + V + V-A) |
| 6 | 3 | normal |
| 7 | 7 | **极密**(V-D + V-E + VI + VI-A + VI-B + VII) |

**Page 3 + 7 极密** — reviewer scan 易 miss 。但内容 critical(Prop 4 + final results)无法压缩。**Trade-off 接受**。

---

## 2. Supplementary structure(5 sections,3 page standalone)

```
§1 The DT-margin Heuristic Baseline                page 1
    Ablation (Tab. 1) + Fig. 1 (ablation bar)
    Δt sensitivity (Tab. 2)
    α/h sensitivity (Tab. 3)
§2 Statistical Significance                          page 2
    Safety-Gymnasium Wilcoxon table (11 baselines)
    Webots transfer McNemar table (120 trials)
    + Friedman omnibus
§3 Generalisation Beyond Point-Robot Dynamics       page 2
    Car Goal 4-method table + 根因诊断 + paper-impact framing
§4 Anticipated Questions                             page 2-3
    Q1 Webots vs real / Q2 novelty over ATACOM /
    Q3 DT-margin marginal / Q4 Prop 4 assumptions /
    Q5 no GPU
§5 Computing Infrastructure                          page 3
    Hardware spec + Software stack + Compute budget
```

### 评估

| 维度 | 状态 |
|---|:---:|
| Section 顺序 logical | ✅ DT-margin context → stats validate → honest scope → FAQ preempt → infra |
| 每 section 自包含 | ✅ 不依赖 main paper specific page numbers |
| Cross-refs to main | ✅ 用 plain text "Table 1 of the main paper",AAAI sections 无 number 不引用 |

---

## 3. Cross-references / Citations 健康

| 指标 | 数 | 评 |
|---|:---:|---|
| `\label{}` 总数 | 27 | 含所有 sections / equations / tables / figures |
| `\ref{}` 引用 | 20 | 正常密度 |
| **Broken refs** | **0** | ✅ 全部 resolve |
| `\cite{}` 引用 | 26 | 跨 29 bib entries |
| Unused labels | 7 | minor(eq:m1 / eq:m2 / fig:webots-overlay 等 anchor,不影响阅读)|

---

## 4. Tables / Figures / Algorithms 分布

| 类 | Main | Supplementary |
|---|:---:|:---:|
| Tables | **4**(Table 1 main / Table 2 design / Table 3 coarse / Table 4 wall-time)| **6**(DT-margin × 3 + Wilcoxon + McNemar + Car)|
| Figures | **2**(Fig 1 trajectory / Fig 2 lshape overlay)| 1(ablation bar)|
| Algorithms | **1**(Algorithm 1 VA-ATACOM step)| 0 |

### 评估

- **Main 4 tables + 2 figures + 1 algorithm** = AAAI ML paper 典型密度
- Table 1 (main results 12 method × 3 task) 是 anchor
- Table 2 (5-design study) isolate velocity augmentation
- Table 3 (Δt sweep) prove Prop 4 prediction
- Table 4 (wall-time) preempt "too slow" 批评
- Figure 1 visualises M1/M2 vs VA-ATACOM mechanism (§V cross-ref + §IV-A forward-ref)
- Figure 2 visualises 240-trial Webots transfer

---

## 5. 可改善但**非必要**的 minor

1. **`\eqref` 用 `\ref`** — paper 现在 0 eqref, 20 ref. 严格 LaTeX 推荐 equation 用 `\eqref` 自动加括号。但 AAAI 多 paper 都直接用 `\ref` 不影响 reviewer 阅读。
2. **fig:webots-overlay unused label** — Figure 2 在 §VI 但 §VI(iv) Webots 段没 explicit `Fig.~\ref{fig:webots-overlay}` cite。可加 1 处 forward ref(类似 §IV-A Fig 1 ref)— **但 figure 在 page 7 同页 reviewer 一眼就看到,加 ref redundancy**。
3. **§VI 内部小标 §VI.A Limitations + §VI.B Sim-to-Real Considerations** — 当前 2 subsection,Limitations 内还有 L1-L4 + Sim-to-Real 内还有 (i)-(iv)。可加更深 \subsubsection,但 AAAI 历年 paper 多 2-level subsection 够用。
4. **§IV.A 含 M1 / M2 两 sub-subsection** — 当前用 `\subsubsection{M1: ...}`,渲染 OK,reviewer 习惯。

---

## 6. 结论

| 维度 | 评分 |
|---|:---:|
| Section 选择 | **9/10** standard ML structure |
| Subsection 深度 | **9/10** appropriate granularity |
| Page 平衡 | **7/10** (page 2/3/7 略密,但内容均 critical) |
| Cross-ref 健康 | **10/10** 0 broken |
| Main vs Supp 划分 | **9/10** transparency 进 supp,argument 进 main |
| Tables/Figures 分布 | **9/10** main + supp 总 10 tables + 3 figures + 1 algorithm 充足 |

**总评:架构 sound,无需重大调整。**Minor 改善 4 项均 cost > benefit,**保持现状**。
