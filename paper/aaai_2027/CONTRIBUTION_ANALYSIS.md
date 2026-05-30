# AAAI 2027 论文贡献结构 + 审稿人攻击面分析

**日期**:2026-05-30
**对象**:`main_v2.tex` 当前稿(10 页,经过 framing B 重构 + 5/28 Push 公平重跑 + 5/29 Webots §VI(iv)插入 + 5/30 EKF 去依赖)

---

## 1. 贡献结构(narrative 链路)

```
                连续时间 CBF/ATACOM 理论
                          │
                     粗 Δt 失效
                          │
                  ┌───────┴───────┐
            贡献 2:M1/M2 诊断    每个 prior filter 都崩
                  └───────┬───────┘
                          │
            贡献 1:速度增广流形 + Prop 4
            (Δt→0 退回 ATACOM;velocity-adaptive
            margin = 其一阶近似 → 解释启发式)
                          │
                  ┌───────┴───────┐
            贡献 3:五设计研究    贡献 4:Safety-Gym 15/15 GO
            (velocity-aug × proj   (唯一全 GO,fair radius match)
             缺一不可)                       │
                                             ▼
                                     §VI(iv) Webots 验证
                                     (sim-to-platform 转移,
                                      P-ctrl + raw GPS + no EKF)
```

## 2. 头号卖点(if asked "ONE thing")

**Prop 4 是 ATACOM 连续不变性定理的首个离散时间对应物**,并:

- **构造性**:给 $(\Delta t, r, v_{\max})$ 直接出 $(d_\text{safe}, \alpha_0)$ 的范围
- **优雅退回**:$\Delta t \to 0$ 时退回原 ATACOM Theorem(Corollary 1)
- **解释启发式**:广用 velocity-adaptive margin $r(1+\alpha\|\boldsymbol v\|)$ 是 Prop 4 的**一阶 Taylor 近似**

---

## 3. 4 条 Contributions 的审稿人攻击面

### 🟢 Contribution 1:Prop 4 / 速度增广流形 — **核心强,但 novelty 防守需打磨**

| | |
|---|---|
| 优势 | Constructive 离散时间定理 + Corollary 1 退回连续 + 启发式是其一阶近似(三位一体) |
| 软肋 | "vs HOCBF (Xiao 2021, relative degree 2) + DCBF (Agrawal 2017, constant γ static barrier) 的差异"现稿讲得不够 |
| 真正的差别 | (a) braking 项把 DCBF 的 constant-γ 变成**state-dependent rate** (b) Prop 4 给**闭式 discretisation defect** $\delta = \tfrac{3}{2}\Delta t^2 a_{\max}$ 与**constructive** 条件;HOCBF/DCBF 的对应参数留作 hyperparameters |
| 行动项 | §II-A 加 1 段正面对比,把上述 (a)(b) 显式写出来 |

### 🟡 Contribution 2:M1/M2 诊断 — **作为问题陈述 OK,作为独立贡献偏轻**

| | |
|---|---|
| 优势 | Props 1-2 给出量化阈值 $\Delta t\|u\| \sim \sqrt{d_\text{safe}\cdot r}$ 与 M2 线性累积 |
| 软肋 | 离散 CBF 圈本来就常提 chord effect,被说"再证一遍"风险中 |
| 建议 | Contribution list 里**降权**(合并入 Contribution 1 作 motivation 子项),或如不动则在描述中强调 M2 的线性累积是首次显式量化 |

### 🟢 Contribution 3:五设计研究 — **强,二维 ablation 很说服**

| | |
|---|---|
| 优势 | velocity-aware × projection 二维分离,只有右下角 (VA-ATACOM) work——这就是为什么"精确形式必要"的核心证据 |
| 软肋 | 5 行表里 CBF-QP 不合二维结构(它是 QP 投影);纯缩放 117.4 过于悬殊有 strawman 嫌疑 |
| 建议 | (a) 解释纯缩放为何 117.4("shrinking action magnitude leaves carried momentum unchanged"一句即可)(b) CBF-QP 划入"QP-based"行单独标注 |

### 🟡 Contribution 4:Safety-Gym 15/15 GO — **headline 漂亮,但 DT-margin 14/15 是软肋**

| | |
|---|---|
| 优势 | 唯一 3 任务全 GO;fair-radius matching 方法学修正(benchmark 长期 bug) |
| 软肋 | **DT-margin 也 14/15,只差 1 个 Push seed**;审稿人:"heuristic 离精确这么近,why bother with the theory?" |
| Reframe | DT-margin **不是独立 baseline 而是 VA-ATACOM 的一阶 Taylor 近似**;它 14/15 恰恰是**理论的实证侧证**——线性化已经拿走大部分性能,残留 1 seed 失败正在 quadratic braking 主导的 regime |
| 行动项 | 在 §V 主结果段 + Table 1 caption 把上述 reframe 显式说出来 |
| 其他风险 | 只 Safety-Gymnasium Point Robot 一个家族 → 单一 benchmark;Webots §VI(iv) 帮了点但只一个 corridor |

### 🟢 §VI(iv) Webots E-puck — **加分项,但不是正式贡献**

| | |
|---|---|
| 优势 | 真物理引擎 + raw GPS noise + 无 EKF 仍 0/20 deep penetration;支撑 "physical-units transfer" 主张 |
| 软肋 | (a) 用 P-controller 不是训练好的 PPO policy(Path B 未做)(b) 一个 corridor world |
| 建议 | 不动也行;若有余力可补 dense/complex world 1 行或上 Path B |

---

## 4. 缺失项(审稿人会问的)

| 缺口 | 严重度 | 估计成本 |
|---|---|---|
| **§II 加一段 vs HOCBF/DCBF 硬正面比较** | 🔴 高 | 1-2 小时写作 |
| **DT-margin reframe 成"理论的实证侧证"** | 🔴 高 | 30 分钟措辞 |
| **PPO policy on Webots(Path B)** | 🟠 中-高 | 1-2 天工程 |
| **多 Webots world 行**(dense / complex) | 🟠 中 | 30 分钟–半天 |
| **Prop 4 多障碍 corollary 显式化** | 🟡 中 | 写半页 |
| **真实硬件**(诚实承认 future work)| 🟢 低 | — |

---

## 5. 我的实诚判断

| 维度 | 评分 | 备注 |
|---|---|---|
| 理论原创性 | **B+** | Prop 4 不错;vs HOCBF/DCBF 差异化没讲透 |
| 实验强度 | **A−** | 15/15 GO 抢眼;DT-margin 14/15 是软肋,需 reframe |
| 写作清晰度 | **A−** | 经过多轮改写,framing 已 tight |
| 范围宽度 | **B** | 单 agent / 单 benchmark 家族 / 单 Webots world |
| Soundness | **A** | F2 修正后无明显事实错误 |

**总评**:**Borderline-accept**,大概**录用概率 35–55%**。

---

## 6. 性价比改进路线(按 ROI 排序)

| # | 改进 | 影响 | 工作量 |
|---|---|---|---|
| **1** ⭐ | §II 加 vs HOCBF/DCBF 硬比较段 → Contribution 1 B+→A− | 高 | 1-2h |
| **2** ⭐ | DT-margin 在 §V 主结果 + Table caption reframe 成"一阶近似实证" | 高 | 30 min |
| 3 | Webots 加 dense/complex world 1 行 | 中 | ~半天 |
| 4 | Path B(PPO on Webots) | 中-高 | 1-2 天 |
| 5 | 多障碍 Prop 4 corollary 显式写 | 中 | 写半页 |

**最低门槛行动**:1 + 2 完成即可把贡献 1 和 4 的明显防守漏洞补上,概率推到 45–60%。

---

## 7. 不在 paper 范围内(诚实划界)

- ❌ 硬件验证(留 future work,§VI(iv) 末尾明说)
- ❌ Multi-body / Push-with-box 多体扩展(单体单障碍 manifold,Prop 4 假设)
- ❌ 多机器人(单 agent 范围)
- ❌ PPO policy on Webots(Path B,未做)
