# Filter "说法" 升级方案：把 DT-ATACOM 重述为 principled 算法

> 动机：当前 framing 软肋——理论优雅的 baseline (ATACOM null-space, CBF-QP)
> 全失败，赢的 DT-ATACOM 核心是 stop-radius + 速度缩放，看似 heuristic。
> 审稿人会问 "principled 在哪？"。本文档记录两条升级路线，供后续扩展。
>
> 状态 (2026-05-21): ① 数学已验证不等式成立（见下），未写入 main_v2.tex。
> ② 仅记录方向，未推导。

---

## 选项 ①：把 stop-radius 重述为 velocity-adaptive discrete-time CBF（推荐）

### 已验证的核心引理

设 barrier `h(q) = ρ(q) − r_base`，safe set `S = {h ≥ 0}`，`r_base = r + d_safe`。
DCBF 条件 (Agrawal-Sreenath 2017)：`h(q_{k+1}) ≥ (1−γ_k) h(q_k)`，`γ_k ∈ [0,1]`。
若 `h(q_0) ≥ 0` 且每步成立 → `h(q_k) ≥ ∏(1−γ_i) h(q_0) ≥ 0` → forward invariance。

**DT-ATACOM 满足，分两区：**

- **Band 内 (filter active, Prop.4 Case 2/3)**：proj_tan 保证 `u_ρ ≥ 0`
  → 由 ρ_{k+1}² = ρ_k² + 2ρ_k Δt u_ρ + Δt²‖u‖² 得 `ρ_{k+1} ≥ ρ_k`
  → `h(q_{k+1}) ≥ h(q_k)`，即 `γ_k = 0`（不许衰减，强于标准 DCBF）。

- **Pass-through (Case 1)**：`h(q_{k+1}) ≥ h(q_k) − Δt·v_max`（三角不等式）；
  pass 条件 `ρ_k ≥ r_eff + d_danger` 给
  `h(q_k) = ρ_k − r_base ≥ r_base·α‖v_k‖ + d_danger ≥ Δt·v_max`（用 Prop.4 (12)₁）。
  取 `γ_k = Δt·v_max / h(q_k) ≤ 1` → DCBF 成立。

**结论**：DT-ATACOM 使 h 成为合法 discrete-time CBF，class-K rate `γ_k`
**状态/速度自适应**（band 内 →0, 远场 ≤1）。✓ 不等式成立。

### Framing 升级（连续→离散 CBF 的正确形式）

- 连续时间 CBF：`ḣ ≥ −α(h)`，常数/固定 class-K `α`。
- 朴素离散化 (DCM, Agrawal 2017)：`h_{k+1} ≥ (1−γ)h_k`，**常数 γ + 静态 barrier**
  → 我们的实验显示 DCM 仅 3/15（M1 chord excursion 让单步 h 下降超过 γh_k 预算）。
- **DT-ATACOM = 正确的离散时间 CBF**：velocity-inflated barrier
  `h̃ = ρ − r_base(1+α‖v‖)` + **velocity-adaptive γ_k**（band 内消失）+
  multi-step BRT 做 recursive-feasibility 证书。

整篇叙事从"我们的 heuristic 打败 principled 方法"升级为：
**"连续时间 CBF 在粗 Δt 失效；我们给出正确的离散时间 CBF 形式
(velocity-adaptive class-K)，并证明 forward invariance (Prop. 4)。"**
既 principled 又解释了为什么所有 baseline 失败。

### 写入 main_v2.tex 的改动（待执行）

1. §IV-B Component 1：把 VAM + 速度缩放重述为 "velocity-adaptive
   discrete-time CBF"；引入 barrier h、DCBF 不等式、上面的 Lemma。
2. §IV-C：Prop. 4 加一句 "equivalently, h is a discrete-time CBF with
   state-dependent rate γ_k"，把现有证明的 Case 1/2/3 映射到 γ_k 的两区。
3. §II Related Work：DCM 段强调 "constant-γ static-barrier DCBF fails;
   velocity-adaptive γ is the fix"。
4. 引 Agrawal2017 (已在 bib) + 可加 Zeng2021 (CBF-CLF-QP discrete) 做 pedigree。
5. Abstract/§I：把 "velocity-adaptive margin" 旁注 "(a velocity-adaptive
   discrete-time control barrier function)"。

代价：~半天写作 + 验证 Lemma 措辞；无需重跑实验（数据不变，只是重新解释机制）。

---

## 选项 ②：predictive safety filter (MPC) 框架（最严格，最重，暂缓）

### 思路

BRT lookahead (9 方向 × h 步 forward-sim) 本质是 Wabersich-Zeilinger 2021
predictive safety filter (PSF) 的**采样近似**：

> PSF: 求最小动作修正 u_safe = argmin ‖u − a‖² s.t. 存在 h 步可行轨迹
> {q_t} 满足 q_{t+1}=f(q_t,u_t), q_t ∈ S ∀t, q_h ∈ S_terminal（safe terminal set）。
> recursive feasibility ⟹ forward invariance（gold-standard 证书）。

我们的 9-direction sim 是把 "存在可行轨迹" 松弛成 "9 个 worst-case 方向都不撞"。

### 升级路线（若要做）

- 把 BRT 重写成真正的 MPC-QP：决策变量 = h 步动作序列，约束 = 每步 CBF
  不等式 + 终端 safe set，目标 = ‖u_0 − a‖²。求解得 u_safe = u_0*。
- "说法"：recursive feasibility of the QP ⟹ forward invariance（Prop. 4 的
  MPC 版本，更强且是标准结果）。
- 与 ① 兼容：终端约束用 ① 的 velocity-adaptive CBF。

### 代价 / 风险（为何暂缓）

- 工程量大：要实现 MPC-QP solver in the RL inner loop。
- 慢：CBF-QP 已是 1ms/call (Table 5)；h-step MPC-QP 更慢，可能 5-10ms。
- 与现有实验数据不兼容：现在的 BRT 是采样版，换 MPC-QP 要重跑全部
  DT-ATACOM cells（Goal/Push/MGoal × 5 seed + α/h sweep + Δt sweep）。
- 适合 ICRA 2027 deploy 版或 journal 扩展，不适合当前 AAAI 投稿。

### 若做，需记录的接口

- `safe_rl/reachability/sim_brt.py` → 新增 `mpc_psf.py`（MPC-QP 版）。
- 终端 safe set = velocity-adaptive CBF 0-superlevel set（接 ①）。
- Table 5 walltime 加 PSF-MPC 行对比 BRT sampling 的开销。

---

## ③ velocity-augmented 约束流形 + higher-RD ATACOM（Task #21, 进行中）

### 动机
vDCBF-QP（纯 kinematic 离散 CBF）已证伪：Safety-Gym Point 是 relative degree 2
(action→velocity→position)，单步约束 action 治不了 carried momentum（smoke C~17）。
正确形式 = velocity-augmented 约束流形：c_aug(q,v)=ρ−r_base−‖v‖²/(2a_max)≥0，
零水平集是 (q,v) 状态空间流形，ATACOM null-space projection 投到其切空间。

### a_max 标定（2026-05-21, T8_amax_calib）
Safety-Gym Point, 驱满推力加速再满刹，测 |Δv|/Δt：
- v_max ≈ 1.13 m/s (p95)
- a_max ≈ 0.94 m/s² (p95 可达减速度, max 1.08); 保守取 0.5–0.9
- **刹车距离 @ v_max = v²/(2a_max) ≈ 0.67 m >> r_base 0.25 m (2.7×)**

### 关键洞见（统一三组件）
真实 braking-distance margin = r_base + v²/(2a_max) = 0.92m @ 满速，远大于:
- VAM margin r_base(1+0.3·1)=0.325m → **VAM 是 braking 距离的线性近似（且严重低估）**
- 所以必须配 velocity-scaling（减速）+ BRT（forward-check）
- braking-distance CBF 天然强制"近障碍减速"：ρ↓ → 必须 v↓ 维持 c_aug≥0
- ATACOM null-space 失败因为只剥径向、不降速、满 momentum 刹不住

→ DT-ATACOM 三 heuristic 组件 = 这个 braking-distance 约束流形的 (线性近似 + 强制减速 + 采样近似)。统一 framing: **"DT-ATACOM = 速度增广约束流形上的离散时间安全投影"**。

### 待做
- [ ] 推导 c_aug 的 Jacobian + null-space projection（RD2: 动作影响 v，需 ḣ/HOCBF-style 或状态增广 ATACOM）
- [ ] 实现 filter (safe_rl/filters/manifold_rd2.py 或类似)
- [ ] Goal smoke（目标 cost << vDCBF-QP 的 ~17，最好接近 DistAdapt 2.9）
- [ ] competitive 再 5-seed × 3 task
- [ ] 写入论文统一 framing

## ④ braking-feasibility filter ★ 正面方法突破（2026-05-21）

三次证伪（null-space ATACOM / kinematic vDCBF-QP / RD2 brake_manifold）共同指向：
**model-based filter 需要 RL 不干净提供的 dynamics/frame 模型 → 失败；model-free 幅值缩放鲁棒**。
据此设计正面方法，保留 model-free 缩放但用**标定物理信号**驱动：

刹车可行性比 β = (ρ_min − r_base) / max(‖v‖²/(2a_max), ε)。
- β ≥ 1：能及时刹停 → pass through（不过保守）
- β < 1：刹不住 → scale = clip(β,0,1) 缩放 action 幅值 → MuJoCo damping 滑行减速 → v↓ → β 恢复

**model-free（只需 ρ,‖v‖,标定 a_max）+ frame-agnostic（缩幅值不投影）+ principled（刹车物理）**。

实现: safe_rl/filters/brake_feasibility.py; config: configs/safety/brake_feasibility.yaml; trainer dispatch 已加。

### Smoke 对比（Goal 20K, seed 0）
| filter | C | R | verdict |
|--------|---|---|---------|
| vDCBF-QP | ~17-19 | -1.7 | NO-GO |
| brake_manifold | 17.2 | -14.6 | NO-GO |
| **brake_feasibility** | **0.00** | **+0.78** | **GO** ★ |

20K 就 zero-cost + 正 reward（比 DT-ATACOM 0.84@200K 还干净）。
确认运行中: t9_brakefeas_goal Goal 200K × 5 seed (bg blq8dx9eq)。

### ★★ 200K 反转：brake_feasibility 证伪（2026-05-21）
20K smoke C=0.00 是**假阳性**（policy 慢、少触发 filter）。Goal 200K × 5 seed:
cost=[132.6, 315.1, 52.6, 68.2, 18.5] mean **117** 0/5 GO R=−10.7。
高速收敛 policy 下，**纯幅值缩放 action→0 抵消不了 carried momentum**，带速径向冲入 → 第四次证伪。
教训：**20K smoke 不可信，必须 200K 验证**。

### 真实张力（核心问题）
| 方法 | null-space 流形? | 200K |
|------|:---:|:---:|
| ATACOM null-space | ✓ | ✗ 0/15 (不减速) |
| vDCBF-QP | ✗ | ✗ |
| brake_manifold RD2 | ✓ | ✗ (frame 错: 用了cartesian, Point 实为 diff-drive) |
| brake_feasibility | ✗ | ✗ 117 |
| DistanceFilter (headline) | ✗ 启发式缩放 | ✓ 0.84 |

**唯一 work 的 (DistanceFilter) 没用约束流形；用流形的都不 work。** 这是论文范式一致性的真问题。
Action 实测为 diff-drive: action=[forward(沿heading θ), turn]，G(θ)=[[cosθ,0],[sinθ,0]]。
brake_manifold 失败因为用 cartesian。

### ★★★ 成立！brake_manifold (diff-drive) Goal 200K × 3: cost=[0,0,0] 3/3 GO R=1.46（2026-05-21）
修成 diff-drive G(θ) 后（约束 forward·(v·ê_θ) ≤ a_max(n̂·v+α₀h)，turn 自由），**200K 验证**:
- Goal: 0.00 cost 3/3 GO, R=1.46 → **比 DT-ATACOM (0.84) 更安全 + 比它 reward 更高 = 严格更优**
- 这是真正含约束流形（velocity-augmented braking manifold 的 null-space/tangent projection）的正面方法

**张力解决 + 强叙事确立**：
- brake_feasibility（无投影纯缩放）117 失败 vs brake_manifold（有 tangent projection + 正确 frame）0.00 成功
- → **证明约束流形切空间投影是关键**（不是可有可无）
- 标准 ATACOM 静态流形 c(q) 失败因为不含速度（不减速）；velocity-augmented c_aug(q,v) 治根
- 完整故事：连续CBF/null-space ATACOM 失效(M1/M2) → 纯缩放也失效(无投影,momentum冲入) → **正确解 = 速度增广 braking-distance 约束流形 + 正确 frame 的 null-space projection**

全面验证完成 (2026-05-21): brake_manifold (diff-drive, 无 BRT) 200K × 5:
| Task | brake_manifold | DT-ATACOM | |
|------|---|---|---|
| Goal | **0.02 (5/5)** R1.44 | 0.84(5/5) | 碾压 40× |
| Push | 85.4 (0/5) | 10.58(2/5) | 灾难差 |
| MultiGoal | 8.46 (2/5) bimodal | 2.59(4/5) | 差 |
| 总 | 7/15 | 11/15 | 总体不如 |

**判定: brake_manifold 是 Goal 专才（核心静态避障 dominant），不泛化。**
- Push 灾难: diff-drive 只约束 forward, 推箱机动受限, box 撞 hazard 没进约束。
- MultiGoal bimodal: goal-switching transient 单步 braking 不够 → 正是 BRT lookahead 能 anticipate 的。
- **反向印证 DT-ATACOM 两组件互补**: 流形投影治 Goal, BRT 治 momentum-transient。

下一步 (option b): brake_manifold + sim_brt 组合（principled braking 流形投影替换 heuristic stop-radius, 保留 BRT shaping）。
跑 Goal/Push/MGoal × 5 = 15 runs 200K (bg, ETA ~40min)。若 MGoal/Push 恢复 → 既含约束流形又泛化的新 headline。
filter: safe_rl/filters/brake_manifold.py (diff-drive 版); config: brake_manifold.yaml。

## 决策记录

- 2026-05-21: ① 数学验证通过（Lemma 不等式成立）。
- 2026-05-21: **① 已写入 main_v2.tex 并编译通过**（9 页, 技术正文仍精确 7 页, 0 warnings）：
  - §4.2 开头加 "A discrete-time CBF view" 段（barrier h + DCBF 条件 + DCM constant-γ 失败 + velocity-adaptive γ_k）
  - §4.3 Prop.4 加 "Remark (DCBF interpretation)"（Case 2-3 → γ_k=0, Case 1 → γ_k=Δt v_max/h_k ≤1）
  - §II DCM 段重写（constant-γ vs velocity-adaptive）
  - abstract 加 velocity-adaptive DCBF 句; conclusion 改 "realising a velocity-adaptive discrete-time control barrier function"
  - 为腾页裁了: §V-G 经验建议段、§6.2 sim-to-real 三因素、§V-D basin 段、§V-E walltime prose、L4、conclusion——均收紧未删信息。
  - framing 升级完成: "heuristic wins" → "我们指出连续时间 CBF 在粗 Δt 失效, 给出正确的离散时间 CBF 形式 (velocity-adaptive class-K)"。
- ② predictive safety filter (MPC) 仍留 ICRA/journal 扩展, 见上。
