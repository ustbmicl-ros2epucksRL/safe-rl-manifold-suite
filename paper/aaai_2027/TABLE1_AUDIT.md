# Table 1 (Main Results) 数字 audit — 2026-05-31

**目的**:逐 cell 把论文 Table 1 cited 数字与 `safe-rl-2027/runs/` raw json 对一遍,catch stale/wrong cite。每个 cell 给出:
- paper 写的数字
- 我找到的 `run_eval.json` 实测数字(5 seed mean)
- 是否 match(✓ / ✗ / ~)
- 如果 ✗,指明正确 dir 或建议 action

**Audit threshold**:|Δ| < 0.06 = ✓,|Δ| < 0.5 = ~,否则 ✗。

---

## Table 1 cells (12 methods × 3 tasks + GO/15)

| Method | Cell | Paper | Measured | dir | 状态 |
|---|---|---:|---:|---|:---:|
| PPO baseline | Goal | 22.8 | 22.77 | `phase3_t1_none/goal_*` | ✓ |
| PPO baseline | Push | 46.9 | 46.93 | `phase3_t1_none_push` | ✓ |
| PPO baseline | MGoal | 58.5 | 58.52 | `phase3_t1_none_multigoal` | ✓ |
| ATACOM (null) | Goal | 17.0 | 28.72 | `phase3_t1_atacom` | **✗ Δ+11.7** |
| ATACOM (null) | Push | 66.3 | 66.30 | `refair_push_atacom` | ✓ |
| ATACOM (null) | MGoal | 50.5 | 84.27 | `phase3_t2x_atacom_multigoal` | **✗ Δ+33.8** |
| ATACOM-VD | Goal | 27.5 | 27.51 | `phase3_t3_vd_brt_goal` | ✓ |
| ATACOM-VD | Push | 57.9 | 57.87 | `refair_push_atacom_vd` | ✓ |
| ATACOM-VD | MGoal | 65.0 | 64.96 | `phase3_t3_vd_brt_multigoal` | ✓ |
| ATACOM-S | Goal | 18.3 | 18.29 | `phase3_t3s_brt_goal` | ✓ |
| ATACOM-S | Push | 11.5 | 11.45 | `refair_push_atacom_s` | ✓ |
| ATACOM-S | MGoal | 35.8 | 32.89 | `phase3_t3s_brt_multigoal` | ~ Δ−2.9 |
| ATACOM-LA | Goal | 37.7 | 37.74 | `phase3_t3la_brt_goal` | ✓ |
| ATACOM-LA | Push | 24.2 | 24.18 | `refair_push_atacom_la` | ✓ |
| ATACOM-LA | MGoal | 40.8 | 40.80 | `phase3_t3la_brt_multigoal` | ✓ |
| HOCBF | Goal | 10.4 | 10.42 | `phase3_t6_hocbf_brt_goal` | ✓ |
| HOCBF | Push | 7.4 | 7.41 | `refair_push_hocbf` | ✓ |
| HOCBF | MGoal | 19.0 | 28.49 | `phase3_t6_hocbf_brt_multigoal` | **✗ Δ+9.5** |
| DCM | Goal | 13.7 | 13.67 | `phase3_t4_dcm_brt_goal` | ✓ |
| DCM | Push | 11.9 | 11.90 | `refair_push_dcm` | ✓ |
| DCM | MGoal | 47.1 | 47.10 | `phase3_t4_dcm_brt_multigoal` | ✓ |
| Predictive ATAC | Goal | 15.4 | 20.94 | `phase3_t1_predictive_atacom` | **✗ Δ+5.5** |
| Predictive ATAC | Push | 12.9 | 12.94 | `refair_push_predictive_atacom` | ✓ |
| Predictive ATAC | MGoal | 36.8 | — | (no dir found) | **? 待 dig** |
| CBF-QP | Goal | 22.3 | — | (no dir found) | **? 待 dig** |
| CBF-QP | Push | 29.4 | 29.36 | `refair_push_cbf_qp` | ✓ |
| CBF-QP | MGoal | 27.7 | — | (no dir found) | **? 待 dig** |
| PPO-Lag | Goal | 35.6 | 35.60 | `phase3_t9_ppolag_goal` | ✓ |
| PPO-Lag | Push | 38.3 | 38.26 | `phase3_t9_ppolag_push` | ✓ |
| PPO-Lag | MGoal | 76.0 | 75.97 | `phase3_t9_ppolag_multigoal` | ✓ |
| **DT-margin** | Goal | 0.84 | **0.836** | `phase3_t1_ours` | ✓ |
| **DT-margin** | Push | 3.36 | 3.36 | `refair_push_distance_adaptive` | ✓ |
| **DT-margin** | MGoal | 2.59 | **2.588** | `phase3_t2_ours_multigoal` | ✓ |
| **VA-ATACOM** | Goal | 0.00 | 0.000 | `t9_brakemanifold_brt_goal` | ✓ |
| **VA-ATACOM** | Push | 0.00 | 0.00 | `refair_push_brake_manifold` | ✓ |
| **VA-ATACOM** | MGoal | 0.40 | 0.400 | `t9_brakemanifold_brt_multigoal` | ✓ |

**Mismatch summary**(13 cells 中 4 个 ✗ + 3 个 ? 待 dig):
- ATACOM (null) Goal: paper 17.0, measured 28.7 → **paper 数字源不明,dir `phase3_t1_atacom` 给 28.7**
- ATACOM (null) MGoal: paper 50.5, measured 84.3 → 同上
- HOCBF MGoal: paper 19.0, measured 28.5(5 seeds: 19.0/35.6/20.8/42.1/25.0)
- Predictive ATAC Goal: paper 15.4, measured 20.94
- Predictive ATAC MGoal: paper 36.8, dir not found
- CBF-QP Goal/MGoal: paper 22.3/27.7, dir not found

---

## GO/15 总计 audit

| Method | measured GO | paper GO | Δ | 影响 |
|---|---:|---:|---:|---|
| PPO baseline | 2 | 2 | ✓ | — |
| ATACOM (null) | 0 | 1 | −1 | Goal/MGoal 不一致 → GO 减 |
| ATACOM-VD | 0 | 0 | ✓ | — |
| ATACOM-S | 1 | 2 | −1 | MGoal 32.9 NO-GO,paper 35.8 NO-GO same;Goal 18.3 NO-GO 一致 → paper 多 1 GO 来源不明 |
| ATACOM-LA | 1 | 1 | ✓ | — |
| HOCBF | 4 | 5 | −1 | MGoal 不一致 |
| DCM | 3 | 5 | −2 | DCM Push GO 我算 2/5,Goal 1/5,MGoal 0/5 → 3/15。paper 5/15 出处不明 |
| Predictive ATAC | ≥3 | 5 | ≥−2 | MGoal dir 缺 |
| CBF-QP | ≥1 | 2 | ≥−1 | Goal/MGoal dir 缺 |
| PPO-Lag | 1 | 1 | ✓ | — |
| DT-margin | 13 | 14 | −1 | dir 全对、cost 全对,GO 我算 5+4+4=13 paper 14 → 哪个 cell GO 算错? |
| **VA-ATACOM** | 15 | 15 | ✓ | **主结论安全** |

---

## 关键 paper claim 是否受影响?

paper 写:
> "every published method fails the GO threshold ($\bar{C} \le 5$) on at least one task; **their best total reaches 5/15 cells (HOCBF, DCM, Predictive ATACOM)**"

按 audit:
- HOCBF: 4/15(不是 5)
- DCM: 3/15
- Predictive ATACOM: ≥3/15(MGoal dir 缺)
- 实际 best baseline 是 **HOCBF 4/15** 或 **DT-margin 13/15**(若 DT-margin 算 baseline;但 paper 现把 DT-margin 列为 VA-ATACOM 的一阶近似,不是 baseline)

→ 若保守按 audit:**"their best total reaches 4/15 cells (HOCBF)"** 比 paper 现稿"5/15 (HOCBF, DCM, Predictive)"更 conservative + 更 accurate。VA-ATACOM 的 15/15 = the only method 主结论**完全不动摇**(从 14 vs 15 → 11 vs 15 反而对比更强)。

---

## 待 user 决定

**Option A: 接受 audit, 修 paper Table 1 数字 + GO totals**

  - 把 4 个 ✗ cell 用 measured 数字替换(若 paper 数字 source 找不到)
  - 把 3 个 ? cell 补 dir 重 audit;若无 dir → 重跑 5 seeds 200K(每 cell ~ 2-3 h CPU)
  - 修 GO 总计 + 修 "best 5/15" 叙述为 "best 4/15"
  - **风险**:VA-ATACOM 的对比反而更强了(15 vs 4/15);但需要 1-2 h paper 改写

**Option B: 维持现状,只 fix dir 找不到的 3 个 cell**

  - 重跑 CBF-QP Goal/MGoal + Predictive ATAC MGoal(~10 h CPU)
  - paper 数字保留(假设来源是其他 ablation 或不同 BRT config)
  - **风险**:数字源不可复现,reviewer 复现实验时可能 catch

**Option C: 投稿前先不 fix,只在 supplementary 里加 audit doc**

  - 标 paper Table 1 数字为"as run on the configurations described in Sec. X.X"
  - 把 audit doc 当 transparency artifact 投 supplementary
  - **风险**:reviewer 看 supplementary 时数字对不上 main paper

**我推荐 A** — 数字诚实,VA-ATACOM 对比反而 *更强* (15 vs 4 cells); 也是 reviewer 一查就 catch 的潜在炸点。

(audit 报告耗时 ~30 min;Option A 修 paper 估 1-2 h;Option B 重跑估 ~10 h CPU + 1 h paper update。)
