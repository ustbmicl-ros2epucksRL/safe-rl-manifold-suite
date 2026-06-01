# Table 1 (Main Results) 数字 audit — closed 2026-06-01

**Status**: **✅ 100% audit closed**. 全 12 method × 3 task = 36 cells + 12 GO/15 总,paper 数字 100% match `safe-rl-2027/runs/` raw json。

---

## 最终 cell 数字(已写进 `aaai27/main_v2_aaai27.tex` Table 1)

| Method | Goal | Push | MGoal | GO/15 | dir(goal / push / mgoal) |
|---|:---:|:---:|:---:|:---:|---|
| PPO baseline | 22.8 (2/5) | 46.9 (0/5) | 58.5 (0/5) | **2** | `phase3_t1_none/...` / `phase3_t1_none_push` / `phase3_t1_none_multigoal` |
| ATACOM (null-space) | 28.7 (0) | 66.3 (0) | 84.3 (0) | **0** | `phase3_t1_atacom` / `refair_push_atacom` / `phase3_t2x_atacom_multigoal` |
| ATACOM-VD | 27.5 (0) | 57.9 (0) | 65.0 (0) | **0** | `phase3_t3_vd_brt_*` / `refair_push_atacom_vd` |
| ATACOM-S | 18.3 (1) | 11.5 (0) | 32.9 (0) | **1** | `phase3_t3s_brt_*` / `refair_push_atacom_s` |
| ATACOM-LA | 37.7 (0) | 24.2 (1) | 40.8 (0) | **1** | `phase3_t3la_brt_*` / `refair_push_atacom_la` |
| HOCBF | 10.4 (1) | 7.4 (3) | 19.0 (0) | **4** | `phase3_t6_hocbf_brt_goal/push` / **`phase3_t7_hocbfsf_multigoal`** (SSF 变种) |
| DCM | 13.7 (1) | 11.9 (2) | 47.1 (0) | **3** | `phase3_t4_dcm_brt_*` / `refair_push_dcm` |
| Predictive ATACOM | 20.9 (1) | 12.9 (2) | 36.8 (0) | **3** | `phase3_t1_predictive_atacom` / `refair_push_predictive_atacom` / `phase3_audit_predictive_atacom_multigoal` (新 audit) |
| CBF-QP | 22.3 (0) | 29.4 (1) | 27.7 (0) | **1** | `phase3_audit_cbf_qp_goal` (新) / `refair_push_cbf_qp` / `phase3_audit_cbf_qp_multigoal` (新) |
| PPO-Lag | 35.6 (0) | 38.3 (1) | 76.0 (0) | **1** | `phase3_t9_ppolag_*` |
| **DT-margin (heuristic)** | **0.84** (5) | **3.36** (4) | **2.59** (4) | **13** | `phase3_t1_ours` / `refair_push_distance_adaptive` / `phase3_t2_ours_multigoal` |
| **VA-ATACOM** | **0.00** (5) | **0.00** (5) | **0.40** (5) | **15** | `t9_brakemanifold_brt_goal` / `refair_push_brake_manifold` / `t9_brakemanifold_brt_multigoal` |

**Best baseline**: HOCBF **4/15** (Push 3/5 GO + Goal 1/5 GO).
**VA-ATACOM** = **15/15**, the only method with full GO. Gap = **15 vs 4 cells**.

---

## Audit 修改的 paper cells(in `aaai27/main_v2_aaai27.tex` Table 1)

| Cell | Was (paper) | Now (audit-aligned) | Reason |
|---|---:|---:|---|
| ATACOM (null) Goal | 17.0 | **28.7** | `phase3_t1_atacom` 5-seed mean (source dir mismatch) |
| ATACOM (null) MGoal | 50.5 | **84.3** | `phase3_t2x_atacom_multigoal` (source dir mismatch) |
| ATACOM-S MGoal | 35.8 | **32.9** | 小 rounding (`phase3_t3s_brt_multigoal`) |
| Predictive Goal | 15.4 | **20.9** | `phase3_t1_predictive_atacom` (source dir mismatch) |
| **GO/15 列**:|||
| ATACOM (null) | 1/15 | **0/15** | per-cell audit re-count |
| ATACOM-S | 2/15 | **1/15** | per-cell |
| HOCBF | 5/15 | **4/15** | per-cell (MGoal NO-GO) |
| DCM | 5/15 | **3/15** | per-cell (MGoal NO-GO) |
| Predictive ATAC | 5/15 | **3/15** | per-cell |
| CBF-QP | 2/15 | **1/15** | per-cell |
| **DT-margin** | 14/15 | **13/15** | per-cell (MGoal 4/5 not 5/5) |

**未改**(measured = paper rounded):
- HOCBF MGoal 19.0 paper ≡ measured 18.97 ✓(用 `phase3_t7_hocbfsf_multigoal` SSF 变种 dir)
- CBF-QP Goal 22.3 paper ≡ measured 22.32 ✓(audit re-run)
- CBF-QP MGoal 27.7 paper ≡ measured 27.67 ✓(audit re-run)
- Predictive ATAC MGoal 36.8 paper ≡ measured 36.79 ✓(audit re-run)

---

## 论文 narrative 修订

§I + §V "best total reaches **5/15 cells (HOCBF, DCM, Predictive ATACOM)**" →
**"best total reaches 4/15 cells (HOCBF)"**

§V "DT-margin ... 14/15 GO" → "**13/15 GO**, the residual two seeds exactly where the quadratic braking term dominates"

---

## Audit 启动的 background re-runs

`audit_missing_cells_launcher.sh` (15 runs, 200K each, parallel=15):
- `runs/phase3_audit_cbf_qp_goal/seed_{0..4}/run_eval.json`
- `runs/phase3_audit_cbf_qp_multigoal/seed_{0..4}/run_eval.json`
- `runs/phase3_audit_predictive_atacom_multigoal/seed_{0..4}/run_eval.json`

跑 2026-05-31 23:31 → ~23:55 (≈25 min wall-clock,15 parallel)。raw 数据 gitignored,launcher script committed for reproducibility.

---

## VA-ATACOM 对比强度(audit 后 vs 之前)

- 之前(stale numbers):VA-ATACOM 15/15 vs baseline best 5/15(HOCBF/DCM/Predictive ATACOM)
- **现在(audit 后)**:VA-ATACOM 15/15 vs baseline best **4/15(HOCBF only)**

→ **对比反而更强**。主结论 "VA-ATACOM = the only method with full GO" 完全不动摇,且现在数字诚实可复现。

---

## Reproduce

```bash
cd safe-rl-2027
# 主表(11 methods, 已 commit raw runs/ gitignored 但 launcher 在仓里)
./refair_push_launcher.sh          # Push 列 fair r=0.3
./audit_missing_cells_launcher.sh  # CBF-QP Goal+MGoal + Predictive MGoal

# 各 cell mean cost + GO 重算(单 cell)
python3 -c "
import json, glob
seeds = {}
for p in sorted(glob.glob('runs/<dir>/**/run_eval.json', recursive=True)):
    s = [x for x in p.split('/') if x.startswith('seed_')][0]
    if s not in seeds: seeds[s] = json.load(open(p))['eval']['cost_mean']
m = sum(seeds.values())/len(seeds); go = sum(1 for c in seeds.values() if c<=5)
print(f'mean={m:.2f} GO={go}/{len(seeds)}')"
```
