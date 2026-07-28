# ABER-LR Goal1: GO / NO-GO 结果备忘

日期：2026-07-26

> 后续状态（2026-07-27）：Push1、Goal2 和 sim_brt 训练模式的剩余
> 25 checkpoints 已完成参数冻结的多 seed 评估。跨任务结论见
> `ABER_LATCHED_RESTORE_FULL_MATRIX_RESULTS.md`；本文保留为 Goal1 阶段性记录。

## 结论

Goal1 上已经得到可复现的 **GO**，但论文必须把目标定义为“安全完成协同”，
不能声称过滤器相对无过滤策略普遍提高原始成功率。

建议方法名为 **liveness-aware ABER (ABER-LR)**。它保留 ABER 的认证投影作为
内层安全机制，并增加 margin-triggered latched restore 作为经验性的外层活性恢复。
AAAI 论文中不需要、也不应引入 VA-ATACOM 名称。

论文主指标应为

\[
\mathrm{SCR}
=\Pr(\text{success}\land\neg\text{geometric collision})
=\mathbb{E}[I_{\rm success}(1-I_{\rm collision})].
\]

SCR 必须与 raw success、collision、unsafe success、timeout 分开报告，不能用一个
人为加权标量隐藏某一项退化。

## 方法与失败机制

旧版严格 ABER 在进入不可认证状态后执行 `(0, 0)`。SafetyPointGoal1 中，主要问题
不是持续多撞，而是 recovery 长尾：机器人接近停止后仍无法重新进入认证集合，
于是 recovery streak 接近整个 1000-step horizon。

ABER-LR 使用：

- safety margin：`0.10`；
- brake deceleration：`a_b = 0.9`；
- near-stop speed：`0.05`；
- restore forward scale：`0.25`；
- restore turn gain：`1.0`；
- 在 near-stop 且不可认证时，按远离最近障碍的方向产生小幅
  `(forward, turn)` 恢复动作；
- 一旦进入 restore，在重新获得认证前保持 latch，避免
  “恢复一步—硬刹一步”的振荡。

`margin=0.10` 使介入早于几何碰撞，latch 负责从停滞状态连续退出。这两个部件是
协同关系：仅加入 restore 会提高碰撞，仅扩大 margin 会继续卡死。

## 五训练种子主结果

五个独立训练策略；每个 checkpoint 评估 250 个相同 reset seeds
（44000--44249），共 1,250 个配对 episode。

| 对照 | Raw success | Collision | SCR | Unsafe success | Timeout |
|---|---:|---:|---:|---:|---:|
| 无过滤 | 76.24% | 39.68% | 47.52% | 28.72% | 23.76% |
| 严格 ABER `(0,0)` | 37.68% | 11.76% | 37.68% | 0.00% | 62.32% |
| ABER-LR | 57.20% | 2.96% | 56.64% | 0.56% | 42.80% |

### ABER-LR 相对严格 ABER

| 指标 | 平均变化 | 配对 bootstrap 95% CI | 训练种子方向 |
|---|---:|---:|---:|
| Raw success | **+19.52 pp** | [17.20, 21.92] pp | 5/5 提高 |
| Collision | **-8.80 pp** | [-10.56, -7.12] pp | 5/5 降低 |
| SCR | **+18.96 pp** | [16.64, 21.28] pp | 5/5 提高 |
| Unsafe success | +0.56 pp | [0.16, 1.04] pp | 2/5 提高 |
| Timeout | **-19.52 pp** | [-21.92, -17.12] pp | 5/5 降低 |

预先冻结的 candidate-vs-strict 门槛全部通过：

- mean raw-success gain ≥ 10 pp；
- collision increase ≤ 0 pp；
- mean SCR gain ≥ 5 pp；
- 至少 4/5 训练种子的 SCR 提高。

实际为 5/5。完整性审计还确认了 checkpoint、reset seed、初始状态一致，并且两批
实验的 filter-off 轨迹逐条完全一致。

活性机制也得到直接支持：

| 指标 | 严格 ABER | ABER-LR |
|---|---:|---:|
| Recovery step fraction | 72.07% | 34.88% |
| Median max recovery streak | 662 | 34 |
| Q95 max recovery streak | 943 | 125 |
| Mean episode steps | 687.18 | 591.58 |

### ABER-LR 相对无过滤

- raw success：`-19.04 pp`，95% CI `[-21.84, -16.32] pp`；
- collision：`-36.72 pp`，95% CI `[-39.60, -33.84] pp`；
- SCR：`+9.12 pp`，95% CI `[6.72, 11.52] pp`；
- unsafe success：`-28.16 pp`，95% CI `[-30.72, -25.60] pp`；
- SCR 在 5/5 训练策略上提高，逐 seed 为
  `+13.2, +8.4, +2.4, +13.2, +8.4 pp`。

因此，不能写“ABER-LR 相对无过滤提高 raw success”；可以写：

> ABER-LR 将大量碰撞完成转化为无碰撞完成。尽管 raw success 相对无过滤策略下降，
> collision-free success 在五个独立策略上全部提高，平均提高 9.12 pp，同时
> collision 降低 36.72 pp。

## 单 checkpoint 独立确认

参数筛选后，在新 reset seeds 43000--43249 上进行了 250-pair 确认：

- 相对严格 ABER：success `+12.8 pp`、collision `-26.0 pp`、
  SCR `+11.6 pp`、timeout `-12.8 pp`；
- 相对无过滤：success `+6.0 pp`、collision `-38.8 pp`、
  SCR `+13.6 pp`；
- held-out confirmation gate 通过。

此确认用于证明筛选结果不是同一批 reset seed 上的偶然波动；五训练种子结果才是
论文的主汇总。

## AAAI 论文可用定位

建议中心贡献写成：

1. 揭示安全过滤与 RL 组合中的指标错配：降低 collision 不等于提高安全完成，
   `(0,0)` recovery 会通过长尾 timeout 破坏任务活性。
2. 提出 ABER-LR，以提前安全裕量和持续恢复的协调机制同时控制 safety 与 liveness。
3. 使用联合事件 SCR，而非任意加权 reward，验证“成功且无碰撞”的真正完成；
   严格 ABER 对照下 success、collision、SCR 和 timeout 同时改善。

摘要/正文可用的一句话：

> We identify recovery-induced horizon saturation as the main failure mode of
> shielded RL and introduce a liveness-aware ABER recovery mechanism that
> coordinates early safety intervention with persistent restoration. Across
> five Goal1 policies, it improves collision-free success by 18.96 points over
> strict ABER while reducing collisions by 8.80 points.

## 必须保留的边界

- 当前多 seed 证据只覆盖 SafetyPointGoal1，不能外推到 Push1 或 Goal2。
- Restore 在不可认证集合内执行，不是 A2 recursive-feasibility witness；不能把 restore
  步骤写成具有原 ABER 理论安全保证。理论保证只适用于认证投影内层。
- 相对无过滤，raw success 和 timeout 仍退化。论文应正面报告，并把主张限定为
  safe-completion coordination。
- candidate 已完成后才冻结 candidate-vs-strict 比较；该对照使用固定历史基线和
  完全相同的 checkpoint/reset pairs，但应在实验选择说明中标注这一点。

## 可复核产物

- 五 seed ABER-LR vs 无过滤：
  `safe-rl-2027/results/aaai27_aber_latched_restore_goal1_multiseed_v1/multiseed_audit.json`
- 五 seed ABER-LR vs 严格 ABER：
  `safe-rl-2027/results/aaai27_aber_strict_goal1_same_seed_v1/vs_candidate_audit.json`
- 单 checkpoint 独立确认：
  `safe-rl-2027/results/aaai27_aber_latched_restore_m10_confirm_v1/confirm_audit.json`
- 冻结协议：
  `ABER_LATCHED_RESTORE_M10_CONFIRM_V1_PROTOCOL.json`、
  `ABER_LATCHED_RESTORE_GOAL1_MULTISEED_V1_PROTOCOL.json`、
  `ABER_LATCHED_RESTORE_VS_STRICT_GOAL1_V1_PROTOCOL.json`
