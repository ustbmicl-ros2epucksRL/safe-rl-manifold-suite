# ABER-LR 完整 30-Checkpoint 多任务结果

日期：2026-07-27

## 结论

固定的 ABER-LR 参数已经在完整的
`2 training modes × 3 tasks × 5 training seeds = 30 checkpoints`
上完成评估。

最可靠的论文结论是：

> ABER-LR 修复了严格 `(0,0)` ABER 的 recovery-liveness 失衡。相对严格
> ABER，六个 mode/task 组的平均 success 与 collision-free success 均提高；
> 30 个 checkpoint 中 28 个提高 collision-free success，平均提高
> 9.67 pp，同时 collision 平均降低 2.89 pp、timeout 降低 9.79 pp。

不能声称固定配置相对无过滤策略在每个任务上都提高 collision-free success：
六组中四组的组均值提高，两个组基本持平或轻微下降。

## 实验规模与完整性

- `none/{Goal1, Push1, Goal2}` 与
  `sim_brt/{Goal1, Push1, Goal2}`；
- 每组 5 个独立训练 checkpoints；
- 每 checkpoint、每过滤配置 250 个确定性策略 episode；
- Goal1/none 使用 reset seeds 44000--44249；
- 其余 25 checkpoints 使用 reset seeds 45000--45249；
- strict 与 candidate 在每个组内使用相同 checkpoint、reset seed 和初始状态；
- strict/candidate 两次评估的 filter-off 轨迹逐条完全一致；
- 本次新增 strict 6,250 对、candidate 6,250 对，共 12,500 对 episode；
- 参数在跨任务执行前冻结，未进行任务专用调参。

## 六组结果

SCR 定义为

\[
\mathrm{SCR}=\Pr(\mathrm{success}\land\neg\mathrm{geometric\ collision}).
\]

| Training/task | Off SCR | Strict SCR | ABER-LR SCR | LR − strict SCR | LR − strict collision | LR − off SCR | 强门槛 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| none/Goal1 | 47.52% | 37.68% | **56.64%** | **+18.96 pp** | **−8.80 pp** | **+9.12 pp** | Pass |
| none/Push1 | 4.16% | 3.36% | 4.08% | +0.72 pp | +0.48 pp | −0.08 pp | Fail |
| none/Goal2 | 39.60% | 27.60% | **46.08%** | **+18.48 pp** | **−2.80 pp** | **+6.48 pp** | Pass |
| sim_brt/Goal1 | 37.68% | 32.64% | **43.20%** | **+10.56 pp** | **−0.72 pp** | **+5.52 pp** | Pass |
| sim_brt/Push1 | 2.48% | 2.16% | 3.44% | +1.28 pp | **−4.08 pp** | +0.96 pp | Fail |
| sim_brt/Goal2 | 24.32% | 15.84% | 23.84% | **+8.00 pp** | **−1.44 pp** | −0.48 pp | Fail |

强门槛沿用 Goal1 的预设要求：

- success gain ≥ 10 pp；
- collision increase ≤ 0；
- SCR gain ≥ 5 pp；
- 至少 4/5 training seeds 的 SCR 提高。

因此 3/6 组通过强门槛。其余三组不是同一种失败：

- `none/Push1`：SCR 仅 +0.72 pp，collision +0.48 pp；
- `sim_brt/Push1`：方向正确，但 success/SCR 绝对增益受极低策略成功率限制，
  未达到 10/5 pp 的幅度门槛；
- `sim_brt/Goal2`：5/5 seeds 的 SCR 均提高，但均值 +8.00 pp，
  未达到 success ≥ 10 pp 门槛。

## 30-Checkpoint 汇总

以下为六组等权汇总；每组恰好包含 5 checkpoints 和相同 episode 数，因此也等于
30 checkpoint cells 的等权均值。

| 方法 | Raw success | Collision | SCR | Unsafe success | Timeout |
|---|---:|---:|---:|---:|---:|
| 无过滤 | 42.48% | 43.19% | 25.96% | 16.52% | 57.52% |
| 严格 ABER | 19.92% | 7.11% | 19.88% | 0.04% | 80.08% |
| ABER-LR | **29.71%** | **4.21%** | **29.55%** | 0.16% | **70.29%** |

### ABER-LR 相对严格 ABER

| 指标 | 30-cell 平均变化 | checkpoint bootstrap 95% CI | 正向 checkpoint |
|---|---:|---:|---:|
| Raw success | **+9.79 pp** | `[+6.49, +13.37] pp` | 28/30 |
| Collision | **−2.89 pp** | `[−5.13, −1.11] pp` | 26/30 降低或持平 |
| SCR | **+9.67 pp** | `[+6.37, +13.33] pp` | 28/30 |
| Timeout | **−9.79 pp** | `[−13.40, −6.47] pp` | 28/30 降低 |

六个组的 success 与 SCR 组均值均提高；五个组的 collision 组均值降低，
`none/Push1` 增加 0.48 pp。

### ABER-LR 相对无过滤

| 指标 | 30-cell 平均变化 | checkpoint bootstrap 95% CI |
|---|---:|---:|
| Raw success | −12.77 pp | `[−18.32, −7.55] pp` |
| Collision | **−38.97 pp** | `[−44.04, −34.41] pp` |
| SCR | **+3.59 pp** | `[+1.03, +6.37] pp` |
| Unsafe success | **−16.36 pp** | `[−22.37, −10.71] pp` |
| Timeout | +12.77 pp | `[+7.63, +18.28] pp` |

这个总体均值不能替代逐任务结论：

- SCR 组均值在 4/6 组提高；
- 17/30 checkpoints 严格提高，19/30 提高或持平；
- `none/Push1` 为 −0.08 pp，`sim_brt/Goal2` 为 −0.48 pp；
- 只有 `none/Goal1` 和 `sim_brt/Push1` 通过严格的
  “≥4/5 seeds 提高且 pooled CI 下界 > 0”门槛。

## 论文应如何定位

主对照应该是 **strict ABER → ABER-LR**，因为研究问题是修复安全过滤器自身造成的
活性损失。完整矩阵支持以下三个贡献：

1. 严格 `(0,0)` recovery 会造成 horizon saturation：
   相对无过滤，strict ABER 的总体 timeout 从 57.52% 升至 80.08%。
2. Latched directional restore 对该失衡具有广泛一致的修复方向：
   六组 SCR 均值全部提高，28/30 checkpoints 提高。
3. 指标必须协同报告：
   ABER-LR 相对无过滤大幅降低 collision，同时总体提高 SCR，但仍损失 raw success
   并增加 timeout；单独报告 success 或 collision 都会得出不完整结论。

论文可用表述：

> Across 30 checkpoints spanning two training modes and three Safety-Gym
> tasks, ABER-LR improves collision-free success over strict zero-action ABER
> in all six mode--task groups and in 28 of 30 checkpoint cells. The
> checkpoint-level mean gain is 9.67 points, accompanied by a 2.89-point
> collision reduction and a 9.79-point timeout reduction.

相对无过滤的表述必须更克制：

> Relative to unfiltered execution, ABER-LR reduces collisions by 38.97 points
> and improves collision-free success by 3.59 points on average, but the gain
> is not universal across tasks and comes with lower raw success.

## 限制

- Push1 无过滤策略成功率仅 4.64%--5.12%，任务性能首先受策略能力限制；当前恢复层
  无法独自解决 Push 的控制/接触问题。
- Push1 的可移动物体拓扑不属于静态障碍 recovery certificate，Push 结果只能作为
  经验性证据。
- Restore 在不可认证区域内执行，不是 A2 recursive-feasibility witness。
- 30-checkpoint bootstrap 以 checkpoint cell 为单位，是描述性不确定性区间；
  不应包装成跨所有任务分布的总体定理。
- Goal1 与其余组使用不同的 held-out reset 区间；所有组内对照严格配对，但完整矩阵
  汇总属于跨组描述性统计。

## 可复核产物

- 冻结协议：
  `ABER_LATCHED_RESTORE_CROSS_TASK_MULTISEED_V1_PROTOCOL.json`
- 新增 25-checkpoint 审计：
  `safe-rl-2027/results/aaai27_aber_latched_restore_cross_task_v1/cross_task_audit.json`
- 完整 30-checkpoint 汇总：
  `safe-rl-2027/results/aaai27_aber_latched_restore_cross_task_v1/full_30_checkpoint_aggregate.json`
- 完整 raw/step/summary：
  `safe-rl-2027/results/aaai27_aber_latched_restore_cross_task_v1/`
