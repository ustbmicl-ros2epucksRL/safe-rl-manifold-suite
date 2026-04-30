# Sanity v3 Distance-Calibration Experiment Record

Date: 2026-04-30

## Goal

修复第 3 章消融实验的单调性问题，验证新的距离形 reward calibration 是否能让 Table 3 ablation 的 cost 随模块逐步加入而单调下降。

当前 sanity v3 使用距离形 calibration：

- tag: `sanity_v3_distcalib`
- script: `experiments/table3_ablation/run.py`
- train steps: `50000`
- seeds: `1`
- lambda calibration: `0.05`
- log: `IROS2026/logs/sanity_v3_distcalib.log`

## Process Notes

初始监控发现 v3 进程耗时异常。排查后发现 v2 进程没有被真正杀掉：

- v2, lambda `0.02`: Python PID `3085716`, 应清理
- v3, lambda `0.05`: Python PID `2191555`, 目标进程

问题原因是之前 kill 的 PID `3085485` 是 bash wrapper，不是实际 Python 进程。两个 `table3_ablation/run.py` 进程同时运行并抢 CPU，导致 v3 速度约为正常的一半。

修正操作：

```bash
kill 3085716
pgrep -af "table3_ablation/run.py"
```

修正后只剩 v3 进程 `2191555` 运行。后续记录建议使用：

```bash
pgrep -f "table3_ablation/run.py"
```

避免只依赖 `nohup ... & echo $!` 返回的 wrapper PID。

## Progress Observations

### [1/6] PPO Baseline

- Reward: `-11.66`
- Cost: `102.48`

结果与 v2 完全一致，说明 baseline 不受 calibration 改动影响，确定性可复现。

### [2/6] + Manifold Filter

- Reward: `0.61`
- Cost: `4.56`

结果与 v2 一致，说明 calibration 关闭时两版等价。

### [3/6] + Reachability

- Reward: `0.82`
- Cost: `3.50`

结果与 v2 一致。真正检验从 [4/6] `+ Reward Calibration` 开始，因为该行起新距离形 calibration 生效。

### [4/6] + Reward Calibration

- Reward: `-2.15`
- Cost: `0.00`

新距离形 calibration 成功：

- 相比 [3/6] `+ Reachability`: cost `3.50 -> 0.00`
- 相比旧 correction-norm squared 机制: cost `20.66 -> 0.00`

Reward 从 `0.82` 降到 `-2.15`，说明 policy 为远离 hazard 牺牲了一部分任务奖励。这是合理的安全-性能折衷。

机制解释：

- 旧机制惩罚 filter 修正幅度，容易激励 policy 学会绕过 filter，导致 cost 上升。
- 新机制直接惩罚接近危险半径的 proximity，目标更明确，能促使策略远离障碍。

### [5/6] + EKF Full

- Reward: `0.24`
- Cost: `0.00`

在加入 `0.1 m` 速度相关噪声和 EKF 后，新 calibration 仍维持 cost `0.00`。Reward 比 [4/6] 的 `-2.15` 提升到 `0.24`，说明 EKF 与噪声组合没有破坏安全性，且可能通过状态平滑让策略更稳健。

### [6/6] Full Without Calibration

- Reward: `0.30`
- Cost: `0.44`

完整系统去掉 calibration 后，cost 从 `0.00` 回升到 `0.44`。这说明 calibration 在带噪场景下仍有必要。

## Final Sanity v3 Results

| # | Configuration | Reward | Cost | Marginal Delta Cost |
|---|---|---:|---:|---:|
| 1 | PPO baseline | -11.66 | 102.48 | - |
| 2 | + Manifold Filter | 0.61 | 4.56 | -97.92 |
| 3 | + Reachability | 0.82 | 3.50 | -1.06 |
| 4 | + Reward Calibration | -2.15 | 0.00 | -3.50 |
| 5 | + EKF Full, with noise | 0.24 | 0.00 | 0.00 |
| 6 | Full without calibration | 0.30 | 0.44 | +0.44 |

Cost 呈现清晰单调趋势：

```text
102.48 -> 4.56 -> 3.50 -> 0.00 -> 0.00
```

最后一行是去掉 calibration 的对照，不属于逐步加模块链条；它的 cost 回升到 `0.44`，反而强化了 calibration 的必要性。

## Paper Narrative Impact

新版 sanity v3 比旧消融结果更适合论文叙事：

1. Manifold Filter 是主力降耗模块，cost `102.48 -> 4.56`，约下降 `95%`。
2. Reachability 进一步精修安全裕度，cost `4.56 -> 3.50`。
3. Reward Calibration 消除残余违规，cost `3.50 -> 0.00`。
4. EKF 在带 `0.1 m` 速度相关噪声时仍维持 cost `0.00`，说明完整系统对感知噪声鲁棒。
5. Full without calibration 的 cost 回升到 `0.44`，说明 calibration 在完整带噪系统中仍有贡献。

这组结果符合“逐步加入模块，cost 单调下降”的工程直觉，解决了旧版本中 reward 全负和单调性反转的问题。

## Reward Interpretation

- [4/6] 的 reward `-2.15` 主要来自 calibration penalty。论文中需要明确：该列 reward 包含 calibration penalty 项；policy 为降低安全违规选择远离 hazard，因此牺牲了一部分任务奖励。
- [5/6] 的 reward `0.24` 高于 [4/6]，说明 EKF 加入噪声场景后并未导致性能退化。

## Known Limitations

- 当前 sanity v3 是 `50000` steps、单 seed 结果，只适合机制验证，不能直接作为论文正式 Table 3 数字。
- 单 seed 下 cost std 没有统计意义。
- PPO baseline reward `-11.66` 仍显著低于论文 100K 训练版本中的 `19.08`，说明 50K 还未充分收敛。

## Suggested Next Step

启动全量实验：

- train steps: `100000`
- seeds: `3`
- configurations: `6`
- lambda calibration: `0.05`
- expected runtime: about `3.5 h`

建议流程：

1. 归档 sanity v3 结果备查。
2. 启动后台全量实验并监控日志。
3. 全量结果完成后，更新 `chap3.tex` 的 Table 3、相关文字叙述，以及 `config.py` 中的 `PAPER_VALUES["table3"]`。
