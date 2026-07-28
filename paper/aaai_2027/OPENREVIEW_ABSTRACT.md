# AAAI-27 OpenReview 摘要字段（完整可提交版）

> **用途**：填 OpenReview 摘要表单时直接复制。  
> **状态**：完整实质内容（非 `test`/`TBD` 空占位；空占位会被删除）。  
> **主线**：ABER（`aaai27/main_aber_aaai27.tex`）  
> **更新**：2026-07-22

---

## 1. Title

```
ABER: Executable Sampled-Data Safety Filtering at the Forward/Turn Command Interface
```

---

## 2. TL;DR

```
Hard safety filter on forward/turn commands with closed-form caps and executable recovery.
```

字符数：约 88（含空格）

---

## 3. Abstract

```
Safe reinforcement learning for mobile robots must constrain the actions a policy can execute, not only actions that are feasible in an idealized Cartesian model. Soft CMDP methods improve average cost but do not guarantee per-step invariance, while many hard filters are derived for Cartesian acceleration and then mapped onto sampled forward/turn velocity setpoints, so a theoretically feasible fallback need not be robot-executable. We introduce ABER (Actuation-Aware Braking-Envelope Recovery), a hard per-step safety shield on the deployed interface: it converts a calibrated stopping envelope into a closed-form cap on the next forward speed, retains the requested turn when the normal branch is certified, and otherwise issues the executable recovery command (0,0). Under measurable velocity-servo envelopes, we prove recursive feasibility and collision avoidance for any finite set of static obstacles. The classical braking-distance formula is not claimed as novel; the contribution is its interface-level projection, recovery witness, and closed-loop multi-obstacle certificate. A seeded property audit finds no required violation in 31,354 cap and projection checks. In a 295,200-record component study, calibrated ABER yields zero certificate violations and collisions over 1,200 navigation episodes, whereas removing the sampled-displacement term or executable recovery causes violations in 1,200 and 1,002 episodes. Across 30 independently trained policies and 7,500 matched Safety-Gymnasium pairs, online ABER reduces geometric collision frequency from 45.0% to 6.5%, but does not eliminate collisions and lowers success in every task/mode cell, exposing the safety-liveness tradeoff and the empirical-envelope boundary of the method.
```

词数：约 250

---

## 4. Keywords（建议）

```
safe reinforcement learning, safety filter, sampled-data control, mobile robots, braking envelope, executable recovery
```

---

## 5. Primary / Secondary Topics（建议勾选，按站点列表微调）

| 角色 | 建议 |
|------|------|
| Primary | Machine Learning → Reinforcement Learning |
| Secondary | Robotics；或 Safe / Trustworthy AI（若列表有对应项） |

> 摘要截止后 **topics 不可改**，提交前务必核对 OpenReview 下拉选项原文。

---

## 6. Submission type / Track

```
Main Technical Track — Regular Paper
```

---

## 7. 一句话贡献（内部核对，勿单独提交）

ABER 把经典制动储备落到真实 `(forward, turn)` 接口：闭式速度帽、可执行 `(0,0)` 恢复、多障碍递归证书；不是回报/成功率 SOTA。

---

## 8. 中文摘要（仅内部，勿贴 OpenReview）

安全强化学习必须约束策略**可执行**的动作，而非仅笛卡尔模型上可行的动作。软 CMDP 只改善期望代价；许多硬滤波器在笛卡尔加速度空间推导后再映射到采样前进/转向设定，理论可行回退未必可执行。本文提出 ABER：在部署接口上由标定制动包络得到闭式前进速度帽，正常分支保留转向，否则发送 `(0,0)`。在可测速度伺服包络下，对任意有限静态障碍证明递归可行性与避碰。经典制动距离非贡献；贡献是接口级投影、恢复见证与多障碍闭环证书。性质审计 31,354 项无必违；295,200 条消融中完整 ABER 在 1,200 回合零违例/碰撞。30 策略、7,500 成对实验中碰撞率 45.0%→6.5%，但未消尽碰撞且各单元成功率下降。

---

## 9. 粘贴检查

- [ ] Title / TL;DR / Abstract 与上文一致
- [ ] 纯文本，无 LaTeX
- [ ] 非 `test` / `xxx` / `TBD`
- [ ] Primary/secondary topics、reciprocal reviewer 已填
- [ ] 与终稿主张一致：硬盾 + 证书，非性能 SOTA
