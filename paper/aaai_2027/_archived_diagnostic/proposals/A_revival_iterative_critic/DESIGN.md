# Proposal A-revival — Iterative In-Distribution Critic for PRCF

**Status.** Drafted 2026-05-15 as the follow-up algorithm
contribution after the AAAI 2027 failure-taxonomy paper is
submitted (target 2026-08-15). Resurrects Proposal A
(originally selected 2026-05-01, rejected 2026-05-14 at PRCF
empirical failure; see `../A_offline_critic/REJECTED_AT_PIVOT.md`)
with a concrete fix for the root-cause hypothesis H1
(in-distribution critic shift, identified in
`paper/aaai_2027/sections/VII-C_prcf_appendix.md` §C.4).

## 1. One-line summary

We resurrect the Predictive Reachability-Critic Filter (PRCF)
by replacing the single-shot offline IQL critic with an
**iterative in-distribution critic retraining loop**: alternate
between (i) running PPO + PRCF for $K$ env steps to collect
fresh rollouts under the filter's actual deployment
distribution, and (ii) retraining the IQL critic on the
combined data, until convergence. This is the standard
offline→online fix in modern offline-RL (Wang et al. 2021;
Kostrikov et al. 2022) applied for the first time to a safety
filter setting.

## 2. Why this direction (post-mortem of the original)

### What we learned from PRCF's failure (Appendix VII-C)

The original PRCF, trained on a one-shot offline dataset
collected under a random + DistanceFilter behaviour policy
(D1 collection, 2026-05-01), achieved AUC=0.87 on its
own held-out test set but **0/5 GO on Safety-Gym Goal × 5
seeds × 200K** when actually deployed inside a PPO trainer.

The training-log signature is unambiguous:

- Filter trigger rate stable at 80–84% throughout training.
- SLSQP fallback fires on **0%** of steps — the linearised
  QP always converges within residual threshold.
- Yet the realised cost (C=17.9) is statistically
  indistinguishable from the no-filter PPO baseline (C=22.8).

A filter that fires 80% of the time and still produces
no-filter-equivalent cost is **applying corrections in the
wrong direction**. The simplest explanation consistent with
the rate / fallback / cost triple is that $V_c$ is evaluated
on states outside its IQL training support: the PPO+PRCF
policy spends ~80% of its time at the keepout boundary (which
the filter pulls it to), whereas the D1 collection policy
spent far less time there.

### H1 is the standard offline→online distribution shift

In offline RL, the analogous problem is well-documented and
several established fixes exist:

| Source | Mechanism | Domain |
|---|---|---|
| Levine et al. 2020 (CQL) | Conservative penalty on OOD actions | Offline policy learning |
| Kostrikov et al. 2022 (IQL) | Expectile regression for OOD-robust V | Offline policy learning |
| Wang et al. 2021 (CRR) | Importance-weighted retraining | Offline-to-online |
| Lee et al. 2022 (TD3-BC) | BC regularisation on online data | Offline-to-online |
| **This proposal** | **Iterative critic retraining under filter rollouts** | **Safety filtering** |

All five fix the same underlying problem (value function
outside training support); ours is the first application to a
*safety filter's* value function, where the consequences of
miscalibration are violated safety constraints rather than
suboptimal policy.

## 3. The algorithm

### A-revival pseudocode

```
Inputs:
    D_0           — initial offline dataset (D1 collection, 50K transitions)
    V_c^{(0)}     — initial IQL critic (trained on D_0)
    PPO config    — same as Phase 3 T8
    K             — rollout chunk size (50K env steps)
    L             — retrain iterations per round (30K IQL updates)
    R             — outer rounds (3-5, with early stop on convergence)
    ε_quantile    — CBF threshold percentile (default 0.75)

Outputs:
    V_c^{(R)}     — final in-distribution critic
    π_R           — final PPO policy
    metrics       — per-round (C_eval, R_eval, AUC_critic, |D_r|)

Algorithm:
    for r = 1, ..., R:
        # 1. Calibrate epsilon on the current critic and dataset.
        ε_r = quantile(V_c^{(r-1)}(s) for s in D_{r-1}[safe_mask], ε_quantile)

        # 2. Run PPO + PRCF for K env steps; collect rollouts.
        D'_r = []                                        # new transitions
        π_r = PPOPolicy.init_from(π_{r-1})               # warm start
        filter = CriticQPFilter(V_c^{(r-1)}, ε_r, ...)
        for t = 1, ..., K:
            s, a_raw = π_r.act(o_t)
            a_safe = filter.project(s, a_raw, ŝ_{t+1|t})
            s_{t+1}, r_t, c_t = env.step(a_safe)
            D'_r.append((s, a_safe, r_t, c_t, s_{t+1}, label_t))
            π_r.update_if_buffer_full()

        # 3. Update training data: D_r = D_{r-1} ∪ D'_r (with optional
        #    importance weighting to up-weight the new rollouts).
        D_r = D_{r-1} ∪ D'_r

        # 4. Retrain critic for L IQL updates on D_r.
        V_c^{(r)} = IQLTrainer.train(V_c^{(r-1)}, D_r, L, expectile=0.7)

        # 5. Evaluate.
        eval_metrics[r] = evaluate_deterministic(π_r, n_eval=50)
        critic_auc[r]   = critic_auc_on_held_out(V_c^{(r)}, D_test)

        # 6. Early stop if (a) eval-C drops below GO threshold AND
        #    (b) critic AUC has plateaued (Δ < 0.01 over 2 rounds).
        if eval_metrics[r].cost ≤ 5.0 and is_plateau(critic_auc, r):
            break
```

### Why we expect H1 to converge

The fixed-point argument is standard:

1. At round $r$, $V_c^{(r-1)}$ is calibrated on $D_{r-1}$.
2. PPO+PRCF rollouts under $V_c^{(r-1)}$ visit a state
   distribution $\rho_r$ that is biased toward the keepout
   boundary (the filter's "edge of feasibility").
3. Retraining $V_c$ on $D_{r-1} \cup D'_r$ reduces the IQL loss
   on the boundary region, sharpening $V_c$ there.
4. Sharper $V_c$ → better filter decisions at boundary states
   → policy explores a less-pathological state distribution
   → $\rho_{r+1}$ is closer to a *safe* mode.
5. Fixed point: $V_c$ is calibrated on the same distribution
   the filter induces.

Convergence is not guaranteed in general (the IQL loss is
non-convex), but the iteration is empirically stable for
similar offline-to-online settings (Lee 2022 §IV-A).

## 4. Empirical protocol

### Setup (matched to AAAI Paper 1 §IV-D)

- Env: SafetyPointGoal1-v0, SafetyPointPush1-v0,
  SafetyPointGoal2-v0 (MultiGoal). Same as PRCF T8.
- Budget: 5 seeds × 200K env steps **per outer round**.
  Total compute = 5 × 200K × R rounds. At R=4 this is 4M
  env steps per task — ~10× the T8 budget.
- Eval: 50 deterministic episodes per seed, same as T8.
- Initial $V_c$: D1 critic checkpoint
  (`experiments/diagnostics/D1_reach_learnable/results/critic_checkpoint.pt`).

### Per-task critic (resolves the Paper 1 Push/MGoal gap)

The original PRCF T8 only had Goal data because D1 collection
was Goal-only. Under A-revival each task gets its own
critic, initialised either from (a) D1's Goal critic and
warm-started, or (b) a per-task D1-style collection at
$K=50K$ random rollouts (~30 min wall per task). Default to
(b) for cleaner per-task baselines.

### Comparison baselines

| Method | Description | Already have data? |
|---|---|---|
| PPO baseline | no filter | ✅ (Paper 1 §IV-D row 1) |
| Ours (recipe) | DistAdapt + sim-BRT | ✅ (Paper 1 row 12) |
| **Single-shot PRCF (T8)** | offline-only critic | ✅ (Paper 1 §VII-C) |
| **A-revival (this proposal)** | iterative critic | **NEW** |
| Online Neural CBF (Robey 2020) | online learned safety | Need to port (3-4 days) |
| PPO-Lag | Lagrangian baseline at 1M steps | Need to extend (T9 → T9-extended, ~3h) |

If A-revival lands at $C \le 1$ on Goal and similarly on Push
/ MGoal, it strictly dominates Ours and the proposal becomes a
clean algorithmic contribution.

## 5. Success criteria & decision thresholds

Three tiers, evaluated at the end of round R=4 (or earlier on
early-stop):

| Tier | Goal cost | Push cost | MGoal cost | Decision |
|---|---|---|---|---|
| 🟢 Strict win | ≤ 0.84 (5/5 GO) | ≤ 10.58 (3/5 GO) | ≤ 2.59 (4/5 GO) | A-revival is the AAAI headline algorithm. §V framing: "we close the offline→online gap for safety filters". |
| 🟡 Match | ≤ 1.5 (5/5 GO) | ≤ 12 (3/5 GO) | ≤ 5 (3/5 GO) | A-revival ties Ours; paper 2 framing: "principled algorithm with comparable empirical performance, plus the composability bound from Proposal A C2". |
| 🔴 Fail | Goal C > 5 at R=4 | — | — | H1 fix did not converge. Pivot to Direction B (Learnable Recipe) or Direction C (MD-ATACOM). |

## 6. Risks & fallbacks

| Risk | Probability | Mitigation |
|---|---|---|
| **H1 alone is insufficient; H2/H3/H4 also matter** | Medium | Run H2 (ε=q50) as a 1-day side experiment in parallel; abandon if A-revival round-1 already at NO-GO |
| **Compute budget overrun** (5×200K×4 rounds = 4M steps/task, ~16h wall per task on -P 5) | Medium | Start with Goal only; expand to Push/MGoal only after Goal convergence is shown |
| **PPO drift between rounds** (warm-start may inherit bad policies) | Low | Add policy-distillation regulariser; or reset PPO each round from D1-trained baseline |
| **Critic over-fitting to recent rollouts** (D_r grows; recent samples may not dominate) | Low | Importance-reweighting by round age, or sliding-window dataset (last 2 rounds only) |
| **Venue mismatch** (algorithm-focused paper not fitting AAAI niche) | Low | AAAI 2027 has accepted Safe-RL algorithm papers historically (PCPO, RCPO). NeurIPS 2027 (May 2027) and ICLR 2028 (Sep 2027) as fallback venues |

## 7. Timeline (revised 2026-05-16, targeting AAAI 2027)

The original 9-week post-AAAI timeline below was written when this
proposal was the *post-submission* algorithm follow-up. With the
AAAI 2027 venue retained and the algorithm work brought *into* the
main paper, the timeline is folded into the AAAI submission window
itself (see STATUS.md §6.1 for the full 13-week schedule). Key
calibration: D6 IICF smoke (2026-05-16) cut Goal cost 28→10 in
100 s wall, projecting full-budget Goal × 5 seeds to ~1 h. If that
projection holds, Push + MGoal can fit in Week 2 rather than the
~4 weeks originally allocated.

Original 9-week schedule (pre-pivot, kept for historical reference):

| Week | Dates (2026) | Output |
|---|---|---|
| 1 | 8/16 – 8/22 | A-revival pipeline implementation. |
| 2 | 8/23 – 8/29 | Goal cell run. |
| 3 | 8/30 – 9/5 | Push + MGoal if Goal converged. |
| 4 | 9/6 – 9/12 | Cross-task data complete; §IV draft. |
| 5 | 9/13 – 9/19 | Robey 2020 baseline. |
| 6 | 9/20 – 9/26 | §V experiments, §III theory. |
| 7-8 | 9/27 – 10/10 | Full draft. |
| 9 | 10/11 – 10/17 | Submission. |

## 8. Connection back to AAAI Paper 1

The success of A-revival would let us cite Paper 1's
M1/M2/Prop 4 directly as the motivation:

> "Paper 1 [our AAAI 2027] identified that tangent-projection
> safety filters fail in discrete-time mobile-agent RL via
> three structural modes (M1–M3). Set-based methods (Family 2
> in [Paper 1]) have a different failure profile: they
> compose offline-learned $V_c$ with online deployment, which
> introduces a distinct distribution-shift problem (H1, see
> [Paper 1] Appendix VII-C). This paper closes H1 by an
> iterative in-distribution critic retraining loop, recovering
> the principled set-based safety guarantee under the
> discrete-time RL deployment regime."

Historical note (pre-2026-05-17): an earlier revision of this
proposal framed A-revival as a "Paper 2" follow-up after a
diagnostic-only AAAI submission. That two-paper framing is
withdrawn. A-revival is either in-scope for AAAI 2027 §V
(if it lands) or out-of-scope future work (if it does not
land). Scope-tightening per `feedback_aaai_scope.md`.

## 9. Concrete next-step deliverables (post-AAAI)

In rough order:

1. **Rollout collection loop** in
   `safe_rl/algos/ppo_critic_qp.py` — add a callback that
   serialises `(s, a_safe, r, c, s', is_unsafe_label)` tuples
   to disk every K=50K env steps. ~80 LoC.

2. **IQL retrain wrapper** in
   `safe_rl/safety_critic/iql_trainer.py` — accept an
   existing checkpoint and a new dataset, retrain for L
   updates, save new checkpoint. ~60 LoC.

3. **A-revival orchestrator** in
   `experiments/diagnostics/D6_iterative_critic/run.py` — outer
   loop, eval, early-stop, metrics logging. ~250 LoC.

4. **Per-task D1 collection** in
   `experiments/diagnostics/D1_reach_learnable/collect_per_task.py`
   — extend to Push and MGoal. ~50 LoC modifications.

5. **Goal cell launch script**:
   `experiments/aaai/phase_3/launch_a_revival_goal.sh`.
   Mirrors `launch_phase3_t8.sh`. ~60 LoC.

Total: ~500 LoC new + 50 modifications. Spread across week 1.

## 10. Open design decisions

- **Should the critic be re-trained from scratch each round or warm-started?**
  Default: warm-start (parameter continuity), but with a
  larger learning rate in early epochs. Ablation in round 1.
- **Importance reweighting for $D'_r$?** Default: equal weights;
  test up-weighting of recent rollouts at $\beta = 2$ in round 2.
- **Should PPO inherit the previous round's policy or restart?**
  Default: warm-start. Restart-from-D1-baseline is a fallback if
  PPO drifts into a bad policy basin.
- **Critic architecture changes?** No — keep MLP[256,256] from
  D1 to allow direct checkpoint transfer; an ablation on
  larger nets is for §V appendix only if time permits.

---

### Open TODOs at proposal stage

- [ ] Confirm IQL trainer entry point names against
      `safe_rl/safety_critic/iql_trainer.py` current API
      (the proposal pseudocode assumes `IQLTrainer.train`).
- [ ] Decide per-task vs single-critic strategy with PI
      before committing to per-task D1 collection.
- [ ] Estimate the compute budget more precisely once
      Phase 3 main runs free up.
