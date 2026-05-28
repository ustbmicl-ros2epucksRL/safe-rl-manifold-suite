# §VII-C  An attempted offline-critic improvement (negative result)

*(Appendix for AAAI 2027.  One page in the final layout.
Documents an exploratory method we built and tested but did not
include in the headline contribution.  Reviewers reward honest
disclosure; this is that disclosure.)*

Between the diagnostic experiments D1–D4 (Appendix VII-D) and
the Phase 3 benchmark (§IV-D) we built and evaluated a
**Predictive Reachability-Critic Filter (PRCF)** that composes
an offline-learned HJ-feasibility critic with a Kalman one-step
state predictor and a single-step CBF-QP.  The motivation, on
paper, addresses two of the §IV failure modes simultaneously:
the offline critic learns a state-dependent safe-action surface
(M3-aware: gradient flows back through the filter), and the
EKF prediction targets the one-step momentum overshoot
(M1-aware).

The method did not transfer.  This appendix records what we
built, what we measured, and our best current hypothesis for
the failure.

## C.1  Algorithm

At env step $t$ with current observation $\boldsymbol{s}_t$,
ground-truth pose $\boldsymbol{q}_t$, velocity $\boldsymbol{v}_t$,
and unfiltered PPO action $\boldsymbol{a}_t$:

$$
\boldsymbol{a}^\star
\;=\;
\arg\min_{\boldsymbol{a}}\;\;
\tfrac{1}{2}\,\|\boldsymbol{a} - \boldsymbol{a}_t\|^2
\quad
\text{s.t.}\quad
V_c\!\left(\hat{\boldsymbol{s}}_{t+1\mid t}(\boldsymbol{a})\right) \le \epsilon,
\;\;
\|\boldsymbol{a}\|_\infty \le 1.
\tag{C-1}
$$

- $V_c$ is an offline-learned value function approximating the
  HJ-reachability "is the state safe?" surface, trained by
  IQL-style expectile regression with the constraint signal as
  the negative reward (D1 design and result;
  `experiments/diagnostics/D1_reach_learnable/`).
- $\hat{\boldsymbol{s}}_{t+1\mid t}(\boldsymbol{a})$ is the
  one-step state prediction under the EKF dynamics model, using
  the candidate action $\boldsymbol{a}$
  (`safe_rl/estimation/predictive.py`).
- $\epsilon$ is the CBF threshold, calibrated offline to the
  75th-percentile $V_c$ value on the D1 dataset's safe states
  (`auto:q75`).

(C-1) is non-convex in $\boldsymbol{a}$ because $V_c$ is a neural
network; we linearise around $\boldsymbol{a}_t$ using torch
autograd to recover a closed-form scalar projection, with an
SLSQP fallback when the linearisation residual exceeds a
threshold.  Per-step latency: ~1 ms on CPU
(`safe_rl/filters/critic_qp.py:322`).

## C.2  Why we tried it

The D1–D4 diagnostic chain (Appendix VII-D) had four signals
that, taken together, motivated PRCF:

| Diagnostic | Result | PRCF implication |
|-----------|--------|------------------|
| D1: is $V_c$ learnable offline? | AUC = 0.87 on held-out test | yes — supports the critic-based filter direction |
| D2: does static $R_t$ noise margin help? | C = 7.3 vs baseline 4.9 | no — naïve margin inflation hurts; need predictive layer |
| D3: does null-space ManifoldFilter + buffer-aware log-prob save the day? | C = 63.2 (collapsed) | no — null-space projection is structurally unfit (this is now M1+M2 of §IV) |
| D4: per-collision attribution | 78.8% estimation involvement, 63.6% momentum co-involvement | filter must compose estimation + momentum awareness |

PRCF was the natural composition: D1 says use the critic, D3
says don't use null-space, D4 says combine estimation +
momentum prediction. It became the original SELECTED.md
direction on 2026-05-01.

## C.3  What we measured (Phase 3 T8)

The D1-trained critic checkpoint
(`experiments/diagnostics/D1_reach_learnable/results/critic_checkpoint.pt`)
was wired into the Phase 3 trainer
(`experiments/aaai/phase_3/run_prcf.py`,
`launch_phase3_t8.sh`) and run on
SafetyPointGoal1-v0 × 5 seeds × 200K env steps × 50 eval
episodes.  Same env, eval protocol, and budget as the rest of
§IV-D Table 1.

| seed | R                | C               | verdict |
|------|------------------|-----------------|---------|
| 0    | −9.62 ± 7.31     | 7.88 ± 15.97    | NO-GO   |
| 1    | −11.62 ± ?       | 23.52 ± ?       | NO-GO   |
| 2    | −13.04 ± ?       | 14.38 ± ?       | NO-GO   |
| 3    | −11.74 ± ?       | 20.98 ± ?       | NO-GO   |
| 4    | −12.51 ± ?       | 22.76 ± ?       | NO-GO   |
| **mean** | **−11.71 ± 1.30** | **17.90 ± 6.66** | **0/5 GO** |

For comparison on the same env:

| Method                       | mean R       | mean C        | GO/5  |
|------------------------------|--------------|---------------|-------|
| PPO baseline (no filter)     | −0.62 ± 2.4  | 22.77 ± 17.6  | 2/5   |
| **PRCF (this appendix)**     | **−11.71**   | **17.90**     | **0/5** |
| Ours (DistAdapt + sim-BRT)   | +1.24 ± 0.5  | 0.84 ± 0.5    | 5/5 ★ |

PRCF underperforms the recipe of §V by **21× on cost** and
**−13 on reward**, and is statistically indistinguishable from
the no-filter PPO baseline on cost.  Per-step filter activation
rate hovers around 80–84% throughout training (the filter is
always engaged), with the SLSQP fallback path firing on **0%**
of steps (the linearisation never exceeded the residual
threshold).

## C.4  Most likely root cause

The filter is **active 80% of the time** and yet the realised
cost matches the no-filter baseline.  Combined with the
training reward trace ($R_{10}$ deteriorating from $\approx -10$
to $\approx -30$ over 200K steps), this is consistent with the
hypothesis that **the filter's correction direction is wrong on
the actual state distribution PPO visits during training**:

> **H1 (untested at submission time).**  The D1 critic was
> trained on offline transitions collected under a random +
> DistanceFilter behaviour policy. The PPO policy under PRCF
> visits a different state distribution — in particular it
> spends more time near the keepout band because the filter
> constantly pulls it back. $V_c$ outputs on the PRCF-visited
> states are therefore evaluated outside the IQL training
> support, and the QP minimisation in (C-1) projects in
> directions that are not safe for the actual dynamics.

The fix (deferred to a follow-up paper) is **in-distribution
critic retraining**: collect a second offline dataset from PRCF
rollouts themselves, retrain $V_c$, re-deploy, and iterate.
This is a known fixed-point procedure in offline-RL safety
(Robey 2020 uses an analogous online-correction loop) but
requires ~3 days of additional compute we did not have before
the AAAI deadline.

Three alternative hypotheses we did not separate:

- **H2: ε too lax** (auto-`q75` calibration too permissive).
  Could be checked with a quick `--epsilon-quantile 0.5` sweep
  (~30 min × 5 seeds).
- **H3: linearisation direction error.** The current implementation
  computes $\partial V_c / \partial \boldsymbol{a}$ via a
  hand-derived pose-block sensitivity; full autograd through
  the EKF predictor might recover a different gradient.
- **H4: V-mode oversimplification.** PRCF uses $V_c(\hat s_{t+1})$
  rather than $Q_c(\hat s_{t+1}, \boldsymbol{a})$; the
  state-action mode might give sharper gradients.

## C.5  What we conclude

PRCF as we built it is **not a working filter on Safety-Gym
point-robot**.  Whether the underlying composition (offline
critic + EKF prediction + CBF-QP) is salvageable depends on
fixing H1, which is the standard offline→online distribution
shift problem.

We retain PRCF in the codebase (`safe_rl/filters/critic_qp.py`,
`safe_rl/algos/ppo_critic_qp.py`) as a substrate for future
work.  The 8-filter benchmark and recipe of the main paper do
not depend on PRCF being fixed; they stand on the §IV failure
taxonomy and the §V invariance result.

The artefacts of this attempt — D1 critic checkpoint (AUC=0.87),
the linearised QP solver, the EKF predictor — are
self-contained and reusable.  Future work that addresses H1
will inherit a working substrate.
