# Proposal C — Filter Distillation: Safe Without Runtime Filter

## 1. One-line summary
Train a policy under a safety filter at training time, but distil the
filter's behaviour into the policy via an auxiliary loss so that at
deployment the policy is safe *without any runtime filter or QP*.

## 2. AAAI fit and positioning

### Why this is the highest-novelty direction
- **A new contract for safe RL.** Prior work either runs a filter at
  deployment (CBF, ATACOM, our Proposals A and B) or uses soft
  Lagrangian penalties that do not guarantee safety (PPO-Lag, CPO).
  This proposal targets the corner that no published method occupies:
  *zero-overhead deployment with provable training-time safety
  internalisation*.
- **Compelling deployment story.** A single forward pass instead of
  CBF-QP. 50× lower latency. Particularly attractive on
  microcontrollers and underactuated robotics platforms.
- **Theoretically tractable.** The distillation gap is measurable;
  the policy's safety violation can be bounded in terms of that gap.

### Why this is the riskiest direction
- Distillation may not converge: the unfiltered policy class can
  collapse to a less-safe local optimum.
- "Safety preservation under distillation" needs a clean theorem;
  proving it under stochastic policies + bounded function class
  is non-trivial.
- One bad seed at deployment can ruin the central claim.

### Positioning vs. closest related work
- **vs. Imitation learning from a safe expert (Ho 2016, Ross 2011):**
  We do not assume an expert dataset; the "expert" is the filter
  applied to the on-policy distribution, computed online during PPO.
- **vs. Implicit constraint satisfaction (Ray 2019):** They use cost
  in the reward; we use a hard filter at training and an explicit
  distillation loss.
- **vs. Safety distillation in autonomous driving (Wabersich
  2018):** They pre-compute a safe-set in offline driving data;
  we distil from a runtime filter on Safety-Gym.
- **vs. Behavior cloning (Pomerleau 1989):** BC matches expert
  actions; we match filtered actions of *the same policy class*
  (a.k.a. self-distillation), avoiding the distribution-shift
  problem of standard BC.

## 3. Method

### 3.1 Two-stream training

For each rollout step:
1. Policy $\pi_\theta$ outputs $a_{\text{raw}}$.
2. Filter $F$ (any of DistanceFilter / Q_c-CBF-QP / ATACOM) projects:
   $a_{\text{safe}} = F(s, a_{\text{raw}})$.
3. Environment receives $a_{\text{safe}}$.
4. We **separately store** both $a_{\text{raw}}$ and $a_{\text{safe}}$
   in the rollout buffer.

### 3.2 Distillation loss
$$\mathcal{L}_{\text{distill}} = \mathbb{E}_{(s, a_{\text{raw}}, a_{\text{safe}}) \sim \mathcal{D}}
   \bigl[ \|a_{\text{raw}} - a_{\text{safe}}\|^2 \bigr].$$

This pushes $\pi_\theta(s)$ toward outputs the filter does not need
to correct. Gradients flow through $a_{\text{raw}}$ only.

### 3.3 Total loss
$$\mathcal{L} = \mathcal{L}_{\text{PPO}} + \beta(t)\,\mathcal{L}_{\text{distill}}.$$

$\beta(t)$ ramps from $0$ to a target value $\beta^*$ over the first
half of training, so PPO can establish reward-maximising behaviour
before being constrained.

### 3.4 Curriculum schedule for filter strength
Optional: gradually relax filter strictness so the policy "earns"
more autonomy as it internalises constraints. E.g.,
$\epsilon(t) = \epsilon_0 + (\epsilon_{\max} - \epsilon_0)
\frac{t}{T}$, where $\epsilon$ is the filter's safety margin.

### 3.5 Deployment

At deployment time the filter is *removed*. The robot executes
$\pi_\theta(s)$ directly. We monitor cost statistics and report
deployment-time safety as our key metric.

## 4. Theory

### Theorem 4.1 (Distillation safety bound)
Let $\pi_\theta^*$ be the converged distilled policy. Let
$\Delta = \sup_s \|\pi_\theta^*(s) - F(s, \pi_\theta^*(s))\|$.
Assume the filter $F$ is forward-invariant on the safe set
$\mathcal{S}_{\text{safe}}$. Then
$$\Pr[c(s_t) > 0 \mid s_0 \in \mathcal{S}_{\text{safe}}]
   \le L_c \Delta + O(\Delta t),$$
where $L_c$ is the Lipschitz constant of the constraint function $c$
along the trajectory.

*Proof sketch.* Each step the unfiltered action differs from the safe
action by at most $\Delta$. The single-step constraint perturbation
is $L_c \Delta$ by Lipschitz. Discrete-time CBF analogue gives the
overall bound.

### Theorem 4.2 (Distillation is consistent with PPO)
Adding the distillation loss to the PPO objective leaves the policy
gradient unbiased *for the reward-only direction*: the mixed gradient
$(\nabla \mathcal{L}_{\text{PPO}}) + \beta (\nabla \mathcal{L}_{\text{distill}})$
remains in the stable convex hull required by the trust-region
analysis (Schulman 2015), provided $\beta \le \beta^*$ for some
constant $\beta^*$ depending on the action space radius and the
filter's projection norm.

*Proof sketch.* The distillation gradient is bounded by the action
space radius, and adding a bounded gradient to the PPO update
preserves the trust-region bound.

### Theorem 4.3 (Convergence rate)
With $\beta(t)$ scheduled as in §3.3, the distillation gap $\Delta$
contracts at rate $\Delta_T \le \Delta_0 \rho^T$ for some
$\rho < 1$, provided the policy network has sufficient capacity.

*Proof sketch.* Self-distillation with bounded loss is a contraction
mapping on the action space; standard fixed-point argument.

## 5. Code architecture

### To add
```
safe_rl/algos/
└── ppo_distill.py             # PPO + distillation loss

safe_rl/buffers/
└── distillation_buffer.py     # stores both a_raw and a_safe

experiments/aaai/
├── distill_main.py            # main results
├── distill_curriculum.py      # beta and epsilon schedules
└── distill_no_filter_eval.py  # deployment-time eval

tests/
└── test_distill_consistency.py
```

### To reuse
- All of Phase 1: filters, env, EKF, base PPO.

### Key implementation details
- Two action heads in the rollout buffer: `actions_raw`, `actions_safe`.
- The base PPO advantage is computed against the executed action
  (i.e. $a_{\text{safe}}$), but the policy gradient uses
  $\log \pi_\theta(a_{\text{raw}}|s)$ to keep on-policy structure.
- KL divergence diagnostic: track the implicit constraint gap during
  training.

## 6. Experiments

### Main results — Table 1 (4 tasks × 7 methods × 5 seeds)

| Method | Train-time safety | Deployment cost | Latency |
|---|---|---|---|
| PPO baseline | none | high | 1× |
| PPO-Lag | soft | medium | 1× |
| Distance filter (deployed) | hard | low | 5× |
| Q_c-CBF-QP (deployed, Proposal A) | hard | very low | 5× |
| **Ours (distilled, no filter)** | hard | low (target) | **1×** |
| ATACOM | hard | low | 3× |
| Filter then BC fine-tune | hard | low | 1× |

Latency = relative wall-clock per step.

### Ablation — Table 2
| Row | Filter at train | Distillation $\beta$ | Curriculum | Filter at deploy |
|---|---|---|---|---|
| 1 | DistanceFilter | 0 | none | yes |
| 2 | DistanceFilter | 0 | none | no |
| 3 | DistanceFilter | constant | none | no |
| 4 | DistanceFilter | scheduled | none | no |
| 5 | DistanceFilter | scheduled | yes | no |

### Distillation-gap curve — Figure 1
Plot $\Delta(t)$ during training. Validate Theorem 4.3 contraction.

### Distillation generalisation — Table 3
Train on Goal with filter; deploy without filter on Goal AND
zero-shot on Push, Circle. Demonstrate that internalised safety
transfers.

### Latency benchmark — Table 4
Wall-clock latency per step:
- DistanceFilter QP: ~5 ms
- Q_c-CBF-QP: ~10 ms
- Distilled policy: ~1 ms

### Real-robot — paragraph
Webots E-puck with the distilled policy, no filter. 50 trials per
scenario, demonstrate that the deployment-time cost matches the
filtered baseline within 2× margin.

## 7. Timeline (4–5 weeks of focused work)

| Week | Dates | Deliverable |
|---|---|---|
| 1 | 5/2–5/8 | Distillation loss + double-buffer + smoke train. |
| 2 | 5/9–5/15 | $\beta$/curriculum schedules, single-task convergence verification. |
| 3 | 5/16–5/22 | Main results table on 4 tasks × 5 seeds. |
| 4 | 5/23–5/29 | Generalisation experiments + latency benchmark. |
| 5 | 5/30–6/5  | Theory writing + paper drafting. |
| 6 | 6/6–6/12  | Internal review. |
| Buffer | 6/13–8/14 | Reviewer experiments. |

AAAI deadline target: ~2026-08-15.

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Distilled policy is not safe enough at deployment (Δ too large) | High | Hybrid mode: keep a lightweight "monitor" that re-engages the filter only when policy confidence drops; report both pure-distillation and monitored-distillation numbers. |
| Distillation collapses policy diversity (low entropy → poor reward) | High | Careful $\beta$ schedule; entropy regulariser; maybe use KL-distillation against $\pi_\theta(\cdot|s)$ shifted toward $a_{\text{safe}}$ rather than hard L2. |
| Theorem 4.3 contraction unrealistic | Medium | Replace with empirical convergence curve in main paper; theorem moves to appendix as "monotonic decrease under standard assumptions". |
| Reviewers prefer Proposal A | Medium | Compose: Proposal A's $Q_c$-CBF-QP can be the *training-time* filter; we then distil it. Best of both worlds. (We may pursue this hybrid as a stretch goal.) |
| Real-robot demo fails (cost too high) | Medium | Run with the safety monitor enabled but trigger-rate near 0; report monitor activations honestly. |

## 9. Deliverables and acceptance criteria

### Code
- `safe_rl/algos/ppo_distill.py`, distillation buffer, full training loop.
- Unit tests: distillation gradient does not blow up, buffer stores
  both action streams, no-filter deployment runs without exceptions.

### Empirical
- Distillation gap $\Delta(t)$ decreasing to $\le 0.05$ on at least
  3 of 4 tasks.
- Deployment cost without filter $\le 2 \times$ deployment cost with
  filter on the same task.
- Latency benchmark showing ≥ 5× reduction vs. CBF-QP at
  deployment.
- Generalisation: zero-shot on a held-out task with cost ≤ 5 × baseline.

### Paper
- 7-page main + appendix with full proofs and stability analysis.
- Latency vs. safety scatter plot (key visual).

### Acceptance gates before submission
1. Distilled policy safe enough on all 4 tasks (cost ≤ 2 × filter-deployed cost).
2. Theorem 4.1 written and proved.
3. Curriculum schedule's contribution shown to be > 30 % cost reduction in ablation row 5 vs row 4.
4. Real-robot Webots demo: distilled policy runs at ≤ 1 collision per 50 trials in dense scenario.
