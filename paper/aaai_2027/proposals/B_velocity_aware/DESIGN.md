# Proposal B — Velocity-Aware Filter + Learned Sensor Noise

## 1. One-line summary
Couple a velocity-adaptive safety margin (one-step BRT proxy) to a
learned velocity-dependent measurement-noise model in a single
state-aware safety pipeline, and prove a robust forward-invariance
result that closes the sim-to-real safety gap on noisy real-robot
deployments.

## 2. AAAI fit and positioning

### Why this could work for AAAI
- **Genuine novelty: NoiseAdapter CNN.** Learning a state-dependent
  measurement-noise covariance $R(s)$ from sensor data is uncommon —
  most safe-RL papers assume fixed $R$. Pairing it with a safety
  filter is, to our knowledge, new.
- **End-to-end empirical chain.** Sensor → EKF → filter → PPO with
  every layer aware of the velocity. The Webots zero-shot transfer
  is a clean sim-to-real story.
- **Reuses everything we already have.** All Phase 0/1 modules
  (DistanceFilter, NoiseAdapter, EKF, env wrapper) carry over
  unchanged; we add a small theoretical layer.

### Why it might not be enough for AAAI
- AAAI's bar is theoretical novelty. NoiseAdapter is empirically
  novel but the theory is pedestrian (a parametric NLL fit).
- "Velocity-adaptive margin" without a learned BRT is heuristic;
  reviewers may say "this is engineering, not method".
- ICRA / IROS / RA-L are honestly a better venue for this story.
  Submitting to AAAI is a stretch.

### Positioning vs. closest related work
- **vs. Robust MPC (Mayne 2005):** Robust MPC inflates uncertainty
  tubes by hand-tuned worst-case bounds. We learn the bound from
  data.
- **vs. Adaptive CBFs (Lopez 2020, Castaneda 2021):** Adaptive CBFs
  estimate model parameters online; we estimate sensor parameters
  offline from a CNN.
- **vs. Probabilistic Safety Certificates (Lederer 2021):** They use
  GP regression for uncertainty; we use CNN with NLL loss.

## 3. Method

### 3.1 NoiseAdapter — learned measurement noise

A 1-D CNN takes a 10-step IMU window
$h_t = [\text{IMU}_{t-9}, \dots, \text{IMU}_t]$ and outputs a 3 × 3
diagonal covariance $R_t$:
$$R_t = \mathrm{diag}\bigl(\exp(\mathrm{CNN}_\phi(h_t))\bigr).$$

Trained on offline data with NLL:
$$\mathcal{L}_\phi = \tfrac{1}{2}\,(z_t - p_t)^\top R_t^{-1}(z_t - p_t)
   + \tfrac{1}{2}\log\det R_t,$$
where $p_t$ is ground truth (sim) or the EKF posterior (real).

This is the existing `safe_rl/estimation/noise_adapter.py`; we keep
it.

### 3.2 EKF with $R_t$

Standard EKF prediction; the update step uses the time-varying
$R_t$:
$$K_t = P_t^- H^\top (H P_t^- H^\top + R_t)^{-1}.$$

### 3.3 Velocity-adaptive safety margin

Combine the EKF posterior $\hat v_t$ with the NoiseAdapter posterior
$R_t$:
$$d_{\text{danger}}(\hat v_t, R_t) = d_0
   + \kappa_v \|\hat v_t\| \Delta t
   + \kappa_R \sqrt{\mathrm{tr}(R_t)},$$
where $\kappa_v, \kappa_R \ge 0$ are user-chosen scalars.

Interpretation: the first term is the deterministic single-step
reach (kinematic), the second is a noise-induced enlargement (one
standard deviation of the position estimate).

The `DistanceFilter` then operates with this dynamic margin; the
existing `AdaptiveDistanceFilter` is generalised to take both terms.

### 3.4 PPO integration

Identical to legacy v4. PPO outputs $a_{\text{unsafe}}$, the filter
projects, the env steps. NoiseAdapter is pretrained offline and
frozen during PPO training (cheap, deterministic).

## 4. Theory

### Theorem 4.1 (Robust forward invariance)
Assume the true measurement noise satisfies
$\mathbb{E}[\|z - p\|^2 | s] \le \mathrm{tr}(R^*(s))$ and the
NoiseAdapter satisfies $\sup_s \|R_\phi(s) - R^*(s)\|_F \le \delta$.
Let the danger margin be set with $\kappa_R \ge \sqrt{n_z}$, where
$n_z$ is the measurement dimension. Then with probability
$\ge 1 - \alpha$ over the noise realisation,
$$\Pr[c(s_t) > 0 \mid s_0 \in \mathcal{S}_{\text{safe}}] \le
   O(\delta) + O(\Delta t^2).$$

*Proof sketch.* Standard concentration argument: the EKF posterior
mean $\hat v_t$ is unbiased; the additional margin
$\kappa_R \sqrt{\mathrm{tr}(R_t)}$ covers $1\sigma$ of the position
uncertainty by Markov inequality with constant $\sqrt{n_z}$. The
filter then enforces $c \le 0$ on the inflated state, which implies
$c \le \delta$ on the true state.

### Theorem 4.2 (NoiseAdapter consistency)
The NLL estimator is consistent: as offline data $N \to \infty$,
$R_\phi \to R^*$ in $L_2$ if the CNN function class is universal.

*Proof sketch.* Standard MLE consistency for parametric Gaussian
models. The CNN's universal approximation suffices.

### Theorem 4.3 (Composite system stability)
If both NoiseAdapter (Theorem 4.2) and the safety filter
(forward-invariance) are individually satisfied, the closed loop is
input-to-state stable in expectation, with stability margin
proportional to $\kappa_R$.

*Proof sketch.* Compose the two error bounds with a Lyapunov
argument on $V(s) = \max(0, c(s))$. Detailed in appendix.

## 5. Code architecture (mostly reuse)

### To add (small)
```
safe_rl/filters/
└── velocity_aware.py          # the d(v, R) margin filter

safe_rl/estimation/
└── adapter_ekf.py             # already exists; extend to expose R_t

experiments/aaai/
├── ablation_velocity.py       # filter + EKF + NoiseAdapter ablation
├── noise_sweep.py             # σ_base sweep
└── sim_to_real.py             # Webots transfer table
```

### To reuse unchanged
- `safe_rl/filters/distance.py`
- `safe_rl/algos/{ppo, ppo_safe}.py`
- `safe_rl/envs/safety_gym.py`
- `safe_rl/estimation/{ekf, noise_adapter}.py` (already implemented)

### To deprecate
- `safe_rl/filters/manifold.py` and `cbf_qp.py` — appendix only.
- `safe_rl/reachability/sim_brt.py` — replaced by the velocity-aware
  margin in this proposal.

## 6. Experiments

### Main results — Table 1 (4 tasks × 5 methods × 5 seeds)
| Task | Goal | Circle | Push | MultiGoal |
|---|---|---|---|---|
| PPO baseline | | | | |
| PPO-Lag | | | | |
| ATACOM | | | | |
| Distance filter (static) | | | | |
| **Ours (velocity-aware + EKF)** | | | | |

5 methods × 4 tasks × 5 seeds = 100 runs × 200 K ≈ 40 h CPU.

### Ablation — Table 2 (Goal task with noise)
| Row | Filter | EKF | NoiseAdapter | adaptive margin |
|---|---|---|---|---|
| 1 | none | — | — | — |
| 2 | DistanceFilter (static) | off | off | off |
| 3 | DistanceFilter (static) | fixed-R | off | off |
| 4 | DistanceFilter (adaptive_v) | fixed-R | off | $\kappa_R{=}0$ |
| 5 | Velocity-aware | learned | on | $\kappa_R{>}0$ |

### Noise robustness — Figure 1
Sweep $\sigma_{\text{base}} \in \{0.05, 0.1, 0.2, 0.4\}$.
Plot cost vs. noise level for the 5 ablation rows.

### Sim-to-real — Table 3 (Webots E-puck, 50 trials per scenario)
| Scenario | None | Filter | Filter + fixed EKF | Filter + learned EKF |
|---|---|---|---|---|
| Sparse (6 obs) | | | | |
| Dense (10 obs) | | | | |
| Corridor | | | | |
| Dynamic obs | | | | |

Metrics: success %, collision %, min distance, position-error mean.

### NoiseAdapter behaviour — Figure 2
Visualise learned $R_t$ vs. ground-truth velocity-dependent noise on
held-out trajectories.

## 7. Timeline (3–4 weeks of focused work)

| Week | Dates | Deliverable |
|---|---|---|
| 1 | 5/2–5/8 | Velocity-aware margin filter + integration test. NoiseAdapter retraining smoke. |
| 2 | 5/9–5/15 | Main results: 4 tasks × 5 baselines × 5 seeds. |
| 3 | 5/16–5/22 | Ablation, noise sweep, NoiseAdapter visualisation. Webots scenarios. |
| 4 | 5/23–5/29 | Theory writing + paper drafting. |
| 5 | 5/30–6/5  | Internal review + polish. |
| Buffer | 6/6–8/14 | Reviewer-anticipation experiments. |

AAAI deadline target: ~2026-08-15. This proposal leaves the longest
buffer of the three, useful if reviewers demand more baselines.

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Reviewers reject "velocity-aware margin" as too engineering | High | Lead with NoiseAdapter as the technical novelty; the margin is "the natural way to use the learned noise". Add a related-work paragraph distinguishing from robust MPC. |
| NoiseAdapter does not transfer from sim to Webots | Medium | Pre-train on Webots data offline once; freeze. We already have the IMU-collection pipeline. |
| Safety-Gym noise model already too simple | Medium | The strong play is Webots; Safety-Gym is just for ablation depth. |
| AAAI reviewers prefer ICRA for this story | High | This is the actual structural risk. We accept it: if rejected at AAAI we resubmit to ICRA / IROS / RA-L. |

## 9. Deliverables and acceptance criteria

### Code
- `safe_rl/filters/velocity_aware.py` integrated and tested.
- `experiments/aaai/{ablation_velocity, noise_sweep, sim_to_real}.py` runnable.

### Empirical
- Main results table with mean ± std on 4 tasks.
- Ablation showing strict monotone cost reduction across rows 1 → 5.
- Sim-to-real Webots table with > 95 % success on at least 3 of 4 scenarios.
- NoiseAdapter visualisation with quantitative comparison to ground truth.

### Paper
- 7-page main; appendix with NoiseAdapter training details and Webots
  ROS2 deployment recipe.

### Acceptance gates before submission
1. NoiseAdapter NLL converges with held-out R-MSE within 2 × of theoretical Cramér–Rao bound.
2. Velocity-aware filter cost ≤ 0.5 × static-margin filter cost in noise-on regime.
3. Webots success rate ≥ 90 % on dense scenario (vs. 80 % unfiltered baseline).
4. Theorems 4.1–4.3 written and proved in appendix.
