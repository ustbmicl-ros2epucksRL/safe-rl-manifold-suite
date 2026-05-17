# Fallback algorithm paths — alternatives if IICF stalls

**Date:** 2026-05-16
**Trigger for this document:** D6 v1 IICF Goal × 5 seeds returned
1/5 GO at C=18.18, statistically tied with single-shot PRCF
(T8: 0/5 GO, C=17.90) on the aggregate but with one full-GO seed.
v2 (anti-drift safeguards: sliding window + critic reset) is
running; smoke at 3K × 2 × 1K showed final C=0.00 in 60 s wall.

**Purpose:** before committing the AAAI 2027 paper §V to IICF
as the headline algorithm, document the next two viable algorithm
contributions so the pivot is fast if IICF v2 also stalls. Each
candidate inherits the §IV taxonomy and the Prop. 4 forward-invariance
result from the existing draft; only §V's algorithm gets swapped.

---

## Decision tree (top-down)

```
D6 v2 IICF Goal × 5 seeds (running 2026-05-16 ~10:30)
│
├── 🟢 ≥4/5 GO, mean C ≤ 5  →  IICF is headline algorithm (Path α)
│       Action: Push + MGoal × 5 seeds; per-task D1 collection;
│       Robey 2020 baseline; finalise §V as IICF; submit AAAI 2027.
│
├── 🟡 1-3 GO, C ∈ [5, 15]  →  IICF partial; need 1 more variant
│       Action: try variants in order
│         (i)  --window-size 0 --reset-critic       (reset-only)
│         (ii) --window-size 1 --no-reset           (window-only)
│         (iii) --window-size 2                     (slightly larger)
│       Budget: ~3 × 1h wall. If one lands 🟢 → Path α.
│       Otherwise → Path β below.
│
├── 🟠 same as v1 (1/5 GO, C ≈ 18)  →  IICF mechanism caps here
│       Pivot to Path β (Learnable Recipe).
│
└── 🔴 worse than v1 (0/5 GO, C > 18)  →  IICF fundamentally broken
        Pivot directly to Path γ (MD-ATACOM) or back to Path β.
```

The cost of running Path α (continuing with IICF) when IICF
actually doesn't work is large — ~3 weeks of failed Push/MGoal
experiments. The cost of switching to Path β/γ early is small —
each is a 2-3 week pipeline that reuses most of §IV.

---

## Path α — IICF (current direction)

**Status:** Goal v2 running 2026-05-16. Smoke OK; full data TBD.

**Algorithm pitch (recap for context):** alternate between
PPO+PRCF rollout collection and IQL critic retraining on a
sliding window of fresh rollouts. The critic converges to an
in-distribution feasibility surface; the filter inherits the
correct safety direction.

**Paper position if it works:** novel application of
offline-to-online RL fix (Wang 2021, Kostrikov 2022, Lee 2022)
to a *safety filter*. The composability bound from the original
PRCF design is recovered. **Strong algorithm contribution.**

**Paper position if it doesn't work:** demote to §VII appendix
as an extended negative result (replaces the current §VII-C),
keep the taxonomy as the headline → equivalent to the
AAAI taxonomy pivot (`SELECTED_AAAI_TAXONOMY_PIVOT.md`).

---

## Path β — Learnable Recipe (Adaptive Margin RL Filter, AMRF)

**Status:** not started. Design below.

### One-line summary

Take the §V recipe (DistAdapt + sim-BRT, which already lands
12/15 GO on the §IV-D benchmark) and replace the hardcoded
$\alpha=0.3$ and $h=3$ with a small neural network that
outputs per-state $(\alpha, h)$, trained under a Lagrangian
that enforces Prop. 4's sufficient conditions
$\alpha \ge \Delta t$ and $h \ge \lceil\Delta t^{-1}\sqrt{d_\text{safe}/r}\rceil$.

### Why this is a real algorithm contribution

1. **Learned discrete-time safety filter that provably satisfies
   Prop. 4 by construction.** Existing learned-CBF methods
   (Robey 2020) do not have an explicit discrete-time guarantee
   — the recipe is the first explicit one (Prop. 4), and AMRF
   is the first learned instance.

2. **Connects naturally to Paper §IV.** The Lagrangian penalty
   form mirrors M1 and M2 directly:
   $L = L_\text{policy} + \lambda_1\max(0, \Delta t - \alpha)
        + \lambda_2\max(0, h^* - h)$, where
   $h^* = \lceil\Delta t^{-1}\sqrt{d_\text{safe}/r}\rceil$.

3. **Significantly lower risk than IICF.** Starting from Ours
   (already 12/15 GO), AMRF should *at worst* match it, not
   regress. The contribution is in *adaptivity* — speed-up in
   sample efficiency on hard tasks (Push variance, MGoal
   transient), not in absolute cost reduction.

### Algorithm sketch

```python
class AMRFFilter:
    def __init__(self, state_dim, dt=0.1, r=0.2, d_safe=0.3):
        self.policy_net = AMRFPolicy(state_dim)  # outputs (alpha, h)
        self.dt = dt
        self.r = r
        self.d_safe = d_safe
        self.h_required = math.ceil(math.sqrt(d_safe / r) / dt)

    def project(self, state, action_unsafe, velocity):
        alpha, h = self.policy_net(state)
        # clip to sufficient-condition lower bounds
        alpha = max(alpha, self.dt)
        h = max(h, self.h_required)
        # apply velocity-adaptive margin + BRT lookahead
        r_eff = self.r_base * (1 + alpha * np.linalg.norm(velocity))
        safe = sim_brt_rollout(state, action_unsafe, h, r_eff)
        return safe
```

Training: standard PPO with the action being filtered through
AMRFFilter. Add an auxiliary loss
$L_\text{aux} = \beta_1 \text{KL}(\alpha\|\alpha_\text{base})
              + \beta_2 (h - h_\text{base})^2$
that regularises toward the recipe defaults; the policy can
deviate when it finds a better trade-off but cannot collapse
to $\alpha=0$ or $h=1$.

### Implementation budget

| Step | LoC | Time |
|---|---|---|
| `safe_rl/filters/amrf.py` — filter class | ~120 | half-day |
| `safe_rl/algos/ppo_amrf.py` — PPO + AMRF trainer | ~150 | 1 day |
| `safe_rl/networks/amrf_policy.py` — small MLP | ~40 | 1 hour |
| `experiments/aaai/phase_3/launch_phase3_t11.sh` — launcher | ~50 | 1 hour |
| Smoke + tune + per-task runs | — | 3-4 days |
| **Total** | **~360 LoC** | **~5-6 days** |

### Empirical comparison

| Method | Goal | Push | MGoal | Note |
|---|---|---|---|---|
| Ours recipe (fixed α, h) | 0.84 | 10.58 | 2.59 | baseline |
| AMRF (Path β candidate) | ≤ 0.84 (by Prop. 4 construction) | TBD | TBD | adaptive |
| IICF (Path α) | TBD | TBD | TBD | iterative |
| PRCF T8 (single-shot) | 17.90 | — | — | upper bound |

### Risk

- **Insufficiency**: AMRF might match Ours but not strictly beat
  it, making the contribution incremental. Mitigation: emphasise
  the Push/MGoal *variance* reduction, which our analysis of
  §VI-C says are the cliff cells.
- **Overlap with existing literature**: any "learn the
  filter's parameters" paper exists (Robey 2020 learns the
  whole filter; Liu thesis 2024 §VI experiments with adaptive
  $r_\text{eff}$). Differentiator: Prop. 4 + the explicit
  discrete-time constraint enforcement.

---

## Path γ — MD-ATACOM (Manifold-Drift ATACOM)

**Status:** not started. Design below.

### One-line summary

Take vanilla ATACOM's null-space projection and add an
**explicit tangent-rotation correction** that compensates for
M2 (Prop. 2). The correction is fully analytic, no learned
component; it is the simplest possible new algorithm motivated
directly by §IV-B's math.

### Why this is a real algorithm contribution

1. **First algorithm-level response to M2.** Every existing
   ATACOM variant (VD, S, LA) attempts to address M1 or
   M2 implicitly through damping or lookahead; MD-ATACOM
   addresses M2 directly with the closed-form correction term
   that Prop. 2 prescribes.

2. **Stays within the ATACOM null-space framework.** Reviewers
   from the manipulator-control / null-space community can
   immediately read and verify the modification; the algorithm
   doesn't require a new conceptual framework.

3. **Falsifies M2 as a hypothesis.** If MD-ATACOM works on
   Safety-Gym, it proves M2 is the dominant failure mode (not
   M1, not "filter family is fundamentally unfit"). If it does
   not work, M1 alone or some other mode dominates, and the
   paper's Prop. 2 is qualitative rather than quantitative.

### Algorithm sketch

```python
class MDATACOMFilter:
    """ATACOM null-space projection + tangent-rotation correction."""
    def project(self, action_unsafe, q, v):
        # Standard ATACOM step:
        J_c = constraint_jacobian(q)
        N_c = orthogonal_null_space(J_c)
        a_proj = N_c @ alpha + ATACOM_correction(q, J_c)

        # Prop. 2 (M2) correction: explicit inward drift compensation
        for hazard in nearby_hazards(q):
            rho = ||q - hazard||
            r_hat = (q - hazard) / rho
            drift_speed = ||a_proj||^2 * dt / rho   # Prop. 2 (IV-4)
            a_proj -= drift_speed * r_hat            # outward push

        return a_proj
```

The drift correction is **per-hazard** and only fires when
$\rho < \rho_\text{trigger}$ (e.g. $r + d_\text{safe}$). The
correction magnitude follows directly from Prop. 2's
$\|u\|^2\Delta t / \rho$ formula.

### Implementation budget

| Step | LoC | Time |
|---|---|---|
| `safe_rl/filters/md_atacom.py` | ~80 | 4-6 hours |
| Wire into `ppo_safe.py` filter dispatch | ~15 | 30 min |
| `configs/safety/md_atacom.yaml` | ~10 | 15 min |
| Launcher + Phase-3 cells | ~40 | 1 hour |
| Smoke + tune + per-task runs | — | 1-2 days |
| **Total** | **~145 LoC** | **~2-3 days** |

### Empirical prediction

If M2 is real, MD-ATACOM should land:
- Goal: $\bar C \le 5$ (close to GO, much better than vanilla
  ATACOM's 28.7)
- Push, MGoal: also significantly better than vanilla, possibly
  on par with Ours

If MD-ATACOM does *not* improve over vanilla ATACOM, the paper
must revise §IV-B Prop. 2 from a quantitative bound to a
qualitative explanation.

### Risk

- **Most likely to fail.** ATACOM-S (PM8) already added a
  similar outward push and only reached 18.3 on Goal; MD-ATACOM
  is a refined version with the Prop. 2 formula but the same
  underlying mechanism. The probability that ATACOM-S got the
  right ε but wrong magnitude is moderate.
- **Quick to disprove.** This is also the *fastest* of the three
  paths to evaluate (2-3 days). Even a negative result here is
  a useful supplementary experiment for §IV-B.

---

## Recommended ordering

If IICF v2 lands 🟢 → ship α, ignore β/γ. AAAI 2027 hits.

If IICF v2 lands 🟡 → run β (Learnable Recipe) in parallel with
IICF variants (i)-(iii) from the decision tree. AMRF and IICF
become complementary: IICF improves the *critic*, AMRF improves
the *recipe parameters*. They could even be combined.

If IICF v2 lands 🟠 or 🔴 → switch primary contribution to β
(Learnable Recipe). Document the IICF mechanism as a 1-page
§VII-D appendix on what *almost* worked. Run γ (MD-ATACOM) as a
~2-3 day sanity check for the Prop. 2 quantification.

If both β and γ fail → revert to taxonomy-only AAAI 2027 paper
per FALLBACK_NARRATIVE.md. The §IV diagnostic taxonomy still
holds; the algorithm contribution downgrades to "an
empirically-validated recipe with explicit discrete-time
forward-invariance theorem" (Prop. 4 + Ours), which is the
state we had before the IICF attempt.

---

## Decision metadata

- Today's date when this doc was drafted: 2026-05-16
- D6 v2 expected completion: ~11:00–11:30
- Decision point for α vs β/γ: as soon as D6 v2 aggregate is in
- Path β earliest start (if needed): 2026-05-17
- Path γ earliest start (if needed): 2026-05-17 (parallel with β)
- Latest decision date that keeps AAAI 2027 timeline on track:
  2026-06-01 (gives 11 weeks for whichever algorithm + writing)
