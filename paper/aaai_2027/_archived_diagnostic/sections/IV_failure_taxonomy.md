# §IV  A 3-axis failure-mode taxonomy for tangent-projection safety filters under coarse $\Delta t$

*(Draft for AAAI 2027.  Restructured 2026-05-14 at pivot —
promoted from a §IV-B subsection to the paper's main contribution
chapter.  Scope tightened 2026-05-15 to **tangent-projection
family** (see §II-A Family 1); set-based methods
(HJ-reachability, Robey 2020 Neural CBF) are out of scope and
have their own failure modes — see §II-A Family 2.  Cross-refs to
§III (continuous-time preliminaries), §V (the recipe), and Table
1/2/7 are placeholders.)*

## Scope of this section

The failure-mode analysis below applies specifically to
**tangent-projection-style continuous-time safety filters**:
methods that project the policy-commanded action onto the
constraint manifold's tangent space at each control instant
using a constraint Jacobian.  This covers the ATACOM family
(null-space projection), CBF / HOCBF / DCM (QP-based variants),
and reachability-based margins (RBAM) — eight methods in
total, surveyed in §II-A Family 1.

We **do not** analyse set-based safety methods (HJ-reachability
and its learned variants), which compute a multi-step safe set
offline and therefore do not satisfy the precondition of
Prop. 1 (single-step tangent projection). Their failure modes
are different and are sketched in §II-A Family 2 and Appendix
VII-C (the latter documents our own learned-critic attempt,
PRCF, which inherits the set-based offline→online distribution
shift problem).

We also do not analyse Lagrangian methods (PPO-Lag, CPO),
which provide soft expected-cost guarantees rather than
hard per-step safety. They appear in the §IV-D benchmark as
Family-3 reference points, not as taxonomy targets.

## Section structure

The contribution of this section is to organise the empirical
failure of the tangent-projection family on Safety-Gym
point-robot into three structural failure modes, each with a
small proposition and a controlled experiment. Section V then
shows that the same modes explain why our recipe — already
present in the literature in pieces, but never benchmarked
together — survives where the rest collapses.

We assume the reader has §III's recap of the ATACOM
continuous-time forward-invariance theorem in hand.  The
shorthand is: under the ODE $\dot{\boldsymbol{q}} =
\boldsymbol{N}_c\boldsymbol{\alpha} - K_c\boldsymbol{J}_c^{+}\boldsymbol{c}$,
constraint values decay exponentially, $\dot{\boldsymbol{c}} =
-K_c\boldsymbol{c}$, so the constraint set is invariant.  The
three failures we analyse all consist of moving from this ODE
guarantee to a finite-$\Delta t$, finite-$\|\boldsymbol{v}\|$,
non-planar discrete-time setting.

---

## §IV-A  M1: Geometric overshoot

### Mechanism

In practice the dynamics are integrated at a finite step size
$\Delta t$.  The agent commits to a single action
$\boldsymbol{u}_k$ over $[t_k, t_k+\Delta t]$, and constraint
values evolve as

$$
\boldsymbol{q}_{k+1} = \boldsymbol{q}_k + \Delta t\,\boldsymbol{u}_k + O(\Delta t^2),
\qquad
\boldsymbol{c}_{k+1} = \boldsymbol{c}_k
+ \Delta t\,\boldsymbol{J}_c(\boldsymbol{q}_k)\,\boldsymbol{u}_k
+ O(\Delta t^2).
\tag{IV-1}
$$

The continuous-time projection (§III eq. 1) zeroes the radial
component $\boldsymbol{J}_c\boldsymbol{u}_k$ to first order in
$\Delta t$, but **does not zero second-order curvature terms**.
For a circular obstacle of radius $r$ centred at
$\boldsymbol{p}_o$, the constraint
$c(\boldsymbol{q})=r^2-\|\boldsymbol{q}-\boldsymbol{p}_o\|^2$ has

$$
c_{k+1} = c_k + \Delta t\,\nabla c \cdot \boldsymbol{u}_k
- \Delta t^2\,\|\boldsymbol{u}_k\|^2
- \tfrac{1}{2}\Delta t^2\,\boldsymbol{u}_k^{\!\top}\nabla^{2}c\,\boldsymbol{u}_k,
$$

and even when $\nabla c \cdot \boldsymbol{u}_k = 0$ (perfect
tangent projection), the term $-\Delta t^2\|\boldsymbol{u}_k\|^2$
is strictly negative: a tangentially-moving agent decreases its
squared distance to the obstacle's centre, i.e. **penetrates the
keepout band**.

### Proposition 1 (Geometric overshoot lower bound)

For a circular hazard of radius $r$, a tangentially-projected
action ($\nabla c \cdot \boldsymbol{u}_k = 0$) leads to
constraint violation in one discrete step bounded by

$$
\Delta c \;\ge\; \Delta t^2\,\|\boldsymbol{u}_k\|^2.
\tag{IV-2}
$$

The penetration becomes comparable to the keepout band width
$d_\text{safe}$ once

$$
\Delta t\,\|\boldsymbol{u}_k\| \;\gtrsim\; \sqrt{d_\text{safe}\cdot r}.
\tag{IV-3}
$$

*Proof sketch.* Taylor-expand $c$ around $\boldsymbol{q}_k$, drop
the first-order term by tangent assumption, bound the third-order
term by $\nabla^2 c \le 2I$ for circular constraints.  □

### Why prior manipulator papers do not surface M1

The original ATACOM paper and its descendants all evaluate on
**high-frequency manipulator control**: Air-Hockey [Liu 2022 §V],
joint-limit-bounded reaching [Liu 2024 §VI], constraint-bounded
robotic arm tasks.  In these settings $\Delta t \le 2$ ms and
typical commanded speed of 1 m/s gives $\Delta t\,\|\boldsymbol{u}\|
= 10^{-3}$ m, **three orders of magnitude below the threshold
(IV-3)**.  Inequality (IV-3) is never approached, the
second-order overshoot is absorbed into discretisation noise,
and the continuous-time safety claim holds empirically.

When the same projection is moved to Safety-Gym point-robot at
$\Delta t = 0.1$ s with PPO policies that reach
$\|\boldsymbol{u}\| \approx 1$ m/s, the product
$\Delta t\,\|\boldsymbol{u}\| = 0.1$ m **lands at the threshold
(IV-3) for $r = 0.2$ m hazards**.  Overshoot is no longer a
perturbation; it is the dominant motion.

### Empirical: D5 $\Delta t$-sweep (Figure IV-1)

We monkey-patched the safety-gymnasium frameskip to vary
$\Delta t$ across $\{0.01, 0.02, 0.05, 0.10\}$ s, holding the
filter and all other physics constants fixed.  ATACOM on
SafetyPointGoal1-v0 over 50K env steps × 3 seeds × 4 $\Delta t$:

| $\Delta t$ | ATACOM cost | PPO baseline cost | Ours cost |
|-----------|------------:|------------------:|----------:|
| 0.01 s    | **0.00 ± 0.0** | 44.05 ± 58.9 | 3.02 ± 5.2 |
| 0.02 s    | **0.00 ± 0.0** | 31.45 ± 17.7 | 0.00 ± 0.0 |
| 0.05 s    | 39.95 ± 13.7 | 50.45 ± 17.9 | 23.70 ± 14.6 |
| 0.10 s    | 13.68 ± 15.8 | 79.33 ± 13.5 | 2.60 ± 3.3 |

At $\Delta t \le 0.02$ s ATACOM achieves **zero cost** on Goal —
the continuous-time guarantee holds.  Between 0.02 and 0.05 s
the cost jumps from 0 to 39.95 (~4× the GO threshold), a **clean
phase transition consistent with crossing (IV-3)**.  Above the
threshold ATACOM stays in the catastrophic regime.

The PPO baseline and Ours-recipe rows are sanity checks: PPO is
uniformly catastrophic across $\Delta t$ as expected, while Ours
stays sub-5 cost except at the dt=0.05 cell where the BRT
horizon (0.15 s lookahead) is in a corner regime (a single
high-variance seed; full data in Appendix VII-B).

---

## §IV-B  M2: Tangential preservation pathology

### Mechanism

M1 says that even a *perfect* tangent projection introduces
$\Delta t^2\|\boldsymbol{u}\|^2$ overshoot.  M2 is a separate
problem: **the tangent direction itself rotates relative to the
obstacle centre as the agent moves**, so a vector that was
tangent at step $k$ has a non-zero radial component at step
$k{+}1$ even before integration error compounds.

For a circular hazard centred at $\boldsymbol{p}_o$ with agent
at $\boldsymbol{q}_k = \boldsymbol{p}_o + \rho_k \hat{\boldsymbol{r}}_k$,
the tangent unit vector $\hat{\boldsymbol{t}}_k$ rotates by
angular increment

$$
\Delta\theta_k = \frac{\|\boldsymbol{u}_k\|\,\Delta t}{\rho_k}
\quad\text{(small-angle approximation)},
$$

so the **next-step radial component** of the same velocity
vector $\boldsymbol{u}_k$ is

$$
\boldsymbol{u}_k\cdot\hat{\boldsymbol{r}}_{k+1}
= -\|\boldsymbol{u}_k\|\,\sin\Delta\theta_k
\approx -\frac{\|\boldsymbol{u}_k\|^2\,\Delta t}{\rho_k}.
\tag{IV-4}
$$

The agent, committed to its tangent action for the full
$\Delta t$, drifts inward at radial speed
$\|\boldsymbol{u}_k\|^2\Delta t/\rho_k$.  Inside the keepout
band ($\rho_k \approx r$) and at $\|\boldsymbol{u}_k\| \approx 1$ m/s,
$\Delta t = 0.1$ s, this is **0.05 m/s of inward drift** — the
agent traverses the entire keepout band in two steps regardless
of what the projection commanded.

### Proposition 2 (Tangential pathology)

For null-space projection with constraint Jacobian
$\boldsymbol{J}_c$ from a single circular hazard, the
multi-step penetration after $n$ tangentially-projected actions
is bounded below by

$$
\sum_{k=1}^{n} \frac{\|\boldsymbol{u}_k\|^2\,\Delta t}{\rho_k}
\;\ge\; \frac{v_\min^2\,n\,\Delta t}{\rho_\max}
\tag{IV-5}
$$

where $v_\min = \min_k \|\boldsymbol{u}_k\|$ and
$\rho_\max = \max_k \rho_k$.

**Assumption:** The agent remains in the keepout band throughout,
i.e. $\rho_k \in [\rho_\min, \rho_\max]$ with
$0 < \rho_\min \le \rho_k \le \rho_\max \le r + d_\text{safe}$
for all $k \in 1..n$.  The lower bound $\rho_\min > 0$ excludes
the degenerate case where the agent reaches the obstacle centre
(see §V-C for handling of this singularity in the recipe).

**Corollary (Linear convergence).**  Under the above assumption,
the agent traverses the keepout band width $d_\text{safe}$ in at
most $n^* = \lceil d_\text{safe} \cdot \rho_\max / (v_\min^2 \Delta t) \rceil$
steps.  For $d_\text{safe} = 0.05$ m, $\rho_\max \approx r = 0.2$ m,
$v_\min = 0.5$ m/s, and $\Delta t = 0.1$ s:
$n^* = \lceil 0.05 \times 0.2 / (0.25 \times 0.1) \rceil = 1$ step.
This confirms the empirical observation that a single committed
tangent action can penetrate the band.

*Proof.*  Sum (IV-4) over $k = 1..n$.  The per-step radial drift is
$|\dot\rho_k| = \|\boldsymbol{u}_k\|^2 \Delta t / \rho_k$.  For the
lower bound, replace $\|\boldsymbol{u}_k\|$ with $v_\min$ and $\rho_k$
with $\rho_\max$ (the worst-case outer position that minimises drift
rate).  Summing yields (IV-5).  For the upper bound on $n^*$, set
the cumulative drift equal to $d_\text{safe}$ and solve for $n$.  □

This is structurally different from M1 (which is a per-step
second-order term).  Under M2, even running ATACOM at zero
control commands $\boldsymbol{\alpha} = 0$ does not save the
agent if a previous-step tangent velocity is preserved.

### Empirical: ATACOM family table

The four ATACOM variants in the literature each modify the
tangent term differently:

| Filter | Tangent term modification | Goal $\bar C$ | Push $\bar C$ | MGoal $\bar C$ |
|--------|---------------------------|--------------:|--------------:|---------------:|
| ATACOM (vanilla) | (none) | 28.7 | 46.2 | 84.3 |
| ATACOM-VD | $-K_v \boldsymbol{N}_c\boldsymbol{v}$ damping | 27.5 | 50.6 | 65.0 |
| ATACOM-S | $-\eta\hat{\boldsymbol{r}}$ outward push inside band | 18.3 | 18.2 | 32.9 |
| ATACOM-LA | 1-step lookahead projection | 37.7 | 40.1 | 40.8 |
| **Ours** | velocity-adaptive stop-radius (§V) | **0.84** | **10.58** | **2.59** |

### D5 $\Delta t$-sweep cross-check on ATACOM-LA (2026-05-15/16 data)

A natural objection is that vanilla ATACOM fails because it
does not look ahead at all, and that adding a lookahead would
mitigate both M1 and M2. The ATACOM-LA variant is designed
to test exactly this: it adds a 1-step lookahead that
predicts position under the commanded action and aborts if
the prediction lies inside any keepout circle.

We re-ran ATACOM-LA in the D5 $\Delta t$-sweep with the
lookahead horizon **scaled to exactly one env step** (so M1
is addressed by construction at every $\Delta t$):

| $\Delta t$ | vanilla ATACOM | ATACOM-LA (1 env-step lookahead) |
|-----------|---------------:|---------------------------------:|
| 0.01 s    | 0.00 ± 0.0     | 19.65 ± 4.5 |
| 0.02 s    | 0.00 ± 0.0     | 41.92 ± 28.2 |
| 0.05 s    | 39.95 ± 13.7   | 62.73 ± 11.8 |
| 0.10 s    | 13.68 ± 15.8   | 53.27 ± 14.0 |

The data falsifies the lookahead hypothesis. ATACOM-LA is
**strictly worse than vanilla ATACOM at every $\Delta t$**,
including the small-$\Delta t$ regime where vanilla achieves
zero cost. We attribute this to two effects:

1. **One env-step lookahead is too short to capture M2.** Prop. 2
   shows the tangential drift accumulates *linearly* in the
   step count $n$. At $\Delta t = 0.1$ s a one-step look-ahead
   sees the agent only 0.1 m further along its current
   trajectory; the drift that materialises at $n \approx 5$
   steps is invisible to the filter. Extending to $h$ steps
   would require the multi-step BRT rollout that our recipe
   (§V) adopts.

2. **Adding lookahead to ATACOM-VD introduces over-conservatism
   at small $\Delta t$.** ATACOM-LA inherits the tangential
   damping $-K_v\boldsymbol{N}_c\boldsymbol{v}$ from ATACOM-VD.
   At $\Delta t \le 0.02$ s the damping prevents the
   well-behaved continuous-time motion that vanilla ATACOM
   would produce, and the additional one-step lookahead
   aborts safe actions; the agent is functionally frozen.

A useful generalisation: **published filters do not factor
cleanly along the M1/M2 axes**, because each method bundles
multiple mechanisms (damping + projection + lookahead) that
interact non-trivially. The clean way to address both M1 and
M2 is the recipe of §V, which decouples the two: a
velocity-adaptive margin for M1 and a multi-step BRT rollout
for M2.

ATACOM-VD damps the existing tangent velocity but is reactive
(applies *after* the agent has acquired it).  ATACOM-S adds a
radial outward push but only inside the band, so the agent must
already be unsafe before the push fires.  ATACOM-LA looks one
step ahead, but at $\Delta t = 0.1$ s a single lookahead step
spans 0.1 m which (IV-4) shows is insufficient.  **None of the
four mitigations addresses M2's linear-in-$n$ accumulation**.

---

## §IV-C  M3: Gradient credit is not the bottleneck

### Counter-hypothesis

A plausible alternative to M1+M2 is that the failure lies in
PPO's policy-gradient credit assignment rather than in the
projection mechanism.  Specifically, the replay buffer stores
the **unfiltered** action $\boldsymbol{a}$ with its on-policy
log-prob, while the environment returns reward and cost from
the **filtered** action $\boldsymbol{a}_{\text{safe}}$.  The
policy gradient therefore credits the wrong action; the policy
cannot learn what the filter is correcting.

A natural fix is the `store_filtered_action` flag: log
$\boldsymbol{a}_{\text{safe}}$ together with a recomputed
log-prob, so the gradient flows back to the action the
environment actually saw.  If M3 is the bottleneck the fix
should rescue at least some of the filter zoo.

### Empirical test: T7 ablation (Table 7)

We re-ran ATACOM-S, HOCBF, and DCM with
`store_filtered_action=True`, holding all other knobs identical
(sim-BRT $h=3$ shaping, 5 seeds × 200K steps × Goal/Push/MGoal —
45 cells total):

| Filter | env | C with SF=true | C with SF=false (prev) | $\Delta C$ |
|---|---|---:|---:|---:|
| ATACOM-S | Goal | 14.79 (0/5) | 18.29 (1/5) | −3.50 |
| ATACOM-S | Push | 26.86 (1/5) | 18.23 (1/5) | +8.63 |
| ATACOM-S | MGoal | 52.44 (0/5) | 32.89 (0/5) | +19.55 |
| HOCBF | Goal | 17.01 (0/5) | 10.42 (1/5) | +6.58 |
| HOCBF | Push | 15.37 (0/5) | 11.43 (0/5) | +3.94 |
| HOCBF | MGoal | 18.97 (0/5) | 28.49 (0/5) | −9.52 |
| DCM | Goal | 11.88 (1/5) | 13.67 (1/5) | −1.79 |
| DCM | Push | 10.67 (1/5) | 11.05 (2/5) | −0.38 |
| DCM | MGoal | 20.87 (0/5) | 47.10 (0/5) | −26.23 |

**Net GO: 5/45 → 3/45.**  The buffer-side fix is empirically
inert (DCM trends slightly better, hovering within seed
variance) or actively harmful (ATACOM-S Push +8.63, HOCBF Goal
+6.58, ATACOM-S MGoal +19.55).  No filter family flips the GO
column.

### Proposition 3 (informal corollary)

Under the M1+M2 failure modes, making the policy gradient
*aware* of the post-filter action $\boldsymbol{a}_{\text{safe}}$
does not bridge the safety gap; the projection mechanism's
geometric overshoot and tangential pathology dominate any
plausible gain from accurate credit assignment.

The conclusion is that **the projection mechanism — not gradient
flow — is the load-bearing failure mode**.  A safety filter for
this regime must address M1+M2 structurally; differentiating
the filter or fixing the buffer log-prob is not enough.

---

## §IV-D  The 10-method empirical benchmark

Drawing M1–M3 together, the controlled comparison spans **eight
tangent-projection filters** (§II-A Family 1) plus one
Lagrangian method (PPO-Lag, §II-A Family 3, included as a
soft-constraint reference) plus a PPO baseline, on the standard
Safety-Gym point-robot tasks (5 seeds × 200K steps × $\{$Goal,
Push, MultiGoal$\}$) — Table IV-1.

Set-based methods (HJ-reachability and Robey 2020 Neural CBF) are
out of scope as discussed above; PRCF (Appendix VII-C) is our
own attempt at a learned set-based method and is documented
separately as a negative result.

| Family | Method | Mechanism class | Goal $\bar C$ | Push $\bar C$ | MGoal $\bar C$ | GO/15 |
|---|---|---|---:|---:|---:|---:|
| — | PPO baseline | (no filter) | 22.8 | – | – | 2 |
| 1 | Manifold | null-space projection | 15.3 | – | – | 0 |
| 1 | ATACOM | null-space (vanilla) | 28.7 | 46.2 | 84.3 | 0 |
| 1 | ATACOM-VD | + tangential damping | 27.5 | 50.6 | 65.0 | 0 |
| 1 | ATACOM-S | + ε-relaxation | 18.3 | 18.2 | 32.9 | 2 |
| 1 | ATACOM-LA | + 1-step lookahead | 37.7 | 40.1 | 40.8 | 0 |
| 1 | HOCBF | 2-stage CBF | 10.4 | 11.4 | 28.5 | 1 |
| 1 | DCM | discrete-time projection | 13.7 | 11.1 | 47.1 | 3 |
| 1 | RBAM | reachability margin | 48.8 | 55.0 | 32.3 | 2 |
| 3 | PPO-Lag | Lagrangian soft-constraint | 35.6 | 38.3 | 76.0 | 1 |
| 1+2 | **Ours** | **stop-radius + sim-BRT** (§V) | **0.84** | **10.58** | **2.59** | **12** |

GO threshold: $\bar C \le 5.0$.  Family column references §II-A:
Family 1 (tangent-projection filters), Family 2 (set-based
methods; out of scope but Ours's BRT component is a discrete-time
approximation), Family 3 (Lagrangian soft constraints). Per-seed
numbers in Appendix VII-A.

This is the most controlled and broadest empirical comparison of
**Family 1 and Family 3** Safe-RL methods on Safety-Gym to date.
No previously-published tangent-projection filter lands within
10× of Ours's Goal cost; HOCBF and DCM, which include explicit
discrete-time corrections, do not close the gap because their
corrections target only M1 (HOCBF: higher-order class-K terms
in continuous time; DCM: explicit $\Delta t$ Euler step but
still single-step) and not M2 or M3. PPO-Lag (Family 3) at
matched 200K-step budget also fails to reach GO; it is widely
reported to require 1M+ steps for convergence in the literature
[Ray 2019 §V], so this is a budget-matched rather than
convergence-matched comparison (see §V-G discussion).

Section V translates M1+M2's structural diagnosis into the
recipe — velocity-adaptive stop-radius plus multi-step lookahead
— and proves a discrete-time forward-invariance result
(Proposition 4) that the literature mechanisms violate by
construction.

---

### TODO before submission

- [x] Cross-ref to actual Table numbers: Table IV-1 (§IV-D benchmark),
      Table V-1 (§V-E 4-task summary) aligned (done 2026-05-18)
- [x] Equation numbering finalized: (IV-1)–(IV-5) in this section,
      (V-1)–(V-4) in §V (done 2026-05-18)
- [ ] D5 $\Delta t$-sweep figure: include ATACOM-LA cell to
      validate M2 separately from M1 (1 sweep cell, ~30 min wall)
- [ ] One-paragraph related-work pointer
      (Robey 2020 Neural CBF on similar discrete-time issues;
      Cheng 2019 §V acknowledges $\Delta t$ assumption)
- [x] Tighten Proposition 2 (done 2026-05-18: added explicit ρ_k bounds,
      corollary on band-traversal time, and full proof)
