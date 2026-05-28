# §I  Introduction

*(Draft for AAAI 2027.  Target ~1.5 pages in the AAAI template.
Lays out the gap → contribution chain that the rest of the paper
fills.  Cross-refs to §II–VII are placeholders.)*

## The empirical setting

Safe reinforcement learning has converged on a small set of
benchmarks dominated by Safety-Gymnasium [Ji et al. 2023], a
suite of mobile-agent navigation tasks at $\Delta t = 0.1$ s
control rates with circular hazard obstacles of radius
$r \approx 0.2$ m.  These tasks are the de-facto reference for
the constrained-RL community (PPO-Lagrangian, CPO, PCPO and
their many follow-ons), and a growing fraction of papers
proposing new safety filters — null-space projection,
control-barrier-function (CBF) variants, discrete-time CBF,
reachability-based margins — also report numbers on the same
suite.

A natural question, given that this benchmark culture is now
mature, is **whether the safety filters proposed in the
manipulator-control literature actually work on it**.  Despite
hundreds of papers in the broader Safe-RL space, we are not
aware of a controlled cross-family comparison.

## What we find

We benchmark eight published tangent-projection safety filters
on three Safety-Gymnasium tasks (Goal, Push, MultiGoal), each
under 5 seeds × 200K env steps, alongside a PPO-Lagrangian
soft-constraint reference at matched budget.  **Every method
fails to reach the cost-GO threshold of $\bar C \le 5$ on at
least one task; together they manage 5–14 GO out of 45 cells**,
while the no-filter PPO baseline achieves 2/15.  A simple
recipe combining a velocity-adaptive stop-radius with a 3-step
backward-reachable-tube approximation — both ingredients
already present in the literature individually — reaches
**12/15 GO** under the same budget.

This empirical asymmetry is the starting point of the paper.

## Why all of them fail

The tangent-projection filters share a single mathematical
mechanism: at each control instant they zero the radial
component of the action with respect to the constraint
Jacobian, leaving the tangent component free.  Under the ODE
$\dot{\boldsymbol{q}} = \boldsymbol{N}_c\boldsymbol{\alpha} -
K_c\boldsymbol{J}_c^{+}\boldsymbol{c}$ this is provably
forward-invariant: $\dot{\boldsymbol{c}} = -K_c\boldsymbol{c}$
and the safe set is preserved.  The original ATACOM paper
[Liu 2022] and its derivatives rely on this guarantee.

The benchmark we report falsifies the guarantee empirically.
We diagnose three structural failure modes that the discrete
time step introduces:

- **M1 — Geometric overshoot** (Prop. 1, §IV-A). Even when the
  tangent projection is exact, a Taylor expansion of the
  constraint shows a strictly negative second-order term of
  size $\Delta t^2 \|\boldsymbol{u}\|^2$. At $\Delta t = 0.1$ s
  and the speeds PPO learns to produce, this is comparable to
  the keepout band width. A controlled $\Delta t$-sweep
  (Figure 1) demonstrates the prediction directly: ATACOM
  achieves zero cost on Goal at $\Delta t \le 0.02$ s and
  jumps to $\bar C = 39.95$ at $\Delta t = 0.05$ s, a clean
  phase transition consistent with the threshold (Prop. 1
  inequality).

- **M2 — Tangential preservation pathology** (Prop. 2, §IV-B).
  The tangent direction itself rotates relative to a circular
  hazard's centre as the agent moves; a vector that was
  tangent at step $k$ has a radial component at step $k{+}1$
  even before any projection error compounds.  The per-step
  inward drift is $\|\boldsymbol{u}\|^2\Delta t / \rho$ and
  the multi-step accumulation is linear in $n$.  None of the
  four ATACOM variants (vanilla, velocity-damping,
  $\varepsilon$-relaxation, 1-step lookahead) closes this
  drift: all four sit at $\bar C \in [18, 84]$ on the
  benchmark.

- **M3 — Gradient credit is not the bottleneck** (Prop. 3,
  §IV-C). A plausible alternative explanation is that the
  policy gradient credits the wrong action because the
  replay buffer stores the unfiltered command while the
  environment returns reward for the filtered one.  We
  re-ran the family with `store_filtered_action=True` over
  45 (filter, env, seed) cells; the net effect was a
  decrease from 5 to 3 GO.  The projection mechanism —
  not gradient flow — is the load-bearing failure.

## Why prior work has not surfaced this gap

The mismatch we identify lives at the intersection of two
benchmark cultures.  Tangent-projection filters were
developed and evaluated in **manipulator control**: 500–1000
Hz control rates, planar workspace constraints, and rank-1
Jacobians.  In this regime the M1 threshold is three orders
of magnitude away and M2 reduces to harmless wall-sliding.
Safe-RL benchmarks like Safety-Gymnasium, on the other hand,
have been dominated by Lagrangian methods (PPO-Lag, CPO),
which the filter community has not engaged with at scale.
Neither culture has run the cross-comparison we report.

The closest prior analyses (Liu thesis 2024 §III-D, Cheng
et al. 2019, Choi et al. 2020) observe individual symptoms but
do not formalise the failure modes nor benchmark the family
breadth.

## What this paper does

This paper does **not** propose a new safety filter algorithm.
Its contributions are diagnostic and translational:

1. **C1 — A 3-axis failure-mode taxonomy** (§IV).  Three
   small propositions characterising when tangent-projection
   safety filters fail under coarse $\Delta t$, each backed
   by a controlled experiment from the §V benchmark.

2. **C2 — The 10-method controlled benchmark on Safety-Gym
   point-robot** (§IV-D, Table 1).  Eight tangent-projection
   filters + PPO-Lagrangian + PPO baseline, 4 tasks × 5 seeds
   × 200K steps each, totalling 105 training runs.

3. **C3 — Proposition 4: discrete-time forward invariance**
   (§V).  An explicit discrete-time analogue of the ATACOM
   ODE forward-invariance theorem, prescribing sufficient
   conditions $\alpha \ge \Delta t$ and
   $h \ge \lceil\Delta t^{-1}\sqrt{d_\text{safe}/r}\rceil$ on
   the velocity-margin coefficient and BRT-lookahead horizon
   under which the inflated keepout set is positively
   invariant.  A simple recipe (DistAdapt + sim-BRT) satisfies
   these conditions and serves as an existence proof.

4. **C4 — Two honest negative-result disclosures**.  We tried
   two principled algorithm approaches: a single-shot
   offline-critic CBF-QP filter (PRCF, Appendix VII-C) and an
   iterative in-distribution refinement of the same critic
   (IICF, Appendix VII-D).  Neither beats the §V recipe; both
   are documented with per-seed data and root-cause analyses.
   We also report an ablation (§V-H) showing that a *learned*
   per-state $\alpha$ head (AMRF) does not outperform the
   fixed $\alpha = 0.3$ either, and that the learned values
   show no measurable correlation with state features (all
   $|\rho| < 0.10$).  These three negative results are not
   filler — they directly support §V's "Prop. 4 floor is what
   matters" claim by ruling out the natural algorithmic
   alternatives.

## On manifold-constrained safe RL

A natural reading of our results is that we have **not**
delivered a working manifold-constrained safe-RL algorithm in
the sense of the ATACOM tradition.  Every method we tested
that performs strict manifold projection (null-space ATACOM
and its four extensions, HOCBF, DCM, RBAM — Family 1 in §II-A)
either fails outright or returns to the GO threshold only
under high-frequency control.  The Family-2 set-based variants
we attempted (PRCF, IICF) also fail.

The method that succeeds — our §V recipe — is **not strict
manifold projection**.  It inflates a distance-based keepout
according to instantaneous velocity, then evaluates safety
over a multi-step rollout.  The resulting safe set is a
velocity-parametrised family of forward-invariant inflated
keepouts, not a single fixed manifold.  Proposition 4 proves
forward invariance in discrete time under explicit conditions
on the inflation coefficient and the lookahead horizon.

We read this as evidence that **the manifold-projection
inductive bias is regime-bound**, not universally wrong.  At
$\Delta t \le 0.02$ s, ATACOM achieves zero cost on
SafetyPointGoal1 (§IV-A Figure 1).  In the manipulator-control
literature where the original theorems were proved, this is the
relevant regime.  At the $\Delta t = 0.1$ s mobile-agent regime
that Safety-Gymnasium and decentralised navigation work in,
the assumption is violated and a *different* mechanism is
required.  The recipe is that mechanism; constructing a strict
manifold-projection algorithm with discrete-time guarantees
analogous to Prop. 4 remains future work (Path γ in our
fallback proposal).

## Scope and limitations

The failure-mode analysis (C1) is specifically about
**tangent-projection safety filters**.  Set-based methods —
Hamilton–Jacobi reachability and its learned variants such as
Robey 2020's Neural CBF — compute a multi-step safe set
offline and do not satisfy Prop. 1's single-step precondition.
They have their own failure modes (curse of dimensionality in
the PDE solve; offline→online distribution shift in the
learned variants — see Appendix VII-C, where PRCF inherits the
latter).  Our recipe's sim-BRT layer is itself a discrete-time
approximation to HJ-BRT, bridging the two families in §V.

The benchmark in §IV-D is run on Safety-Gymnasium point-robot
at $\Delta t = 0.1$ s.  Whether the taxonomy generalises to
locomotion ($\Delta t = 0.02$ s and explicit dynamics) or
manipulation ($\Delta t \le 2$ ms) is left to §VI Discussion
and follow-up work.

## Paper structure

Section II surveys the three families of Safe-RL methods and
positions our scope.  Section III recaps the ATACOM
continuous-time forward-invariance theorem as the analytic
reference.  Section IV develops the failure-mode taxonomy
(M1–M3) with the 10-method benchmark as the empirical
substrate.  Section V states the recipe and proves
Proposition 4.  Section VI discusses generalisation and
sim-to-real implications.  The appendix collects per-seed
data, the $\Delta t$-sweep raw numbers, and the PRCF negative
result.

---

### TODO before submission

- [ ] Replace "10-method" with the exact count once
      the table is finalised (it is 10 now; could change if
      we re-include Manifold-Push-MGoal columns)
- [ ] Add 1-paragraph teaser of the §IV phase-transition
      figure once the figure is generated
- [ ] Tighten the opening paragraph; possibly drop the
      "the community has converged on" framing if reviewers
      find it presumptuous
- [x] Cite-format placeholders unified to `[AuthorYear]` format
      (done 2026-05-18)
- [ ] Word-count check (target 800–1000 in AAAI template)
