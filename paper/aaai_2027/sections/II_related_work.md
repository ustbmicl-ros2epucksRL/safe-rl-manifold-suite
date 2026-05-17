# §II  Related work

*(Draft for AAAI 2027.  Scope statement (§II-A) is load-bearing
for the paper's contribution claims — see also
`FALLBACK_NARRATIVE.md` and the scope-tightening notes in
`IV_failure_taxonomy.md`.)*

Safety in reinforcement learning has been approached from three
structurally different directions.  We position our contribution
within the first of these and discuss the other two as parallel
families that motivate but do not overlap with our analysis.

## §II-A  Three families of safe-RL methods

### Family 1 — Tangent-projection safety filters (this paper's scope)

A safety filter intercepts the policy's commanded action and
projects it onto the constraint manifold's tangent space at
each control instant.  Forward invariance is proven in
continuous time by zeroing the radial component of the
constraint Jacobian.

| Sub-family | Representative |
|---|---|
| Null-space projection (operational-space) | Khatib 1987, Sentis 2005 |
| ATACOM family | Liu 2022, Liu 2024 (thesis), and the ATACOM-VD / -S / -LA extensions |
| Control Barrier Functions (CBF) | Ames 2014, 2017; Cheng 2019 |
| Higher-order CBF (HOCBF) | Xiao 2021 |
| Discrete-time CBF (DCM) | Agrawal & Sreenath 2017 |
| Reachability-based margins (RBAM) | Liu thesis 2024 §III-D |

These all share the per-step tangent-projection mechanism.
They are **the subject of §IV's failure-mode taxonomy**.  We
benchmark eight of them in §IV-D.

### Family 2 — Set-based safety (parallel, out of scope)

Set-based methods compute a **safe set** offline (the
backward-reachable tube under worst-case disturbances) and
enforce safety at runtime by querying or projecting onto the
set.  Unlike Family 1, no per-step Jacobian projection is
done; safety is encoded **globally** in the offline value
function.

| Sub-family | Representative |
|---|---|
| Hamilton–Jacobi reachability (PDE solve) | Mitchell 2005, Bansal et al. 2017 (survey) |
| Safety Bellman equation | Fisac et al. 2018 |
| Learned safe sets / Neural CBF | Robey et al. 2020, Lindemann et al. 2023 |
| Distributional safe RL | Yang et al. 2021 |

**Why these are out of scope.**  Set-based methods do not
satisfy the precondition of our §IV-A Prop. 1 (perfect
single-step tangent projection); they compute a *multi-step*
look-ahead implicitly in the offline solve.  M1 (geometric
overshoot from single-step projection) and M2 (tangential
preservation pathology) therefore do not directly apply.
Set-based methods have their own failure modes
(curse of dimensionality in the PDE solve; offline→online
distribution shift in the learned variants — see also
Appendix VII-C on PRCF, our own learned-critic attempt).

**Connection.**  Our recipe's sim-BRT component (§V-B) is a
poor-man's discrete-time approximation of the
Hamilton–Jacobi backward-reachable tube.  We use 9-direction
worst-case forward simulation over $h$ steps instead of
solving the HJB-PDE; the approximation is conservative for
small $h$ but tractable in the RL inner loop.  In this sense
Ours is a discrete-time bridge **between** Family 1 (tangent
projection at the filter call) and Family 2 (set-based
look-ahead).

### Family 3 — Lagrangian soft constraints

Lagrangian methods relax the hard safety constraint into a
penalty term whose multiplier is learned online to satisfy a
budget on expected cost.

| Sub-family | Representative |
|---|---|
| Constrained policy optimisation | Achiam et al. 2017 (CPO), Yang et al. 2020 (PCPO) |
| PPO-Lagrangian | Ray et al. 2019 (Safety-Gym) |
| TRPO-Lagrangian, FOCOPS, P3O | various 2020–2022 |

These provide no per-step safety guarantee; safety is
*expected* not *guaranteed*.  They are however the most
widely-cited Safe-RL baselines, and we benchmark PPO-Lag
(§IV-D Table 1 row "PPO-Lag", from T9, 2026-05-15) under
matched 200K-step training budget as a soft-constraint
reference point.

## §II-B  Why prior work has not identified the discrete-time gap

The continuous-time forward-invariance theorems behind Family 1
were developed in **manipulator control**, where:

- control rates are 500–1000 Hz (so $\Delta t \le 2$ ms);
- constraints are planar / convex (workspace walls, joint
  limits) and tangent motion = safe sliding;
- the Jacobian $J_c$ acts on rank-1 or rank-2 instantaneous
  constraints.

Inequality (IV-3) is never approached in this regime; the
theorems hold empirically.

When the same mechanisms are moved to **Safety-Gym
point-robot** (Ji et al. 2023) or **Safe-Gymnasium
locomotion** (Ray et al. 2019), the
$\Delta t = 0.1$ s control rate and 2D circular obstacle
geometry violate both assumptions simultaneously.  Yet the
Safe-RL benchmark community has principally compared
*Lagrangian* methods (PPO-Lag, CPO, PCPO) against each other,
not against Family 1 filters.  The ATACOM and CBF communities
have continued to benchmark within manipulator control.  The
gap our paper identifies is at the **intersection of two
benchmark cultures**.

The closest prior work to our analysis is:

- Liu 2024 (thesis, §III-D): empirically observes that
  ATACOM's stop-radius needs velocity inflation in
  multi-agent navigation. The thesis does not derive the
  discrete-time penetration bound (our Prop. 1) or formalise
  the tangent-rotation pathology (our Prop. 2).
- Cheng 2019: applies CBF-QP to mobile robot navigation
  *with* explicit $\Delta t$ awareness, but assumes a
  conservative outer margin and does not benchmark against
  filter alternatives.
- Choi et al. 2020: hybrid HJ-reachability + CBF, focused on
  the set-based side of our Family 2.

Our paper closes the gap by providing the formal failure-mode
analysis (Prop. 1–3), the controlled benchmark (10 methods,
including PPO-Lag from Family 3), and a discrete-time
forward-invariance result (Prop. 4) that prescribes when a
filter can be expected to work in this regime.

## §II-C  Negative results and benchmarking-as-contribution

The paper's primary contribution is **diagnostic**, not
algorithmic.  Three threads of prior work justify this framing:

1. **Reproducibility audits in deep RL.** Henderson et al.
   2018, Engstrom et al. 2020 established that controlled
   empirical comparisons of widely-used methods can yield
   field-level insight; our §IV-D plays the same role for
   safety filters.

2. **Loss-landscape and "why does X work?" papers.** Li et al.
   2018, Frankle et al. 2019 are examples of empirical
   syntheses that did not propose new algorithms but
   established why existing ones succeed or fail; Prop. 1–4
   are the analytic counterpart in our setting.

3. **Honest negative-result disclosure.** Bouthillier et al.
   2021, Goodfellow et al. 2018 NeurIPS reproducibility
   reports. We include the PRCF attempt (Appendix VII-C) as
   the analogous disclosure in this paper.

---

### TODO before submission

- [ ] Confirm Robey 2020 is the right learned-CBF reference;
      possibly also cite Lindemann 2023, Tonkens 2022
- [ ] Add discussion of distributional Safe-RL (Yang 2021,
      Bharadhwaj 2021) — currently not in §II-A Family 2
- [ ] Resolve cite-format placeholders (Liu 2022 etc.) once
      the bibliography template is fixed
- [ ] Cross-check Cheng 2019 statement re: explicit Δt
      awareness against the actual paper
