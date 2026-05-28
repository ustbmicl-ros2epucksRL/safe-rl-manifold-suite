# §I Introduction

We present **DT-ATACOM** (Discrete-Time ATACOM), the first safety filter
with explicit discrete-time forward-invariance guarantees for mobile
agents at coarse control rates. DT-ATACOM closes a two-orders-of-magnitude
gap between the continuous-time theory of tangent-projection filters
(ATACOM [Liu2022; Liu2024], CBF [Ames2014; Ames2017], and their variants)
and the $\Delta t = 0.1$ s reality of Safe-RL benchmarks [Ji2023],
where every published filter we tested fails.

**Why DT-ATACOM is needed.** Continuous-time tangent-projection filters
guarantee forward invariance via the ODE
$\dot{\boldsymbol{q}} = \boldsymbol{N}_c\boldsymbol{\alpha} -
K_c\boldsymbol{J}_c^{+}\boldsymbol{c}$, but discretisation at
$\Delta t = 0.1$ s breaks the guarantee. We benchmark eight published
tangent-projection filters on three Safety-Gymnasium tasks (Goal, Push,
MultiGoal) under 5 seeds $\times$ 200K env steps: **every method fails
to reach the GO threshold ($\bar{C} \le 5$) on at least one task;
together they achieve 0–3 GO out of 15 cells.** A PPO baseline with no
filter achieves 2/15. Two structural mechanisms drive these failures,
which DT-ATACOM is designed to address:

- **M1 (chord excursion and bearing rotation).** Even under exact
  tangent projection at step $k$, Pythagoras gives
  $\rho_{k+1}^2 = \rho_k^2 + \Delta t^2 \|u\|^2$: the discrete chord
  overshoots the continuous arc, and the obstacle's bearing rotates by
  $\phi \approx \Delta t\|u\|/\rho$. A committed world-frame velocity
  acquires inward radial component $\approx -\Delta t\|u\|^2/\rho$ at
  the next step. The effect is operationally significant when
  $\Delta t \|u\| \gtrsim \sqrt{d_\text{safe} \cdot r}$; at
  $\Delta t = 0.1$ s, PPO-learned $\|u\| \approx 1$ m/s, $r = 0.2$ m,
  the threshold is reached.

- **M2 (sustained-commitment drift).** The rotation in M1 compounds:
  under $n$ steps of sustained commitment, the cumulative inward drift
  is $\Delta\rho_n \sim n \Delta t^2 \|u\|^2/\rho$ (linear in $n$). On
  Safety-Gym, $n^\star = 2$ steps suffices to exhaust the safety
  margin, so no single-step ($h{=}1$) projector can recover once the
  agent commits to a tangent trajectory.

**The two-component design.** DT-ATACOM addresses M1 and M2 with
complementary mechanisms:

1. **Velocity-adaptive margin (VAM).** The keepout radius is inflated
   to $r_\text{eff}(\boldsymbol{v}) = r_\text{base}(1 + \alpha\|\boldsymbol{v}\|)$,
   absorbing the M1 chord excursion proportionally to speed.

2. **Multi-step BRT lookahead.** A backward-reachable-tube approximation
   forward-simulates $h$ steps under worst-case directions and fires
   the filter before M2 drift accumulates to penetration.

**Theoretical contribution: Proposition 4.** We prove that DT-ATACOM
yields positive invariance of the inflated keepout set under the
explicit *rule-of-thumb* conditions
$$
\alpha \;\gtrsim\; \frac{\Delta t^2 v_\max}{r}
\quad\text{and}\quad
h \;\ge\; \left\lceil \frac{r}{\Delta t \cdot v_\max} \right\rceil,
$$
which we sharpen in §IV-C to a rigorous sufficient condition involving
a danger-zone buffer $d_\text{danger}$. This is the **first discrete-time
analogue** of the ATACOM continuous-time forward-invariance theorem:
as $\Delta t \to 0$, DT-ATACOM reduces to ATACOM (Corollary 1), and at
coarse $\Delta t$ the conditions are constructive — given $\Delta t$,
$r$, and $v_\max$, they prescribe the filter's hyperparameters.

**Experimental validation.** On the same Safety-Gymnasium benchmark,
DT-ATACOM achieves **12/15 GO** (mean cost 0.84 on Goal, 10.58 on Push,
2.59 on MultiGoal), a 4–12$\times$ improvement over the best published
filter (DCM at 3/15). Ablations confirm both mechanisms are necessary;
a $\Delta t$ sweep validates the regime predicted by Proposition 4.

## Contributions

1. **DT-ATACOM algorithm** (§IV): A safety filter that explicitly
   addresses discrete-time dynamics through velocity-adaptive margin
   and multi-step BRT lookahead. Both ingredients exist individually
   in the literature; we are the first to combine them with a
   discrete-time invariance proof.

2. **Proposition 4** (§IV-C): The first discrete-time forward-invariance
   theorem for tangent-projection safety filters, with constructive
   hyperparameter conditions that bridge continuous-time ATACOM theory
   to practical deployment.

3. **Controlled benchmark** (§V): A 10-method comparison on
   Safety-Gymnasium (8 tangent-projection filters + PPO-Lagrangian +
   baseline), the broadest empirical study of this filter family to date.

4. **Failure-mode diagnosis** (§IV-A): Propositions 1–2 characterising
   M1 and M2, explaining why all published filters fail at coarse
   $\Delta t$ and why DT-ATACOM's two mechanisms are each necessary.

## Paper structure

Section II surveys related work and positions DT-ATACOM relative to
the tangent-projection, set-based, and Lagrangian families. Section III
provides preliminaries on the CMDP setting and the ATACOM continuous-time
theorem. Section IV presents the DT-ATACOM algorithm, the M1/M2 failure
diagnosis, and Proposition 4. Section V reports experiments: the 10-method
comparison, ablations, and $\Delta t$ sensitivity. Section VI discusses
limitations and sim-to-real implications.
