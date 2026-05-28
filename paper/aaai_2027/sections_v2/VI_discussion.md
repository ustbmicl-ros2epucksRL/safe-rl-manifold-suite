# §VI Discussion

## §VI-A When to Use DT-ATACOM vs ATACOM

DT-ATACOM is designed for coarse control rates where continuous-time
guarantees fail. The decision boundary follows from Proposition 4:

**Use ATACOM when:**
$$
\Delta t \cdot v_\max < \sqrt{d_\text{safe} \cdot r}
$$
This typically holds for manipulator control ($\Delta t \le 2$ ms) and
high-frequency mobile control ($\Delta t \le 20$ ms).

**Use DT-ATACOM when:**
$$
\Delta t \cdot v_\max \ge \sqrt{d_\text{safe} \cdot r}
$$
This applies to Safety-Gymnasium ($\Delta t = 0.1$ s), many sim-to-real
pipelines, and embedded controllers with limited compute.

**Rule of thumb.** At $r = 0.2$ m and $d_\text{safe} = 0.05$ m:
- $\Delta t \le 30$ ms: ATACOM sufficient
- $\Delta t > 30$ ms: DT-ATACOM recommended

## §VI-B Limitations

**L1: Push task (3/5 GO).** The box-agent coupling creates dynamics not
captured by single-agent BRT. The box can be pushed into hazards before
the filter reacts. Addressing this requires either:
- Multi-body BRT (computationally expensive), or
- Box-aware constraint formulation (environment-specific)

**L2: MultiGoal transients (4/5 GO).** Goal-switching creates sudden
heading changes that can overwhelm the velocity-adaptive margin. The
filter assumes smooth velocity evolution; discontinuities violate this.

**L3: Conservative behaviour.** DT-ATACOM trades reward for safety. On
Goal task, DT-ATACOM achieves reward −0.66 vs ATACOM's +0.34 (before
cost penalty). The inflated margin and BRT lookahead restrict exploration.

**L4: Hyperparameter sensitivity.** While Proposition 4 provides minimal
conditions, practical deployment requires margin ($\alpha = 0.3$ vs
minimal 0.05). This margin accounts for model mismatch but is not
theoretically derived.

## §VI-C Sim-to-Real Considerations

Although experiments are in simulation, Prop. 4's conditions are in
physical units and transfer to hardware once three deployment effects
are folded into the constants:

1. **Effective integration step.** Actuation/sensing latency $\tau$
   makes the filter act on stale state, so the relevant step is
   $\Delta t_\text{eff} = \Delta t + \tau$. A 50 Hz robot
   ($\Delta t = 20$ ms) with $\tau = 30$ ms wireless delay behaves as a
   20 Hz system; substituting $\Delta t_\text{eff}$ into (IV-12) raises
   $h^\star$ and $\alpha^\star$. Latency pushes systems from ATACOM's
   regime into DT-ATACOM's.

2. **State-estimation noise.** With localisation covariance
   $\Sigma_\text{pos}$, inflate $r_\text{base} \mapsto r_\text{base} +
   k_\sigma \sqrt{\lambda_\max(\Sigma_\text{pos})}$. In a noisy-pose
   ablation ($\sigma_\text{pos} \approx 0.1$ m), this cov-aware
   inflation dominated the safety budget.

3. **Velocity tracking error.** The separation condition
   $r \ge d_\text{safe} + \Delta t_\text{eff}\,v_\max^\text{actual}$
   should use a conservative $v_\max^\text{actual}$ to avoid jump-over
   violations.

Concrete envelope: differential-drive robot ($r = 0.3$ m,
$d_\text{safe} = 0.1$ m, $v_\max = 1$ m/s) at 20 Hz with 30 ms latency
→ $\Delta t_\text{eff} = 80$ ms → $h^\star = 4$,
$\alpha^\star_\text{rule} \approx 0.02$; separation $0.3 \ge 0.18$ holds.
Hardware validation (Webots-to-physical differential-drive transfer) is
future work; the present contribution is the discrete-time theory and
its simulation validation.

## §VI-D Relation to Continuous-Time Theory

DT-ATACOM does not replace continuous-time ATACOM theory — it extends it.
The continuous-time forward-invariance theorem (Theorem 1) remains valid
in its regime. DT-ATACOM provides the first **constructive** answer to:
*Given a discrete system, what filter parameters ensure safety?*

This is analogous to the relationship between continuous and discrete
Lyapunov stability: continuous analysis provides insight, but discrete
implementation requires discrete conditions.

## §VI-E Future Work

1. **Learned BRT.** Replace the 9-direction simulation with a neural
   network trained to predict worst-case reach. This could reduce
   computational cost while improving accuracy.

2. **Multi-agent extension.** Extend BRT to account for agent-agent
   interactions, enabling safe multi-robot coordination.

3. **Adaptive $\alpha$.** Learn the margin coefficient $\alpha$ online
   to minimise conservatism while maintaining safety. Our AMRF (Adaptive
   Margin via Reward Feedback) experiments showed this is non-trivial.

4. **Non-circular obstacles.** Generalise the geometric overshoot analysis
   to convex polygons and arbitrary smooth boundaries.

---

# §VII Conclusion

We presented **DT-ATACOM**, the first safety filter with explicit
discrete-time forward invariance guarantees for mobile agents at coarse
control rates. By diagnosing two structural failure modes — M1 (geometric
overshoot) and M2 (tangential drift) — and addressing each with a targeted
mechanism — velocity-adaptive margin and multi-step BRT lookahead — we
achieve 12/15 GO on Safety-Gymnasium, a 4× improvement over the best
published filter.

**Proposition 4** provides constructive hyperparameter selection: given
$\Delta t$, $r$, and $v_\max$, it prescribes the minimal $\alpha$ and $h$
that ensure safety. This bridges the gap between continuous-time ATACOM
theory and practical discrete-time deployment.

The key insight is that single-step analysis is insufficient at coarse
$\Delta t$. Safety requires anticipating multi-step drift before it
accumulates to penetration. DT-ATACOM's two-component design captures
this insight with minimal computational overhead (1.4× vs baseline).

We hope this work encourages the safety filter community to engage with
mobile-agent RL benchmarks, and the RL community to adopt hard safety
guarantees beyond soft Lagrangian constraints.
