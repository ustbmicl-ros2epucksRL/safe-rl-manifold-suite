# §IV DT-ATACOM Algorithm

This section presents DT-ATACOM (Discrete-Time ATACOM), a safety filter
designed for coarse control rates. We first diagnose why continuous-time
filters fail (§IV-A), then present the two-component solution (§IV-B),
and finally prove discrete-time forward invariance (§IV-C).

## §IV-A Why Continuous-Time Filters Fail

When tangent-projection filters are discretised with Euler integration at
$\Delta t = 0.1$ s, two structural failure modes emerge.

### M1: Chord Excursion and Bearing Rotation

**Proposition 1 (Single-step bearing rotation).** Let
$\rho_k := \|\boldsymbol{q}_k - \boldsymbol{p}_o\| > 0$ and decompose the
velocity as $\boldsymbol{u}_k = u_\rho \hat{\boldsymbol{e}}_\rho^{(k)} +
u_\theta \hat{\boldsymbol{e}}_\theta^{(k)}$ with
$\hat{\boldsymbol{e}}_\rho^{(k)} := (\boldsymbol{q}_k - \boldsymbol{p}_o)/\rho_k$
and $\hat{\boldsymbol{e}}_\theta^{(k)} \perp \hat{\boldsymbol{e}}_\rho^{(k)}$.
Under purely tangential motion ($u_\rho = 0$), Pythagoras applied to
$\boldsymbol{q}_{k+1} = \boldsymbol{q}_k + \Delta t\,\boldsymbol{u}_k$ gives
$$
\rho_{k+1}^2 = \rho_k^2 + \Delta t^2 u_\theta^2, \quad
\delta_\text{chord} := \rho_{k+1} - \rho_k \approx
\frac{\Delta t^2 u_\theta^2}{2 \rho_k}.
\tag{IV-1}
$$
The discrete chord trajectory lies *outside* the constant-$\rho$ arc, so
single-step safety is preserved. The geometric cost is a rotation of the
obstacle's bearing by
$$
\phi_k = \arctan\!\bigl(\Delta t\,u_\theta / \rho_k\bigr)
\approx \Delta t\,u_\theta / \rho_k,
\tag{IV-2}
$$
giving the world-frame velocity $\boldsymbol{u}_k$ a radial-inward
component $\approx -\Delta t\,u_\theta^2/\rho_k$ relative to the new radial
direction $\hat{\boldsymbol{e}}_\rho^{(k+1)}$. If the policy commits to
this world-frame direction across steps (momentum at coarse filter
rates), the inward drift compounds into M2.

**Threshold.** M1 becomes operationally significant when:
$$
\Delta t \cdot \|u\| \gtrsim \sqrt{d_\text{safe} \cdot r}
\tag{IV-3}
$$
At $\Delta t = 0.1$ s, $r = 0.2$ m, $d_\text{safe} = 0.05$ m, and
$\|u\| = 1$ m/s, the LHS is 0.1 and RHS is 0.1 — the threshold is reached.

### M2: Multi-Step Drift from Sustained Commitment

**Proposition 2 (Multi-step drift).** Suppose the policy commits to a
fixed world-frame velocity $\boldsymbol{u} = v\,\hat{\boldsymbol{e}}_\theta^{(0)}$
that was tangent at $\boldsymbol{q}_0$. By (IV-2), the radial component of
$\boldsymbol{u}$ relative to the bearing at step $k$ is $-v\sin\phi_k
\approx -\Delta t\,v^2/\rho_k$, contributing per-step inward drift
$\Delta t \cdot v \sin\phi_k \approx \Delta t^2 v^2/\rho_k$. Summing:
$$
\Delta\rho_n := \rho_0 - \rho_n \;\le\;
\sum_{k=0}^{n-1} \frac{\Delta t^2 v^2}{\rho_k}.
\tag{IV-4}
$$

**Assumption.** The agent remains in the keepout band, i.e.,
$\rho_k \in [\rho_\min, \rho_\max]$ with
$0 < \rho_\min \le \rho_k \le \rho_\max \le r + d_\text{safe}$.

Under this assumption:
$$
\frac{n\,\Delta t^2 v^2}{\rho_\max}
\le \Delta\rho_n \le
\frac{n\,\Delta t^2 v^2}{\rho_\min}
\tag{IV-5}
$$
The drift accumulates **linearly** in $n$.

**Corollary (Penetration bound).** If drift must stay within $d_\text{safe}$:
$$
n^\star \le \left\lfloor
  \frac{d_\text{safe} \cdot \rho_\min}{\Delta t^2 v^2}
\right\rfloor
\tag{IV-6}
$$
At $d_\text{safe} = 0.05$ m, $\rho_\min = 0.25$ m, $v = 1$ m/s,
$\Delta t = 0.1$ s: per-step drift $\le 0.04$ m, so $\Delta\rho_1 \le 0.04$
($< d_\text{safe}$) but $\Delta\rho_2 \le 0.08$ ($> d_\text{safe}$) —
penetration at $n^\star = 2$ steps. A single-step ($h{=}1$) filter cannot
anticipate M2.

### Why Single-Step Filters Fail

Both M1 and M2 operate at $O(\Delta t^2)$ per step but accumulate over the
trajectory. A filter that only examines the *current* step cannot anticipate
the drift that will occur over subsequent committed actions. This explains
why DCM (discrete-time CBF for single steps) achieves only 3/15 GO.

## §IV-B DT-ATACOM: Two-Component Design

DT-ATACOM addresses M1 and M2 with complementary mechanisms.

### Component 1: Velocity-Adaptive Margin (addresses M1)

We inflate the keepout radius proportionally to speed:
$$
r_\text{eff}(\boldsymbol{v}) = r_\text{base}(1 + \alpha\|\boldsymbol{v}\|)
\tag{IV-7}
$$
where $\alpha > 0$ is the margin coefficient.

**Intuition.** The M1 penetration scales as $\Delta t^2 \|u\|^2 / \rho$.
By expanding the effective radius by $\alpha \|v\|$, we absorb this
overshoot before it reaches the true constraint boundary.

**Rule-of-thumb (M1 chord absorption).** The chord excursion (IV-1)
scales as $\Delta t^2 \|u\|^2/\rho$. The VAM inflation
$r_\text{base}\alpha\|v\|$ absorbs this rotation budget when
$$
\alpha \gtrsim \frac{\Delta t^2 v_\max}{2r}.
\tag{IV-9}
$$
This is a useful scaling but is neither necessary nor strictly sufficient;
the rigorous condition is given by Proposition 4 below and additionally
engages a danger-zone buffer $d_\text{danger}$ to absorb linear-in-$\Delta t$
direct displacement.

### Component 2: Multi-Step BRT Lookahead (addresses M2)

We approximate the backward reachable tube (BRT) via forward simulation:
$$
\hat{c}(\boldsymbol{p}, \boldsymbol{v}, h) =
\min_{d \in \mathcal{D}} \min_{t=1}^{h} \min_i
\|\boldsymbol{p}_t(d) - \boldsymbol{p}_o^{(i)}\|^2 - r^2
\tag{IV-10}
$$
where:
- $\mathcal{D} = \{(\cos\theta, \sin\theta) : \theta \in \{0, 40°, ..., 320°\}\}
\cup \{(0,0)\}$ is a 9-direction discretisation
- $\boldsymbol{p}_t(d) = \boldsymbol{p} + t \cdot \Delta t \cdot v_\max \cdot d$
is the position after $t$ steps in direction $d$
- $h$ is the lookahead horizon

**Intuition.** M2's drift accumulates over multiple steps. By simulating $h$
steps into the future under worst-case directions, we detect impending
penetration before the agent commits to a trajectory that will violate
constraints.

**Condition.** The lookahead must cover the time window during which drift
can accumulate to penetration. From (IV-6):
$$
h \ge \left\lceil \frac{r}{\Delta t \cdot v_\max} \right\rceil
\tag{IV-11}
$$

### Algorithm 1: DT-ATACOM

```
Input: action a, state s=(p, v), obstacles O, params (α, h)
Output: safe action a_safe

1. Velocity-adaptive margin:
   r_eff = r_base × (1 + α × ||v||)

2. For each obstacle o_i ∈ O:
   d_i = ||p - p_o^(i)|| - r_eff

3. BRT lookahead:
   For each direction d ∈ D:
     For t = 1 to h:
       p_t = p + t × Δt × v_max × d
       ĉ_t = min_i (||p_t - p_o^(i)||² - r²)
     ĉ(d) = min_t ĉ_t
   ĉ_min = min_d ĉ(d)

4. Danger assessment:
   If min(d_i) < d_danger OR ĉ_min < 0:
     // In danger zone: apply velocity scaling
     scale = clip(min(d_i) / d_safe, 0, 1)
     a_safe = scale × project_tangent(a)
   Else:
     a_safe = a

5. Return a_safe
```

**Computational cost.** Step 3 evaluates $9 \times h \times |O|$ distance
computations per filter call. At $h = 3$ and $|O| = 8$: 216 scalar
operations — negligible compared to neural network inference.

## §IV-C Proposition 4: Discrete-Time Forward Invariance

We now state the main theoretical result.

**Theorem (Proposition 4).** Let $\{c_i(\boldsymbol{q}) = r^2 -
\|\boldsymbol{q} - \boldsymbol{p}_o^{(i)}\|^2\}_{i=1}^m$ be $m$ circular
constraints with common radius $r$, safety margin $d_\text{safe} > 0$,
and $r_\text{base} := r + d_\text{safe}$. Define $\rho_i(\boldsymbol{q}) :=
\|\boldsymbol{q} - \boldsymbol{p}_o^{(i)}\|$ and the safe set
$\mathcal{S} := \{\boldsymbol{q} : \min_i \rho_i(\boldsymbol{q}) \ge
r_\text{base}\}$. Consider an agent with discrete dynamics
$\boldsymbol{q}_{k+1} = \boldsymbol{q}_k + \Delta t\,\boldsymbol{u}_k$,
$\|\boldsymbol{u}_k\| \le v_\max$, where $\boldsymbol{u}_k$ is produced
by Algorithm 1 (DT-ATACOM) with hyperparameters
$(\alpha, h, d_\text{danger})$.

If
$$
\begin{aligned}
& \alpha\,r_\text{base}\,v_\max + d_\text{danger}
\;\ge\; \Delta t\, v_\max,
\quad h \;\ge\; \left\lceil \frac{r}{\Delta t \cdot v_\max} \right\rceil, \\
& \text{and}\quad r \;\ge\; d_\text{safe} + \Delta t\, v_\max,
\end{aligned}
\tag{IV-12}
$$
then $\boldsymbol{q}_0 \in \mathcal{S} \Rightarrow \boldsymbol{q}_k \in
\mathcal{S}$ for all $k \ge 0$.

**Proof.** By induction on $k$. The base case is by hypothesis. Suppose
$\rho_k \ge r_\text{base}$. Decompose $\boldsymbol{u}_k = u_\rho
\hat{\boldsymbol{e}}_\rho^{(k)} + \boldsymbol{u}_\perp$ as in (IV-1); the
recursion
$$
\rho_{k+1}^2 = \rho_k^2 + 2\rho_k \Delta t\,u_\rho + \Delta t^2 \|\boldsymbol{u}_k\|^2
\tag{IV-13}
$$
follows from squaring $\boldsymbol{q}_{k+1} - \boldsymbol{p}_o$. Algorithm 1
branches at line 14 into three cases.

*Case 1 (Pass-through; lines 17–18).* If $\min_i d_i \ge d_\text{danger}$
and $\hat c_\min \ge 0$, then $\boldsymbol{u}_k = \boldsymbol{a}_k$
unfiltered. The pass condition gives $\rho_k \ge r_\text{eff}(\boldsymbol{v}_k)
+ d_\text{danger} = r_\text{base}(1 + \alpha\|\boldsymbol{v}_k\|) +
d_\text{danger}$. The triangle inequality yields $\rho_{k+1} \ge \rho_k -
\Delta t\,\|\boldsymbol{u}_k\|$. At the worst case
$\|\boldsymbol{v}_k\| = \|\boldsymbol{u}_k\| = v_\max$:
$$
\rho_{k+1} \ge r_\text{base} + \bigl(\alpha\,r_\text{base}\,v_\max +
d_\text{danger} - \Delta t\,v_\max\bigr) \ge r_\text{base},
$$
where the final inequality is (IV-12)$_1$.

*Case 2 (Filtered, no BRT trigger; lines 14–16).* If $\min_i d_i <
d_\text{danger}$ and $\hat c_\min \ge 0$, the filter outputs
$\boldsymbol{u}_k = \text{scale}\cdot \mathrm{proj}_\text{tan}(\boldsymbol{a}_k)$
with $\mathrm{proj}_\text{tan}(\boldsymbol{a}) := \boldsymbol{a} -
\min(0, \boldsymbol{a}\cdot\hat{\boldsymbol{e}}_\rho^{(k)})
\hat{\boldsymbol{e}}_\rho^{(k)}$ zeroing the inward radial component. By
construction $u_\rho \ge 0$, so (IV-13) yields
$$
\rho_{k+1}^2 \ge \rho_k^2 + \Delta t^2 \|\boldsymbol{u}_k\|^2 \ge \rho_k^2
\ge r_\text{base}^2.
$$

*Case 3 (BRT trigger).* If $\hat c_\min < 0$, the filter executes the
same scaled tangent projection as Case 2, giving $\rho_{k+1} \ge \rho_k
\ge r_\text{base}$.

*Role of the horizon.* Cases 2–3 close the single-step bound. The horizon
condition (IV-12)$_2$ ensures the trigger fires *before* cumulative drift
(Prop. 2) can carry the agent below $r_\text{base}$: each BRT rollout
covers $h\,\Delta t\,v_\max \ge r$ of displacement, detecting any
direction that would collide within the horizon. Combining cases,
$\rho_{k+1} \ge r_\text{base}$. $\square$

**Remark (rule-of-thumb form).** When $d_\text{danger} \ge \Delta t\,v_\max$,
condition (IV-12)$_1$ is automatic for any $\alpha \ge 0$. In practice we
want $\alpha > 0$ regardless, to absorb the M1 chord excursion (IV-1) per
step; the independent M1 heuristic (IV-9) governs this design. Empirical
choices ($\alpha = 0.3$, $d_\text{danger} = 0.05$ on Safety-Gym) satisfy
both the strict bound (IV-12)$_1$ and the heuristic (IV-9) with margin.

**Corollary 1 (Continuous-time limit).** As $\Delta t \to 0$ with $v_\max$
and $r$ fixed:
(i) (IV-12)$_1$ becomes vacuous, satisfied by any $\alpha \ge 0$,
$d_\text{danger} \ge 0$;
(ii) $h^\star := \lceil r/(\Delta t v_\max)\rceil \to \infty$ but
$h^\star \Delta t v_\max \to r$, i.e., the BRT horizon-time stays bounded;
(iii) the VAM inflation $r_\text{eff} - r_\text{base} = r_\text{base}
\alpha v_\max \to 0$ if $\alpha$ scales as $\Delta t^2$. DT-ATACOM thus
reduces to the continuous-time tangent-projection flow
$\dot{\boldsymbol{q}} = \boldsymbol{N}_c\boldsymbol{\alpha} -
K_c \boldsymbol{J}_c^+ \boldsymbol{c}$, recovering ATACOM's Theorem 1.

## §IV-D Constructive Hyperparameter Selection

Given Safety-Gymnasium parameters ($\Delta t = 0.1$ s, $r = 0.2$ m,
$d_\text{safe} = 0.05$ m, $v_\max = 1.0$ m/s), $r_\text{base} = 0.25$ m.

The M1 rule-of-thumb (IV-9) and the BRT horizon (IV-11) prescribe
$$
\alpha \gtrsim \frac{(0.1)^2 \cdot 1.0}{0.2} = 0.05,\quad
h \ge \left\lceil \frac{0.2}{0.1 \cdot 1.0} \right\rceil = 2,
$$
while the rigorous bound (IV-12)$_1$ requires $\alpha \cdot 0.25 +
d_\text{danger} \ge 0.1$.

**Implementation choice.** We use $\alpha = 0.3$, $h = 3$, and
$d_\text{danger} = 0.05$ throughout. This satisfies (IV-12)$_1$ with
slack $0.3 \cdot 0.25 + 0.05 - 0.1 = 0.025$, comfortably exceeds the M1
heuristic (IV-9) by 6$\times$, and over-provisions the horizon by one
step. Section V validates these choices empirically.

## §IV-E Relation to Prior Work

**Vs. ATACOM.** DT-ATACOM inherits the null-space projection structure but
adds the two discrete-time mechanisms. When $\Delta t \to 0$, both
components become unnecessary and DT-ATACOM reduces to ATACOM.

**Vs. DCM.** Agrawal & Sreenath (2017) address single-step discrete CBF
constraints. DT-ATACOM extends this with multi-step lookahead (Component 2),
capturing drift that single-step analysis misses.

**Vs. HJ-BRT.** Hamilton-Jacobi reachability computes the exact BRT offline
via PDE solving. DT-ATACOM approximates this online with forward simulation,
avoiding the curse of dimensionality and distribution shift issues.
