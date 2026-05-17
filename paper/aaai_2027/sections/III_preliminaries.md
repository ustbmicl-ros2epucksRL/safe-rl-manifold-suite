# §III  Preliminaries

*(Draft for AAAI 2027.  Target ~3/4 page in the AAAI template.
Provides the constrained-MDP setup, the §III-B ATACOM theorem that
§IV cites, and the §III-C filter-pipeline notation used in §V.)*

## §III-A  Constrained MDP and the Safety-Gymnasium point-robot benchmark

We work in the constrained Markov decision process (CMDP) setting
[Altman 1999]. An MDP $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ is
augmented with a per-step cost signal
$c: \mathcal{S}\times\mathcal{A}\!\to\!\{0,1\}$ that indicates
safety-constraint violation at the current state. The policy
$\pi_\theta(a\mid s)$ is trained to maximise the expected
discounted reward subject to keeping the expected discounted cost
below a budget; the per-episode safety metric we report is the
undiscounted episode cost
$C = \sum_{t=0}^{T-1} c(s_t, a_t)$, and the GO threshold used
throughout the paper is $\bar C \le 5.0$.

All experiments use Safety-Gymnasium [Ji et al. 2023] in the
Point-Robot regime: 2-D navigation tasks with eight circular
hazard regions of radius $r = 0.2$ m, a goal region, and
optionally a push-box. Control is at $\Delta t = 0.1$ s; the
agent action is two-dimensional (`forward_vel`, `angular_vel`
under the `diff_drive` form, or planar acceleration under
`cartesian`). The observation includes ego-state, hazard
egocentric features, and goal-egocentric features (60-D for Goal
and MultiGoal, 76-D for Push). Per-step cost is
$c(s,a)=\mathbb{1}\big[\|q\!-\!p_o\|\le r\big]$ for the agent
entering any hazard radius.

Three task instances are used in this paper:

| Task | Env id | Hazards | Box | Goal switching |
|------|--------|--------:|:---:|:--------------:|
| Goal | `SafetyPointGoal1-v0` | 8 | no | no |
| Push | `SafetyPointPush1-v0` | 2 + pillar | yes | no |
| MultiGoal | `SafetyPointGoal2-v0` | 10 + vases | no | yes |

The standard episode horizon is $T = 1000$ env steps
(100 s of physical time per episode). Phase 3 budget is
5 seeds × 200K env steps × 50-episode deterministic eval per
seed. The numbers reported in §IV-D and §V-E follow this protocol
unmodified.

## §III-B  The ATACOM continuous-time forward-invariance theorem

The literature that we benchmark in §IV-D (ATACOM, HOCBF, DCM,
RBAM and their variants — Family 1 in §II-A) shares a common
analytic foundation [Liu 2022; Ames 2014; Xiao 2021]. We
recapitulate it in the form §IV's failure analysis relies on.

Let $\boldsymbol{q}\in\mathbb{R}^{n_q}$ be the agent configuration
and $\boldsymbol{c}(\boldsymbol{q})\le\mathbf{0}$ a vector of
inequality constraints. The constraint Jacobian is
$\boldsymbol{J}_c(\boldsymbol{q}) = \partial\boldsymbol{c}/\partial\boldsymbol{q}$,
and its null-space projector is
$\boldsymbol{N}_c = \boldsymbol{I} - \boldsymbol{J}_c^{+}\boldsymbol{J}_c$,
where $\boldsymbol{J}_c^{+}$ is the Moore–Penrose pseudoinverse.

The ATACOM safety filter outputs

$$
\dot{\boldsymbol{q}} \;=\;
\underbrace{\boldsymbol{N}_c(\boldsymbol{q})\,\boldsymbol{\alpha}}_{\text{tangent term}}
\;-\;
\underbrace{K_c\,\boldsymbol{J}_c^{+}(\boldsymbol{q})\,\boldsymbol{c}(\boldsymbol{q})}_{\text{radial correction}},
\tag{III-1}
$$

with $K_c > 0$ a class-K gain and $\boldsymbol{\alpha}\in\mathbb{R}^{n_q}$
the policy-commanded velocity in joint-velocity coordinates.

**Theorem (ATACOM continuous-time forward invariance).** Under
the ODE $\dot{\boldsymbol{q}}=\boldsymbol{u}$ with
$\boldsymbol{u}$ given by (III-1), the constraint value satisfies

$$
\dot{\boldsymbol{c}}
\;=\;
\boldsymbol{J}_c\,\dot{\boldsymbol{q}}
\;=\;
\underbrace{\boldsymbol{J}_c\boldsymbol{N}_c}_{=\,\mathbf{0}}\,\boldsymbol{\alpha}
\;-\;K_c\,\boldsymbol{J}_c\boldsymbol{J}_c^{+}\,\boldsymbol{c}
\;=\;
-K_c\,\boldsymbol{c},
\tag{III-2}
$$

so $\boldsymbol{c}(t)=\boldsymbol{c}(0)\,e^{-K_c t}$ and the
safe set $\{\boldsymbol{q}: \boldsymbol{c}(\boldsymbol{q})\le\mathbf{0}\}$
is forward-invariant *for any* policy $\boldsymbol{\alpha}$.

Two structural reads of (III-2) are useful for §IV.

(i) The first product $\boldsymbol{J}_c\boldsymbol{N}_c = \mathbf{0}$
encodes the *tangent assumption*: the policy contributes nothing
to the radial direction of $\boldsymbol{c}$, no matter what it
commands.  §IV-A will show this assumption breaks at second order
in discrete time (M1).

(ii) The exponential decay $\boldsymbol{c}(t)=\boldsymbol{c}(0)\,e^{-K_c t}$
is a continuous-time statement. Replacing
$\dot{\boldsymbol{c}}=-K_c\boldsymbol{c}$ with its Euler
discretisation
$\boldsymbol{c}_{k+1}=(1-K_c\Delta t)\boldsymbol{c}_k$ requires
$K_c\Delta t<1$, which is met for any reasonable choice; the
*decay rate* of the radial correction is therefore not the
bottleneck. The bottleneck is what happens to the **tangential
displacement** $\boldsymbol{N}_c\boldsymbol{\alpha}\cdot\Delta t$
over one step — §IV-B analyses this directly (M2).

Variants of (III-1) used in §IV-D include HOCBF [Xiao 2021] which
replaces the linear class-K term with a two-stage condition
$\dot{h}\ge-\alpha_1 h$, $\ddot{h}\ge-\alpha_2(\dot{h}+\alpha_1 h)$;
DCM [Agrawal & Sreenath 2017] which replaces the ODE with an
explicit Euler step; and RBAM [Liu 2024 §III-D] which adds a
velocity-dependent margin to $r$. All inherit the
forward-invariance promise of (III-2) in their respective
analytic settings.

## §III-C  The safety-filter pipeline

Throughout the paper, we treat the safety filter as a wrapper
$\boldsymbol{a}_\text{safe} = W(\boldsymbol{s}, \boldsymbol{a}_\text{nom})$
that intercepts the policy-commanded action before
`env.step`:

$$
\boldsymbol{s}_t \;\xrightarrow{\pi_\theta}\; \boldsymbol{a}_\text{nom}
\;\xrightarrow{W}\; \boldsymbol{a}_\text{safe}
\;\xrightarrow{\text{env}}\; (\boldsymbol{s}_{t+1}, r_t, c_t).
\tag{III-3}
$$

The filter $W$ is deterministic and non-differentiable; the
policy is unaware of the projection. A *calibration* term in the
reward,
$r_t \leftarrow r_t - \lambda_\text{calib}\|\boldsymbol{a}_\text{safe}-\boldsymbol{a}_\text{nom}\|^2$,
discourages the policy from issuing commands that the filter
must heavily correct ($\lambda_\text{calib}=0.02$ throughout).
The PPO algorithm [Schulman et al. 2017] is then standard;
training rolls $\boldsymbol{a}_\text{safe}$ through the
environment but stores $\boldsymbol{a}_\text{nom}$ with its
on-policy log-prob in the rollout buffer. §IV-C tests an
alternative (`store_filtered_action=true`) and shows the choice
does not change the failure conclusion.

When a reachability layer is also present (used in our recipe,
§V), $W$ receives a one-step forward-rolled state $\hat
{\boldsymbol{s}}_{t+1\mid t}$ and shapes its keepout radius
accordingly. The implementation details are in §V-B; this
preliminaries section only fixes the notation
$W(\boldsymbol{s}, \boldsymbol{a})$ for the rest of the paper.

---

### TODO before submission

- [ ] Cross-ref (III-1) and (III-2) consistently after §V-C's
      Prop. 4 numbering settles
- [ ] Confirm Liu 2022 vs. Liu 2024 citation style matches the
      rest of §II
- [ ] Tighten the GO-threshold derivation note: $\bar C \le 5$
      is a held-over Phase-2 convention; if a reviewer asks, we
      cite the legacy v4 §IV table footnote in the supplementary
- [ ] Add a one-line pointer to the §V-B sim-BRT definition for
      readers who skim §III-C looking for $W$ details
