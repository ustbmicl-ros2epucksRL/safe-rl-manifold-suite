# §III Preliminaries

## §III-A Constrained MDP Setting

We consider safe reinforcement learning in the constrained MDP (CMDP)
framework. A CMDP is a tuple $(S, A, P, r, c, \gamma, \bar{C})$ where
$S$ is the state space, $A$ the action space, $P$ the transition kernel,
$r$ the reward function, $c: S \times A \to \{0, 1\}$ a binary cost
indicator, and $\bar{C}$ the episode cost threshold.

**Benchmark.** We evaluate on Safety-Gymnasium [Ji2023], specifically the
Point-Robot environment with Goal, Push, and MultiGoal tasks. The setting
features:

- **Time step**: $\Delta t = 0.1$ s (10 Hz control)
- **Obstacles**: 8 circular hazards with radius $r = 0.2$ m
- **Cost function**: $c = 1$ if agent-hazard distance $< r$, else $c = 0$
- **GO threshold**: $\bar{C} \le 5.0$ (mean over 50 evaluation episodes)

**Training budget.** All experiments use 200K environment steps with 5
random seeds. This matches prior work on safe RL sample efficiency and
enables controlled comparison.

## §III-B ATACOM Continuous-Time Theorem

ATACOM (Action Transformation based on Constrained Manifold) [Liu2022]
provides forward invariance of the constraint manifold through null-space
projection. We restate the core result.

**Definition (Constraint function).** Let $\boldsymbol{c}(\boldsymbol{q}):
\mathbb{R}^n \to \mathbb{R}^m$ encode $m$ safety constraints. The safe set
is $\mathcal{S} = \{\boldsymbol{q} : \boldsymbol{c}(\boldsymbol{q}) \le 0\}$.

**Constraint Jacobian.** Define $\boldsymbol{J}_c = \partial\boldsymbol{c}
/\partial\boldsymbol{q} \in \mathbb{R}^{m \times n}$.

**ATACOM dynamics.** Given an RL action $\boldsymbol{\alpha} \in \mathbb{R}^k$,
ATACOM transforms it via:
$$
\dot{\boldsymbol{q}} = \boldsymbol{N}_c \boldsymbol{\alpha}
  - K_c \boldsymbol{J}_c^+ \boldsymbol{c}(\boldsymbol{q})
\tag{III-1}
$$
where $\boldsymbol{N}_c = \boldsymbol{I} - \boldsymbol{J}_c^+
\boldsymbol{J}_c$ is the null-space projector and $K_c > 0$ is a
correction gain.

**Theorem 1 (Liu 2022).** Under the continuous-time dynamics (III-1), the
constraint value evolves as:
$$
\dot{\boldsymbol{c}} = -K_c \boldsymbol{c}
\implies
\boldsymbol{c}(t) = \boldsymbol{c}(0) e^{-K_c t}
\tag{III-2}
$$
Hence if $\boldsymbol{c}(0) \le 0$, then $\boldsymbol{c}(t) \le 0$ for all
$t \ge 0$. The safe set is forward invariant.

**Discretisation gap.** Theorem 1 assumes infinitesimal time steps. When
implemented with Euler integration at $\Delta t = 0.1$ s:
$$
\boldsymbol{q}_{k+1} = \boldsymbol{q}_k + \Delta t \cdot \boldsymbol{u}_k
\tag{III-3}
$$
the continuous-time guarantee degrades. The next section characterises the
two failure modes and proposes DT-ATACOM to address them.

## §III-C Circular Obstacle Geometry

For a single circular hazard at position $\boldsymbol{p}_o$ with radius
$r$, the signed distance constraint is:
$$
c(\boldsymbol{q}) = r^2 - \|\boldsymbol{q} - \boldsymbol{p}_o\|^2
\tag{III-4}
$$
where $c < 0$ means the agent is outside the hazard (safe), $c = 0$ on the
boundary, and $c > 0$ inside (violation).

**Keepout band.** With safety margin $d_\text{safe}$, the effective
constraint becomes $c(\boldsymbol{q}) + d_\text{safe} \le 0$, enforcing
minimum distance $\sqrt{r^2 + d_\text{safe}} - r \approx d_\text{safe}/(2r)$
from the hazard surface.

**Polar coordinates.** Let $\rho = \|\boldsymbol{q} - \boldsymbol{p}_o\|$
be the radial distance and decompose velocity as:
$$
\boldsymbol{u} = u_\rho \hat{\boldsymbol{e}}_\rho
              + u_\theta \hat{\boldsymbol{e}}_\theta
\tag{III-5}
$$
where $u_\rho$ is the radial component and $u_\theta$ the tangential
component. A tangent-projection filter zeroes $u_\rho$ when $c \ge 0$,
commanding motion purely in $\hat{\boldsymbol{e}}_\theta$.

This polar decomposition is central to understanding M1 and M2.
