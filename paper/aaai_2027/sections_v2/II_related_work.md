# §II Related Work

Safety in reinforcement learning has been approached from three
structurally different directions. DT-ATACOM belongs to the first
family but addresses a gap that none of the existing methods fill.

## §II-A Tangent-Projection Safety Filters (DT-ATACOM's family)

A safety filter intercepts the policy's commanded action and projects
it onto the constraint manifold's tangent space at each control instant.
Forward invariance is proven in continuous time by zeroing the radial
component of the constraint Jacobian.

| Sub-family | Representative works |
|---|---|
| Null-space projection | Khatib 1987, Sentis 2005 |
| ATACOM family | Liu 2022, Liu 2024 thesis, ATACOM-VD/-S/-LA |
| Control Barrier Functions | Ames 2014, 2017; Cheng 2019 |
| Higher-order CBF (HOCBF) | Xiao 2021 |
| Discrete-time CBF (DCM) | Agrawal & Sreenath 2017 |
| Reachability-based margins | Liu thesis 2024 §III-D |

**The common assumption.** All these methods derive their guarantees
from continuous-time analysis (ODEs) or single-step discrete analysis.
The original ATACOM paper and its descendants evaluate on high-frequency
manipulator control ($\Delta t \le 2$ ms) where the continuous-time
assumption holds empirically. When ported to Safety-Gymnasium at
$\Delta t = 0.1$ s, the assumption is violated by two orders of
magnitude and the guarantees break down.

**DCM is closest but insufficient.** Agrawal & Sreenath 2017 propose
a discrete-time CBF formulation that explicitly accounts for one-step
Euler integration. However, DCM addresses only single-step constraint
satisfaction; it does not account for the multi-step tangential drift
(M2) that accumulates over several committed actions. In our benchmark,
DCM achieves 3/15 GO — better than vanilla ATACOM (0/15) but far from
DT-ATACOM's 12/15.

**DT-ATACOM's position.** We stay within the tangent-projection family
but add two mechanisms that the existing members lack: (1) velocity-adaptive
margin to absorb M1 overshoot, and (2) multi-step BRT lookahead to
anticipate M2 drift. Proposition 4 provides the first discrete-time
invariance theorem that prescribes when these mechanisms are sufficient.

## §II-B Set-Based Safety Methods

Set-based methods compute a **safe set** offline (the backward-reachable
tube under worst-case disturbances) and enforce safety at runtime by
querying or projecting onto the set.

| Sub-family | Representative works |
|---|---|
| Hamilton–Jacobi reachability | Mitchell 2005, Bansal 2017 survey |
| Safety Bellman equation | Fisac 2018 |
| Learned safe sets / Neural CBF | Robey 2020, Lindemann 2023 |

**Relation to DT-ATACOM.** DT-ATACOM's multi-step BRT lookahead is a
discrete-time, online approximation of the HJ backward-reachable tube.
Instead of solving the HJB PDE offline, we forward-simulate 9 worst-case
directions over $h$ steps at each filter call. This is conservative for
small $h$ but tractable in the RL inner loop (27 position evaluations
per step at $h = 3$).

**Why we don't use offline HJ-BRT.** The offline approach suffers from
the curse of dimensionality (state space gridding) and offline→online
distribution shift when the policy explores states not covered by the
precomputed tube. Our online simulation sidesteps both issues at the
cost of a lookahead approximation.

## §II-C Lagrangian Soft Constraints

Lagrangian methods relax the hard safety constraint into a penalty
term whose multiplier is learned online to satisfy a budget on expected
cost.

| Sub-family | Representative works |
|---|---|
| Constrained policy optimisation | Achiam 2017 (CPO), Yang 2020 (PCPO) |
| PPO-Lagrangian | Ray 2019 (Safety-Gym) |

**Difference from DT-ATACOM.** Lagrangian methods provide no per-step
guarantee; safety is achieved *in expectation* over the episode. They
are widely used because they require no filter mechanism, but they
cannot prevent individual constraint violations. We include PPO-Lag
in the benchmark (§V) as a soft-constraint reference; at matched 200K-step
budget it achieves 1/15 GO, confirming that soft constraints do not
substitute for hard filtering at this training scale.

## §II-D Gap Analysis: Why the Communities Have Not Crossed

The mismatch DT-ATACOM addresses lives at the intersection of two
benchmark cultures:

1. **Manipulator control** ($\Delta t \le 2$ ms): ATACOM/CBF filters
   work perfectly; the M1 threshold is three orders of magnitude away
   and M2 reduces to harmless wall-sliding.

2. **Mobile-agent RL** ($\Delta t = 0.1$ s): Lagrangian methods dominate
   the leaderboard; the filter community has not engaged at scale.

The closest prior analyses (Liu thesis 2024 §III-D on velocity-adaptive
margins; Cheng 2019 on mobile CBF; Choi 2020 on hybrid HJ+CBF) observe
individual symptoms but do not formalise M1/M2 or benchmark the filter
family breadth. DT-ATACOM closes this gap with a combined mechanism
and a discrete-time invariance proof.
