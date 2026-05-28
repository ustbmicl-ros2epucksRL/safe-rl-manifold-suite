# AAAI 2027 — Selected Direction (post-pivot)

**Title (working).** Why tangent-projection safety filters fail
in discrete-time mobile-agent RL: a failure-mode taxonomy.

(Scope tightened 2026-05-15 from "continuous-time safety filters"
to "tangent-projection safety filters" so that set-based methods —
HJ-reachability, Robey 2020 Neural CBF — are explicitly outside
our analytic claims; see §II-A and the scope statement at the top
of §IV.)

**Pivot date.** 2026-05-14.

**Pivot trigger.** Phase 3 T8 (PRCF Goal × 5 seeds × 200K, see
`runs/phase3_t8_prcf_goal/`) returned 0/5 GO at mean C=17.9,
which ties the no-filter PPO baseline (C=22.8) and is 21× worse
than `Ours` (DistAdapt + sim-BRT, C=0.84). The previously
selected "PRCF as new algorithm" framing (preserved in
`SELECTED_ORIGINAL_PRCF.md`) is no longer empirically
defensible; the pivot rationale is in
`paper/aaai_2027/sections/FALLBACK_NARRATIVE.md`.

---

## 1. One-line summary

Across 8 published **tangent-projection** safety-filter
families plus PPO-Lag as a Lagrangian reference, on 4
Safety-Gym tasks, we identify three structural failure modes —
geometric overshoot, tangential preservation pathology, and
gradient-credit irrelevance — that arise when continuous-time
*tangent-projection* filters are applied at the
$\Delta t = 0.1\,\text{s}$ control rates standard in
mobile-agent RL.  We show that a two-line recipe
(velocity-adaptive stop-radius + 3-step BRT lookahead) survives
this regime where every other method collapses, and we
formalise the result as the first explicit discrete-time
analogue of the ATACOM continuous-time forward-invariance
theorem.

**Out of scope** (acknowledged in §II-A and §IV scope statement):
set-based safety methods (HJ-reachability, Robey 2020 Neural
CBF). These have a different failure profile because they
compute a multi-step safe set offline rather than projecting
per-step; M1–M3 of §IV are specific to per-step tangent
projection.  Our recipe's sim-BRT component is itself a
poor-man's approximation to HJ-BRT, bridging Family 1 and
Family 2 in discrete time.

---

## 2. Three named contributions

### C1.  3-axis failure-mode taxonomy (tangent-projection family)

We isolate three structural failures of **continuous-time
tangent-projection** safety filters under coarse $\Delta t$,
each backed by a small proposition and an existing experiment.
Scope: §II-A Family 1 (null-space projection, CBF, HOCBF, DCM,
RBAM and ATACOM extensions). Set-based methods (Family 2) are
out of scope and have separate failure modes documented in
Appendix VII-C.

| Mode | Characterisation | Theorem | Evidence |
|------|------------------|---------|----------|
| **M1: Geometric overshoot** | Even with perfect tangent projection, $c_{k+1}\!-\!c_k \ge \Delta t^2\|u\|^2$ is strictly positive. | §IV-A Prop. 1 | D5 Δt-sweep: ATACOM C 0.00→39.95 at $\Delta t$ crossing 0.05. |
| **M2: Tangential preservation pathology** | Null-space projection preserves the tangent component, but for circular hazards the tangent rotates relative to the obstacle centre over $\Delta t$. | §IV-B Prop. 2 | PM8: 4 ATACOM variants (vanilla, VD, S, LA) all fail; mean C ∈ [18, 84]. |
| **M3: Gradient credit is not the bottleneck** | Buffer-side post-filter log-prob (`store_filtered_action=True`) does not rescue the family. | §IV-C Prop. 3 | PM9 (T7): 5/45 GO → 3/45 GO across 45 (filter, env, seed) cells. |

### C2.  The 8-filter benchmark on Safety-Gym point-robot

The largest controlled empirical comparison of safety filters on
Safety-Gym to date: 8 filter families × 4 tasks × 5 seeds × 200K
steps, plus the D5 Δt-sweep ablation and the PM9
store_filtered_action ablation. See §V Table 1.

### C3.  A discrete-time-aware two-line recipe

We do not propose a new filter algorithm. We propose a design
rule with two ingredients (both individually present in the
literature) whose combination has not been benchmarked:

1.  **Velocity-adaptive stop-radius.** Inflate the keepout radius
    by $\alpha\|\boldsymbol{v}\|$ rather than relying on
    null-space projection. The inflation absorbs the
    $\Delta t\|\boldsymbol{u}\|$ overshoot from M1 directly.
2.  **Multi-step BRT lookahead.** Evaluate safety over the next
    $h\!=\!3$ steps under sampled action rollouts. At $\Delta t = 0.1$ s
    this spans 0.3 s, exceeding the single-step overshoot regime.

**Proposition 4** (informal). For
$\alpha \ge \Delta t$ and
$h \ge \lceil\sqrt{d_\text{safe}/r}\rceil$, the inflated keepout
radius is positively invariant for all velocities
$\|\boldsymbol{v}\|\le v_\max$ that the policy can produce.

This is the **first explicit discrete-time analogue of ATACOM's
continuous-time forward-invariance theorem.** We are translating
the safety guarantee from continuous to discrete time, not
proposing a new algorithm.

---

## 3. What the data already supports

| Claim | Status | Cells |
|---|---|---|
| 8 filter families fail uniformly on Goal/Push/MGoal | ✅ done | 24 (filter, env) × 5 seeds = 120 runs |
| ATACOM C 0.00→39.95 at Δt=0.05 (M1 evidence) | ✅ done | D5 v2 sweep, 48 runs |
| store_filtered_action=true does not rescue (M3 evidence) | ✅ done | T7, 45 runs |
| Ours stays sub-5 cost across Δt ∈ {0.01, 0.02, 0.10} | ✅ done | D5 v2 |
| Ours headline: 5/5 GO on Goal, 4/5 on MGoal, 3/5 on Push | ✅ done | T1/T2, 15 runs |

---

## 4. What needs to be added (1.5–2 days)

1.  **One Δt-sweep cell with ATACOM-LA** to validate M2 is
    distinct from M1 (~30 min).
2.  **Empirical tightness of Proposition 4**: measure
    $\max\|\boldsymbol{v}\|$ in Ours rollouts and verify
    $\alpha\|\boldsymbol{v}\|$ exceeds the M1 penetration bound
    (~1 day).
3.  **One extra-seed pair** on the noisy cells (Push,
    Ours/Δt=0.05) to confirm variance is policy-search not
    methodology (~2 h).

The pivot is largely a **restructuring of existing data**, not a
new experimental programme.

---

## 5. PRCF disposition

PRCF remains in the repo (`safe_rl/filters/critic_qp.py`,
`safe_rl/algos/ppo_critic_qp.py`) and is presented in the paper
as **§VII-C Appendix: an attempted offline-critic improvement
that does not yet transfer**. The most likely root cause is in-
distribution critic shift (H1); the fix is identified and
deferred to a follow-up paper.

This honest disclosure both (i) gives the reader a complete
picture of the design space we explored and (ii) signals to
reviewers that we did not cherry-pick the positive results.

---

## 6. Section structure

```
I.    Introduction
II.   Related work
        II-A. Three families of safe-RL methods
              Family 1: tangent-projection filters (this paper's scope)
              Family 2: set-based safety (parallel, out of scope)
              Family 3: Lagrangian soft constraints (reference)
        II-B. Why prior work has not identified the discrete-time gap
        II-C. Negative results and benchmarking-as-contribution
III.  Preliminaries (Safety-Gym, ATACOM continuous-time theorem)
IV.   The 3-axis failure taxonomy   ★ MAIN CONTRIBUTION
        IV-A. M1: Geometric overshoot (Prop. 1, D5 figure)
        IV-B. M2: Tangential preservation pathology (Prop. 2, PM8)
        IV-C. M3: Gradient credit irrelevance (Prop. 3, PM9 ablation)
        IV-D. The 10-method empirical benchmark (Table 1, includes
              8 Family-1 filters + PPO-Lag + PPO baseline)
V.    The recipe: stop-radius + BRT
        V-A. The two ingredients
        V-B. Proposition 4: discrete-time invariance
        V-C. Why this combination is not in the literature
        V-D. Empirical: 4 tasks × 5 seeds (Table 2)
VI.   Discussion (taxonomy generalisation, cliff cells, sim-to-real)
VII.  Appendix
        VII-A. Full per-seed data
        VII-B. D5 Δt-sweep raw numbers
        VII-C. PRCF — an attempted Family-2-style improvement that
               does not yet transfer
```

6 pages main + appendix (within AAAI 8-page limit).
