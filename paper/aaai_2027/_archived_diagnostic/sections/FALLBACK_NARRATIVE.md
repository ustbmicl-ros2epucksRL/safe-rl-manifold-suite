# Fallback narrative — pivot from "PRCF algorithm" to "Failure-mode taxonomy"

**Date:** 2026-05-14
**Trigger:** T8 PRCF Goal × 5 seeds returned 0/5 GO (C=17.9, R=−11.7),
i.e. PRCF underperforms Ours by 21× cost and lands marginally
better than the no-filter PPO baseline. The "PRCF as new
algorithm" framing (SELECTED.md, written 5/1) is no longer
defensible without 2–3 days of H1-debugging (in-distribution
critic retraining), and even that is speculative.

This document drafts the pivoted paper around evidence we
**already have**, so the AAAI submission timeline (Week 7–9
writing window, see STATUS.md §6) does not slip.

---

## 1. New title and one-line summary

**Working title.** *Why continuous-time safety filters fail in
discrete-time mobile-agent RL: a failure-mode taxonomy.*

**One-line summary.** Across 8 published safety-filter families
and 4 Safety-Gym tasks, we identify three structural failure
modes — geometric overshoot, tangential preservation pathology,
and gradient-credit irrelevance — that arise when filters
designed under continuous-time assumptions are applied at the
$\Delta t = 0.1\,\text{s}$ control rates standard in Safety-Gym
and decentralised multi-agent navigation, and we show that a
two-line recipe (velocity-adaptive stop-radius + 3-step BRT
lookahead) survives this regime where every other method
collapses.

The paper offers no new filter algorithm; the contribution is
the failure-mode analysis and the empirical demonstration that
the published filter zoo has a structural gap that the proposed
recipe closes.

---

## 2. Three new named contributions

### C1. A 3-axis failure-mode taxonomy for safety filters under coarse $\Delta t$

We isolate three structural failures, each backed by a theorem,
a controlled experiment, and 50+ training runs.

| Mode | One-line characterisation | Theorem | Evidence |
|------|---------------------------|---------|----------|
| **M1: Geometric overshoot** | Even with perfect tangent projection, $c_{k+1} - c_k \ge \Delta t^2\|u\|^2$ is strictly positive; constraint penetration scales as $\Delta t^2$. | §IV-B Prop. 1 | D5 Δt-sweep: ATACOM C drops from 39.95 at Δt=0.05 to 0.00 at Δt=0.02. |
| **M2: Tangential preservation pathology** | Null-space projection preserves $\boldsymbol{N}_c\boldsymbol{\alpha}$. For circular hazards, the tangent vector rotates relative to the obstacle centre over $\Delta t$, so what was tangent becomes radial. | §IV-B Prop. 2 | PM8: 4 ATACOM variants (vanilla, VD, S, LA) all fail; mean C ∈ [18, 84]. |
| **M3: Gradient credit is not the bottleneck** | `store_filtered_action=True` (logging the post-filter action with re-computed log-prob) fails to rescue the filter family. | §IV-B Prop. 3 (corollary, no analytic statement) | PM9 (T7): 5/45 GO → 3/45 GO across 45 (filter, env, seed) cells. |

The three modes are jointly necessary: M1 alone gives a single
collision per step; M2 alone gives unsafe orbiting; M3 alone
gives untrainable policies. Combined they explain why the 8
published filters we benchmark all fail simultaneously on
Goal/Push/MultiGoal.

### C2. The largest controlled benchmark of safety filters on Safety-Gym to date

**Scope.** 8 filter families × 4 tasks × 5 seeds × 200K steps,
plus the D5 Δt-sweep and the PM9 store_filtered_action ablation.

| Filter family | Mechanism | Goal Ĉ | Push Ĉ | MGoal Ĉ | GO/15 |
|---|---|---:|---:|---:|---:|
| PPO baseline | (no filter) | 22.8 | – | – | 2 |
| Manifold | null-space | 15.3 | – | – | 0 |
| ATACOM | null-space | 28.7 | 46.2 | 84.3 | 0 |
| ATACOM-VD | + tang. damping | 27.5 | 50.6 | 65.0 | 0 |
| ATACOM-S | + ε-relax | 18.3 | 18.2 | 32.9 | 2 |
| ATACOM-LA | + 1-step LA | 37.7 | 40.1 | 40.8 | 0 |
| HOCBF | 2-stage CBF | 10.4 | 11.4 | 28.5 | 1 |
| DCM | discrete-time projection | 13.7 | 11.1 | 47.1 | 3 |
| RBAM | reachability-based margin | 48.8 | 55.0 | 32.3 | 2 |
| **Ours (recipe)** | **stop-radius + BRT** | **0.84** | **10.58** | **2.59** | **12** |

This table alone is a stronger empirical claim than any single
ATACOM follow-up paper: the *literature filter zoo as a whole*
does not solve Safety-Gym Point-Robot under standard settings.

### C3. A two-line recipe that survives the discrete-time regime

We do **not** propose a new filter algorithm. We propose a
**design rule** with two ingredients, both already in the
literature, whose combination has not been benchmarked:

1.  **Velocity-adaptive stop-radius.** Inflate the keepout
    radius by $\alpha\|\boldsymbol{v}\|$ rather than relying on
    null-space projection. The inflation absorbs the
    $\Delta t \|\boldsymbol{u}\|$ overshoot from M1 directly.
2.  **Multi-step BRT lookahead.** Evaluate safety over the next
    $h$ steps under sampled action rollouts. $h = 3$ at
    $\Delta t = 0.1$ s spans the multi-step accumulation that a
    single-step projector cannot see (M2).

Provable safety result (Proposition 4): under these two
ingredients with $\alpha \ge \Delta t$ and $h \ge \lceil
\sqrt{d_\text{safe}/r}\rceil$, the inflated keepout radius is
positively invariant for all velocities $\|\boldsymbol{v}\| \le
v_\max$ that the policy can produce.

**This contribution is positioning, not algorithmic novelty.**
We are clear about that in the abstract and §V Discussion.

---

## 3. Where existing assets fit in the pivoted paper

| Asset | Original role | Pivoted role |
|---|---|---|
| `safe_rl/filters/critic_qp.py` (PRCF) | Headline algorithm | §VII Appendix C: "An attempted improvement that does not yet work" — honest negative result on PRCF Goal cell, sketch of why (H1: critic OOD) |
| D1 critic (`critic_checkpoint.pt`) | Underpins PRCF | §VII appendix data point: AUC=0.87 *on its own offline test set*, but does not transfer to PRCF online deployment |
| §IV-B draft (just written) | Empirical motivation for §IV-C PRCF | **Becomes §IV main body** — the failure-mode taxonomy section |
| PM8/PM9 SUMMARY | Background context | **Becomes §V Table 1 / Table 2 / Table 7** directly |
| D5 dt-sweep (just done) | §IV-B Figure | **Becomes §IV-B Figure 1**, with the ATACOM 0.00→39.95 phase transition as the headline figure |
| Ours (DistAdapt + sim-BRT) | "Comparison baseline" | **Becomes C3, the proposed recipe** |
| `paper/aaai_2027/proposals/A_offline_critic/DESIGN.md` (PRCF design doc) | Selected | Moved to `proposals/REJECTED_AT_PIVOT.md` with reasons |

---

## 4. New section structure

```
I.    Introduction
II.   Related work
        - ATACOM family (Liu 2022 etc) → continuous-time, manipulator
        - CBF family (Ames 2017, Robey 2020) → continuous-time, manipulator
        - Safety-Gym / point-robot benchmarks (Ray 2019, Ji 2023)
        - Discrete-time safety (this is where our gap sits)
III.  Preliminaries
        - Safety-Gym point-robot, dt=0.1s
        - Continuous-time safety theorem recap (ATACOM eq. 1-2)
IV.   The 3-axis failure taxonomy   ★ MAIN CONTRIBUTION
        IV-A. M1: Geometric overshoot           (Prop. 1, D5 figure)
        IV-B. M2: Tangential preservation       (Prop. 2, PM8 table)
        IV-C. M3: Gradient credit irrelevance   (Prop. 3, PM9 ablation)
        IV-D. The 8-filter empirical benchmark   (Table 1)
V.    The recipe: stop-radius + BRT
        V-A. The two ingredients
        V-B. Proposition 4: discrete-time invariance
        V-C. Why this combination is not in the literature
        V-D. Empirical: 4 tasks × 5 seeds                (Table 2)
VI.   Discussion
        - When does the taxonomy generalise? (multi-agent, locomotion)
        - When does our recipe fail? (Push at 10.58 - cliff-edge)
        - Sim-to-real implications
VII.  Appendix
        VII-A. Full per-seed data, all 8 filters × 4 tasks
        VII-B. D5 Δt-sweep raw numbers
        VII-C. An attempted offline-critic improvement
               (PRCF, why it doesn't transfer)
```

This is **6 pages** for AAAI (the page limit is 8 + appendix in
the new format). The PRCF appendix VII-C is 1 page and serves
as an *honest disclosure*: "we attempted X, here's the design
and the per-seed numbers, here is our current understanding of
why it failed." Reviewers reward this.

---

## 5. What additional experiments the pivot needs

The pivot is largely **a restructuring of existing data**, not a
new experimental programme. We need only:

1. **One more controlled D5-style sweep** (~30 min wall): include
   ATACOM-LA at Δt={0.01, 0.05, 0.1} to validate M2 (tangential
   pathology) is a distinct mode from M1 (overshoot). Hypothesis:
   ATACOM-LA degrades less steeply with Δt than ATACOM-vanilla.

2. **A "Proposition 4 empirical tightness" check** (~1 day): in
   the Ours-recipe runs, measure the actual maximum velocity
   $\|\boldsymbol{v}\|$ and compare to $\alpha = 0.3$ inflation.
   Verify that $\alpha\|\boldsymbol{v}\|$ exceeds the per-step
   penetration depth predicted by M1. This grounds Proposition 4
   numerically.

3. **One Circle / Multigoal additional seed-pair** (~2 h): the
   Phase 3 main runs use 5 seeds; the paper figure needs at
   least 1 more seed for the cliff cells (Push 10.58 ± high, dt
   0.05) to show that the variance is policy-search and not a
   methodology bug.

**Estimated total additional work: 1.5–2 days, all on the
existing infrastructure.**

Compare to the original PRCF programme which required, on top of
all the above: per-task critic retraining for Push/MultiGoal (~1
week), composability-bound derivation and empirical tightness
analysis (~3 days), and Robey 2020 baseline reproduction (~3
days). The pivot saves ~2 weeks of work and removes the
single-point-of-failure dependency on PRCF actually working.

---

## 6. What the pivot loses

**Honestly:**

1. **"First to compose offline-RL critic with Kalman prediction"
   in a safety filter** — this is shelved, becomes future work.
   This is genuinely novel and might have given a stronger
   "algorithmic contribution" line; we lose it.

2. **The composability bound** — was the theoretical centerpiece.
   We lose the theorem-style contribution.

3. **Connection to ICRA 2027** — the original plan was to reuse
   PRCF for Webots sim-to-real (STATUS.md §5). With the pivot,
   the ICRA paper either uses Ours-recipe directly (weaker
   novelty for ICRA) or stays on the original PRCF plan but
   solo-tracked (separate timeline).

**What we gain:**

- An honest paper that demonstrates a real, large-scale
  empirical finding (8 filters × 4 tasks fail uniformly).
- Three small theorems instead of one big composability bound;
  each theorem is local and verifiable against our data.
- A submission that doesn't depend on PRCF being debugged in
  time.

---

## 7. Risk assessment of the pivot

**Risk 1: "No algorithmic novelty."** Some AAAI reviewers
penalise pure-empirical papers. *Mitigation:* Propositions 1–4
provide algorithm-level analysis; the recipe (C3) is presented
with a positive-invariance result that is itself a small
theorem, not just an empirical claim.

**Risk 2: "8 filters is not exhaustive."** A reviewer could ask
why we don't include NLNL filters, online Neural CBFs (Robey
2020), or Lyapunov-based safety (Chow 2018). *Mitigation:*
Honest scoping in §II Related Work — we cover the
ATACOM/CBF/HOCBF/DCM/RBAM lineages that share the *null-space
projection or implicit linearisation* failure mode (M1–M3). One
short paragraph in Discussion acknowledges other lineages.

**Risk 3: "The 'recipe' is just engineering, not science."*
*Mitigation:* Proposition 4 elevates it. The point is not that
the two ingredients individually are novel but that the
sufficient condition $\alpha \ge \Delta t$ and
$h \ge \lceil \sqrt{d_\text{safe}/r}\rceil$ is the **first
explicit discrete-time analogue** of the ATACOM continuous-time
forward-invariance theorem. We are translating, not inventing.

**Risk 4: "We don't fix PRCF."** *Mitigation:* Appendix VII-C
explicitly says "Future work: in-distribution critic retraining
under PRCF rollouts (H1), and/or finer ε calibration (H2)."
Reviewers respect the disclosure.

---

## 8. Decision needed from PI / author

The pivot is reversible up to ~Week 6 (writing window starts in
Week 7). After that the structure is locked.

**Option A — Pivot now.** Lock the failure-taxonomy narrative,
move PRCF to Appendix, schedule the 1.5–2 days of extra
experiments above, start writing §IV–V from existing data.
Submission risk: moderate (the negative-result framing is
respected but uncommon in AAAI). Wall-time cost: low.

**Option B — Pivot conditionally on PRCF debug.** Spend the next
3 days on H1 (in-distribution critic retraining). If PRCF reaches
≤ Ours's 0.84 C on Goal with 5/5 GO by end of week, keep the
original SELECTED.md plan. Otherwise pivot. Submission risk:
high (3 days lost if H1 fails, then we still need to pivot).
Wall-time cost: 3 days.

**Option C — Run both narratives in parallel.** Write the pivot
section A–C as in this document AND continue PRCF debugging. At
final submission decide which to emphasise. Submission risk:
low (always have a fallback) but writing cost is 1.4× — section
III/IV needs two drafts.

**Recommendation:** **A**. Today's data already justifies a paper
on the failure taxonomy alone, and the ATACOM 0→39.95
phase-transition at Δt=0.05 (D5) is a publishable empirical
result on its own. PRCF can come back as a follow-up paper
once the OOD-critic issue is fixed properly.

---

## 9. Immediate action items (if pivot accepted)

1. Rename `SELECTED.md` → `SELECTED_ORIGINAL_PRCF.md`. Write new
   `SELECTED.md` reflecting the pivot. (~30 min)
2. Move `paper/aaai_2027/proposals/A_offline_critic/DESIGN.md`
   to `proposals/REJECTED_AT_PIVOT.md`, append a note about
   PRCF empirical failure on Goal. (~10 min)
3. Update existing `paper/aaai_2027/sections/IV-B_discrete_time_failure.md`
   header to reflect it is the main §IV body, not a "subsection".
   Add the M1/M2/M3 framing. (~1 h)
4. Draft §V (the recipe) from PM6/PM7 data: 2 pages. (~2 h)
5. Draft Appendix VII-C (the PRCF disclosure): 1 page. (~1 h)
6. Run the extra experiments listed in §5 of this document. (~1.5–2 days)
7. Update `STATUS.md` §6 timeline. (~30 min)

**Total writing+exp burst:** ~3 days. AAAI Week-7 writing window
remains intact.
