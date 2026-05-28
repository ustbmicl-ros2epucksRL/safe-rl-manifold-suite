# AAAI 2027 — Selected Direction (algorithm + taxonomy)

**Title (working).** *Iterative In-Distribution Critic Filtering for
Safe RL: Closing the Discrete-Time Gap.*

Alternative working title: *PRCF: Predictive Reachability-Critic
Filtering with Iterative Self-Improvement.*

**Pivot date.** 2026-05-15, corrected 2026-05-16 (this is the
**third** strategic pivot — see `SELECTED_ORIGINAL_PRCF.md` for the
2026-05-01 first selection, `SELECTED_AAAI_TAXONOMY_PIVOT.md` for
the 2026-05-14 taxonomy pivot, and `FALLBACK_NARRATIVE.md` for the
chain of decisions).

**Pivot trigger.** PI judgment that a diagnostic-only paper, while
publishable, does not deliver a *real safe-RL algorithm* — the
stated goal of an algorithm contribution. Reverting toward an algorithm-centric
paper. Target remains **AAAI 2027**; the fixed 2026-08-15 deadline
calculation was dropped in favour of "submit when ready within the
AAAI 2027 calendar". The previous edit briefly retargeted NeurIPS
2027 by mistake; corrected back on 2026-05-16 per PI directive.

---

## 1. One-line summary

Safety filters built from continuous-time hard-constraint theory
fail in discrete-time mobile-agent RL via three structural failure
modes (M1–M3, §IV). We propose a learning algorithm that closes
this gap: an **iterative in-distribution critic filter (IICF)**
that alternates between PPO+filter rollout collection and IQL
critic retraining on the filter-induced state distribution, until
$V_c$ stops shifting. The result is the first safety filter for
this regime with provable discrete-time forward invariance
(Prop. 4) and an in-distribution learned value function.

The paper makes four contributions:

1. **A 3-axis failure-mode taxonomy** (M1 geometric overshoot,
   M2 tangential preservation pathology, M3 gradient credit
   irrelevance) characterising when tangent-projection safety
   filters fail under coarse $\Delta t$.

2. **The 10-method controlled benchmark on Safety-Gym**:
   the largest cross-family safety-filter comparison to date
   (Family 1 tangent-projection + Family 3 Lagrangian, 5 seeds
   × 200K × 4 tasks).

3. **A discrete-time forward-invariance theorem (Prop. 4)** and
   a recipe (DistAdapt + sim-BRT) that satisfies it as a
   simple baseline; the recipe lands 12/15 GO where the rest
   of the field reaches 0–3/15.

4. **The IICF algorithm**: a critic-CBF-QP safety filter with
   an iterative offline→online retraining loop that resolves
   the in-distribution shift problem (H1 in our PRCF
   post-mortem). IICF inherits the composability bound
   ($c \le \epsilon + \delta_Q + L_c \delta_E + O(\Delta t)$,
   §V-C) from the original PRCF design and adds the iterative
   convergence guarantee.

The contribution claim is that **(1)–(3) motivate (4) and (4)
makes (1)–(3) actionable**. The diagnostic taxonomy alone is a
useful framework; the algorithm alone is an offline-RL trick;
together they form a complete story for safe RL at coarse
$\Delta t$.

---

## 2. Why this pivot now (post-mortem of the AAAI direction)

The AAAI 2027 taxonomy pivot (2026-05-14) was driven by deadline
pressure: PRCF T8 returned 0/5 GO; fixing PRCF required ~3 weeks
that the AAAI 8/15 deadline could not absorb without putting
the whole paper at risk. The taxonomy paper was a defensive
move — submission-feasible, but limited to diagnostic
contribution.

The longer-term goal for this work is a *real safe-RL algorithm*.
A diagnostic-only paper does not contribute that. With AAAI
out of scope, we have ~3 months to the typical AAAI 2027
deadline (mid-August 2026) and ~6–8 months to camera-ready —
enough to:

- finish the A-revival implementation (~3-5 weeks),
- iterate critic retraining to convergence (~2-3 weeks),
- benchmark IICF against Online Neural CBF (Robey 2020) and
  PPO-Lag (extended to 1M) for proper comparison (~2 weeks),
- formalise the composability bound + convergence proof
  (~1-2 weeks),
- write and revise the paper (~6-8 weeks).

Total ~3-4 months of focused execution + 4-6 months of buffer.
AAAI 2027 deadlines are typically tight; the buffer is needed.

---

## 3. Algorithm summary (§V-A in the paper)

**Inputs:** initial offline dataset $D_0$ (D1 collection,
50K transitions), warm-start critic $V_c^{(0)}$, PPO config,
rollout chunk $K = 50{,}000$, retrain iterations $L = 30{,}000$,
outer rounds $R = 4$ with early stop.

**Outer loop:** for $r = 1, \ldots, R$:

1. Calibrate $\epsilon_r$ = 75th percentile of
   $V_c^{(r-1)}(s)$ over safe states in $D_{r-1}$.
2. Run PPO + PRCF for $K$ env steps with filter
   $\boldsymbol{a}^\star = \arg\min \tfrac{1}{2}
   \|\boldsymbol{a} - \boldsymbol{a}_{\text{nom}}\|^2$ s.t.
   $V_c^{(r-1)}(\hat{\boldsymbol{s}}_{t+1\mid t}(\boldsymbol{a})) \le \epsilon_r$;
   collect rollouts $D'_r$.
3. Concatenate: $D_r = D_{r-1} \cup D'_r$ (optional
   importance reweighting).
4. Retrain critic: $V_c^{(r)} \leftarrow$ IQL on $D_r$ for $L$
   updates, warm-started from $V_c^{(r-1)}$.
5. Evaluate; early-stop if cost-GO and critic-AUC plateau.

**Convergence argument (informal):** the iteration is a
fixed-point on the filter-induced state distribution
$\rho_r$. At $r$, $V_c^{(r-1)}$ is mis-calibrated on $\rho_r$;
retraining on $D'_r$ reduces the mis-calibration; the new
$V_c^{(r)}$ induces a $\rho_{r+1}$ closer to the safe mode.
The fixed point exists where $V_c^{(r)}$ is calibrated on the
state distribution the filter induces. Not provably convergent
in general but empirically stable in analogous offline-to-online
settings (Lee et al. 2022, Wang et al. 2021).

**Composability bound (§V-C, recovered from original PRCF
design).** For a Lipschitz critic and EKF predictor:

$$
c(\boldsymbol{s}_{t+1}) \;\le\; \epsilon + \delta_Q + L_c\,\delta_E + O(\Delta t).
$$

The four terms are independently measurable from data: $\epsilon$
from the calibration, $\delta_Q$ from the IQL test loss, $L_c$
from gradient norms of $V_c$, $\delta_E$ from EKF posterior
covariance. The bound replaces ATACOM's continuous-time
$\dot{\boldsymbol{c}} = -K_c\boldsymbol{c}$ guarantee with a
discrete-time inequality that **does not require continuous
dynamics**.

---

## 4. Section structure

```
I.    Introduction (existing draft, needs algorithm reframe)
II.   Related work (existing draft, expand A-revival section)
III.  Preliminaries
        III-A. Safety-Gym, ATACOM continuous-time theorem
        III-B. IQL expectile regression for safety critics
IV.   The 3-axis failure taxonomy (existing draft, KEEPS)
        IV-A. M1: Geometric overshoot
        IV-B. M2: Tangential preservation pathology
        IV-C. M3: Gradient credit irrelevance
        IV-D. The 10-method empirical benchmark
V.    The IICF algorithm   ★ NEW MAIN CONTRIBUTION
        V-A. The two components: critic-CBF-QP + iterative retrain
        V-B. The outer loop algorithm
        V-C. Composability bound (Theorem 1)
        V-D. Convergence argument (Theorem 2 informal)
        V-E. The simpler recipe baseline (Prop. 4 ex-§V; reduced
              to baseline status)
VI.   Experiments
        VI-A. IICF on Goal/Push/MultiGoal (Tables)
        VI-B. Convergence across rounds (per-round R/C curves)
        VI-C. Composability bound empirical tightness
        VI-D. Comparison to Online Neural CBF (Robey 2020) and
              PPO-Lag (1M steps)
VII.  Discussion (existing draft, reframe for algorithm story)
VIII. Conclusion
Appendix:
        VIII-A. Full per-seed data
        VIII-B. D5 Δt-sweep raw numbers
        VIII-C. The unsuccessful single-shot PRCF (T8 data, the
                "before fix" snapshot — frames IICF as the
                solution)
```

Page budget: AAAI allows 7 pages main (+ references; 2026 onwards
the call has been 8 pages including everything). Confirm against
the AAAI 2027 specific call when published.
Target: 9 pages with §V occupying ~3.5 pages, §IV ~2.5 pages,
§VI ~2 pages, others ~1 page collectively.

---

## 5. What changes for each existing section

| Section | Pre-pivot status | Post-pivot action |
|---|---|---|
| §I Intro | "diagnostic + recipe" framing | Re-frame around IICF as algorithm contribution; M1-M3 motivate it; recipe is the simple baseline |
| §II Related work | already covers 3 families | Expand A-revival positioning: iterative offline→online (Wang 2021, Lee 2022) is the established fix for the value-function-shift problem |
| §IV Taxonomy | M1/M2/M3 propositions | **No change**, this is the diagnostic backbone |
| §V Recipe | Prop. 4 + DistAdapt+BRT | **Reduce to §V-E baseline**; new §V-A/B/C is IICF |
| §VI Discussion | brittleness + scope + sim2real | Trim brittleness section to half (it was a Tier-2 defense, less needed now); add IICF-specific failure modes |
| §VII-C PRCF appendix | negative result documentation | **Repurpose**: this becomes Appendix VIII-C, the "before fix" snapshot that motivates IICF |

---

## 6. Concrete next-step deliverables

Working backward from the AAAI 2027 submission window
(deadline typically mid-August 2026; if AAAI 2027 publishes a
phase-2 submission window in late 2026, the schedule can shift
accordingly without changing the venue):

| Week | Dates (2026-27) | Output | Status |
|---|---|---|---|
| 0 | 5/15 – 5/22 | Plan-doc updates + rollout collection callback (~80 LoC) | 🟡 in progress |
| 1 | 5/23 – 5/29 | IQL retrain wrapper enhancement (~60 LoC); A-revival orchestrator skeleton (~250 LoC) | ⏳ |
| 2 | 5/30 – 6/5 | Smoke test single seed (50K × 2 rounds); verify the loop converges in principle | ⏳ |
| 3 | 6/6 – 6/12 | Goal cell at full budget (5 seeds × 200K × R=4 = ~16h wall at -P 5) | ⏳ |
| 4 | 6/13 – 6/19 | Push + MGoal cells in parallel; per-task D1 collection where needed | ⏳ |
| 5-6 | 6/20 – 7/3 | First convergence-curve plots; H2/H3/H4 cheap ablations | ⏳ |
| 7-8 | 7/4 – 7/17 | Robey 2020 Neural CBF baseline port + run | ⏳ |
| 9-10 | 7/18 – 7/31 | PPO-Lag extended (1M steps) baseline | ⏳ |
| 11-13 | 8/1 – 8/21 | Composability bound: formalisation + empirical tightness measurement | ⏳ |
| 14-18 | 8/22 – 9/25 | Paper §V (algorithm) + §VI (experiments) drafting | ⏳ |
| 19-22 | 9/26 – 10/24 | Full draft + internal review round 1 | ⏳ |
| 23-26 | 10/25 – 11/22 | Review-1 revisions + ablation gaps | ⏳ |
| 27-32 | 11/23 – 1/3/2027 | Internal review round 2 + final pass | ⏳ |
| 33-44 | 1/4 – 3/29/2027 | **Buffer** for unforeseen experimental issues | ⏳ |
| 45-50 | 3/30 – 5/3/2027 | Camera-ready buffer; supplementary material; reproducibility scripts | ⏳ |
| 13-15 | 8/8 – 8/22 | **AAAI 2027 submission** (~mid-August target; exact dates per AAAI call) | ⏳ |

Total experimental work: ~10 weeks of focused execution + 14
weeks of writing + 30 weeks of buffer. Buffer is intentionally
large because (a) safe RL has known reproducibility issues that
eat time, and (b) the iterative critic might need multiple
debugging rounds beyond the initial H1 fix.

---

## 7. Risks specific to this longer-form paper

| Risk | Probability | Mitigation |
|---|---|---|
| **IICF still NO-GO after iteration** (H1 alone insufficient) | Medium | Run H2 (ε quantile), H3 (full autograd), H4 (V vs Q mode) as parallel ablations in Week 5-6. Fall back to Direction B (Learnable Recipe) if IICF stalls — paper retains §IV taxonomy core |
| **Compute budget**: A-revival is 10× T8 compute, +baselines, ~150-200 GPU-hour-equivalent | Low | Already running on CPU box; iros2026 env handles. Spread runs over weeks 2-10. |
| **Single-author work pace** (no co-author execution) | High | Stick to weekly milestones in §6; replan at 6-week boundaries |
| **AAAI Safe-RL niche thinness** (reviewer pool) | Medium | Frame paper for ML/RL audience, not robotics — emphasise the iterative offline→online aspect as ML contribution. NeurIPS 2027 (May 2027 deadline) and ICLR 2028 (Sep 2027) as backup venues if AAAI 2027 doesn't land |
| **L4DC or ICRA accept-rate trade** | Low | Smaller-venue fallbacks if both AAAI 2027 and NeurIPS 2027 reject |

---

## 8. Connections back to prior history

This is the **third selection**. The prior two are preserved:

- `SELECTED_ORIGINAL_PRCF.md` (2026-05-01): chose PRCF as
  algorithm; rejected when T8 returned 0/5 GO.
- `SELECTED_AAAI_TAXONOMY_PIVOT.md` (2026-05-14): pivoted to
  diagnostic taxonomy paper for AAAI 2027 deadline; rejected
  when PI judged that a diagnostic-only paper does not satisfy
  the longer-term goal of contributing a real safe-RL
  algorithm.
- `SELECTED.md` (this file, 2026-05-15): combines both — the
  taxonomy is the diagnostic that motivates IICF; IICF is the
  algorithm that the diagnostic prescribes.

`FALLBACK_NARRATIVE.md` documents the original pivot logic and
remains valid as the diagnostic-only fallback if the
algorithm work cannot
absorb the algorithm work in time. In the worst case we
re-pivot to AAAI 2028 (deadline 2027-08) with the diagnostic-only
framing.

`proposals/A_offline_critic/REJECTED_AT_PIVOT.md` documents
the H1 root cause from the original PRCF rejection.

`proposals/A_revival_iterative_critic/DESIGN.md` is the
algorithm design doc; **this SELECTED.md is its execution
mandate**.

---

### TODO at this commit

- [x] Paper directory: `paper/aaai_2027/` (reverted 2026-05-16
      after a brief mis-rename to `neurips_2027/`)
- [x] Preserve old SELECTED.md as `SELECTED_AAAI_TAXONOMY_PIVOT.md`
- [x] Write the new SELECTED.md (this file)
- [ ] Update STATUS.md §6 timeline with new deadlines
- [ ] Start A-revival pipeline: rollout collection callback
- [ ] Tag git history with `aaai_pivot_point` for easy rollback
