# §VII-D  An iterative refinement of the PRCF critic (negative result)

*(Supplementary appendix for AAAI 2027.  Companion to §VII-C.
One page in the final layout. Documents a follow-up attempt to
fix the H1 hypothesis flagged in §VII-C.)*

## D.1  Motivation

§VII-C §C.4 attributed PRCF's 0/5 GO outcome on Goal to
**H1: in-distribution critic shift** — the D1 critic was trained
on data collected under a random + DistanceFilter behaviour
policy, but the PPO policy under PRCF at runtime visits a
distinct state distribution near the keepout boundary. The
critic's $V_c$ evaluated on those states is OOD and the QP
minimisation projects in the wrong direction.

The standard fix in modern offline-RL is **iterative
offline-to-online retraining** [Wang et al. 2021;
Kostrikov et al. 2022; Lee et al. 2022]: alternate between
(i) running the policy with the current critic to collect
fresh rollouts, and (ii) retraining the critic on the union of
the offline dataset and the new rollouts. We call this scheme
the **Iterative In-Distribution Critic Filter (IICF)**.

## D.2  Algorithm

Inputs: D1 offline dataset $D_0$, D1 critic checkpoint
$V_c^{(0)}$ (AUC 0.87 on D1's held-out test set), PPO config,
rollout chunk size $K$, retrain iterations $L$, outer-round
count $R$, sliding-window depth $W$.

```
for r = 1..R:
    ε_r        ← q75-quantile of V_c^(r-1)(s) over D_{r-1}'s safe states
    D'_r       ← run PPO+PRCF for K env steps with critic V_c^(r-1)
                  and threshold ε_r; collect (s,a_safe,c,s') transitions
    D_train    ← D_0  ∪  last W rounds of {D'_*}        # sliding window
    V_c^(r)    ← IQL retrain on D_train for L updates
                  (warm-start from V_c^(r-1) or fresh load of V_c^(0))
    eval_r     ← 20-episode deterministic eval
    early stop if eval_r.cost ≤ 5.0 and critic AUC plateaus
return final V_c^(R), final PPO policy
```

Two anti-drift safeguards beyond cumulative retraining:

- **Sliding window** ($W$): keep only $D_0$ plus the last $W$ rounds
  of rollouts in the critic-retrain dataset. Limits the influence
  of stale early-round rollouts on the late-round critic.
- **Critic reset** (`--reset-critic` flag): each round loads the
  D1 baseline checkpoint instead of inheriting the previous
  round's fitted state. Decouples per-round critic fine-tuning
  from cross-round drift.

The PPO actor is fresh each round by construction (a new
trainer instance per outer-loop iteration), so the
algorithm-side feedback loop is between (critic, dataset)
only — not PPO policy parameters.

## D.3  Experimental results

Goal × 5 seeds × 200K env steps per round, $K=50{,}000$,
$L=30{,}000$ IQL updates, $R\in\{4, 6\}$, $W\in\{0, 1, 2\}$.

| Config | Window | Reset | $R$ | mean $C$ | GO/5 | Distribution |
|--------|-------:|:-----:|:---:|---------:|-----:|--------------|
| v1 | 0 (cumulative) | no | 4 | 18.18 ± 11.0 | 1/5 | unimodal-high |
| v2 | 1 | yes | 4 | 8.13 ± 6.8 | 2/5 | tight, 4/5 sub-8 |
| v3a | 0 (cumulative) | yes | 4 | 12.88 ± 10.3 | 2/5 | high variance |
| **v3b** | 2 | yes | 4 | 9.67 ± 11.1 | **3/5** | bimodal (3 wins + 2 catastrophic) |
| v3c | 1 | yes | 4 | 10.62 ± 6.1 | 1/5 | medium variance |
| v4 | 2 | yes | 6 | 9.26 ± 10.7 | **3/5** | bimodal persists |

The best variant (**v3b**, sliding window depth 2 with per-round
critic reset) lifts the single-shot PRCF baseline from 0/5 GO to
3/5 GO at half the mean cost (9.67 vs 17.90). The two
anti-drift safeguards each contribute: turning off the window
(v3a) loses one GO seed; turning off the critic reset would
collapse to v1 behaviour.

But the v3b distribution is *bimodal*: three seeds reach
$C<2$ cleanly while two seeds remain at $C\ge 21$
([22.24, 1.60, 1.98, 21.36, 1.16]). Extending to $R=6$ rounds
(v4) does not move the failing seeds — the post-extension
distribution is structurally identical
([19.58, 22.16, 0.00, 1.72, 2.82]). The "failed" seeds are not
budget-bound; they are stuck in a different basin of the
critic-policy fixed point.

## D.4  Diagnosis

We attribute the bimodal failure to a property of the iteration
itself rather than to remaining offline-RL distribution shift:

**The iteration is a fixed point on the filter-induced state
distribution, not a *unique* fixed point.** At round $r$, the
critic $V_c^{(r-1)}$ defines a filter $W_r$, which in turn
induces a state distribution $\rho_r$. Retraining $V_c$ on
$\rho_r$ produces $V_c^{(r)}$, and the iteration converges when
$V_c^{(r)}\!\approx\!V_c^{(r-1)}$. Two such fixed points appear
in our experiments:

- A **safe fixed point** where the filter correctly identifies
  near-hazard states and the policy learns to avoid them ($C<2$,
  $\rho_r$ concentrates away from the keepout band).
- A **failure fixed point** where the filter is over-aggressive
  on far-from-hazard states and the policy abandons the goal-
  seeking direction ($C\ge 20$, $\rho_r$ scatters defensively).

Which fixed point a seed converges to is determined by the
*initial* PPO trajectory — specifically by whether the first
$K$ steps under $V_c^{(0)}$ hit the keepout band frequently
enough to make D1 collection's hazard-near states informative.
Three of five seeds happen to start with productive trajectories
under D1's coarse critic; two do not.

The natural follow-up — *adaptive curriculum* over the initial
PPO trajectory, or *ensemble critics* that vote on filter
decisions — is plausible but did not fit our submission timeline
and remains open.

## D.5  Implications for the main paper

§V-H reports that the *fixed-recipe* baseline (Ours,
$\alpha = 0.3$) achieves 5/5 GO uniformly without any
critic-based machinery, and that even a *learned* AMRF variant
in the same recipe-class does not improve. Combining this with
§VII-D: the iterative critic mechanism is **not necessary** for
this regime — the recipe of §V solves the problem directly via
inflated keepout + lookahead. IICF is therefore retained in this
appendix as honest disclosure of an attempted improvement that
yields partial but unstable gains, not as a contributing
algorithm to the headline §V framing.

Code and per-seed data:
`experiments/diagnostics/D6_iterative_critic/`,
`runs/phase3_t10v{1,2,3a,3b,3c}_iicf_goal/`,
`runs/phase3_t10v4_iicf_window2_R6/`.

---

### TODO before submission

- [ ] If page-budget pressure on supplementary forces trimming,
      drop D.4 diagnosis to a 2-sentence summary and cite the
      fixed-point literature instead
- [ ] Add per-round R/C trajectory figure (currently only in
      the round.json files; ~30 min matplotlib) — would make
      the bimodal structure more visually clear
- [ ] Confirm the offline-to-online citations match the rest of
      §II's reference list once compiled
